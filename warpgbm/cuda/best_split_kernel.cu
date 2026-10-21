#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Histograms carry B = Bv + 1 bins per feature: Bv value bins followed by one NaN bin.
// Split candidates per (era, feature) are laid out as [2 * Bv]:
//   c in [0, Bv)      : left = value bins <= c,                NaN goes right
//   c in [Bv, 2 * Bv) : left = value bins <= c - Bv + NaN bin, NaN goes left
// The gain of a candidate is the sum over outputs k of the usual second-order gain; outputs whose
// hessian sum on either side is below min_sum_hessian do not contribute. The direction is
// +1 if every contributing output has left leaf > right leaf, -1 if every contributing output has
// left leaf <= right leaf, and 0 otherwise (or for invalid candidates).
//
// One block per (feature, era), one thread per candidate. The feature's histogram is staged in
// shared memory when it fits; prefix sums are accumulated in bin order, so results are
// deterministic.
__global__ void directional_split_kernel(
    const float *__restrict__ G,         // [E, F, B, K]
    const float *__restrict__ H,         // [E, F, B, K]
    const float *__restrict__ C,         // [E, F, B]
    int E, int F, int B, int K,
    float min_child_weight,
    float min_sum_hessian,
    float eps,
    bool use_shared,
    float *__restrict__ per_era_gain,     // [E, F, 2 * (B - 1)]
    float *__restrict__ per_era_direction // [E, F, 2 * (B - 1)]
)
{
    const int f = blockIdx.x; // feature index
    const int e = blockIdx.y; // era index
    const int Bv = B - 1;
    const int nan_bin = Bv;
    const int64_t base_bins = (static_cast<int64_t>(e) * F + f) * B;
    const int64_t base_out = (static_cast<int64_t>(e) * F + f) * (2 * Bv);

    const float *g = G + base_bins * K; // [B, K]
    const float *h = H + base_bins * K; // [B, K]
    const float *c = C + base_bins;     // [B]

    extern __shared__ float sh[];
    if (use_shared)
    {
        float *sh_g = sh;
        float *sh_h = sh + B * K;
        float *sh_c = sh + 2 * B * K;
        for (int i = threadIdx.x; i < B * K; i += blockDim.x)
        {
            sh_g[i] = g[i];
            sh_h[i] = h[i];
        }
        for (int i = threadIdx.x; i < B; i += blockDim.x)
            sh_c[i] = c[i];
        __syncthreads();
        g = sh_g;
        h = sh_h;
        c = sh_c;
    }

    const int cand = threadIdx.x;
    if (cand >= 2 * Bv)
        return;

    const int nan_left = cand >= Bv;
    const int t = cand - nan_left * Bv;

    // ── Child weight constraint (independent of outputs) ──
    float C_total = 0.0f;
    for (int b = 0; b < B; ++b)
        C_total += c[b];
    float C_L = 0.0f;
    for (int b = 0; b <= t; ++b)
        C_L += c[b];
    const float C_nan = c[nan_bin];

    // Histograms obtained by subtraction may carry float noise in bins that are empty in this
    // node. Treat such bins as empty, so that thresholds which produce the same partition
    // (empty bin t == threshold t-1, NaN-left without NaNs == NaN-right) are not candidates.
    const float empty_eps = 1e-6f * fabsf(C_total);
    const float cl = nan_left ? C_L + C_nan : C_L;
    const float cr = C_total - cl;
    const bool valid = (cl >= min_child_weight) && (cr >= min_child_weight)
                    && (cl > empty_eps) && (cr > empty_eps)
                    && (c[t] > empty_eps) && !(nan_left && C_nan <= empty_eps);

    float gain = 0.0f;
    float dir = 0.0f;
    bool conflict = false;

    if (valid)
    {
        for (int k = 0; k < K; ++k)
        {
            float G_total = 0.0f, H_total = 0.0f;
            for (int b = 0; b < B; ++b)
            {
                G_total += g[b * K + k];
                H_total += h[b * K + k];
            }
            float G_L = 0.0f, H_L = 0.0f;
            for (int b = 0; b <= t; ++b)
            {
                G_L += g[b * K + k];
                H_L += h[b * K + k];
            }
            if (nan_left)
            {
                G_L += g[nan_bin * K + k];
                H_L += h[nan_bin * K + k];
            }
            const float G_R = G_total - G_L;
            const float H_R = H_total - H_L;

            if (H_L < min_sum_hessian || H_R < min_sum_hessian)
                continue;

            gain += (G_L * G_L) / (H_L + eps)
                  + (G_R * G_R) / (H_R + eps)
                  - (G_total * G_total) / (H_total + eps);

            const float left_val = G_L / (H_L + eps);
            const float right_val = G_R / (H_R + eps);
            const float d = (left_val > right_val) ? 1.0f : -1.0f;
            if (dir == 0.0f)
                dir = d;
            else if (dir != d)
                conflict = true;
        }
    }

    per_era_gain[base_out + cand] = gain;
    per_era_direction[base_out + cand] = conflict ? 0.0f : dir;
}

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B, K]
    const at::Tensor &H, // [E, F, B, K]
    const at::Tensor &C, // [E, F, B]
    float min_child_weight,
    float min_sum_hessian,
    float eps,
    at::Tensor &per_era_gain,      // [E, F, 2 * (B - 1)]
    at::Tensor &per_era_direction, // [E, F, 2 * (B - 1)]
    int threads = 128)
{
    // Run on the tensors' device and the current stream of that device; this also makes the
    // shared-memory query below refer to that device
    const at::cuda::OptionalCUDAGuard guard(device_of(G));
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&H, &C, &per_era_gain, &per_era_direction})
        TORCH_CHECK(G.is_cuda() && t->device() == G.device(), "split kernel tensors must be on the same CUDA device");
    TORCH_CHECK(G.dim() == 4 && H.sizes() == G.sizes(), "G and H must be [E, F, B, K]");
    TORCH_CHECK(C.dim() == 3, "C must be [E, F, B]");
    TORCH_CHECK(G.is_contiguous() && H.is_contiguous() && C.is_contiguous(), "histograms must be contiguous");

    int E = G.size(0);
    int F = G.size(1);
    int B = G.size(2);
    int K = G.size(3);

    TORCH_CHECK(per_era_gain.size(2) == 2 * (B - 1), "per_era_gain must have 2 * (B - 1) candidates");

    if (E == 0 || F == 0)
        return;

    // One thread per candidate (2 * (B - 1) <= 254), rounded up to full warps
    int block_threads = ((2 * (B - 1) + 31) / 32) * 32;

    int device = 0;
    cudaGetDevice(&device);
    int max_shared = 0;
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);
    int64_t shared_bytes = static_cast<int64_t>(B) * (2 * K + 1) * sizeof(float);
    bool use_shared = shared_bytes <= max_shared;

    dim3 blocks(F, E);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    directional_split_kernel<<<blocks, block_threads, use_shared ? shared_bytes : 0, stream>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        C.data_ptr<float>(),
        E, F, B, K,
        min_child_weight,
        min_sum_hessian,
        eps,
        use_shared,
        per_era_gain.data_ptr<float>(),
        per_era_direction.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM split kernel failed: ", cudaGetErrorString(err));
}
