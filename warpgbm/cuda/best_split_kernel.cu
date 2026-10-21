#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>

// Split finding for all nodes of a tree level.
//
// Histograms are [slots, F, E, B, C] with B = Bv + 1 bins (Bv value bins followed by one NaN bin)
// and C = K + Kh + 1 channels [grad | hess | count] (Kh = 0: the hessian of every output is the
// count channel). int64 histograms hold fixed-point values (see histogram_kernel.cu) and are
// converted with the inverse scales; float32 histograms are used as they are.
//
// Split candidates per (era, feature) are laid out as [2 * Bv]:
//   c in [0, Bv)      : left = value bins <= c,                NaN goes right
//   c in [Bv, 2 * Bv) : left = value bins <= c - Bv + NaN bin, NaN goes left
// The gain of a candidate is the sum over outputs k of the usual second-order gain; outputs whose
// hessian sum on either side is below min_sum_hessian do not contribute. The direction is
// +1 if every contributing output has left leaf > right leaf, -1 if every contributing output has
// left leaf <= right leaf, and 0 otherwise (or for invalid candidates).

template <typename T>
struct HistValue;

// Histogram values as float: int64 fixed point times the inverse (power-of-two) scale, which
// rounds once; float histograms as they are
__device__ __forceinline__ float hist_value(long long v, float inv_scale) { return __ll2float_rn(v) * inv_scale; }
__device__ __forceinline__ float hist_value(float v, float) { return v; }

// One block per (feature, node slot), one thread per candidate. For every era the block converts
// the feature's histogram to float and builds the prefix sums over bins of all channels in shared
// memory (in bin order, so the results are deterministic), then evaluates the candidates. Writes
//   era_gain [slots, F, E, 2Bv]  per-era gain
//   gain     [slots, F, 2Bv]     mean gain over eras
//   agree    [slots, F, 2Bv]     |mean direction| over eras
//   mono_ok  [slots, F, 2Bv]     direction == required direction in every era (if mono_req)
template <typename T>
__global__ void level_split_kernel(
    const T *__restrict__ hist,
    const uint8_t *__restrict__ node_active, // [slots]
    float *__restrict__ prefix_scratch,      // [slots, F, B, C] when the prefix does not fit in shared memory
    bool use_shared,
    int F, int E, int B, int C, int K, int Kh,
    const float *__restrict__ scales, // [3] fixed-point scales (int64 histograms)
    const int8_t *__restrict__ mono_req, // [F] required direction per feature or nullptr
    float min_child_weight,
    float min_sum_hessian,
    float eps,
    float *__restrict__ era_gain,
    float *__restrict__ gain_out,
    float *__restrict__ agree_out,
    uint8_t *__restrict__ mono_ok_out)
{
    const int f = blockIdx.x;
    const int slot = blockIdx.y;
    if (!node_active[slot])
        return;

    const int Bv = B - 1;
    const int n_cand = 2 * Bv;
    const int cand = threadIdx.x;
    const bool has_cand = cand < n_cand;
    const int nan_left = cand >= Bv;
    const int t = has_cand ? cand - nan_left * Bv : 0;
    const int cnt = C - 1;

    extern __shared__ unsigned char shraw[];
    float *pre = use_shared ? reinterpret_cast<float *>(shraw)
                            : prefix_scratch + (static_cast<int64_t>(slot) * F + f) * B * C;

    const int8_t req = mono_req ? mono_req[f] : 0;
    float gain_sum = 0.0f;
    float dir_sum = 0.0f;
    bool mono_ok = true;
    const int64_t node_feat = static_cast<int64_t>(slot) * F + f;

    for (int e = 0; e < E; ++e)
    {
        const T *h = hist + (node_feat * E + e) * B * C;
        __syncthreads(); // previous era is done with pre
        for (int c = threadIdx.x; c < C; c += blockDim.x)
        {
            const float inv = sizeof(T) == 8 ? 1.0f / (c < K ? scales[0] : (c < K + Kh ? scales[1] : scales[2])) : 1.0f;
            float run = 0.0f;
            for (int b = 0; b < B; ++b)
            {
                run += hist_value(h[b * C + c], inv);
                pre[b * C + c] = run;
            }
        }
        __syncthreads();

        if (has_cand)
        {
            // ── Child weight constraint (independent of outputs) ──
            const float C_total = pre[(B - 1) * C + cnt];
            const float C_lt = pre[t * C + cnt];
            const float C_nan = C_total - pre[(Bv - 1) * C + cnt];
            const float c_t = t > 0 ? C_lt - pre[(t - 1) * C + cnt] : C_lt;
            const float cl = nan_left ? C_lt + C_nan : C_lt;
            const float cr = C_total - cl;

            // Prefix sums (and float subtraction in fast mode) may leave noise in bins that are
            // empty in this node. Treat such bins as empty, so that thresholds which produce the
            // same partition (empty bin t == threshold t-1, NaN-left without NaNs == NaN-right)
            // are not candidates.
            const float empty_eps = 1e-6f * fabsf(C_total);
            const bool valid = (cl >= min_child_weight) && (cr >= min_child_weight)
                            && (cl > empty_eps) && (cr > empty_eps)
                            && (c_t > empty_eps) && !(nan_left && C_nan <= empty_eps);

            float gain = 0.0f;
            float dir = 0.0f;
            bool conflict = false;
            if (valid)
            {
                for (int k = 0; k < K; ++k)
                {
                    const int hc = Kh > 0 ? K + k : cnt;
                    const float G_total = pre[(B - 1) * C + k];
                    const float H_total = pre[(B - 1) * C + hc];
                    float G_L = pre[t * C + k];
                    float H_L = pre[t * C + hc];
                    if (nan_left)
                    {
                        G_L += G_total - pre[(Bv - 1) * C + k];
                        H_L += H_total - pre[(Bv - 1) * C + hc];
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
            if (conflict)
                dir = 0.0f;
            era_gain[(node_feat * E + e) * n_cand + cand] = gain;
            gain_sum += gain;
            dir_sum += dir;
            mono_ok = mono_ok && (req == 0 || dir == static_cast<float>(req));
        }
    }

    if (has_cand)
    {
        const int64_t o = node_feat * n_cand + cand;
        gain_out[o] = gain_sum / static_cast<float>(E);
        agree_out[o] = fabsf(dir_sum / static_cast<float>(E));
        if (mono_ok_out)
            mono_ok_out[o] = mono_ok ? 1 : 0;
    }
}

void launch_level_split(
    const at::Tensor &hist,        // [slots, F, E, B, C] int64 or float32
    const at::Tensor &node_active, // [slots] uint8
    const at::Tensor &scales,      // [3] float32
    const c10::optional<at::Tensor> &mono_req, // [F] int8
    float min_child_weight,
    float min_sum_hessian,
    float eps,
    int K,
    at::Tensor &era_gain, // [slots, F, E, 2Bv]
    at::Tensor &gain,     // [slots, F, 2Bv]
    at::Tensor &agree,    // [slots, F, 2Bv]
    const c10::optional<at::Tensor> &mono_ok) // [slots, F, 2Bv] uint8
{
    const at::cuda::OptionalCUDAGuard guard(device_of(hist));
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&node_active, &scales, &era_gain, &gain, &agree})
        TORCH_CHECK(hist.is_cuda() && t->device() == hist.device(), "split kernel tensors must be on the same CUDA device");
    TORCH_CHECK(hist.dim() == 5 && hist.is_contiguous(), "hist must be a contiguous [slots, F, E, B, C] tensor");
    const bool det = hist.scalar_type() == at::kLong;
    TORCH_CHECK(det || hist.scalar_type() == at::kFloat, "hist must be int64 or float32");
    TORCH_CHECK(node_active.scalar_type() == at::kByte, "node_active must be uint8");
    TORCH_CHECK(mono_req.has_value() == mono_ok.has_value(), "mono_req and mono_ok go together");

    const int slots = static_cast<int>(hist.size(0));
    const int F = static_cast<int>(hist.size(1));
    const int E = static_cast<int>(hist.size(2));
    const int B = static_cast<int>(hist.size(3));
    const int C = static_cast<int>(hist.size(4));
    const int Kh = C - 1 - K;
    TORCH_CHECK(Kh == 0 || Kh == K, "hist must have 2K + 1 or K + 1 channels");
    TORCH_CHECK(gain.size(2) == 2 * (B - 1), "gain must have 2 * (B - 1) candidates");
    TORCH_CHECK(slots <= 65535, "at most 65535 nodes per split launch");
    if (slots == 0 || F == 0 || E == 0)
        return;

    int device = 0;
    cudaGetDevice(&device);
    int max_shared = 0;
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);
    const int64_t shared_bytes = static_cast<int64_t>(B) * C * sizeof(float);
    const bool use_shared = shared_bytes <= max_shared;
    at::Tensor scratch;
    if (!use_shared)
        scratch = at::empty({slots, F, B, C}, hist.options().dtype(at::kFloat));

    // One thread per candidate (2 * (B - 1) <= 252), rounded up to full warps
    const int threads = std::max(32, ((2 * (B - 1) + 31) / 32) * 32);
    dim3 grid(F, slots);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int8_t *req = mono_req.has_value() ? mono_req->data_ptr<int8_t>() : nullptr;
    uint8_t *ok = mono_ok.has_value() ? mono_ok->data_ptr<uint8_t>() : nullptr;
    if (det)
        level_split_kernel<long long><<<grid, threads, use_shared ? shared_bytes : 0, stream>>>(
            reinterpret_cast<const long long *>(hist.data_ptr<int64_t>()), node_active.data_ptr<uint8_t>(),
            use_shared ? nullptr : scratch.data_ptr<float>(), use_shared,
            F, E, B, C, K, Kh, scales.data_ptr<float>(), req, min_child_weight, min_sum_hessian, eps,
            era_gain.data_ptr<float>(), gain.data_ptr<float>(), agree.data_ptr<float>(), ok);
    else
        level_split_kernel<float><<<grid, threads, use_shared ? shared_bytes : 0, stream>>>(
            hist.data_ptr<float>(), node_active.data_ptr<uint8_t>(),
            use_shared ? nullptr : scratch.data_ptr<float>(), use_shared,
            F, E, B, C, K, Kh, scales.data_ptr<float>(), req, min_child_weight, min_sum_hessian, eps,
            era_gain.data_ptr<float>(), gain.data_ptr<float>(), agree.data_ptr<float>(), ok);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM split kernel failed: ", cudaGetErrorString(err));
}

// ─── Best split per node ──────────────────────────────────────────────────────
// One block per node slot. With several eras, only candidates with the maximal directional
// agreement of the node qualify (computed over all candidates). A candidate qualifies if its
// gain exceeds min_split_gain and it satisfies the monotone constraints; the best is the first
// maximal gain in (feature, candidate) order. Writes best[slot] = feature * 2Bv + candidate or
// -1, and the per-era gains of the best candidate (era order) and their sum.

__global__ void select_split_kernel(
    const float *__restrict__ gain,     // [slots, n]
    const float *__restrict__ agree,    // [slots, n]
    const uint8_t *__restrict__ mono_ok, // [slots, n] or nullptr
    const float *__restrict__ era_gain, // [slots, F, E, n_cand]
    const uint8_t *__restrict__ node_active,
    int n, int n_cand, int E, bool use_agreement, float min_split_gain,
    int32_t *__restrict__ best,       // [slots]
    float *__restrict__ best_gain,    // [slots] sum over eras of the best's per-era gains
    float *__restrict__ best_era_gain) // [slots, E]
{
    const int slot = blockIdx.x;
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    const int tid = threadIdx.x;
    if (!node_active[slot])
    {
        if (tid == 0)
        {
            best[slot] = -1;
            best_gain[slot] = 0.0f;
        }
        for (int e = tid; e < E; e += blockDim.x)
            best_era_gain[static_cast<int64_t>(slot) * E + e] = 0.0f;
        return;
    }
    const float *g = gain + static_cast<int64_t>(slot) * n;
    const float *a = agree + static_cast<int64_t>(slot) * n;
    const uint8_t *m = mono_ok ? mono_ok + static_cast<int64_t>(slot) * n : nullptr;

    float max_agree = 0.0f;
    if (use_agreement)
    {
        float local = -FLT_MAX;
        for (int i = tid; i < n; i += blockDim.x)
            local = fmaxf(local, a[i]);
        s_val[tid] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
                s_val[tid] = fmaxf(s_val[tid], s_val[tid + stride]);
            __syncthreads();
        }
        max_agree = s_val[0];
        __syncthreads();
    }

    // Each thread scans its indices in increasing order, keeping the first maximum
    float bv = -INFINITY;
    int bi = -1;
    for (int i = tid; i < n; i += blockDim.x)
    {
        const float v = g[i];
        const bool ok = (v > min_split_gain) && (!use_agreement || a[i] == max_agree) && (!m || m[i]);
        if (ok && v > bv)
        {
            bv = v;
            bi = i;
        }
    }
    s_val[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            const float ov = s_val[tid + stride];
            const int oi = s_idx[tid + stride];
            if (oi >= 0 && (s_idx[tid] < 0 || ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])))
            {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }
    const int b = s_idx[0];
    if (tid == 0)
    {
        best[slot] = b;
        float total = 0.0f;
        if (b >= 0)
        {
            const int f = b / n_cand;
            const int c = b - f * n_cand;
            const float *eg = era_gain + (static_cast<int64_t>(slot) * (n / n_cand) + f) * E * n_cand + c;
            for (int e = 0; e < E; ++e)
            {
                const float v = eg[static_cast<int64_t>(e) * n_cand];
                best_era_gain[static_cast<int64_t>(slot) * E + e] = v;
                total += v;
            }
        }
        else
        {
            for (int e = 0; e < E; ++e)
                best_era_gain[static_cast<int64_t>(slot) * E + e] = 0.0f;
        }
        best_gain[slot] = total;
    }
}

void launch_select_split(
    const at::Tensor &gain,      // [slots, F, 2Bv]
    const at::Tensor &agree,     // [slots, F, 2Bv]
    const c10::optional<at::Tensor> &mono_ok,
    const at::Tensor &era_gain,  // [slots, F, E, 2Bv]
    const at::Tensor &node_active,
    float min_split_gain,
    at::Tensor &best,            // [slots] int32
    at::Tensor &best_gain,       // [slots] float32
    at::Tensor &best_era_gain)   // [slots, E] float32
{
    const at::cuda::OptionalCUDAGuard guard(device_of(gain));
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&agree, &era_gain, &node_active, &best, &best_gain, &best_era_gain})
        TORCH_CHECK(gain.is_cuda() && t->device() == gain.device(), "select tensors must be on the same CUDA device");
    const int slots = static_cast<int>(gain.size(0));
    const int n_cand = static_cast<int>(gain.size(2));
    const int n = static_cast<int>(gain.size(1)) * n_cand;
    const int E = static_cast<int>(era_gain.size(2));
    if (slots == 0)
        return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    select_split_kernel<<<slots, 256, 0, stream>>>(
        gain.data_ptr<float>(), agree.data_ptr<float>(),
        mono_ok.has_value() ? mono_ok->data_ptr<uint8_t>() : nullptr,
        era_gain.data_ptr<float>(), node_active.data_ptr<uint8_t>(),
        n, n_cand, E, E > 1, min_split_gain,
        best.data_ptr<int32_t>(), best_gain.data_ptr<float>(), best_era_gain.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM select kernel failed: ", cudaGetErrorString(err));
}
