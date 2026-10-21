#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Histogram layouts:
//   grad_hist / hess_hist : [E, F, B, K]  (K = number of gradient channels / outputs)
//   count_hist            : [E, F, B]     (sum of row weights, used for min_child_weight)
// grid = (F, row_tiles, K): each block handles one feature, one tile of rows and one output channel.
// The count channel is only accumulated by blocks with blockIdx.z == 0.
//
// Determinism: values are accumulated as fixed-point integers round(value * scale), where the
// power-of-two scale is chosen on the host such that the sum of one block's rows fits into 31 bits
// (|value| * scale * rows_per_block <= 2^30). Blocks accumulate in 32-bit shared memory and
// flush into 64-bit global accumulators. Integer addition is associative, so the histograms -
// and therefore all split decisions - do not depend on the non-deterministic order of atomics.
// A final pass converts back to float.

typedef unsigned long long fixed64_t;

__device__ __forceinline__ unsigned int to_fixed32(float v, float scale)
{
    // v * scale is exact for power-of-two scales; two's complement wrap-around makes
    // unsigned addition of negative values correct.
    return static_cast<unsigned int>(__float2int_rn(v * scale));
}

__device__ __forceinline__ fixed64_t to_fixed64(float v, float scale)
{
    return static_cast<fixed64_t>(static_cast<long long>(__float2int_rn(v * scale)));
}

__device__ __forceinline__ fixed64_t widen(unsigned int v)
{
    // sign-extend the 32-bit two's complement block sum
    return static_cast<fixed64_t>(static_cast<long long>(static_cast<int>(v)));
}

__global__ void histogram_shared_kernel(
    const int8_t *__restrict__ bin_indices,      // [N_total, F_master]
    const float *__restrict__ grads,             // [N_total, K]
    const float *__restrict__ hess,              // [N_total, K]
    const float *__restrict__ row_weight,        // [N_total]
    const int32_t *__restrict__ sample_indices,  // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices,     // [N_total]
    fixed64_t *__restrict__ grad_acc,            // [E * F * B * K]
    fixed64_t *__restrict__ hess_acc,            // [E * F * B * K]
    fixed64_t *__restrict__ count_acc,           // [E * F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t num_eras, int64_t K,
    int rows_per_thread, int64_t row_tiles,
    float grad_scale, float hess_scale, float count_scale)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[hist_feat_idx]; // 1 block per feature
    int64_t k = blockIdx.z;
    bool do_count = (k == 0);

    extern __shared__ unsigned int shmem[];
    unsigned int *sh_grad = shmem;                  // [num_eras * B]
    unsigned int *sh_hess = &sh_grad[num_eras * B]; // [num_eras * B]
    unsigned int *sh_cnt = &sh_hess[num_eras * B];  // [num_eras * B]

    // Initialize shared memory histograms
    for (int64_t i = threadIdx.x; i < num_eras * B; i += blockDim.x)
    {
        sh_grad[i] = 0u;
        sh_hess[i] = 0u;
        sh_cnt[i] = 0u;
    }

    __syncthreads();

    // Each thread processes multiple rows (grid-stride over row tiles)
    for (int64_t tile = blockIdx.y; tile < row_tiles; tile += gridDim.y)
    {
        int64_t row_start = (tile * blockDim.x + threadIdx.x) * rows_per_thread;
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int64_t row = row_start + r;
            if (row < N)
            {
                int64_t sample = sample_indices[row];
                int8_t bin = bin_indices[sample * F_master + feat];
                int32_t era = era_indices[sample];
                if (bin >= 0 && bin < B)
                {
                    atomicAdd(&sh_grad[era * B + bin], to_fixed32(grads[sample * K + k], grad_scale));
                    atomicAdd(&sh_hess[era * B + bin], to_fixed32(hess[sample * K + k], hess_scale));
                    if (do_count)
                        atomicAdd(&sh_cnt[era * B + bin], to_fixed32(row_weight[sample], count_scale));
                }
            }
        }
    }
    __syncthreads();

    // One thread per bin writes results back to global memory
    for (int64_t b = threadIdx.x; b < num_eras * B; b += blockDim.x)
    {
        int64_t e = b / B;
        int64_t bin = b % B;
        int64_t idx = (e * F + hist_feat_idx) * B + bin;

        if (sh_grad[b] != 0u)
            atomicAdd(&grad_acc[idx * K + k], widen(sh_grad[b]));
        if (sh_hess[b] != 0u)
            atomicAdd(&hess_acc[idx * K + k], widen(sh_hess[b]));
        if (do_count && sh_cnt[b] != 0u)
            atomicAdd(&count_acc[idx], widen(sh_cnt[b]));
    }
}

// Fallback for configurations whose per-block histogram does not fit into shared memory
// (e.g. many eras). Accumulates directly into the 64-bit global accumulators.
__global__ void histogram_global_kernel(
    const int8_t *__restrict__ bin_indices,
    const float *__restrict__ grads,
    const float *__restrict__ hess,
    const float *__restrict__ row_weight,
    const int32_t *__restrict__ sample_indices,
    const int32_t *__restrict__ feature_indices,
    const int32_t *__restrict__ era_indices,
    fixed64_t *__restrict__ grad_acc,
    fixed64_t *__restrict__ hess_acc,
    fixed64_t *__restrict__ count_acc,
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t K,
    int rows_per_thread, int64_t row_tiles,
    float grad_scale, float hess_scale, float count_scale)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[hist_feat_idx];
    int64_t k = blockIdx.z;
    bool do_count = (k == 0);

    for (int64_t tile = blockIdx.y; tile < row_tiles; tile += gridDim.y)
    {
        int64_t row_start = (tile * blockDim.x + threadIdx.x) * rows_per_thread;
        for (int r = 0; r < rows_per_thread; ++r)
        {
            int64_t row = row_start + r;
            if (row < N)
            {
                int64_t sample = sample_indices[row];
                int8_t bin = bin_indices[sample * F_master + feat];
                int64_t era = era_indices[sample];
                if (bin >= 0 && bin < B)
                {
                    int64_t idx = (era * F + hist_feat_idx) * B + bin;
                    atomicAdd(&grad_acc[idx * K + k], to_fixed64(grads[sample * K + k], grad_scale));
                    atomicAdd(&hess_acc[idx * K + k], to_fixed64(hess[sample * K + k], hess_scale));
                    if (do_count)
                        atomicAdd(&count_acc[idx], to_fixed64(row_weight[sample], count_scale));
                }
            }
        }
    }
}

// Converts the three accumulators (stored back to back in one workspace) to float.
__global__ void fixed_to_float_kernel(
    const fixed64_t *__restrict__ acc,
    float *__restrict__ grad_hist,
    float *__restrict__ hess_hist,
    float *__restrict__ count_hist,
    int64_t n_gh, int64_t n_c,
    double inv_grad_scale, double inv_hess_scale, double inv_count_scale)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 2 * n_gh + n_c)
        return;

    double v = static_cast<double>(static_cast<long long>(acc[i]));
    if (i < n_gh)
        grad_hist[i] = static_cast<float>(v * inv_grad_scale);
    else if (i < 2 * n_gh)
        hess_hist[i - n_gh] = static_cast<float>(v * inv_hess_scale);
    else
        count_hist[i - 2 * n_gh] = static_cast<float>(v * inv_count_scale);
}

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,     // [N_total, F_master] int8
    const at::Tensor &grads,           // [N_total, K] float32
    const at::Tensor &hess,            // [N_total, K] float32
    const at::Tensor &row_weight,      // [N_total] float32
    const at::Tensor &sample_indices,  // [N] int32
    const at::Tensor &feature_indices, // [F] int32
    const at::Tensor &era_indices,     // [N_total] int32
    at::Tensor &grad_hist,             // [E, F, B, K]  (overwritten)
    at::Tensor &hess_hist,             // [E, F, B, K]  (overwritten)
    at::Tensor &count_hist,            // [E, F, B]     (overwritten)
    int num_bins,
    float grad_scale,                  // power-of-two fixed-point scales, see header comment
    float hess_scale,
    float count_scale,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{
    // Run on the tensors' device and the current stream of that device; this also makes the
    // workspace allocation and the shared-memory query below refer to that device
    const at::cuda::OptionalCUDAGuard guard(device_of(bin_indices));
    const std::initializer_list<const at::Tensor *> tensors = {
        &grads, &hess, &row_weight, &sample_indices, &feature_indices,
        &era_indices, &grad_hist, &hess_hist, &count_hist};
    for (const at::Tensor *t : tensors)
        TORCH_CHECK(bin_indices.is_cuda() && t->device() == bin_indices.device(),
                    "histogram kernel tensors must be on the same CUDA device");
    TORCH_CHECK(grads.is_contiguous() && hess.is_contiguous(), "grads and hess must be contiguous");
    TORCH_CHECK(grads.dim() == 2 && hess.sizes() == grads.sizes(), "grads and hess must be [N, K] with equal shapes");
    TORCH_CHECK(grad_hist.dim() == 4 && grad_hist.size(3) == grads.size(1), "grad_hist must be [E, F, B, K]");
    TORCH_CHECK(grad_hist.is_contiguous() && hess_hist.is_contiguous() && count_hist.is_contiguous(),
                "histogram outputs must be contiguous");

    int64_t N = sample_indices.size(0);
    int64_t F = feature_indices.size(0);
    int64_t K = grads.size(1);
    int64_t num_features_master = bin_indices.size(1);
    int64_t num_eras = grad_hist.size(0); // inferred from output tensor
    int64_t n_gh = grad_hist.numel();
    int64_t n_c = count_hist.numel();

    if (N == 0 || F == 0 || K == 0)
    {
        grad_hist.zero_();
        hess_hist.zero_();
        count_hist.zero_();
        return;
    }

    int64_t rows_per_block = static_cast<int64_t>(threads_per_block) * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;

    // Every block initializes and flushes num_eras * num_bins entries; with many eras let each
    // block process several row tiles so that this overhead is amortized.
    int64_t min_tiles_per_block = (num_eras * num_bins) / (2 * rows_per_block);
    if (min_tiles_per_block < 1)
        min_tiles_per_block = 1;
    int64_t grid_y = (row_tiles + min_tiles_per_block - 1) / min_tiles_per_block;
    if (grid_y > 65535)
        grid_y = 65535;

    // The host scales assume one tile per block; with grid-stride over several tiles,
    // shrink them by a power of two so the 32-bit block sums still cannot overflow.
    int64_t tiles_per_block = (row_tiles + grid_y - 1) / grid_y;
    float shrink = 1.0f;
    while (shrink < static_cast<float>(tiles_per_block))
        shrink *= 2.0f;
    grad_scale /= shrink;
    hess_scale /= shrink;
    count_scale /= shrink;

    dim3 blocks(F, grid_y, K);
    dim3 threads(threads_per_block);
    int64_t shared_mem_bytes = 3 * num_eras * num_bins * static_cast<int64_t>(sizeof(unsigned int));

    // One workspace for the three 64-bit accumulators: [grad | hess | count]
    at::Tensor acc = at::zeros({2 * n_gh + n_c}, grad_hist.options().dtype(at::kLong));
    fixed64_t *grad_acc = reinterpret_cast<fixed64_t *>(acc.data_ptr<int64_t>());
    fixed64_t *hess_acc = grad_acc + n_gh;
    fixed64_t *count_acc = hess_acc + n_gh;

    int device = 0;
    cudaGetDevice(&device);
    int max_shared = 0;
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (shared_mem_bytes <= max_shared)
    {
        histogram_shared_kernel<<<blocks, threads, shared_mem_bytes, stream>>>(
            bin_indices.data_ptr<int8_t>(),
            grads.data_ptr<float>(),
            hess.data_ptr<float>(),
            row_weight.data_ptr<float>(),
            sample_indices.data_ptr<int32_t>(),
            feature_indices.data_ptr<int32_t>(),
            era_indices.data_ptr<int32_t>(),
            grad_acc,
            hess_acc,
            count_acc,
            N, num_features_master, F, num_bins, num_eras, K,
            rows_per_thread, row_tiles,
            grad_scale, hess_scale, count_scale);
    }
    else
    {
        histogram_global_kernel<<<blocks, threads, 0, stream>>>(
            bin_indices.data_ptr<int8_t>(),
            grads.data_ptr<float>(),
            hess.data_ptr<float>(),
            row_weight.data_ptr<float>(),
            sample_indices.data_ptr<int32_t>(),
            feature_indices.data_ptr<int32_t>(),
            era_indices.data_ptr<int32_t>(),
            grad_acc,
            hess_acc,
            count_acc,
            N, num_features_master, F, num_bins, K,
            rows_per_thread, row_tiles,
            grad_scale, hess_scale, count_scale);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM histogram kernel failed: ", cudaGetErrorString(err));

    const int conv_threads = 256;
    int64_t conv_blocks = (2 * n_gh + n_c + conv_threads - 1) / conv_threads;
    fixed_to_float_kernel<<<conv_blocks, conv_threads, 0, stream>>>(
        grad_acc,
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        count_hist.data_ptr<float>(),
        n_gh, n_c,
        1.0 / static_cast<double>(grad_scale),
        1.0 / static_cast<double>(hess_scale),
        1.0 / static_cast<double>(count_scale));

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM histogram conversion failed: ", cudaGetErrorString(err));
}
