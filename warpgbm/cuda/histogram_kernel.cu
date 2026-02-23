#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void histogram_tiled_configurable_kernel(
    const int8_t *__restrict__ bin_indices,      // [N, F_master]
    const float *__restrict__ grads,             // [N]
    const float *__restrict__ hess,              // [N]
    const int32_t *__restrict__ sample_indices,  // [N]
    const int32_t *__restrict__ feature_indices, // [F]
    const int32_t *__restrict__ era_indices,     // [N]
    float *__restrict__ grad_hist,               // [F * B]
    float *__restrict__ hess_hist,               // [F * B]
    int64_t N, int64_t F_master, int64_t F, int64_t B, int64_t num_eras,
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat = feature_indices[ hist_feat_idx ]; // 1 block per feature
    int row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;

    extern __shared__ float shmem[];
    float *sh_grad = shmem;                             // [num_eras * B]
    float *sh_hess = &sh_grad[num_eras * B];            // [num_eras * B]

    // Initialize shared memory histograms
    for (int i = threadIdx.x; i < num_eras * B; i += blockDim.x) {
        sh_grad[i] = 0.0f;
        sh_hess[i] = 0.0f;
    }
    
    __syncthreads();

    // Each thread processes multiple rows
    for (int r = 0; r < rows_per_thread; ++r)
    {
        int row = row_start + r;
        if (row < N)
        {
            int sample = sample_indices[row];
            int8_t bin = bin_indices[sample * F_master + feat];
            int32_t era = era_indices[sample];
            if (bin >= 0 && bin < B)
            {
                atomicAdd(&sh_grad[era * B + bin], grads[sample]);
                atomicAdd(&sh_hess[era * B + bin], hess[sample]);
            }
        }
    }
    __syncthreads();

    // One thread per bin writes results back to global memory
    for (int b = threadIdx.x; b < num_eras * B; b += blockDim.x)
    {
        int e = b / B;
        int bin = b % B;
        int64_t idx = e * F * B + hist_feat_idx * B + bin;

        atomicAdd(&grad_hist[idx], sh_grad[b]);
        atomicAdd(&hess_hist[idx], sh_hess[b]);
    }

}

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices,
    const at::Tensor &grads,
    const at::Tensor &hess,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1)
{

    int64_t N = sample_indices.size(0);
    int64_t F = feature_indices.size(0);
    int num_features_master = bin_indices.size(1);

    int64_t rows_per_block = threads_per_block * rows_per_thread;
    int64_t row_tiles = (N + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles); // grid.x = F, grid.y = row_tiles
    dim3 threads(threads_per_block);
    int num_eras = grad_hist.size(0); // inferred from output tensor
    int shared_mem_bytes = 2 * num_eras * num_bins * sizeof(float);

    histogram_tiled_configurable_kernel<<<blocks, threads, shared_mem_bytes>>>(
        bin_indices.data_ptr<int8_t>(),
        grads.data_ptr<float>(),
        hess.data_ptr<float>(),
        sample_indices.data_ptr<int32_t>(),
        feature_indices.data_ptr<int32_t>(),
        era_indices.data_ptr<int32_t>(),
        grad_hist.data_ptr<float>(),
        hess_hist.data_ptr<float>(),
        N, num_features_master, F, num_bins, num_eras,
        rows_per_thread);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
