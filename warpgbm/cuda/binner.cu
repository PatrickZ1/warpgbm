#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>

// Bins a whole row-major feature matrix in one launch: bin = index of the first edge > x, i.e. the
// number of edges <= x for sorted edges; NaN -> nan_bin
__global__ void bin_matrix_kernel(
    const float *__restrict__ X,         // [N, F]
    const float *__restrict__ bin_edges, // [F, E], sorted per row
    int8_t *__restrict__ bin_indices,    // [N, F]
    int64_t total,                       // N * F
    int F,
    int E,
    int nan_bin)
{
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride)
    {
        const float val = X[idx];

        // Missing values get their own dedicated bin
        if (isnan(val))
        {
            bin_indices[idx] = static_cast<int8_t>(nan_bin);
            continue;
        }

        const float *edges = bin_edges + (idx % F) * E;
        int bin = 0;
        // Linear scan over the edges of this feature
        while (bin < E && val >= edges[bin])
        {
            ++bin;
        }

        bin_indices[idx] = static_cast<int8_t>(bin);
    }
}

// C++ launcher for calling from Python
void launch_bin_matrix_kernel(
    at::Tensor X,           // [N, F] float32
    at::Tensor bin_edges,   // [F, E] float32
    at::Tensor bin_indices, // [N, F] int8, output
    int nan_bin)
{
    // Run on the tensors' device and the current stream of that device, independent of the
    // caller's current device
    const at::cuda::OptionalCUDAGuard guard(device_of(X));
    TORCH_CHECK(X.is_cuda() && bin_edges.device() == X.device() && bin_indices.device() == X.device(),
                "X, bin_edges and bin_indices must be on the same CUDA device");
    TORCH_CHECK(X.dim() == 2 && X.scalar_type() == at::kFloat && X.is_contiguous(),
                "X must be a contiguous 2-D float32 tensor");
    TORCH_CHECK(bin_edges.dim() == 2 && bin_edges.scalar_type() == at::kFloat && bin_edges.is_contiguous(),
                "bin_edges must be a contiguous 2-D float32 tensor");
    TORCH_CHECK(bin_indices.scalar_type() == at::kChar && bin_indices.is_contiguous() &&
                    bin_indices.sizes() == X.sizes(),
                "bin_indices must be a contiguous int8 tensor shaped like X");

    const int64_t N = X.size(0);
    const int F = static_cast<int>(X.size(1));
    const int E = static_cast<int>(bin_edges.size(1));
    TORCH_CHECK(bin_edges.size(0) == F, "bin_edges must have one row per feature (", F, "), got ", bin_edges.size(0));
    TORCH_CHECK(E <= 127 && nan_bin >= -128 && nan_bin <= 127, "bin indices must fit int8");

    const int64_t total = N * F;
    if (total == 0)
        return;

    const int threads = 256;
    const int64_t max_blocks =
        static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * 32;
    const int blocks = static_cast<int>(std::min((total + threads - 1) / threads, max_blocks));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bin_matrix_kernel<<<blocks, threads, 0, stream>>>(
        X.data_ptr<float>(),
        bin_edges.data_ptr<float>(),
        bin_indices.data_ptr<int8_t>(),
        total,
        F,
        E,
        nan_bin);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM binning kernel failed: ", cudaGetErrorString(err));
}
