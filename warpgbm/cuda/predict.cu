#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Flattened tree node layout (NODE_WIDTH floats per node):
//   0: feature, 1: split_bin, 2: left_id, 3: right_id, 4: is_leaf, 5: leaf_value, 6: nan_left
#define NODE_WIDTH 7

__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bin_indices, // [N x F]
    const float *__restrict__ tree_tensor,  // [T x max_nodes x NODE_WIDTH]
    int64_t N, int64_t F, int64_t T, int64_t max_nodes,
    int nan_bin,
    float learning_rate,
    float *__restrict__ out // [N]
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_jobs = N * T;
    if (idx >= total_jobs)
        return;

    int64_t i = idx % N; // sample index
    int64_t t = idx / N; // tree index

    const float *tree = tree_tensor + t * max_nodes * NODE_WIDTH;

    int node_id = 0;
    while (true)
    {
        const float *node = tree + node_id * NODE_WIDTH;
        float is_leaf = node[4];
        if (is_leaf > 0.5f)
        {
            atomicAdd(&out[i], learning_rate * node[5]);
            return;
        }

        int feat = static_cast<int>(node[0]);
        int split_bin = static_cast<int>(node[1]);
        int left_id = static_cast<int>(node[2]);
        int right_id = static_cast<int>(node[3]);
        bool nan_left = node[6] > 0.5f;

        // prevent overflow
        int64_t bin_idx = i * F + feat;
        int bin = bin_indices[bin_idx];

        bool go_left = (bin == nan_bin) ? nan_left : (bin <= split_bin);
        node_id = go_left ? left_id : right_id;
    }
}


void predict_with_forest(
    const at::Tensor &bin_indices,
    const at::Tensor &tree_tensor,
    int nan_bin,
    float learning_rate,
    at::Tensor &out
)
{
    TORCH_CHECK(tree_tensor.dim() == 3 && tree_tensor.size(2) == NODE_WIDTH,
                "tree_tensor must be [T, max_nodes, 7]");

    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int64_t T = tree_tensor.size(0);
    int64_t max_nodes = tree_tensor.size(1);

    int64_t total_jobs = N * T;
    if (total_jobs == 0)
        return;

    int threads_per_block = 256;
    int64_t blocks = (total_jobs + threads_per_block - 1) / threads_per_block;

    predict_forest_kernel<<<blocks, threads_per_block>>>(
        bin_indices.data_ptr<int8_t>(),
        tree_tensor.data_ptr<float>(),
        N, F, T, max_nodes,
        nan_bin,
        learning_rate,
        out.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM predict kernel failed: ", cudaGetErrorString(err));
}
