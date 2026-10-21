#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Flattened tree node layout (NODE_WIDTH floats per node):
//   0: feature, 1: split_bin, 2: left_id, 3: right_id, 4: is_leaf, 5: leaf_value, 6: nan_left
#define NODE_WIDTH 7

// One thread per sample; trees are summed in order so predictions are bit-reproducible.
__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bin_indices, // [N x F]
    const float *__restrict__ tree_tensor,  // [T x max_nodes x NODE_WIDTH]
    int64_t N, int64_t F, int64_t T, int64_t max_nodes,
    int nan_bin,
    float learning_rate,
    float *__restrict__ out // [N]
)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    // prevent overflow
    const int8_t *row = bin_indices + i * F;
    float acc = 0.0f;
    for (int64_t t = 0; t < T; ++t)
    {
        const float *tree = tree_tensor + t * max_nodes * NODE_WIDTH;
        int node_id = 0;
        while (true)
        {
            const float *node = tree + node_id * NODE_WIDTH;
            if (node[4] > 0.5f) // is_leaf
            {
                acc += learning_rate * node[5];
                break;
            }

            int feat = static_cast<int>(node[0]);
            int split_bin = static_cast<int>(node[1]);
            bool nan_left = node[6] > 0.5f;
            int bin = row[feat];

            bool go_left = (bin == nan_bin) ? nan_left : (bin <= split_bin);
            node_id = go_left ? static_cast<int>(node[2]) : static_cast<int>(node[3]);
        }
    }
    out[i] += acc;
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

    if (N == 0 || T == 0)
        return;

    int threads_per_block = 256;
    int64_t blocks = (N + threads_per_block - 1) / threads_per_block;

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
