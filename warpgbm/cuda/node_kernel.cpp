#include <torch/extension.h>
#include <vector>

// Declare the functions from the .cu files

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B, K]
    const at::Tensor &H, // [E, F, B, K]
    const at::Tensor &C, // [E, F, B]
    float min_child_weight,
    float min_sum_hessian,
    float eps,
    at::Tensor &per_era_gain,      // [E, F, 2 * (B - 1)]
    at::Tensor &per_era_direction, // [E, F, 2 * (B - 1)]
    int threads = 128);

void launch_histogram_kernel_cuda_configurable(
    const at::Tensor &bin_indices, const at::Tensor &grads,
    const at::Tensor &hess, const at::Tensor &row_weight,
    const at::Tensor &sample_indices, const at::Tensor &feature_indices,
    const at::Tensor &era_indices, at::Tensor &grad_hist,
    at::Tensor &hess_hist, at::Tensor &count_hist, int num_bins,
    float grad_scale, float hess_scale, float count_scale,
    int threads_per_block = 256, int rows_per_thread = 1);

void launch_bin_matrix_kernel(
    at::Tensor X,           // [N, F] float32
    at::Tensor bin_edges,   // [F, E] float32
    at::Tensor bin_indices, // [N, F] int8
    int nan_bin);

void predict_with_forest(
    const at::Tensor &bin_indices, // [N x F], int8
    const at::Tensor &tree_tensor, // [T x max_nodes x 7], float32
    int nan_bin,
    float learning_rate,
    at::Tensor &out // [N], float32
);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("compute_histogram3", &launch_histogram_kernel_cuda_configurable, "Histogram Feature Shared Mem");
    m.def("compute_split", &launch_directional_split_kernel, "Best Split (CUDA)");
    m.def("bin_matrix", &launch_bin_matrix_kernel, "Quantile binning of a feature matrix (CUDA)");
    m.def("predict_forest", &predict_with_forest, "CUDA Predictions");
}
