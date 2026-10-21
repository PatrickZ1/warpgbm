#include <torch/extension.h>
#include <vector>

// Declare the functions from the .cu files

void launch_histogram_level(
    const at::Tensor &bins, const at::Tensor &grads, const at::Tensor &hess,
    const at::Tensor &row_weight, const at::Tensor &order, const at::Tensor &seg_start,
    const at::Tensor &seg_len, const at::Tensor &seg_slot, const at::Tensor &tile_offset,
    const at::Tensor &feature_indices, const at::Tensor &era_indices, at::Tensor &hist,
    const at::Tensor &scales, int64_t max_tiles, int tile_rows);

void launch_node_sums(
    const at::Tensor &node_of_row, const at::Tensor &grads, const at::Tensor &hess,
    const at::Tensor &row_weight, at::Tensor &sums, const at::Tensor &scales, int tile_rows);

void launch_sibling_subtract(at::Tensor &hist, const at::Tensor &parent, const at::Tensor &left_built,
                             const at::Tensor &split);

void launch_level_split(
    const at::Tensor &hist, const at::Tensor &node_active, const at::Tensor &scales,
    const c10::optional<at::Tensor> &mono_req, float min_child_weight, float min_sum_hessian,
    float eps, int K, at::Tensor &era_gain, at::Tensor &gain, at::Tensor &agree,
    const c10::optional<at::Tensor> &mono_ok);

void launch_select_split(
    const at::Tensor &gain, const at::Tensor &agree, const c10::optional<at::Tensor> &mono_ok,
    const at::Tensor &era_gain, const at::Tensor &node_active, float min_split_gain,
    at::Tensor &best, at::Tensor &best_gain, at::Tensor &best_era_gain);

void launch_bin_matrix_kernel(
    at::Tensor X,           // [N, F] float32
    at::Tensor bin_edges,   // [F, E] float32
    at::Tensor bin_indices, // [N, F] int8
    int nan_bin);

void predict_with_forest(
    const at::Tensor &bins, int64_t row_stride, int64_t feat_stride, int64_t num_rows,
    const at::Tensor &nodes, const at::Tensor &leaf, const c10::optional<at::Tensor> &out_col,
    const at::Tensor &base, int nan_bin, const at::Tensor &ckpt, at::Tensor &out);

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("histogram_level", &launch_histogram_level, "Histograms of the nodes of a tree level (CUDA)");
    m.def("node_sums", &launch_node_sums, "Gradient / hessian / weight totals per node (CUDA)");
    m.def("sibling_subtract", &launch_sibling_subtract, "Sibling histograms: parent - built child (CUDA)");
    m.def("level_split", &launch_level_split, "Split candidates of the nodes of a tree level (CUDA)");
    m.def("select_split", &launch_select_split, "Best split per node (CUDA)");
    m.def("bin_matrix", &launch_bin_matrix_kernel, "Quantile binning of a feature matrix (CUDA)");
    m.def("predict_forest", &predict_with_forest, "Forest predictions with checkpoints (CUDA)");
}
