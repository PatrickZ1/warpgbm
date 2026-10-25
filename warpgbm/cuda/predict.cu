#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forest block (see warpgbm/forest.py):
//   nodes  [n, 2] int32 in pre-order per tree, the left child of node i is i + 1:
//          split (feature, bin | nan_left << 8 | (right - i) << 9), leaf (-1 - leaf index, rows)
//   values leaf values (not multiplied by the learning rate), [n_leaves, n_cols] per tree
//   meta   [T, 4] int32 per tree: first node, first value, column mask (K <= 32) or first entry
//          of cols (K > 32), n_cols (K = all columns in order, 0 = the tree adds nothing)
//   lr     [T] float32 learning rate of the tree's round
//   cols   column indices of the trees with 0 < n_cols < K (K > 32 only)
//   init   [N, K] tree sums the block continues from (null: 0), e.g. the sums of the blocks before
//   base   [K] (base_stride 0) or [N, K] (base_stride K) score added at the checkpoints
// One thread per sample. Trees are added in order, acc = acc + leaf * lr with one float32 rounding
// per operation (no FMA, the same values as the v2.8 kernel that read pre-multiplied leaves), and
// the base score is added last, so predictions are bit-reproducible, independent of the batch and
// of how the forest is split into blocks. Checkpoint p (tree count ckpt[p], ascending, <= T)
// receives base + sum of the first ckpt[p] trees: out is [P, N, K]. acc_out (optional) receives the
// sums after all T trees, without the base score.

__device__ __forceinline__ int leaf_index(const int2 *__restrict__ tree, const int8_t *__restrict__ row,
                                          int64_t feat_stride, int nan_bin)
{
    int i = 0;
    while (true)
    {
        const int2 nd = tree[i];
        if (nd.x < 0)
            return -1 - nd.x;
        const int bin = row[nd.x * feat_stride];
        const bool go_left = (bin == nan_bin) ? ((nd.y >> 8) & 1) : (bin <= (nd.y & 0xff));
        i = go_left ? i + 1 : i + (nd.y >> 9);
    }
}

template <int KMAX>
__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bins, int64_t row_stride, int64_t feat_stride, int64_t N,
    const int2 *__restrict__ nodes,
    const float *__restrict__ values,
    const int4 *__restrict__ meta,
    const float *__restrict__ lr,
    int64_t T, int K,
    const float *__restrict__ init,
    const float *__restrict__ base_all, int64_t base_stride,
    int nan_bin,
    const int64_t *__restrict__ ckpt, int P,
    float *__restrict__ out,
    float *__restrict__ acc_out)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const int8_t *row = bins + i * row_stride;
    const float *base = base_all + i * base_stride;

    float acc[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; ++k)
        acc[k] = (init != nullptr && k < K) ? init[i * K + k] : 0.0f;

    int p = 0;
    for (int64_t t = 0; t < T; ++t)
    {
        while (p < P && ckpt[p] == t)
        {
            float *o = out + (static_cast<int64_t>(p) * N + i) * K;
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < K)
                    o[k] = base[k] + acc[k];
            ++p;
        }
        const int4 m = meta[t];
        if (m.w == 0)
            continue;
        const int leaf = leaf_index(nodes + m.x, row, feat_stride, nan_bin);
        const float *lv = values + m.y + static_cast<int64_t>(leaf) * m.w;
        const float rate = lr[t];
        const unsigned int mask = static_cast<unsigned int>(m.z);
        int j = 0;
#pragma unroll
        for (int k = 0; k < KMAX; ++k)
            if ((mask >> k) & 1u)
            {
                acc[k] = __fadd_rn(acc[k], __fmul_rn(lv[j], rate));
                ++j;
            }
    }
    for (; p < P; ++p)
    {
        float *o = out + (static_cast<int64_t>(p) * N + i) * K;
#pragma unroll
        for (int k = 0; k < KMAX; ++k)
            if (k < K)
                o[k] = base[k] + acc[k];
    }
    if (acc_out != nullptr)
    {
#pragma unroll
        for (int k = 0; k < KMAX; ++k)
            if (k < K)
                acc_out[i * K + k] = acc[k];
    }
}

// More than 32 outputs: accumulate in acc_out (the row is owned by this thread) instead of
// registers; sparse trees list their columns in cols
__global__ void predict_forest_wide_kernel(
    const int8_t *__restrict__ bins, int64_t row_stride, int64_t feat_stride, int64_t N,
    const int2 *__restrict__ nodes,
    const float *__restrict__ values,
    const int4 *__restrict__ meta,
    const float *__restrict__ lr,
    const int32_t *__restrict__ cols,
    int64_t T, int K,
    const float *__restrict__ init,
    const float *__restrict__ base_all, int64_t base_stride,
    int nan_bin,
    const int64_t *__restrict__ ckpt, int P,
    float *__restrict__ out,
    float *__restrict__ acc_all)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const int8_t *row = bins + i * row_stride;
    const float *base = base_all + i * base_stride;
    float *acc = acc_all + i * K;
    for (int k = 0; k < K; ++k)
        acc[k] = init != nullptr ? init[i * K + k] : 0.0f;

    int p = 0;
    for (int64_t t = 0; t < T; ++t)
    {
        while (p < P && ckpt[p] == t)
        {
            float *o = out + (static_cast<int64_t>(p) * N + i) * K;
            for (int k = 0; k < K; ++k)
                o[k] = base[k] + acc[k];
            ++p;
        }
        const int4 m = meta[t];
        if (m.w == 0)
            continue;
        const int leaf = leaf_index(nodes + m.x, row, feat_stride, nan_bin);
        const float *lv = values + m.y + static_cast<int64_t>(leaf) * m.w;
        const float rate = lr[t];
        if (m.w == K)
            for (int k = 0; k < K; ++k)
                acc[k] = __fadd_rn(acc[k], __fmul_rn(lv[k], rate));
        else
            for (int j = 0; j < m.w; ++j)
            {
                const int c = cols[m.z + j];
                acc[c] = __fadd_rn(acc[c], __fmul_rn(lv[j], rate));
            }
    }
    for (; p < P; ++p)
    {
        float *o = out + (static_cast<int64_t>(p) * N + i) * K;
        for (int k = 0; k < K; ++k)
            o[k] = base[k] + acc[k];
    }
}

void predict_with_forest(
    const at::Tensor &bins,    // int8, element (i, f) at i * row_stride + f * feat_stride
    int64_t row_stride,
    int64_t feat_stride,
    int64_t num_rows,
    const at::Tensor &nodes,   // [n, 2] int32
    const at::Tensor &values,  // float32
    const at::Tensor &meta,    // [T, 4] int32
    const at::Tensor &lr,      // [T] float32
    const c10::optional<at::Tensor> &cols, // int32 (K > 32 with sparse trees)
    int64_t K,
    const c10::optional<at::Tensor> &init, // [N, K] float32
    const at::Tensor &base,    // [K] or [N, K] float32
    int nan_bin,
    const at::Tensor &ckpt,    // [P] int64 tree counts, ascending, <= T (P may be 0)
    const c10::optional<at::Tensor> &out,     // [P, N, K] float32 (P > 0)
    const c10::optional<at::Tensor> &acc_out) // [N, K] float32 (required for K > 32)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(bins));
    std::vector<const at::Tensor *> tensors = {&nodes, &values, &meta, &lr, &base, &ckpt};
    for (const auto *opt : {&cols, &init, &out, &acc_out})
        if (opt->has_value())
            tensors.push_back(&opt->value());
    for (const at::Tensor *t : tensors)
        TORCH_CHECK(bins.is_cuda() && t->device() == bins.device(),
                    "predict tensors must be on the same CUDA device");
    TORCH_CHECK(nodes.dim() == 2 && nodes.size(1) == 2 && nodes.scalar_type() == at::kInt && nodes.is_contiguous(),
                "nodes must be a contiguous [n, 2] int32 tensor");
    TORCH_CHECK(values.scalar_type() == at::kFloat && values.is_contiguous(), "values must be contiguous float32");
    TORCH_CHECK(meta.dim() == 2 && meta.size(1) == 4 && meta.scalar_type() == at::kInt && meta.is_contiguous(),
                "meta must be a contiguous [T, 4] int32 tensor");
    const int64_t T = meta.size(0);
    TORCH_CHECK(lr.scalar_type() == at::kFloat && lr.is_contiguous() && lr.numel() == T, "lr must be [T] float32");
    TORCH_CHECK(K >= 1, "K must be positive");
    TORCH_CHECK(ckpt.scalar_type() == at::kLong && ckpt.dim() == 1 && ckpt.is_contiguous(), "ckpt must be [P] int64");
    const int P = static_cast<int>(ckpt.size(0));
    float *out_ptr = nullptr;
    if (P > 0)
    {
        TORCH_CHECK(out.has_value(), "out is required with checkpoints");
        TORCH_CHECK(out->dim() == 3 && out->is_contiguous() && out->scalar_type() == at::kFloat &&
                        out->size(0) == P && out->size(1) == num_rows && out->size(2) == K,
                    "out must be a contiguous [P, N, K] float32 tensor");
        out_ptr = out->data_ptr<float>();
    }
    float *acc_ptr = nullptr;
    if (acc_out.has_value())
    {
        TORCH_CHECK(acc_out->is_contiguous() && acc_out->scalar_type() == at::kFloat && acc_out->numel() == num_rows * K,
                    "acc_out must be a contiguous [N, K] float32 tensor");
        acc_ptr = acc_out->data_ptr<float>();
    }
    const float *init_ptr = nullptr;
    if (init.has_value())
    {
        TORCH_CHECK(init->is_contiguous() && init->scalar_type() == at::kFloat && init->numel() == num_rows * K,
                    "init must be a contiguous [N, K] float32 tensor");
        init_ptr = init->data_ptr<float>();
    }
    TORCH_CHECK(base.is_contiguous() && base.scalar_type() == at::kFloat &&
                    (base.numel() == K || (base.dim() == 2 && base.size(0) == num_rows && base.size(1) == K)),
                "base must be [K] or [N, K] float32");
    const int64_t base_stride = base.dim() == 2 ? K : 0;
    if (num_rows == 0)
        return;

    const int threads = 128;
    const int64_t blocks = (num_rows + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int2 *nodes_ptr = reinterpret_cast<const int2 *>(nodes.data_ptr<int32_t>());
    const int4 *meta_ptr = reinterpret_cast<const int4 *>(meta.data_ptr<int32_t>());
    const int Ki = static_cast<int>(K);
    if (K <= 32)
    {
#define WARPGBM_PREDICT_ARGS                                                                       \
    bins.data_ptr<int8_t>(), row_stride, feat_stride, num_rows, nodes_ptr, values.data_ptr<float>(), \
        meta_ptr, lr.data_ptr<float>(), T, Ki, init_ptr, base.data_ptr<float>(), base_stride, nan_bin, \
        ckpt.data_ptr<int64_t>(), P, out_ptr, acc_ptr
        if (K <= 1)
            predict_forest_kernel<1><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
        else if (K <= 4)
            predict_forest_kernel<4><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
        else if (K <= 8)
            predict_forest_kernel<8><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
        else if (K <= 16)
            predict_forest_kernel<16><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
        else
            predict_forest_kernel<32><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
#undef WARPGBM_PREDICT_ARGS
    }
    else
    {
        TORCH_CHECK(acc_ptr != nullptr, "acc_out is required for more than 32 outputs");
        TORCH_CHECK(!cols.has_value() || (cols->scalar_type() == at::kInt && cols->is_contiguous()),
                    "cols must be contiguous int32");
        predict_forest_wide_kernel<<<blocks, threads, 0, stream>>>(
            bins.data_ptr<int8_t>(), row_stride, feat_stride, num_rows, nodes_ptr, values.data_ptr<float>(),
            meta_ptr, lr.data_ptr<float>(), cols.has_value() ? cols->data_ptr<int32_t>() : nullptr, T, Ki,
            init_ptr, base.data_ptr<float>(), base_stride, nan_bin, ckpt.data_ptr<int64_t>(), P, out_ptr, acc_ptr);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM predict kernel failed: ", cudaGetErrorString(err));
}
