#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Packed forest (see WarpGBM._pack_round):
//   nodes [T, M, 4] int32 per node: feature (-1 = leaf), split_bin | nan_left << 8, left, right
//   leaf  [T, M, Lw] float32 leaf values, already multiplied by the learning rate
//   out_col [T] int32 score column of each tree (per-output forests, Lw = 1) or empty: the
//           Lw = K leaf values of a tree are added to all K columns
//   base    [K] (base_stride 0) or [N, K] (base_stride K) score the trees are added to
// One thread per sample. Trees are added in order with plain additions (acc += leaf), starting
// from 0, and the base score is added last, so predictions are bit-reproducible and do not
// depend on the batch. Checkpoint p (tree count ckpt[p], ascending, the last one = T) receives
// base + sum of the first ckpt[p] trees; the output is [P, N, K].

__device__ __forceinline__ int leaf_of(const int4 *__restrict__ tree, const int8_t *__restrict__ row,
                                       int64_t feat_stride, int nan_bin)
{
    int node_id = 0;
    while (true)
    {
        const int4 nd = tree[node_id];
        if (nd.x < 0)
            return node_id;
        const int bin = row[nd.x * feat_stride];
        const int split_bin = nd.y & 0xff;
        const bool nan_left = (nd.y >> 8) & 1;
        const bool go_left = (bin == nan_bin) ? nan_left : (bin <= split_bin);
        node_id = go_left ? nd.z : nd.w;
    }
}

template <int KMAX>
__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bins, int64_t row_stride, int64_t feat_stride, int64_t N,
    const int4 *__restrict__ nodes,
    const float *__restrict__ leaf,
    const int32_t *__restrict__ out_col,
    int64_t T, int M, int Lw, int K,
    const float *__restrict__ base_all, int64_t base_stride,
    int nan_bin,
    const int64_t *__restrict__ ckpt, int P,
    float *__restrict__ out)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const int8_t *row = bins + i * row_stride;
    const float *base = base_all + i * base_stride;

    float acc[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; ++k)
        acc[k] = 0.0f;

    int p = 0;
    for (int64_t t = 0; t < T; ++t)
    {
        while (p < P - 1 && ckpt[p] == t)
        {
            float *o = out + (static_cast<int64_t>(p) * N + i) * K;
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < K)
                    o[k] = base[k] + acc[k];
            ++p;
        }
        const int n = leaf_of(nodes + t * M, row, feat_stride, nan_bin);
        const float *lv = leaf + (t * M + n) * Lw;
        if (out_col)
        {
            const int col = out_col[t];
            const float v = lv[0];
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k == col)
                    acc[k] += v;
        }
        else
        {
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < Lw)
                    acc[k] += lv[k];
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
}

// More than 32 outputs: accumulate in the output row (owned by this thread) instead of registers
__global__ void predict_forest_wide_kernel(
    const int8_t *__restrict__ bins, int64_t row_stride, int64_t feat_stride, int64_t N,
    const int4 *__restrict__ nodes,
    const float *__restrict__ leaf,
    const int32_t *__restrict__ out_col,
    int64_t T, int M, int Lw, int K,
    const float *__restrict__ base_all, int64_t base_stride,
    int nan_bin,
    const int64_t *__restrict__ ckpt, int P,
    float *__restrict__ out)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const int8_t *row = bins + i * row_stride;
    const float *base = base_all + i * base_stride;
    float *acc = out + (static_cast<int64_t>(P - 1) * N + i) * K;
    for (int k = 0; k < K; ++k)
        acc[k] = 0.0f;

    int p = 0;
    for (int64_t t = 0; t < T; ++t)
    {
        while (p < P - 1 && ckpt[p] == t)
        {
            float *o = out + (static_cast<int64_t>(p) * N + i) * K;
            for (int k = 0; k < K; ++k)
                o[k] = base[k] + acc[k];
            ++p;
        }
        const int n = leaf_of(nodes + t * M, row, feat_stride, nan_bin);
        const float *lv = leaf + (t * M + n) * Lw;
        if (out_col)
            acc[out_col[t]] += lv[0];
        else
            for (int k = 0; k < Lw; ++k)
                acc[k] += lv[k];
    }
    for (; p < P - 1; ++p)
    {
        float *o = out + (static_cast<int64_t>(p) * N + i) * K;
        for (int k = 0; k < K; ++k)
            o[k] = base[k] + acc[k];
    }
    for (int k = 0; k < K; ++k)
        acc[k] = base[k] + acc[k];
}

void predict_with_forest(
    const at::Tensor &bins,    // int8, element (i, f) at i * row_stride + f * feat_stride
    int64_t row_stride,
    int64_t feat_stride,
    int64_t num_rows,
    const at::Tensor &nodes,   // [T, M, 4] int32
    const at::Tensor &leaf,    // [T, M, Lw] float32
    const c10::optional<at::Tensor> &out_col, // [T] int32
    const at::Tensor &base,    // [K] or [N, K] float32
    int nan_bin,
    const at::Tensor &ckpt,    // [P] int64 tree counts, ascending, last == T
    at::Tensor &out)           // [P, N, K] float32
{
    const at::cuda::OptionalCUDAGuard guard(device_of(bins));
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&nodes, &leaf, &base, &ckpt, &out})
        TORCH_CHECK(bins.is_cuda() && t->device() == bins.device(),
                    "predict tensors must be on the same CUDA device");
    TORCH_CHECK(nodes.dim() == 3 && nodes.size(2) == 4 && nodes.scalar_type() == at::kInt && nodes.is_contiguous(),
                "nodes must be a contiguous [T, M, 4] int32 tensor");
    TORCH_CHECK(leaf.dim() == 3 && leaf.size(0) == nodes.size(0) && leaf.size(1) == nodes.size(1) && leaf.is_contiguous(),
                "leaf must be a contiguous [T, M, Lw] float32 tensor");
    TORCH_CHECK(out.dim() == 3 && out.is_contiguous() && out.size(0) == ckpt.size(0) && out.size(1) == num_rows,
                "out must be [P, N, K]");
    TORCH_CHECK(ckpt.scalar_type() == at::kLong && ckpt.size(0) >= 1, "ckpt must hold at least one int64 tree count");

    const int64_t T = nodes.size(0);
    const int M = static_cast<int>(nodes.size(1));
    const int Lw = static_cast<int>(leaf.size(2));
    const int K = static_cast<int>(out.size(2));
    const int P = static_cast<int>(ckpt.size(0));
    const int32_t *col = nullptr;
    if (out_col.has_value())
    {
        TORCH_CHECK(out_col->size(0) == T && out_col->scalar_type() == at::kInt, "out_col must be [T] int32");
        col = out_col->data_ptr<int32_t>();
    }
    else
        TORCH_CHECK(Lw == K, "vector trees need K leaf values");
    TORCH_CHECK(base.is_contiguous() && (base.numel() == K || (base.dim() == 2 && base.size(0) == num_rows && base.size(1) == K)),
                "base must be [K] or [N, K]");
    const int64_t base_stride = base.dim() == 2 ? K : 0;
    if (num_rows == 0)
        return;

    const int threads = 128;
    const int64_t blocks = (num_rows + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define WARPGBM_PREDICT_ARGS                                                                      \
    bins.data_ptr<int8_t>(), row_stride, feat_stride, num_rows,                                   \
        reinterpret_cast<const int4 *>(nodes.data_ptr<int32_t>()), leaf.data_ptr<float>(), col, T, M, Lw, K, \
        base.data_ptr<float>(), base_stride, nan_bin, ckpt.data_ptr<int64_t>(), P, out.data_ptr<float>()
    if (K <= 1)
        predict_forest_kernel<1><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
    else if (K <= 4)
        predict_forest_kernel<4><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
    else if (K <= 8)
        predict_forest_kernel<8><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
    else if (K <= 16)
        predict_forest_kernel<16><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
    else if (K <= 32)
        predict_forest_kernel<32><<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
    else
        predict_forest_wide_kernel<<<blocks, threads, 0, stream>>>(WARPGBM_PREDICT_ARGS);
#undef WARPGBM_PREDICT_ARGS

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM predict kernel failed: ", cudaGetErrorString(err));
}
