#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Level-wise histograms of several tree nodes in one launch.
//
// Inputs
//   bins       : [F_master, N_total] int8, feature-major (one contiguous row per feature)
//   grads      : [N_total, K]  float32
//   hess       : [N_total, Kh] float32, Kh = K, or Kh = 0 when every hessian equals the row
//                weight (the hessian histograms are then read from the count channel)
//   row_weight : [N_total] float32 (count channel, used for min_child_weight)
//   order      : row ids grouped by segment; segment s holds order[seg_start[s] : + seg_len[s]]
//   seg_slot   : output slot of segment s; tile_offset[s] = first tile of segment s (exclusive
//                prefix sum of ceil(seg_len / tile_rows), S + 1 entries)
// Output
//   hist       : [slots, F, E, B, C], C = K + Kh + 1 channels [grad | hess | count], zeroed.
//                int64 -> deterministic fixed-point sums, float32 -> float sums (see Acc).
//
// grid = (tile, feature group, channel chunk). Every block handles one tile of at most tile_rows
// rows of one segment and a group of FG features, accumulates into shared [FG, E, B, chunk]
// histograms and flushes them with one global atomic per non-zero entry. Within a warp,
// G = min(32, next_pow2(chunk)) lanes share one row and take its channels: the shared-memory
// addresses of a warp instruction are consecutive (no bank conflicts), the reads of grads[row, :]
// are coalesced, and a row's values are loaded and converted once for all features of the group.
//
// Values are accumulated in shared memory as 32-bit fixed-point integers round(v * scale), with
// power-of-two scales chosen such that one tile's sum fits into 31 bits
// (|v| * scale * tile_rows <= 2^30). With int64 output the block sums are added as integers:
// integer addition is associative, so the result does not depend on the order of the atomics,
// and histograms can be subtracted exactly.

typedef unsigned long long fixed64_t;

__device__ __forceinline__ unsigned int to_fixed32(float v, float scale)
{
    // v * scale is exact for power-of-two scales; two's complement wrap-around makes
    // unsigned addition of negative values correct.
    return static_cast<unsigned int>(__float2int_rn(v * scale));
}

__device__ __forceinline__ fixed64_t to_fixed64(float v, float scale)
{
    return static_cast<fixed64_t>(__float2ll_rn(v * scale));
}

__device__ __forceinline__ fixed64_t widen(unsigned int v)
{
    // sign-extend the 32-bit two's complement block sum
    return static_cast<fixed64_t>(static_cast<long long>(static_cast<int>(v)));
}

// Both modes accumulate a block's rows as 32-bit fixed-point integers in shared memory: integer
// shared-memory atomics are native, float ones are emulated with compare-and-swap loops on many
// GPUs. They differ in the global accumulation of the block sums:
//   DET  : int64 fixed point (exact, order independent)
//   !DET : float atomics of the block sums (order dependent rounding, half the memory)
template <bool DET>
struct Acc;

template <>
struct Acc<true>
{
    typedef long long out_t;
    __device__ static void flush(out_t *p, unsigned int v, float)
    {
        if (v != 0u)
            atomicAdd(reinterpret_cast<fixed64_t *>(p), widen(v));
    }
    __device__ static void add_global(out_t *p, float v, float scale)
    {
        atomicAdd(reinterpret_cast<fixed64_t *>(p), to_fixed64(v, scale));
    }
};

template <>
struct Acc<false>
{
    typedef float out_t;
    __device__ static void flush(out_t *p, unsigned int v, float inv_scale)
    {
        if (v != 0u)
            atomicAdd(p, static_cast<float>(static_cast<int>(v)) * inv_scale);
    }
    __device__ static void add_global(out_t *p, float v, float) { atomicAdd(p, v); }
};

// Fixed-point scale of channel ch
__device__ __forceinline__ float channel_scale(const float *scales, int ch, int K, int Kh)
{
    return ch < K ? scales[0] : (ch < K + Kh ? scales[1] : scales[2]);
}

// Value and fixed-point scale of channel ch of a row
__device__ __forceinline__ float channel_value(
    const float *__restrict__ grads, const float *__restrict__ hess, const float *__restrict__ row_weight,
    int64_t row, int ch, int K, int Kh, const float *scales, float &scale)
{
    if (ch < K)
    {
        scale = scales[0];
        return grads[row * K + ch];
    }
    if (ch < K + Kh)
    {
        scale = scales[1];
        return hess[row * Kh + (ch - K)];
    }
    scale = scales[2];
    return row_weight[row];
}

// Segment of a tile: tile_offset is sorted, find s with tile_offset[s] <= tile < tile_offset[s + 1]
__device__ __forceinline__ int find_segment(const int32_t *tile_offset, int S, int tile)
{
    int lo = 0, hi = S; // invariant: tile_offset[lo] <= tile < tile_offset[hi]
    while (hi - lo > 1)
    {
        int mid = (lo + hi) >> 1;
        if (tile_offset[mid] <= tile)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// Channels per lane: a block's chunk of channels is spread over G = min(32, next_pow2(chunk))
// lanes; each lane accumulates at most MAX_VALS channels (chunk <= 32 * MAX_VALS)
#define MAX_VALS 4

template <bool DET, int FG, int NV>
__global__ void histogram_level_kernel(
    const int8_t *__restrict__ bins, int64_t N_total,
    const float *__restrict__ grads,
    const float *__restrict__ hess,
    const float *__restrict__ row_weight,
    const int32_t *__restrict__ order,
    const int32_t *__restrict__ seg_start,
    const int32_t *__restrict__ seg_len,
    const int32_t *__restrict__ seg_slot,
    const int32_t *__restrict__ tile_offset,
    int S,
    const int32_t *__restrict__ feature_indices,
    const int32_t *__restrict__ era_indices,
    typename Acc<DET>::out_t *__restrict__ hist,
    int F, int E, int B, int C, int K, int Kh,
    int chunk, int G, int tile_rows,
    const float *__restrict__ scales)
{
    typedef unsigned int shared_t;
    const int tile = blockIdx.x;
    if (tile >= tile_offset[S])
        return;
    const int s = find_segment(tile_offset, S, tile);
    const int f0 = blockIdx.y * FG;
    const int nf = min(FG, F - f0);
    const int c0 = blockIdx.z * chunk;
    const int cc = min(chunk, C - c0);

    const int64_t r0 = static_cast<int64_t>(seg_start[s]) + static_cast<int64_t>(tile - tile_offset[s]) * tile_rows;
    const int64_t r1 = min(r0 + tile_rows, static_cast<int64_t>(seg_start[s]) + seg_len[s]);

    extern __shared__ unsigned char shraw[];
    // [FG, E, B, cc] followed by a scratch tail of 32 * NV entries: additions of inactive lanes
    // (value 0) may reach past the last bin, and features beyond the group (nf < FG) add there,
    // so the atomics below need no branches
    shared_t *sh = reinterpret_cast<shared_t *>(shraw);
    const int per_feature = E * B * cc;
    const int n_sh = nf * per_feature;
    const int trash = FG * per_feature;
    for (int i = threadIdx.x; i < trash + 32 * NV; i += blockDim.x)
        sh[i] = 0;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int rows_per_step = 32 / G;
    const int sub = lane / G; // row of this lane within a step
    const int cl = lane % G;  // first channel of this lane

    // Source of every channel of this lane, resolved once: value = src[row * stride] * scale
    const float *src[NV];
    int stride[NV];
    float scale[NV];
    bool on[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v)
    {
        const int c = cl + v * G;
        const int ch = c0 + c;
        on[v] = c < cc;
        if (ch < K)
        {
            src[v] = grads + ch;
            stride[v] = K;
            scale[v] = scales[0];
        }
        else if (ch < K + Kh)
        {
            src[v] = hess + (ch - K);
            stride[v] = Kh;
            scale[v] = scales[1];
        }
        else
        {
            src[v] = row_weight;
            stride[v] = 1;
            scale[v] = scales[2];
        }
    }
    const int8_t *fbins[FG];
#pragma unroll
    for (int q = 0; q < FG; ++q)
        fbins[q] = bins + static_cast<int64_t>(feature_indices[f0 + min(q, nf - 1)]) * N_total;
    __syncthreads();

    for (int64_t base = r0 + static_cast<int64_t>(warp) * 32; base < r1; base += static_cast<int64_t>(nwarps) * 32)
    {
        // Coalesced loads of 32 rows (row id, era and the bins of all features of the group),
        // then broadcast row by row
        const int64_t i = base + lane;
        const bool in = i < r1;
        const int row = in ? order[i] : 0;
        const int era_off = in ? era_indices[row] * B * cc : 0;
        int bin_off[FG];
#pragma unroll
        for (int q = 0; q < FG; ++q)
            bin_off[q] = q < nf ? q * per_feature + (in ? fbins[q][row] : 0) * cc : trash;

        const int n_rows = static_cast<int>(min(static_cast<int64_t>(32), r1 - base));
        for (int j = 0; j < n_rows; j += rows_per_step)
        {
            const int jr = j + sub;
            const int from = jr & 31;
            const int r = __shfl_sync(0xffffffffu, row, from);
            const int eo = __shfl_sync(0xffffffffu, era_off, from);
            const bool has_row = jr < n_rows;

            // This lane's channel values of row r, converted once for all features
            shared_t add[NV];
#pragma unroll
            for (int v = 0; v < NV; ++v)
            {
                const float x = (has_row && on[v]) ? src[v][static_cast<int64_t>(r) * stride[v]] : 0.0f;
                add[v] = to_fixed32(x, scale[v]);
            }
            // Inactive lanes add 0 (no branches around the atomics)
#pragma unroll
            for (int q = 0; q < FG; ++q)
            {
                const int off = __shfl_sync(0xffffffffu, bin_off[q], from);
                shared_t *h = sh + (q < nf ? eo + off : trash) + cl;
#pragma unroll
                for (int v = 0; v < NV; ++v)
                    atomicAdd(&h[v * G], add[v]);
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_sh; i += blockDim.x)
    {
        const int q = i / per_feature;
        const int rest = i - q * per_feature;
        const int eb = rest / cc;
        const int c = rest - eb * cc;
        typename Acc<DET>::out_t *out =
            hist + ((static_cast<int64_t>(seg_slot[s]) * F + f0 + q) * E) * B * C;
        Acc<DET>::flush(&out[static_cast<int64_t>(eb) * C + c0 + c], sh[i],
                        1.0f / channel_scale(scales, c0 + c, K, Kh));
    }
}

// Fallback when not even one channel of [E, B] fits into shared memory (very many eras):
// accumulates directly into the global histogram, one thread per (row, channel).
template <bool DET>
__global__ void histogram_level_global_kernel(
    const int8_t *__restrict__ bins, int64_t N_total,
    const float *__restrict__ grads,
    const float *__restrict__ hess,
    const float *__restrict__ row_weight,
    const int32_t *__restrict__ order,
    const int32_t *__restrict__ seg_start,
    const int32_t *__restrict__ seg_len,
    const int32_t *__restrict__ seg_slot,
    const int32_t *__restrict__ tile_offset,
    int S,
    const int32_t *__restrict__ feature_indices,
    const int32_t *__restrict__ era_indices,
    typename Acc<DET>::out_t *__restrict__ hist,
    int F, int E, int B, int C, int K, int Kh,
    int tile_rows,
    const float *__restrict__ scales)
{
    const int tile = blockIdx.x;
    if (tile >= tile_offset[S])
        return;
    const int s = find_segment(tile_offset, S, tile);
    const int f = blockIdx.y;
    const int feat = feature_indices[f];
    const int64_t r0 = static_cast<int64_t>(seg_start[s]) + static_cast<int64_t>(tile - tile_offset[s]) * tile_rows;
    const int64_t r1 = min(r0 + tile_rows, static_cast<int64_t>(seg_start[s]) + seg_len[s]);

    float sc[3] = {1.0f, 1.0f, 1.0f};
    if (DET)
    {
        sc[0] = scales[0];
        sc[1] = scales[1];
        sc[2] = scales[2];
    }
    const int8_t *fbins = bins + static_cast<int64_t>(feat) * N_total;
    typename Acc<DET>::out_t *out = hist + ((static_cast<int64_t>(seg_slot[s]) * F + f) * E) * B * C;
    const int64_t n = (r1 - r0) * C;
    for (int64_t idx = threadIdx.x; idx < n; idx += blockDim.x)
    {
        const int64_t i = r0 + idx / C;
        const int c = static_cast<int>(idx % C);
        const int row = order[i];
        const int b = fbins[row];
        if (b < 0 || b >= B)
            continue;
        float scale;
        const float v = channel_value(grads, hess, row_weight, row, c, K, Kh, sc, scale);
        Acc<DET>::add_global(&out[(static_cast<int64_t>(era_indices[row]) * B + b) * C + c], v, scale);
    }
}

static int next_pow2(int v)
{
    int p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

static int max_shared_bytes(int device)
{
    int max_shared = 0;
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device);
    return max_shared;
}

void launch_histogram_level(
    const at::Tensor &bins,            // [F_master, N_total] int8
    const at::Tensor &grads,           // [N_total, K] float32
    const at::Tensor &hess,            // [N_total, Kh] float32 (Kh may be 0)
    const at::Tensor &row_weight,      // [N_total] float32
    const at::Tensor &order,           // [>= sum(seg_len)] int32
    const at::Tensor &seg_start,       // [S] int32
    const at::Tensor &seg_len,         // [S] int32
    const at::Tensor &seg_slot,        // [S] int32
    const at::Tensor &tile_offset,     // [S + 1] int32
    const at::Tensor &feature_indices, // [F] int32
    const at::Tensor &era_indices,     // [N_total] int32
    at::Tensor &hist,                  // [slots, F, E, B, C] int64 or float32, zero-initialized
    const at::Tensor &scales,          // [3] float32 fixed-point scales (grad, hess, count)
    int64_t max_tiles,                 // upper bound of tile_offset[S]
    int tile_rows)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(bins));
    const std::initializer_list<const at::Tensor *> tensors = {
        &grads, &hess, &row_weight, &order, &seg_start, &seg_len, &seg_slot, &tile_offset,
        &feature_indices, &era_indices, &hist, &scales};
    for (const at::Tensor *t : tensors)
        TORCH_CHECK(bins.is_cuda() && t->device() == bins.device(),
                    "histogram kernel tensors must be on the same CUDA device");
    TORCH_CHECK(bins.dim() == 2 && bins.scalar_type() == at::kChar && bins.is_contiguous(),
                "bins must be a contiguous [F, N] int8 tensor");
    TORCH_CHECK(grads.dim() == 2 && grads.is_contiguous() && hess.is_contiguous() && row_weight.is_contiguous(),
                "grads, hess and row_weight must be contiguous");
    TORCH_CHECK(hist.dim() == 5 && hist.is_contiguous(), "hist must be a contiguous [slots, F, E, B, C] tensor");
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&order, &seg_start, &seg_len, &seg_slot, &tile_offset, &feature_indices, &era_indices})
        TORCH_CHECK(t->scalar_type() == at::kInt && t->is_contiguous(), "index tensors must be contiguous int32");

    const int S = static_cast<int>(seg_start.size(0));
    const int F = static_cast<int>(hist.size(1));
    const int E = static_cast<int>(hist.size(2));
    const int B = static_cast<int>(hist.size(3));
    const int C = static_cast<int>(hist.size(4));
    const int K = static_cast<int>(grads.size(1));
    const int Kh = hess.dim() == 2 ? static_cast<int>(hess.size(1)) : 0;
    const int64_t N_total = bins.size(1);
    TORCH_CHECK(C == K + Kh + 1, "hist must have K + Kh + 1 channels");
    TORCH_CHECK(feature_indices.size(0) == F, "feature_indices must match hist");
    TORCH_CHECK(F <= 65535, "at most 65535 features per histogram launch");
    const bool det = hist.scalar_type() == at::kLong;
    TORCH_CHECK(det || hist.scalar_type() == at::kFloat, "hist must be int64 (deterministic) or float32");

    if (S == 0 || F == 0 || max_tiles <= 0)
        return;

    int device = 0;
    cudaGetDevice(&device);
    const int64_t sh_limit = max_shared_bytes(device);
    const int64_t entry = sizeof(unsigned int);
    const int64_t eb_bytes = static_cast<int64_t>(E) * B * entry;
    // Channels per block: limited by shared memory and by MAX_VALS registers per lane
    const int64_t tail_bytes = 32 * MAX_VALS * entry;
    int chunk = static_cast<int>(std::min<int64_t>({static_cast<int64_t>(C), (sh_limit - tail_bytes) / eb_bytes, 32 * MAX_VALS}));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;

#define WARPGBM_HIST_ARGS(DET_)                                                                 \
    bins.data_ptr<int8_t>(), N_total, grads.data_ptr<float>(), hess.data_ptr<float>(),          \
        row_weight.data_ptr<float>(), order.data_ptr<int32_t>(), seg_start.data_ptr<int32_t>(), \
        seg_len.data_ptr<int32_t>(), seg_slot.data_ptr<int32_t>(), tile_offset.data_ptr<int32_t>(), S, \
        feature_indices.data_ptr<int32_t>(), era_indices.data_ptr<int32_t>(),                  \
        reinterpret_cast<Acc<DET_>::out_t *>(hist.data_ptr())
    if (chunk >= 1)
    {
        // Balance the channel chunks: e.g. 41 channels with room for 32 -> 2 x 21
        const int n_chunks = (C + chunk - 1) / chunk;
        chunk = (C + n_chunks - 1) / n_chunks;
        // Lanes per row; chunk <= 32 * MAX_VALS, so a lane holds at most MAX_VALS channels
        const int G = std::min(32, next_pow2(chunk));
        // Features per block: up to 8 that fit into 32 KB (several blocks per SM)
        const int64_t feature_bytes = eb_bytes * chunk;
        int FG = 1;
        while (FG < 8 && FG < F && 2 * FG * feature_bytes + tail_bytes <= std::min<int64_t>(sh_limit, 32 * 1024))
            FG *= 2;
        const int NV = (chunk + G - 1) / G;
        dim3 grid(static_cast<unsigned int>(max_tiles), (F + FG - 1) / FG, n_chunks);
        const size_t shared = static_cast<size_t>(FG * feature_bytes + tail_bytes);
#define WARPGBM_HIST_LAUNCH(DET_, FG_, NV_)                                         \
    histogram_level_kernel<DET_, FG_, NV_><<<grid, threads, shared, stream>>>(      \
        WARPGBM_HIST_ARGS(DET_), F, E, B, C, K, Kh, chunk, G, tile_rows, scales.data_ptr<float>())
#define WARPGBM_HIST_NV(DET_, FG_)                                   \
    switch (NV)                                                      \
    {                                                                \
    case 1: WARPGBM_HIST_LAUNCH(DET_, FG_, 1); break;                \
    case 2: WARPGBM_HIST_LAUNCH(DET_, FG_, 2); break;                \
    default: WARPGBM_HIST_LAUNCH(DET_, FG_, 4); break;               \
    }
#define WARPGBM_HIST_FG(DET_)                         \
    switch (FG)                                       \
    {                                                 \
    case 1: WARPGBM_HIST_NV(DET_, 1); break;          \
    case 2: WARPGBM_HIST_NV(DET_, 2); break;          \
    case 4: WARPGBM_HIST_NV(DET_, 4); break;          \
    default: WARPGBM_HIST_NV(DET_, 8); break;         \
    }
        if (det)
        {
            WARPGBM_HIST_FG(true)
        }
        else
        {
            WARPGBM_HIST_FG(false)
        }
#undef WARPGBM_HIST_FG
#undef WARPGBM_HIST_NV
#undef WARPGBM_HIST_LAUNCH
    }
    else
    {
        dim3 grid(static_cast<unsigned int>(max_tiles), F, 1);
        if (det)
            histogram_level_global_kernel<true><<<grid, threads, 0, stream>>>(
                WARPGBM_HIST_ARGS(true), F, E, B, C, K, Kh, tile_rows, scales.data_ptr<float>());
        else
            histogram_level_global_kernel<false><<<grid, threads, 0, stream>>>(
                WARPGBM_HIST_ARGS(false), F, E, B, C, K, Kh, tile_rows, scales.data_ptr<float>());
    }
#undef WARPGBM_HIST_ARGS

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM histogram kernel failed: ", cudaGetErrorString(err));
}

// ─── Node totals ───────────────────────────────────────────────────────────────
// sums[node, c] = sum over rows with node_of_row == node of channel c ([grad | hess | count]).
// Rows with node_of_row < 0 are skipped. Same fixed-point scheme as the histograms, with one tile
// of tile_rows rows per block.
template <bool DET>
__global__ void node_sums_kernel(
    const int32_t *__restrict__ node_of_row, int64_t N,
    const float *__restrict__ grads,
    const float *__restrict__ hess,
    const float *__restrict__ row_weight,
    int K, int Kh, int C, int n_nodes, int tile_rows, bool use_shared,
    typename Acc<DET>::out_t *__restrict__ sums,
    const float *__restrict__ scales)
{
    extern __shared__ unsigned char shraw[];
    unsigned int *sh = reinterpret_cast<unsigned int *>(shraw); // [n_nodes, C]
    if (use_shared)
    {
        for (int i = threadIdx.x; i < n_nodes * C; i += blockDim.x)
            sh[i] = 0;
        __syncthreads();
    }
    const float sc[3] = {scales[0], scales[1], scales[2]};

    const int64_t r0 = static_cast<int64_t>(blockIdx.x) * tile_rows;
    const int64_t r1 = min(r0 + tile_rows, N);
    const int64_t n = (r1 - r0) * C;
    for (int64_t idx = threadIdx.x; idx < n; idx += blockDim.x)
    {
        const int64_t row = r0 + idx / C;
        const int c = static_cast<int>(idx % C);
        const int node = node_of_row[row];
        if (node < 0)
            continue;
        float scale;
        const float v = channel_value(grads, hess, row_weight, row, c, K, Kh, sc, scale);
        if (use_shared)
            atomicAdd(&sh[node * C + c], to_fixed32(v, scale));
        else
            Acc<DET>::add_global(&sums[static_cast<int64_t>(node) * C + c], v, scale);
    }
    if (use_shared)
    {
        __syncthreads();
        for (int i = threadIdx.x; i < n_nodes * C; i += blockDim.x)
            Acc<DET>::flush(&sums[i], sh[i], 1.0f / channel_scale(scales, i % C, K, Kh));
    }
}

void launch_node_sums(
    const at::Tensor &node_of_row, // [N] int32, -1 = skip
    const at::Tensor &grads,       // [N, K]
    const at::Tensor &hess,        // [N, Kh]
    const at::Tensor &row_weight,  // [N]
    at::Tensor &sums,              // [n_nodes, C] int64 or float32, zero-initialized
    const at::Tensor &scales,      // [3]
    int tile_rows)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(node_of_row));
    for (const at::Tensor *t : std::initializer_list<const at::Tensor *>{&grads, &hess, &row_weight, &sums, &scales})
        TORCH_CHECK(node_of_row.is_cuda() && t->device() == node_of_row.device(),
                    "node_sums tensors must be on the same CUDA device");
    TORCH_CHECK(node_of_row.scalar_type() == at::kInt && node_of_row.is_contiguous(), "node_of_row must be int32");
    TORCH_CHECK(sums.dim() == 2 && sums.is_contiguous(), "sums must be [n_nodes, C]");
    const int64_t N = node_of_row.size(0);
    const int K = static_cast<int>(grads.size(1));
    const int Kh = hess.dim() == 2 ? static_cast<int>(hess.size(1)) : 0;
    const int n_nodes = static_cast<int>(sums.size(0));
    const int C = static_cast<int>(sums.size(1));
    TORCH_CHECK(C == K + Kh + 1, "sums must have K + Kh + 1 channels");
    const bool det = sums.scalar_type() == at::kLong;
    TORCH_CHECK(det || sums.scalar_type() == at::kFloat, "sums must be int64 or float32");
    if (N == 0 || n_nodes == 0)
        return;

    int device = 0;
    cudaGetDevice(&device);
    const size_t shared = static_cast<size_t>(n_nodes) * C * sizeof(unsigned int);
    const bool use_shared = shared <= static_cast<size_t>(max_shared_bytes(device));
    const int64_t blocks = (N + tile_rows - 1) / tile_rows;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (det)
        node_sums_kernel<true><<<blocks, 256, use_shared ? shared : 0, stream>>>(
            node_of_row.data_ptr<int32_t>(), N, grads.data_ptr<float>(), hess.data_ptr<float>(),
            row_weight.data_ptr<float>(), K, Kh, C, n_nodes, tile_rows, use_shared,
            reinterpret_cast<long long *>(sums.data_ptr<int64_t>()), scales.data_ptr<float>());
    else
        node_sums_kernel<false><<<blocks, 256, use_shared ? shared : 0, stream>>>(
            node_of_row.data_ptr<int32_t>(), N, grads.data_ptr<float>(), hess.data_ptr<float>(),
            row_weight.data_ptr<float>(), K, Kh, C, n_nodes, tile_rows, use_shared,
            sums.data_ptr<float>(), scales.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM node_sums kernel failed: ", cudaGetErrorString(err));
}

// ─── Sibling histograms ────────────────────────────────────────────────────────
// hist [2P, ...]: for every parent p that was split (split[p]), one child was built (2p if
// left_built[p], else 2p + 1); the other one becomes parent - built child:
// hist[sibling] = parent[p] - hist[built]. Children of parents without a split are left as they are.
template <typename T>
__global__ void sibling_subtract_kernel(
    T *__restrict__ hist, const T *__restrict__ parent, const bool *__restrict__ left_built,
    const bool *__restrict__ split, int64_t P, int64_t node_size)
{
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < P * node_size; i += stride)
    {
        const int64_t p = i / node_size;
        if (!split[p])
            continue;
        const int64_t j = i - p * node_size;
        const int64_t built = 2 * p + (left_built[p] ? 0 : 1);
        const int64_t sibling = 4 * p + 1 - built; // 2p + (1 - side)
        hist[sibling * node_size + j] = parent[i] - hist[built * node_size + j];
    }
}

void launch_sibling_subtract(at::Tensor &hist, const at::Tensor &parent, const at::Tensor &left_built,
                             const at::Tensor &split)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(hist));
    TORCH_CHECK(hist.is_contiguous() && parent.is_contiguous() && hist.scalar_type() == parent.scalar_type(),
                "hist and parent must be contiguous with the same dtype");
    TORCH_CHECK(left_built.scalar_type() == at::kBool && left_built.is_contiguous()
                    && split.scalar_type() == at::kBool && split.is_contiguous(),
                "left_built and split must be contiguous bool tensors");
    const int64_t P = parent.size(0);
    TORCH_CHECK(hist.size(0) == 2 * P && left_built.numel() == P, "hist must have two children per parent");
    if (P == 0)
        return;
    const int64_t node_size = parent.numel() / P;
    const int threads = 256;
    const int64_t blocks = std::min<int64_t>((P * node_size + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (hist.scalar_type() == at::kLong)
        sibling_subtract_kernel<long long><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<long long *>(hist.data_ptr<int64_t>()),
            reinterpret_cast<const long long *>(parent.data_ptr<int64_t>()),
            left_built.data_ptr<bool>(), split.data_ptr<bool>(), P, node_size);
    else
        sibling_subtract_kernel<float><<<blocks, threads, 0, stream>>>(
            hist.data_ptr<float>(), parent.data_ptr<float>(), left_built.data_ptr<bool>(),
            split.data_ptr<bool>(), P, node_size);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WarpGBM sibling kernel failed: ", cudaGetErrorString(err));
}
