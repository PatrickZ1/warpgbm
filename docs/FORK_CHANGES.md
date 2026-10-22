# Fork changes: review, fixes and new features

This document covers everything this fork changes relative to upstream WarpGBM v2.2.0
(`15aa526`, jefferythewind/warpgbm):

1. a review of the fork's first commits (Feb–Apr 2026), with a verdict per commit;
2. the bugs found during that review (issues **A–H** and a few smaller ones), with root causes
   and fixes;
3. the features added afterwards (custom objective API, NaN feature bins, deterministic
   histograms, …);
4. behaviour changes that matter for existing users, and how everything was verified.

See [CUSTOM_OBJECTIVES.md](CUSTOM_OBJECTIVES.md) for the custom-loss guide.

---

## 1. Review of the initial fork commits

| Commit | Change | Verdict |
|---|---|---|
| `46a90db` basic sample_weight implementation | `sample_weight` / `sample_weight_eval` (normalized to mean 1); weighted metrics (`weighted_*_torch`); the histogram kernel now takes a **hessian** array instead of counting rows; Newton leaves `Σg/(Σh+L2)`; lazy loading of the CUDA extension | Plumbing and validation are correct. **Bug A** (min_child_weight now compared with the hessian sum) and **Bug E** (unweighted base prediction) come from this commit. The multiclass path changed from "mean of gradients" to proper Newton leaves, which is an improvement |
| `a2ff160` extend tests | weight-effect tests (noise data, constant model) | Fine |
| `31583c2` monotonic constraints | per-feature `-1/0/+1`, vector or dict; split candidates whose left/right leaf order violates the constraint are rejected (in every era when eras are used); save/load and warm-start validation | API and validation are correct. **Bug C**: the check is only local to each split; no bounds are passed to the subtree, so monotonicity breaks when features interact. The tests only probed slices without interactions |
| `792a947` / `3238815` binning timing prints | added, then removed | No net change |
| `2403bc0` streaming prediction performance | caches the flattened forest between `predict` calls (invalidated on training), vectorized binning, `device=` instead of hard-coded `"cuda"`, batched per-class prediction | Correct (chunked predictions are identical). Minor: during training with an eval set the cache was invalidated every tree, so the whole forest was re-flattened at each evaluation (O(T²)). Fixed by incremental eval scores |
| `7da33df` multi-output regression | `objective="multi_regression"`, 2-D targets with NaN masking, 1-D or 2-D weights, vector leaves, warm start, save/load, multi-output MSE metric | Masking, weights and leaves are correct. **Bug B**: the split gain uses `(Σ_k G_k)² / Σ_k H_k`, so outputs with opposite gradients cancel and the model can learn nothing. **Bug D**: monotonicity uses that summed direction too. `min_child_weight` compared the hessian *summed over outputs* (≈ K × rows) |
| `543d1c5` README | "Changes" section | Fine |

Test status of the fork at `543d1c5` (fresh build): **49 passed, 1 failed**
(`test_feature_importance.py::test_iris_feature_importance`, caused by Bug A). The in-repo
`warpgbm/cuda/*.so` (dated Feb 23) was stale and segfaulted; it is a gitignored build artifact and
has been rebuilt.

---

## 2. Issues found and fixed

Each issue has a regression test in `tests/test_regressions.py` (and `tests/test_nan_features.py`
for H).

| # | Problem | Origin | Evidence before → after | Fix |
|---|---|---|---|---|
| **A** | `min_child_weight` was compared with the **hessian** sum. Softmax hessians are p(1−p) ≤ 0.25, so with default parameters small data sets never split | `46a90db` | iris, defaults: accuracy **0.33 → 0.98** | separate *count* histogram channel accumulating row sample weights; `min_child_weight` is the minimum weight sum per child (≈ row count when unweighted, i.e. the upstream behaviour). New `min_sum_hessian` (default 1e-3) guards the hessian side |
| **B** | Multi-output gain `(ΣG)²/ΣH` lets outputs with opposite gradients cancel | `7da33df` | Y = [f, −f]: MSE **4.05 (= variance) → 0.168** (same as a single-output model) | gain = Σ_k (G_Lk²/(H_Lk+λ) + G_Rk²/(H_Rk+λ) − G_k²/(H_k+λ)) from per-output histograms `[E, F, B, K]` |
| **C** | Monotonicity was checked only locally at each split, with no bounds for the subtree | `31583c2` | increasing constraint: prediction fell **9.7 → 4.9** along x0; after the fix min Δ ≥ 0 on all slices | LightGBM-style bound propagation: at a constrained split, mid = (v_L + v_R)/2 bounds the children (increasing: left ≤ mid ≤ right); leaves are clamped to their bounds |
| **D** | Multi-output monotonicity used the summed direction | `7da33df` | per-output min Δ ≥ 0 after the fix | the split kernel reports a direction only if **all** outputs agree; bounds are per output |
| **E** | Weighted regression `base_prediction` was the unweighted mean | `46a90db` | **0.008 → 0.672** (weighted mean 0.672) | weighted, masked mean in `SquaredError.init_score` |
| **F** | Classification trained from log class priors, but `predict_proba` started from log(1/K) | upstream | binary at 10 % positives: mean p **0.50 → 0.106** | the initial scores (weighted log priors) are stored in `base_prediction` and used at prediction |
| **G** | The histogram kernel needs 2·E·B floats of shared memory. Above 48 KB (≈ >190 eras at 32 bins) the launch failed, which was only `printf`'d, and training silently used empty histograms | upstream | 400 eras: MSE **1.009 → 0.274** | automatic global-memory fallback kernel; all CUDA launch errors now raise (`TORCH_CHECK`) |
| **H** | A single NaN in a feature made all its quantile edges NaN, so the feature became useless | upstream | edges `[nan, …]` → finite; train MSE 9.12 → 0.84 (var 9.37) | NaN feature bin (see §3.3) |

Smaller issues fixed along the way:

* **Run-to-run nondeterminism.** Float `atomicAdd` order made histograms differ in the last bits,
  and exact ties (common in round 0 of softmax, where every row has the same probability) were
  then broken randomly. `test_random_state_multiclass` failed ~40 % of the time once Bug A was
  fixed (the old code passed only because it hardly split). The fix is deterministic
  fixed-point histograms (§3.4); 8/8 repeated fits now give identical forests.
* Warm start used to **recompute bin edges** from the new data, which silently changed what the
  stored split bins mean. The edges of the first fit are now reused.
* `save_model` pickled CUDA tensors (bin edges, base score); they are now stored on the CPU and
  moved to `device` on load. Tree split features are stored as Python ints instead of 0-d CUDA
  tensors.
* `WarpGBM.__init__` allocated an unused CUDA tensor (`best_bin`), so even constructing a model
  required a GPU. Removed.
* 1-D regression targets with NaN were not masked (everything became NaN); they are now treated
  as missing, as in multi-output regression.

---

## 3. New features

### 3.1 Unified objective engine and custom loss functions

`warpgbm/objectives.py` defines the `Objective` interface (GPU tensors, `grad = dL/dscore`,
`hess = d²L/dscore²`). **All** objectives, built-in and custom, run through a single training
loop in `core.py`. This replaces the three previous copies (`_fit_regression`,
`_fit_multi_regression`, `_fit_classification` and their `grow_forest*` / `compute_eval*` twins):

* built-ins: `SquaredError` (`"regression"`, `"multi_regression"`) and `Softmax`
  (`"multiclass"`, `"binary"`);
* custom losses are **per-sample** torch expressions `loss(y, pred)` returning `[N, K]` or `[N]`.
  Gradients and the exact diagonal hessian come from autograd (one extra backward pass per
  output). Pass the function directly (`WarpGBM(objective=loss)`), wrap it in
  `CustomObjective(loss_fn, grad_hess_fn=None, hessian="autograd"|"ones", min_hessian, init_fn,
  transform_fn, output_strategy)`, or subclass `Objective` and implement `loss` and/or
  `grad_hess`;
* the engine applies sample weights (`[N]` or `[N, K]`) and masks missing targets, so custom
  losses contain neither;
* `output_strategy="vector"` (one tree per round with K-dimensional leaves) or `"per_output"`
  (K trees per round);
* `fit(..., eval_metric=callable)` with signature `(y_true, pred, weight, mask) -> float`,
  `eval_metric=None` (objective default) and `"loss"` (weighted mean of the objective's loss);
* `predict_raw(X)` returns the untransformed scores (logits for classification).

The forest storage formats are unchanged: scalar leaves for single output, list leaves for vector
trees, a list of K trees per round for `per_output`. Models saved by the previous version still
load (tested).

### 3.2 Early stopping with rollback, metric direction and auxiliary data

* **Best iteration.** Every evaluation (every `eval_every_n_trees` rounds and at the last round)
  is recorded in `evals_result_` (`"train"` / `"eval"`) and `eval_iterations_`; `best_iteration_`
  (rounds, 1-based) and `best_score_` hold the best one. With `early_stopping_rounds`, training
  stops after that many evaluations without improvement and the **default model is the best
  round**: `predict` / `predict_raw` / `predict_proba` / importances / `train_predictions_` use
  the first `best_iteration_` rounds. Before, all trees up to the stop were used (the hedge project
  worked around this by setting `_trees_trained`).
* **`num_trees=`** on all prediction and importance methods selects any prefix of rounds;
  `truncate(num_trees=None)` drops the rounds after `best_iteration_`; `num_trees_` is the number
  of trained rounds.
* **Importances follow the used trees.** Each split node stores its total gain, so
  `get_feature_importance(num_trees=n)` is exact for any `n`. Per-era importances are stored as
  snapshots at `best_iteration_` and at all rounds.
* **Metric direction.** `"accuracy"` and `"corr"` are maximized and reported as natural values
  (before: `1 - accuracy`, `1 - corr`); callables are minimized unless wrapped in
  `Metric(fn, higher_is_better=True, name=...)`.
* **Auxiliary per-row data.** `fit(..., eval_aux=..., train_aux=...)` passes `[n]` / `[n, A]`
  arrays to a callable metric as `aux=`, e.g. raw targets and scales for a competition metric
  when the model trains on scaled targets. The eval metric is also computed on the training set
  (`evals_result_["train"]`), unless it requires `aux` and no `train_aux` was given.
* **`train_predictions_`**: in-sample raw scores of the default model, `[n_samples, n_outputs]`,
  taken from the engine's training scores (no second predict pass).
* Persistence: `save_model` keeps `best_iteration_`, `best_score_`, the eval history and the
  importance snapshots; older pickles load as "all trees, no best round".

### 3.3 NaN features get their own bin

* Bin edges are computed with `torch.nanquantile`; NaN values map to a dedicated bin
  `num_bins` (histograms have `num_bins + 1` bins; `num_bins ≤ 127` still fits int8).
* Each split evaluates every threshold twice: once with NaN going right and once with NaN going
  left. The learned default direction is stored per node (`"nan_left"`) and used by the CUDA
  prediction kernel (the tree tensor now has 7 columns). The threshold "all values left, NaN
  right" isolates missing values.
* NaN-left candidates are not considered when the node has no NaN for that feature, and neither
  are thresholds whose bin is empty in the node. So without NaNs the candidate set, the
  tie-breaking and the resulting trees are the same as before. A NaN that appears only at
  predict time follows the right branch, like `+inf`.

### 3.4 Deterministic, faster histogram kernel

* Gradients, hessians and row weights are accumulated as fixed-point integers with a power-of-two
  scale per tree (32-bit in shared memory, 64-bit in global memory), then converted back to
  float. Integer addition is associative, so histograms and splits no longer depend on the order
  of atomic operations. The quantization step is about 2.4e-7 × max|g| per row at the default
  `threads_per_block × rows_per_thread = 256`.
* One kernel launch covers all K output channels (`grid.z`) plus the count channel. With many
  eras, a block processes several row tiles to amortize the per-block histogram init/flush.
* The split kernel uses one thread per candidate with the feature histogram staged in shared
  memory (previously one thread per feature walked all bins serially).
* Per-node host syncs were cut from ~6 to ~1: leaf values are fetched once per tree, feature
  importance is accumulated on the GPU, and the monotone check is skipped on the host when the
  tree has no constrained feature.

### 3.5 Other

* The prediction kernel sums the trees of a sample in a fixed order (one thread per sample)
  instead of using float `atomicAdd` across trees. Repeated `predict` calls are now
  bit-identical; before, they differed by a few ULPs.
* `min_sum_hessian` hyperparameter (default 1e-3). `min_child_weight` now also accepts floats
  (≥ 0).
* Training-time evaluation adds only the newest round to cached eval scores instead of
  re-flattening the forest.

### 3.6 Several GPUs and concurrent fits (v2.5.0)

* **Per-model RNG.** `fit` creates `self._rng = torch.Generator(device=self.device)`, seeded
  with `random_state` (or a nondeterministic seed when it is `None`), fresh on every fit
  including warm starts. The `colsample_bytree` draws (`torch.randperm`) use it, and it is the
  only random draw in training. The generator is released at the end of `fit` and not pickled.
  `fit` no longer calls `torch.manual_seed`, `np.random.seed` or `torch.cuda.manual_seed(_all)`.
* **Device guard.** Every launcher in `warpgbm/cuda/*.cu` (binner, histogram including the
  fixed-point conversion, split, predict) starts with
  `at::cuda::OptionalCUDAGuard guard(device_of(<first input>))`, checks that all its tensors
  are on that device, and launches on `at::cuda::getCurrentCUDAStream()`. The histogram
  workspace and the shared-memory size queries therefore refer to the tensors' device.
  `WarpGBM(device="cuda:1")` works from any thread without `torch.cuda.set_device`.
* **Current device (fix after 2.7.0).** The kernel guards were not enough: on 2.7.0 a fit on
  `cuda:1` from a thread whose current device is `cuda:0` died with an illegal memory access
  (2 × T4; the `cuda:1` tests had only ever been skipped). Now `fit`, the GPU prediction path,
  `compute_eval`, the binning methods, `fit_bins` and `apply_bins` run with the model device as
  the current device (`_on_model_device`, the caller's device is restored). The event that
  guards the host copy of a tree's records is recorded on the model device's stream (it was
  recorded on the current device's stream, so the records could be read before the copy had
  finished). CUDA graphs are captured on a side stream per device (`_capture_stream`):
  `torch.cuda.graph`'s default capture stream is one stream per process, created on the device
  of the first capture. Checked on two GPUs (Kaggle 2 × T4, torch 2.11) with
  `tests/multi_gpu_check.py`. Its diagnosis step (device scope removed, `CUDA_LAUNCH_BLOCKING=1`)
  located the 2.7.0 fault in PyTorch itself: `torch.ldexp` in `_device_scale` (a `cuda:1`
  tensor, current device `cuda:0`), before the first tree. Hence the scope on the entry points
  rather than per-op fixes.
* Together with the deterministic histograms (§3.4), models with the same `random_state` fitted
  concurrently in two threads (on one GPU or on two) give forests identical to sequential fits.
* **`verbose`** constructor parameter (default `True`) gates every `print` in `core.py`. It is
  not saved by `save_model`; the loading instance's value applies.

### 3.7 Explicit pre-binned input, public binning, training-loss switch (v2.6.0)

* **`prebinned=True`** (constructor, saved by `save_model`). Integer input holds bin indices
  `0 … num_bins-1` and `-1` for a missing value. `-1` is mapped to the internal NaN bin `num_bins`
  on the GPU (`masked_fill_`). The same contract and validation apply wherever data enters:
  `X` and `X_eval` in `fit`, `predict`, `predict_raw`, `predict_proba`. Non-integer dtypes raise
  `TypeError`. Values outside `[-1, num_bins-1]` raise `ValueError` naming the column; wider
  dtypes are checked before narrowing to int8, so 300 does not wrap to 44. The mode is stored as
  `prebinned_`, and `bin_edges` is `None` for such models.
* **The mode is decided once, at fit.** `prebinned=None` (default) keeps the old auto-detection
  (integers all in `[0, num_bins-1]`), but stores its decision in `prebinned_`. Integer input that
  fails the check is quantile-binned with a `UserWarning` that points to `prebinned=True/False`.
  `prebinned=False` always quantile-bins. Before, `predict` re-ran the heuristic on every call, and
  that caused two bugs:
  - a model fitted on floats treated small integer input as bin indices;
  - pre-binned data containing the NaN bin (value `num_bins`) failed the check and was silently
    quantile-binned again.
  Negative integers also passed the old check (`max < num_bins`) and reached the histogram kernel
  as negative bin indices.
* **`warpgbm.fit_bins(X, num_bins, device)`** returns the `[F, num_bins-1]` edge tensor, and
  **`warpgbm.apply_bins(X, edges, device)`** returns int8 bins with `-1` (`warpgbm.MISSING_BIN`)
  for NaN: numpy in, numpy out; tensor in, tensor out. Both live in `warpgbm/binning.py`, which
  `fit` / `predict` use too, so external binning cannot drift from the internal one. A model fitted
  on `apply_bins(X, fit_bins(X, B))` with `prebinned=True` has the same forest, eval history and
  predictions as one fitted on `X` (tested).
* **Vectorised binning.** `nanquantile` runs on blocks of features (≤ 2^24 elements per block, to
  bound the sort workspace) instead of one column at a time. The CUDA binner bins the whole
  row-major `[N, F]` matrix in one launch (grid-stride, 64-bit indices) instead of one launch per
  column on a transposed copy. Same rule (linear scan, first edge > x), so edges, bins, forests and
  predictions are bit-identical to v2.5.0 (checked at 7/64/127 bins with NaNs, ties and an
  all-NaN column). Applying edges is ~2× faster (200k × 100: 29 → 15 ms). Edge computation is
  dominated by the sort and is unchanged.
* **`record_training_loss`** (`"eval"` default, `"all"`, `"none"`). `training_loss` used to cost a
  full training-set loss plus a `.item()` host sync after every tree. It is now computed only at
  evaluation checkpoints, aligned with `eval_iterations_`, and reuses the training score when the
  eval metric is that same loss. Training itself is unchanged (tested). Fits with an eval set were
  ~5–10 % faster at 20k rows × 300 trees; the saving grows with N × K and with costly custom losses.
* **Fix: NaNs with the sign bit set.** NaNs with the sign bit set, which numpy and polars produce
  for 0/0, were sorted before all numbers by the CUDA sort inside `torch.nanquantile`, so they were
  counted as values. The lower edges then became NaN → 0 and the edges were unsorted. `fit` mis-binned
  the feature silently, and `apply_bins` rejected the edges. CPU `nanquantile` was not affected.
  `_quantile_edges` now canonicalises NaNs per feature block before taking the quantiles. The binning
  kernel uses `isnan()` and was always correct. Edges for data with ordinary NaNs are unchanged.
* `_get_node_kernel` moved to `warpgbm/cuda/__init__.py` (`warpgbm.core._get_node_kernel` still
  imports). `predict*` now checks the feature count of its input.

### 3.8 Training and prediction performance, `deterministic` switch (v2.7.0)

The target workload is small, wide data: ~10k rows, ~1k features, K = 20 outputs, a few eras.
v2.6.0 spent most of its time there in the histogram kernel and in per-node host work.
`benchmarks/bench_small_wide.py` reproduces the numbers below.

**Diagnosis (v2.6.0, GTX 1660 Ti, 10k × 1000, K = 20, 3 eras, 16 bins, depth 4, colsample 0.4).**
53 ms per tree. 84 % of the GPU time went to the histogram kernel. One block per feature read the
row-major `[N, F]` bins one byte per thread (stride F, uncoalesced), once per output (grid.z = K).
Every 256-row block also flushed `E·B·3` 64-bit global atomics. Each node then added several host
syncs (best-split `.tolist()`, boolean indexing, the fixed-point scales). Prediction re-flattened
the forest in Python for every new `num_trees` (370 ms for 80 trees × 20 outputs) and launched one
kernel per output.

**Training engine (both modes).**

* **Level-wise growth.** All nodes of a depth are processed together, with a few kernel launches
  and no host sync (`_grow_tree_device`):
  1. node totals (`node_sums`) give the leaf values and the monotone bounds;
  2. histograms of the level;
  3. `level_split`: the candidates of all nodes;
  4. `select_split`: the best split per node;
  5. rows move to their child (heap numbering).

  The trees are full-depth with independent node splits, so this grows the same trees as the
  former depth-first recursion. The tree is copied to the host once, at the end, as one record
  tensor. Python builds the same node dicts as before, so pickles, importances, `truncate` and
  warm start are unchanged.
* **CUDA graphs.** Every tree of a fit launches the same kernels on tensors of the same shapes. The
  second tree of each shape is captured as a CUDA graph; later trees copy their gradients,
  hessians and sampled features into its static buffers and replay it. That is one launch instead
  of about 100 small ones. A process-wide lock serialises captures, because concurrent captures
  in several threads break PyTorch's caching allocator. If a capture fails, the trees run
  eagerly.
* **Pipelined rounds.** Round i + 1 is enqueued before the host turns round i into dicts and
  evaluates it. Records reach the host through pinned memory, eval uploads are asynchronous, and
  the eval metric gets a snapshot of the training scores after round i. When early stopping ends
  training at round i, the speculative round i + 1 is dropped. Training then rolls back to the
  best round, so its score update does not matter.
* **Histogram kernel** (`histogram_level`):
  * bins are stored feature-major `[F, N]` during fit, so reads are coalesced;
  * one block takes a tile of rows of one node and a group of up to 8 features, with all output
    channels;
  * the lanes of a warp share a row and take its channels: consecutive shared-memory addresses,
    coalesced `grads[row, :]`, and each value converted once for the whole feature group;
  * the inner loop is branch-free: inactive lanes add 0, features beyond the group add into a
    scratch tail;
  * on the target workload, histogram time per tree went from 37 ms to 2.5 ms. It is now limited by
    shared-memory throughput (78 % in Nsight Compute).
* **Integer shared-memory accumulation in both modes.** On Turing, shared-memory float
  `atomicAdd` is about 2× slower than integer atomics; measured, it is presumably emulated with a
  compare-and-swap loop. So both modes accumulate a tile's rows as 32-bit fixed-point integers.
  The power-of-two scales come from `max|g|` and the tile size, computed on the device without a
  host sync.
* **Unit-hessian shortcut.** For squared error without missing targets and with the same weight
  for every output, every hessian equals the row weight. The hessian sums are then read from the
  row-weight ("count") channel: K + 1 channels instead of 2K + 1. Objectives declare this with the
  private hint `Objective._hessian_is_one()`.
* **Sibling subtraction** is one fused kernel (`sibling_subtract`). Only the built slots are
  zeroed, and parents without a split are skipped.
* **Split kernel.** One block per (feature, node) loops over the eras. It converts each histogram
  bin to float once and builds the prefix sums in shared memory, in bin order (deterministic). It
  then evaluates the candidates and writes the per-era gains, the era-mean gain, the directional
  agreement and the monotone flag. `select_split` reproduces the former `find_best_split`
  semantics: maximal agreement over all candidates of the node, then `gain > min_split_gain`,
  then the monotone mask, then the first maximal gain.
* **Memory guard.** A level keeps its histograms (needed for subtraction) only if
  parent + child levels fit in 40 % of the free GPU memory at fit. Otherwise the nodes of the
  level are processed in chunks and both children are built directly. In deterministic mode the
  forests are identical either way (tested).
* `gc.collect()` at the end of `fit` is gone; it cost about 55 ms per fit.
* The `threads_per_block` and `rows_per_thread` parameters were removed: the kernels choose their
  launch configuration. Passing them raises `TypeError`.

**`deterministic` (constructor, default `True`, saved by `save_model`; older pickles load as `True`).**

* `True`:
  * histograms are int64 fixed-point sums; sibling histograms are exact (integer subtraction);
  * node totals are fixed-point too;
  * repeated fits, concurrent fits, CUDA-graph replay and eager runs give identical forests.
* `False`:
  * the tile sums are flushed into float32 histograms with float atomics, so rounding depends on
    the order of the atomics;
  * half the histogram memory and traffic (memset, subtraction, split kernel);
  * same algorithm: predictions differ from deterministic mode by float rounding, and near-tie
    splits can flip.
* `random_state` controls the column sampling in both modes.
* Lower-precision arithmetic (fp16 / bf16) was considered and not adopted. Sums over thousands
  of rows need fp32, and the bottleneck is shared-memory atomics, not FLOPs or bandwidth. Turing
  (GTX 16xx, T4) has no tensor-core path that applies here.

**Prediction.**

* **Packed forest.** Each round is flattened once, for all outputs, into structure-of-arrays numpy
  data (`_pack_round`):
  * `nodes [t, M, 4]`;
  * leaves `[t, M, Lw]`, already multiplied by the learning rate;
  * `out_col` for per-output trees.

  It is cached with a lazily uploaded GPU copy and extended as rounds are added. Truncation keeps
  the rounds before the cut. `num_trees=c` is a slice. Empty rounds cannot shift slices, because
  the cache stores the tree count after each round.
* **One kernel for all outputs** (`predict_forest`):
  * one thread per row, register accumulators for up to 32 outputs;
  * the trees are added in order with plain additions (no FMA with the learning rate), then the
    base score;
  * explicit strides, so it reads row-major and feature-major bins;
  * checkpoints write several prefixes in one pass.
* **`staged_predict_raw` / `staged_predict` / `staged_predict_proba(X, rounds)`** return
  `(len(rounds), n, …)` from one pass. Each entry is bit-identical to
  `predict_raw(X, num_trees=r)`. This replaces the hedge project's `tune.py::staged_predict`
  workaround, which used the removed internals `_get_cached_tree_tensors` /
  `_accumulate_tree_tensors`.
* **Opt-in CPU path.** `device="cpu"` on `predict`, `predict_raw`, `predict_proba` and
  `staged_*` makes no CUDA call:
  * NumPy binning with the same rule as the CUDA binner;
  * traversal vectorised over trees on flat per-field arrays;
  * sequential float32 accumulation, so raw scores are bit-identical to the GPU kernel (softmax
    probabilities may differ in the last bit).

  This is the low-latency option for a few rows.
* **Training eval.** The eval scores accumulate each new round with the prediction kernel (per-row
  base). They equal `predict_raw(X_eval, num_trees=i + 1)` exactly, without re-flattening.

**Results** (GTX 1660 Ti; ms per tree including the per-fit setup amortised over 30 trees, idle
GPU; `python benchmarks/bench_small_wide.py --large`). Base config: 10k × 1000, K = 20, 3 eras,
16 bins, depth 4, colsample 0.4, `min_child_weight=50`, 2 % NaN features.

| variant | v2.6.0 | v2.7.0 `deterministic=True` | v2.7.0 `deterministic=False` |
|---|---|---|---|
| base config | 53.1 | 6.6 (8.0×) | 5.1 (10.4×) |
| depth 7 | 124.7 | 15.3 (8.2×) | 12.9 (9.7×) |
| 64 bins | 160.7 | 11.7 (13.7×) | 10.7 (15.0×) |
| 8 eras | 74.2 | 9.0 (8.2×) | 8.3 (8.9×) |
| colsample 1.0 | 111.1 | 9.6 (11.6×) | 9.0 (12.3×) |
| K = 1 | 13.6 | 3.0 (4.5×) | 2.7 (5.0×) |
| depth 7, 64 bins, 8 eras, colsample 1.0 | 778.3 | 171.5 (4.5×) | 121.1 (6.4×) |

Without the per-fit setup (binning, upload, first eager tree and graph capture, about
15–30 ms per fit), the base config trains at 5.1 / 4.6 ms per tree, the GPU time.

Hedge-like configuration: 1681 pre-binned features, colsample 0.4, K = 20 with 5 % NaN targets
(so no unit-hessian shortcut), 16 bins, depth 4, `min_child_weight=300`, 3 eras, 10k rows,
100 trees.

| | v2.6.0 | v2.7.0 `deterministic=True` | v2.7.0 `deterministic=False` |
|---|---|---|---|
| fit, ms per tree | 59.0 | 10.8 | 9.6 |
| single-row `predict`, GPU | 2.36 ms | 0.35 ms | 0.36 ms |
| single-row `predict(device="cpu")` | — | 0.15 ms | 0.15 ms |

Prediction at 16 checkpoints of an 80-tree, K = 20 model (2000 rows) took 8.6 s in v2.6.0 (cold
cache, mostly re-flattening). With `predict(num_trees=c)` it now takes 27 ms, and with one
`staged_predict` call 1.9 ms.

**Accuracy.** The reference cases were fitted with v2.6.0 and v2.7.0 on the same data:

* regression, multi-regression K = 20 with 3 and 8 eras, multiclass, monotone constraints;
* NaN features;
* both v2.7.0 modes.

Held-out scores are identical to 5 decimals. The largest prediction difference from v2.6.0 is
7e-7 (mean 1e-7).

**Split semantics (unchanged, documented here because they matter for tuning).**

* `min_child_weight` applies to the summed row weights of a child **in every era**, not to the
  node total.
* The split gain is summed over the K outputs and averaged over eras. `min_split_gain` is compared
  with that value, so its useful scale depends on K, the era sizes and the target variance.
* With several eras, an era's direction is ±1 only if all K outputs agree (else 0). Only
  candidates with the node's maximal directional agreement qualify. With many outputs this can be
  restrictive.
* `L2_reg` is the λ in G / (H + λ). With the squared-error hessian of 1 per row it matters once λ
  approaches the number of rows per leaf.

---

## 4. Behaviour changes to be aware of

| Area | Before | Now |
|---|---|---|
| `min_child_weight` | hessian sum (fork) / row count (upstream) | sum of row sample weights (≈ row count) |
| multi-output split gain | gain of summed gradients | sum of per-output gains; results of `multi_regression` models change (they get much better when outputs disagree) |
| multiclass / binary prediction | started from log(1/K) | starts from the stored training priors. Old pickles without a stored prior keep the old behaviour |
| `fit(eval_metric=...)` default | `"mse"` | `None`, meaning the objective default (`"mse"` for regression, `"logloss"` for classification, `"loss"` for custom objectives) |
| `num_outputs` | `None` for classifiers | K (number of classes) for every model |
| warm start | recomputed bin edges | reuses the first fit's bin edges; the objective cannot change between warm-start fits |
| early stopping | stop when the value `r` evals back beats the latest one; all trees kept and used | stop after `r` evals without improving on the best; predictions / importances use `best_iteration_`; the last round is always evaluated |
| `eval_loss` for `accuracy` / `corr` | `1 - accuracy`, `1 - corr` | natural values (maximized) |
| `feature_importance_`, `per_era_feature_importance_` | all trained trees | the default model (best round after early stopping) |
| tree nodes | `"feature"` was a 0-d CUDA tensor | Python int, plus a `"nan_left"` flag |
| `predict` | float atomics over trees (last-bit noise between calls) | deterministic summation order |
| CUDA API (`node_kernel`) | — | changed signatures: histogram (K channels, count, fixed-point scales), split (`[E,F,B,K]`, count, `min_sum_hessian`, 2·B candidates), binner (`nan_bin`; v2.6.0: `custom_cuda_binner` per column → `bin_matrix` for a whole `[N, F]` matrix), predict (`nan_bin`, 7-column trees) |
| global RNG state | `fit` seeded torch / numpy / CUDA global RNGs with `random_state` | untouched; code that relied on this side effect must seed itself. With `random_state=None` the column draws no longer depend on earlier global seeding |
| single-output regression without NaNs/weights | — | same predictions as before up to float rounding (max abs diff 6e-5 at prediction scale 540) |
| pre-binned detection (v2.6.0) | heuristic `integer and max < num_bins` at fit **and** at every predict | explicit `prebinned=True` (-1 = missing), or auto-detection once at fit (`[0, num_bins-1]`), stored as `prebinned_`. Float input to a pre-binned model raises; integer input to a raw-feature model is binned with its edges. Pre-binned models have `bin_edges=None`. Older pickles keep binning with their stored (identity) edges |
| `training_loss` (v2.6.0) | one value per round | one value per eval checkpoint by default; `record_training_loss="all"` restores per-round values. Older pickles load with `"all"` |
| forests (v2.7.0) | — | close to v2.6.0, not bit-identical: fixed-point tiles of 1024–8192 rows (quantisation step max abs(g) · tile / 2^30, before 256 rows), leaf values from fixed-point node totals, split gains from float prefix sums. Held-out scores of the reference cases are identical to 5 decimals |
| predictions (v2.7.0) | `acc += lr * leaf` in the kernel, per output | leaves pre-multiplied by the learning rate, one kernel for all outputs; differences of float rounding (≤ 1e-6 relative) |
| `threads_per_block`, `rows_per_thread` (v2.7.0) | constructor parameters (histogram launch configuration) | removed; passing them raises `TypeError` |
| internal API (v2.7.0) | `compute_histograms`, `find_best_split`, `grow_tree`, `_get_cached_tree_tensors`, `_accumulate_tree_tensors`, CUDA `compute_histogram3` / `compute_split` / `predict_forest` | replaced by the level-wise engine (`_grow_tree_device`) and the packed forest (`_packed_forest`); CUDA `histogram_level`, `node_sums`, `sibling_subtract`, `level_split`, `select_split`, `predict_forest` (new signature). `flatten_tree` is kept but unused |
| end of `fit` (v2.7.0) | `gc.collect()` | removed (cost about 55 ms per fit) |

---

## 5. Verification

Environment: GTX 1660 Ti (6 GB), PyTorch 2.14 + CUDA 13.0, Python 3.11 (`machine_learning`
pyenv), installed with `pip install --no-build-isolation`.

* **Test suite:** 108 tests pass, both against the in-place build and against the installed
  package: 50 pre-existing, plus 58 new in `test_regressions.py`, `test_nan_features.py`,
  `test_custom_objective.py` and `test_early_stopping.py`. Not run: `tests/test_invariant.py`
  (downloads data) and the `numerai_test.py` / `full_numerai_test.py` scripts.
  v2.6.0: 142 pass (26 new in `test_prebinned.py`) and 2 skip: the `cuda:1` tests in
  `test_multi_gpu.py` need a second GPU.
* **Determinism:** 8 repeated multiclass fits with the same `random_state` give identical trees;
  repeated `predict` calls are bit-identical.
* **Refactor sanity:** single-output regression (plain, with eras, with `colsample_bytree`)
  matches the previous version's predictions to float rounding.
* **Benchmark** (300k × 50, 100 trees, depth 6, 64 bins; multiclass 50 rounds, depth 5, 32 bins):

  | | before | after |
  |---|---|---|
  | regression fit | 5.41 s | 4.71 s |
  | multi-output fit (3 outputs, one of them `-y`) | 5.27 s, MSE 0.858 | 5.29 s, MSE 0.0126 |
  | multiclass fit | 3.46 s | 3.61 s |
  | histogram kernel, 1 era / 100 eras | 1.83 / 6.58 ms | 1.45 / 4.72 ms |

---

* **v2.7.0:** 166 tests pass and 2 skip (the `cuda:1` tests need a second GPU), against the in-place
  build. 18 tests are new in `tests/test_performance_modes.py`:
  * identical repeated fits, CUDA-graph replay = eager, chunked levels = subtraction;
  * quality of the fast mode against the deterministic one;
  * the unit-hessian shortcut against an autograd loss;
  * staged and CPU predictions bit-identical to the GPU at every round count;
  * eval scores equal to predictions;
  * early stopping with the pipelined rounds;
  * the `deterministic` parameter in old pickles.

  The histogram-kernel test now covers both modes, two segments and the unit-hessian layout. A
  4-thread stress test (repeated deterministic and fast fits on one GPU) matched sequential
  forests.

## 6. Known limitations

* Training and prediction require CUDA; there is no CPU code path. Since v2.5.0 the constructor
  rejects non-CUDA devices such as `device="cpu"` (mentioned upstream) with a `ValueError`.
* For classifiers, monotonic constraints apply to each class's logit, not to the probabilities.
* Custom losses must be per sample (a row's loss depends only on that row). Dataset-level losses
  such as a Pearson correlation over all rows are not supported: they were tried and did not
  train better than MSE on suitably standardized targets, at the cost of a much more complex
  interface.
* The level arrays are dense (2^depth nodes per level), so very deep trees (depth ≳ 12) waste memory
  and launches on inactive nodes. The memory guard keeps them working, slowly.
* CUDA-graph capture is serialised per process. Other code that captures CUDA graphs in a
  parallel thread may still conflict; set `WarpGBM._use_cuda_graphs = False` to run every tree
  eagerly.
* At most 65535 (sampled) features per tree; the histogram grid has features on its y axis.
* Older PyTorch versions limit `torch.nanquantile` to ~16M rows per feature column. PyTorch 2.14
  has no such limit (tested with 33M rows). Pre-binned integer input avoids the quantiles anyway.
