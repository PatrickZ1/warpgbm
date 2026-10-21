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

## 6. Known limitations

* Training and prediction require CUDA; there is no CPU code path. Since v2.5.0 the constructor
  rejects non-CUDA devices such as `device="cpu"` (mentioned upstream) with a `ValueError`.
* For classifiers, monotonic constraints apply to each class's logit, not to the probabilities.
* Custom losses must be per sample (a row's loss depends only on that row). Dataset-level losses
  such as a Pearson correlation over all rows are not supported: they were tried and did not
  train better than MSE on suitably standardized targets, at the cost of a much more complex
  interface.
* Older PyTorch versions limit `torch.nanquantile` to ~16M rows per feature column. PyTorch 2.14
  has no such limit (tested with 33M rows). Pre-binned integer input avoids the quantiles anyway.
