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
| **H** | A single NaN in a feature made all its quantile edges NaN, so the feature became useless | upstream | edges `[nan, …]` → finite; train MSE 9.12 → 0.84 (var 9.37) | NaN feature bin (see §3.2) |

Smaller issues fixed along the way:

* **Run-to-run nondeterminism.** Float `atomicAdd` order made histograms differ in the last bits,
  and exact ties (common in round 0 of softmax, where every row has the same probability) were
  then broken randomly. `test_random_state_multiclass` failed ~40 % of the time once Bug A was
  fixed (the old code passed only because it hardly split). The fix is deterministic
  fixed-point histograms (§3.3); 8/8 repeated fits now give identical forests.
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
`hess = d²L/dscore²`, `ObjectiveContext` with weights, masks, eras and the round index).
**All** objectives, built-in and custom, run through a single training loop in `core.py`. This
replaces the three previous copies (`_fit_regression`, `_fit_multi_regression`,
`_fit_classification` and their `grow_forest*` / `compute_eval*` twins):

* built-ins: `SquaredError` (`"regression"`, `"multi_regression"`) and `Softmax`
  (`"multiclass"`, `"binary"`);
* custom: subclass `Objective`, `GradHessObjective(fn)`, `AutogradObjective(loss_fn)` (autograd
  gradient + diagonal hessian), or a plain callable `(y, pred, ctx) -> (grad, hess)`;
* dataset-level losses: `DatasetLossObjective(loss_fn)`, e.g. multi-output Pearson correlation
  over the full dataset plus an MSE term, with NaN targets, 2-D weights and era splitting. It
  provides gradient rescaling, an `"auto"` curvature estimate and a squared-error warm-up
  (`warmup_rounds`, default 10), because correlations are undefined at constant predictions;
* `warpgbm/losses.py`: GPU helpers `pearson_corr` (weighted, masked, per output, optionally per
  group/era) and `weighted_mse`;
* `output_strategy="vector"` (one tree per round with K-dimensional leaves) or `"per_output"`
  (K trees per round);
* objective switching on warm start (e.g. MSE pre-training, then a custom loss);
* `fit(..., eval_metric=callable)` with an optional `era` argument, `fit(..., era_id_eval=...)`,
  `eval_metric=None` (objective default) and `"loss"` (the objective's own loss);
* `predict_raw(X)` returns the untransformed scores (logits for classification).

The forest storage formats are unchanged: scalar leaves for single output, list leaves for vector
trees, a list of K trees per round for `per_output`. Models saved by the previous version still
load (tested).

### 3.2 NaN features get their own bin

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

### 3.3 Deterministic, faster histogram kernel

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

### 3.4 Other

* `min_sum_hessian` hyperparameter (default 1e-3). `min_child_weight` now also accepts floats
  (≥ 0).
* Training-time evaluation adds only the newest round to cached eval scores instead of
  re-flattening the forest.

---

## 4. Behaviour changes to be aware of

| Area | Before | Now |
|---|---|---|
| `min_child_weight` | hessian sum (fork) / row count (upstream) | sum of row sample weights (≈ row count) |
| multi-output split gain | gain of summed gradients | sum of per-output gains; results of `multi_regression` models change (they get much better when outputs disagree) |
| multiclass / binary prediction | started from log(1/K) | starts from the stored training priors. Old pickles without a stored prior keep the old behaviour |
| `fit(eval_metric=...)` default | `"mse"` | `None`, meaning the objective default (`"mse"` for regression, `"logloss"` for classification, `"loss"` for custom objectives) |
| `num_outputs` | `None` for classifiers | K (number of classes) for every model |
| warm start | recomputed bin edges | reuses the first fit's bin edges |
| tree nodes | `"feature"` was a 0-d CUDA tensor | Python int, plus a `"nan_left"` flag |
| CUDA API (`node_kernel`) | — | changed signatures: histogram (K channels, count, fixed-point scales), split (`[E,F,B,K]`, count, `min_sum_hessian`, 2·B candidates), binner (`nan_bin`), predict (`nan_bin`, 7-column trees) |
| single-output regression without NaNs/weights | — | same predictions as before up to float rounding (max abs diff 6e-5 at prediction scale 540) |

---

## 5. Verification

Environment: GTX 1660 Ti (6 GB), PyTorch 2.14 + CUDA 13.0, Python 3.11 (`machine_learning`
pyenv), installed with `pip install --no-build-isolation`.

* **Test suite:** 98 tests pass, both against the in-place build and against the installed
  package: 50 pre-existing, plus 48 new in `test_regressions.py`, `test_nan_features.py`,
  `test_custom_objective.py` and `test_dataset_objective.py`. Not run: `tests/test_invariant.py`
  (downloads data) and the `numerai_test.py` / `full_numerai_test.py` scripts.
* **Determinism:** 8 repeated multiclass fits with the same `random_state` give identical trees.
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

* Training requires CUDA (the kernels have no CPU path), despite `device="cpu"` being mentioned
  upstream.
* For classifiers, monotonic constraints apply to each class's logit, not to the probabilities.
* Correlation-type `DatasetLossObjective`s need the warm-up (or a warm start from another
  objective); with a small MSE weight λ, the prediction scale grows slowly.
* `torch.nanquantile` limits a single feature column to ~16M rows for quantile binning. This is a
  pre-existing limitation; pre-binned integer input avoids it.
