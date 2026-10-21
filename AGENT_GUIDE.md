# WarpGBM Agent Development Guide 🤖

> **Comprehensive guide for AI agents and developers working on WarpGBM**

This guide provides everything you need to understand, extend, and contribute to WarpGBM. Whether you're an AI coding agent or a human developer, this is your roadmap to the codebase.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Code Organization](#code-organization)
4. [Key Concepts](#key-concepts)
5. [Development Workflow](#development-workflow)
6. [Adding New Features](#adding-new-features)
7. [Testing Strategy](#testing-strategy)
8. [GPU/CUDA Details](#gpucuda-details)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### What is WarpGBM?

WarpGBM is a **GPU-accelerated gradient boosting library** that combines:
- **Speed**: Custom CUDA kernels for all bottleneck operations
- **Intelligence**: Directional Era-Splitting for invariant learning
- **Versatility**: Unified support for regression and classification
- **Compatibility**: Scikit-learn API for easy adoption

### Design Philosophy

1. **GPU-First**: Everything performance-critical runs on GPU
2. **No Kernel Changes**: Add features by reusing existing CUDA infrastructure
3. **Backward Compatible**: New features never break existing functionality
4. **Test-Driven**: Every feature has comprehensive tests
5. **Production-Ready**: Performance + correctness + usability

---

## 🏗️ Architecture

### High-Level Flow

```
User Code
    ↓
WarpGBM.fit()
    ↓
┌─────────────────────────────────────┐
│ 1. Data Preprocessing               │
│    - Label encoding (classification)│
│    - Binning (quantization)         │
│    - GPU transfer                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Forest Growing Loop              │
│    For each boosting round:         │
│    ├─ Compute gradients/hessians    │
│    ├─ Grow the tree level by level  │
│    │  on the GPU (CUDA graph replay)│
│    │  histograms → splits → rows    │
│    ├─ Update training scores        │
│    └─ (host) tree dict + eval of    │
│       the previous round (pipeline) │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Prediction                       │
│    - Traverse trees (CUDA kernel)   │
│    - Aggregate outputs              │
│    - Softmax (classification)       │
└─────────────────────────────────────┘
```

### Components

**Python Layer** (`warpgbm/core.py`):
- API interface
- Training orchestration (one engine for all objectives)
- Tree data structures

**Objectives** (`warpgbm/objectives.py`):
- Built-in and custom per-sample loss functions (gradients/hessians on the GPU, autograd by default)

**CUDA Layer** (`warpgbm/cuda/`):
- `binner.cu`: Feature quantization
- `histogram_kernel.cu`: Gradient/hessian accumulation per tree level, node totals, sibling subtraction
- `best_split_kernel.cu`: Split evaluation of all nodes of a level, best split per node
- `predict.cu`: Tree traversal for inference
- `node_kernel.cpp`: Python bindings

**Utilities** (`warpgbm/metrics.py`):
- Evaluation metrics (weighted, NaN-aware for multi-output)

---

## 📁 Code Organization

```
warpgbm/
├── __init__.py              # Package exports (WarpGBM, Objective, CustomObjective, fit_bins, apply_bins)
├── core.py                  # Main WarpGBM class: ONE training engine for all objectives
│   ├── fit()                # validation → targets/weights/masks → grow_forest()
│   ├── grow_forest()        # pipelined boosting loop: objective.grad_hess → one vector tree or K per-output trees
│   ├── _run_tree()          # CUDA graph capture / replay of _grow_tree_device
│   ├── _grow_tree_device()  # level-wise growth on the GPU: node sums, histograms, splits, row partition
│   ├── _level_splits()      # histograms of a level (subtraction or memory-bounded chunks) + split kernels
│   ├── _tree_from_records() # host: tree dict + feature importance from the per-tree record tensor
│   ├── compute_eval()       # incremental eval scores, built-in / callable metrics, early stopping
│   ├── _packed_forest()     # flattened forest cache (per round, all outputs) for the predict kernel
│   ├── predict(), predict_raw(), predict_proba(), staged_predict*()  # device=None (GPU) or "cpu"
│   └── save_model() / load_model()
│
├── objectives.py            # Objective interface + built-ins
│   ├── Objective            # base class (loss, grad_hess = autograd of loss, init_score, transform, output_strategy, ...)
│   ├── SquaredError         # 'regression' / 'multi_regression'
│   ├── Softmax              # 'multiclass' / 'binary' (per_output strategy)
│   ├── CustomObjective      # custom loss from plain functions (loss_fn, optional grad_hess_fn, init_fn, ...)
│   ├── autograd_grad_hess() # gradient + exact diagonal hessian of a per-sample loss
│   └── compute_grad_hess()  # applies weights + NaN-target masks, finiteness check
│
├── metrics.py               # Evaluation metrics (weighted MSE / corr / RMSLE / logloss / accuracy)
├── binning.py               # fit_bins / apply_bins (public), pre-binned input validation
│
└── cuda/
    ├── node_kernel.cpp      # Python/CUDA bridge
    ├── __init__.py          # _get_node_kernel(): lazy import of the compiled extension
    ├── binner.cu            # Quantile binning of a whole matrix (NaN -> dedicated bin)
    ├── histogram_kernel.cu  # Deterministic fixed-point histograms [E,F,B,K] + count channel
    ├── best_split_kernel.cu # Per-candidate gains (2 NaN directions), summed over outputs, DES directions
    └── predict.cu           # Tree traversal (7-column flattened trees with NaN direction)

docs/
├── CUSTOM_OBJECTIVES.md     # Guide for custom losses
└── FORK_CHANGES.md          # Review of fork changes, fixed bugs, behaviour changes

tests/                       # see "Test Suite Overview" below
examples/
└── Spiral Dataset.ipynb     # OOD generalization demo
```

---

## 🧠 Key Concepts

### 1. Objective Types

Every objective is an `Objective` (see `warpgbm/objectives.py` and `docs/CUSTOM_OBJECTIVES.md`)
returning `grad = dL/dscore`, `hess = d²L/dscore²` on the GPU for raw scores `[N, K]`. The engine
multiplies by sample weights, zeroes missing targets and grows trees on `-grad`
(leaf = `Σ(-g) / (Σh + L2_reg)`).

**Regression** (`objective='regression'` / `'multi_regression'` → `SquaredError`):
- Gradients `pred - y`, hessians 1; init = weighted mean per output
- Multi-output: one tree per round with K-dimensional leaves; gain summed over outputs

**Binary / Multiclass** (`'binary'` / `'multiclass'` → `Softmax`, `output_strategy="per_output"`):
- K trees per round (one per class), `g_k = p_k - y_k`, `h_k = p_k (1 - p_k)`
- Init = weighted log class priors, stored in `base_prediction` and used at prediction

**Custom**: a per-sample loss function `(y, pred) -> loss`, a `CustomObjective(loss_fn, ...)`, or an
`Objective` subclass. Gradients and the diagonal hessian come from autograd unless `grad_hess` is
given; the engine applies weights and masks missing targets.

### 2. Directional Era-Splitting (DES)

**Standard GBDT**: Evaluates splits globally across all data.

**WarpGBM with DES**: 
1. Computes split gain separately for each era
2. Only accepts splits where:
   - Gain > 0 in ALL eras
   - Split direction is consistent across eras
3. Result: Model learns only invariant signals

**Usage**:
```python
model.fit(X, y, era_id=era_labels)  # era_labels: integer array
```

If `era_id` is omitted, behaves like standard GBDT (single era).

### 3. Tree Structure

Trees are stored as nested Python dictionaries:

```python
{
    'feature': 5,           # Which feature to split on (int)
    'bin': 3,               # Split threshold (in bin space): bin <= 3 goes left
    'nan_left': False,      # Learned direction of missing values (NaN bin)
    'left': {...},          # Left subtree (recursive)
    'right': {...}          # Right subtree (recursive)
}

# Leaf node:
{
    'leaf_value': 0.234,    # Prediction value (a list of K values for vector-leaf trees)
    'samples': 150          # Number of samples in leaf
}
```

**per_output strategy (multiclass/binary)**: `forest[round]` is a list of K trees (one per class).

**vector strategy (regression, multi_regression, most custom objectives)**: `forest[round]` is a single tree.

### 4. Binning

WarpGBM quantizes continuous features into discrete bins for histogram-based training.

**Automatic Binning** (raw features):
- `warpgbm/binning.py` is the single implementation, used by `fit` / `predict` and exposed as
  `warpgbm.fit_bins(X, num_bins, device)` and `warpgbm.apply_bins(X, edges, device)`
- Edges: `torch.nanquantile` per feature (NaNs excluded, all-NaN columns → 0), computed for blocks of
  features to bound the sort workspace
- Assignment: one CUDA launch over the whole `[N, F]` matrix (`bin_matrix`), bin = number of edges ≤ x;
  NaN → the internal bin `num_bins` (`model.nan_bin`), or -1 (`MISSING_BIN`) in `apply_bins`' output
- Stored in `self.bin_edges`; reused on warm start

**Pre-binned Data** (`prebinned=True`, stored as `prebinned_`):
- Integer input = bin indices `0 … num_bins-1`, `-1` = missing; mapped to the internal NaN bin on the GPU
- Applies to `X`, `X_eval`, `predict*`; float input or values outside `[-1, num_bins-1]` raise
  (`binning._prebinned_to_gpu`); `bin_edges` is `None`
- `prebinned=None` (default) auto-detects at fit time (integers in `[0, num_bins-1]`); the decision is
  stored in `prebinned_`, so prediction never guesses
- The NaN bin index `num_bins` is a label, not an ordering: splits route it by `nan_left`

### 5. Histograms

For each feature bin, accumulate:
- Gradient sum per output: `G = Σ gradient_i`
- Hessian sum per output: `H = Σ hessian_i`
- Count channel: `C = Σ row_weight_i` (used for `min_child_weight`)

**Shapes**: one tensor per tree level, `(nodes, num_features, num_eras, num_bins + 1, C)` with the
channels `C = K + Kh + 1` = `[grad | hess | count]`. `Kh = 0` when every hessian equals the row
weight (squared error without missing targets, `Objective._hessian_is_one()`); the hessian sums are
then the count channel.

**CUDA Kernel**: `histogram_level` parallelizes over (row tile of a node, group of up to 8
features, channel chunk), reading feature-major bins `[F, N]` (`self.bins_fm` during fit). Blocks
accumulate 32-bit fixed-point integers in shared memory (integer shared atomics are much faster than
float ones on Turing) and flush into int64 (`deterministic=True`, exact) or float32
(`deterministic=False`) histograms. A global-memory fallback is used when one channel of
`[eras, bins]` does not fit into shared memory (many eras).

**Split candidates**: `2 * num_bins` per feature — every threshold with NaN going right
(`c < num_bins`) and with NaN going left (`c >= num_bins`).

**Subtraction Trick**: per level, the smaller child of every split is built and its sibling is
`parent - child` (`sibling_subtract`; exact in int64). When parent + child levels exceed the memory
budget (`_HIST_MEMORY_FRACTION` of free memory at fit), the level is processed in node chunks and
both children are built directly.

**Level-wise growth** (`_grow_tree_device`): nodes are numbered densely per level (children of `j`
are `2j`, `2j+1`); nodes below a leaf are inactive. Per level: `node_sums` → values / monotone
bounds, histograms, `level_split` + `select_split` → best split per node, rows move to their child.
No host sync inside a tree, so `_run_tree` captures the second tree as a CUDA graph and replays it
(static input buffers: gradients, hessians, sampled features). The tree comes back as one record
tensor (`_tree_from_records`), and `grow_forest` pipelines rounds: round i + 1 is enqueued before
round i is turned into dicts and evaluated.

### 6. Multiclass Implementation

`Softmax.output_strategy == "per_output"`: per round the engine computes `g, h` for all classes
once and grows one tree per class on the column `(-g[:, k], h[:, k])`, updating only score column k.
The same code path (`_launch_tree(..., score_col=k)`) serves any custom per-output objective.

### 7. Feature Importance

**Gain-Based Importance**: WarpGBM tracks the total gain (loss reduction) attributable to each feature across all splits.

**Implementation**:
- `feature_importance_`: Array of shape `(n_features,)` with total gain per feature
- `per_era_feature_importance_`: Array of shape `(n_eras, n_features)` with per-era gains

**Accumulation Logic** (in `_tree_from_records()`): the per-era gains of every accepted split come
back with the tree's records and are added on the host, in pre-order (deterministic):
```python
importance[:, feature] += per_era_gains.astype(np.float32)   # self._feature_importance_acc [E, F]
```

**Aggregation** (after training, in grow_forest()):
```python
self._per_era_importance_all = base + self._feature_importance_acc
self.feature_importance_ = self._total_importance(n)   # from the per-split "gain" of the used rounds
```

**Multiclass**: Importance is accumulated across all K trees per iteration. This means a feature used in multiple class trees gets higher importance (which is correct behavior).

**Per-Era Importance**: Unique to WarpGBM! Allows identifying which features are:
- **Invariant**: High importance across ALL eras (robust signals)
- **Era-specific**: High importance in only some eras (potentially spurious)

**API Methods**:
- `get_feature_importance(normalize=True)`: Returns total importance across eras
- `get_per_era_feature_importance(normalize=True)`: Returns per-era breakdown

**Normalization**: When `normalize=True`, importances sum to 1.0 (per-era normalization is done independently for each era).

---

## 💾 Warm Start & Checkpointing

**Purpose**: Enable incremental training and model checkpointing for iterative development, hyperparameter tuning, and production workflows.

### Implementation Details

**Key Components**:
1. `warm_start` parameter (bool, default `False`)
2. `_is_fitted` flag to track if model has been trained
3. `_trees_trained` counter to track number of trees already trained
4. `save_model()` and `load_model()` methods for serialization

### How Warm Start Works

**Without warm_start** (default):
```python
model = WarpGBM(n_estimators=100)
model.fit(X, y)  # Trains 100 trees
model.fit(X, y)  # Retrains from scratch, 100 new trees
```

**With warm_start**:
```python
model = WarpGBM(n_estimators=50, warm_start=True)
model.fit(X, y)  # Trains 50 trees
model.n_estimators = 100
model.fit(X, y)  # Adds 50 more trees (total: 100)
```

### Critical Implementation Detail: Score Restoration

**The Challenge**: When continuing training, the raw scores `self.scores` `[N, K]` must be restored
from the existing forest. Otherwise, new trees would be trained as if no previous trees exist.

**Solution** (in `fit()`): `self.scores = self._predict_raw_binned(self.bin_indices, self._trees_trained)`, i.e. the stored
`base_prediction` plus all existing trees evaluated with the CUDA prediction kernel. The bin edges
of the first fit are reused (`preprocess_gpu_data(..., reuse_edges=True)`).

**Continuation point**: always after all trained rounds; `truncate()` drops the rounds after
`best_iteration_` first if training should continue from the best round. The objective cannot
change between warm-start fits (`_resolve_objective` raises).

### Early Stopping, Best Iteration and `num_trees`

- `compute_eval()` evaluates every `eval_every_n_trees` rounds and at the last round, records
  `evals_result_` / `eval_iterations_` and the best round (`best_iteration_`, `best_score_`,
  direction from `_HIGHER_IS_BETTER` or `Metric.higher_is_better`), and stops after
  `early_stopping_rounds` evaluations without improvement.
- `training_loss` (objective loss on the training set) costs a full pass and a host sync, so with
  `record_training_loss="eval"` (default) it is only computed at checkpoints (reusing the training
  score when the eval metric is the same loss); `"all"` records every round, `"none"` nothing.
- `_rollback` (set when `early_stopping_rounds` was given) makes `_default_num_trees()` return
  `best_iteration_`; every prediction / importance method resolves `num_trees=None` through it.
  The tree-tensor cache is keyed on the number of rounds.
- Each split node stores its total gain (`"gain"`), so summed importances are exact for any
  prefix; per-era importances are snapshots (`_per_era_importance_all/_best`).
- `train_predictions_` is a copy of the training scores at the best round (or the final scores).

### Save/Load Implementation

**save_model()**:
- Pickles all model state (hyperparameters + trained state)
- Includes: `forest`, `objective_` (the Objective instance — custom classes must be importable), `bin_edges` and `base_prediction` (on CPU), `prebinned_`, `num_outputs`, `label_encoder`, feature importances, `_trees_trained`, etc.
- Uses standard Python pickle (simple, portable)

**load_model()**:
- Restores all state from pickle file
- User can then:
  - Use for predictions immediately
  - Enable `warm_start` and continue training
  - Modify `n_estimators` to add more trees

### Test Strategy

**test_warm_start.py** covers:
1. **Equivalence**: Train 100 trees at once vs 50+50 with warm_start → identical predictions
2. **Save/Load**: Predictions before and after save/load are identical
3. **Resume Training**: Load checkpoint, continue training, verify correctness
4. **Both objectives**: Regression and multiclass classification
5. **Feature Importance**: Accumulated correctly across warm start sessions

### Usage Examples

**Incremental Training**:
```python
model = WarpGBM(n_estimators=50, warm_start=True)
model.fit(X, y)
# Evaluate, decide if more trees needed
model.n_estimators = 100
model.fit(X, y)  # Trains 50 more
```

**Checkpointing**:
```python
model = WarpGBM(n_estimators=200, warm_start=True)
for checkpoint in [50, 100, 150, 200]:
    model.n_estimators = checkpoint
    model.fit(X_train, y_train)
    model.save_model(f'ckpt_{checkpoint}.pkl')
    eval_score = evaluate(model, X_val, y_val)
    print(f"Checkpoint {checkpoint}: {eval_score:.4f}")
```

**Load and Resume**:
```python
model = WarpGBM()
model.load_model('ckpt_100.pkl')  # Has 100 trees
model.warm_start = True
model.n_estimators = 200  # Train 100 more
model.fit(X, y)
```

### Design Notes

- **Sklearn-compatible**: Matches sklearn's `warm_start` API
- **No CUDA state saved**: Trees are Python dicts, no GPU tensors in checkpoint
- **Backward compatible**: Default `warm_start=False` maintains existing behavior
- **Works with all objectives**: Regression, multi-output regression, binary, multiclass and custom objectives (the objective must stay the same)

---

## 🔨 Development Workflow

### Environment Setup

```bash
# Create conda environment
conda create -n warpgbm_dev python=3.11
conda activate warpgbm_dev

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install WarpGBM in dev mode
cd /path/to/warpgbm
pip install -e .

# Install dev dependencies
pip install pytest scikit-learn pandas numerapi
```

### Branch Strategy

- `main`: Stable, production-ready code
- Feature branches: `multiclass`, `distributed`, etc.
- Always create a new branch for new features

```bash
git checkout -b feature_name
# ... make changes ...
git add relevant_files
git commit -m "Descriptive message"
git push -u origin feature_name
```

### Testing Workflow

```bash
# Run specific test
pytest tests/test_multiclass.py -v
pytest tests/test_feature_importance.py -v

# Run all tests except slow ones
pytest tests/test_fit_predict_corr.py tests/test_invariant.py tests/test_multiclass.py -v

# Run all tests (slow, includes Numerai download)
pytest tests/ -v
```

**Test Suite Overview**:
- `test_fit_predict_corr.py`: Basic regression functionality
- `test_invariant.py`: Era-splitting and DES algorithm
- `test_multiclass.py`: Classification with softmax
- `test_comparison_lightgbm.py`: Classification predictions vs LightGBM
- `test_feature_importance.py`: Gain-based importance and per-era tracking
- `test_random_state.py`: Reproducibility with random_state parameter
- `test_warm_start.py`: Incremental training, save/load, and checkpointing
- `test_sample_weight.py`, `test_monotonic_constraints.py`, `test_multi_regression.py`, `test_predict_chunking.py`: fork features
- `test_custom_objective.py`: autograd and explicit objectives, multi-output, NaN targets, weights, per-output strategy, warm start
- `test_early_stopping.py`: best iteration / rollback, metric direction, `num_trees`, importances, `eval_aux` / `train_aux`, `train_predictions_`, `truncate()`
- `test_nan_features.py`: NaN feature bin and learned default direction
- `test_regressions.py`: Regression tests for the bugs listed in docs/FORK_CHANGES.md (incl. kernel vs reference, legacy pickles)
- `numerai_test.py` / `full_numerai_test.py`: Integration scripts (slow, require download)

**Important**: Always verify regression tests still pass when adding features!

### Code Style

- Follow existing patterns
- Add docstrings to public methods
- Use type hints where helpful
- Keep functions focused (single responsibility)
- Comment non-obvious GPU operations

---

## ✨ Adding New Features

### Example: Adding a New Objective

No engine changes are needed — implement an `Objective` (see `docs/CUSTOM_OBJECTIVES.md`):

```python
import torch
from warpgbm import Objective, WarpGBM

class Poisson(Objective):
    name = "poisson"

    def loss(self, y, pred):                    # per sample, pred = log-rate [N, K]; autograd gives grad/hess
        return torch.exp(pred) - y * pred

    def init_score(self, y, weight):            # [K]; weight is [N, K], 0 for missing targets
        return torch.log((y * weight).sum(0) / weight.sum(0))

    def transform(self, raw):                   # predict() returns the rate
        return torch.exp(raw)

model = WarpGBM(objective=Poisson()).fit(X, y)
```

To make it a named built-in, add it to `resolve_objective()` in `objectives.py`, extend the
string check in `_validate_hyperparams()`, and add tests.

### Adding New Metrics

Pass a callable `metric(y_true, pred, weight, mask[, aux]) -> float` (minimized) or
`Metric(fn, higher_is_better=True)` as `eval_metric`, or add a string metric to `_metric_value()`
in `core.py` (and to the metric lists at the top of the file, plus `_HIGHER_IS_BETTER` if it is
maximized). `_metric_value()` returns natural values.

### Adding New Regularization

Example: L1 regularization (already in constructor, needs implementation):

```python
# In _grow_tree_device(), update the node value computation (val = G / (H + L2_reg)):
def compute_leaf_value(self, grad_sum, hess_sum):
    # Current: -G / (H + λ_L2)
    # With L1: soft thresholding
    raw_value = -grad_sum / (hess_sum + self.L2_reg)
    if self.L1_reg > 0:
        raw_value = np.sign(raw_value) * max(0, abs(raw_value) - self.L1_reg)
    return raw_value
```

---

## 🧪 Testing Strategy

### Test Organization

**Unit Tests**: Test individual functions
- `test_fit_predict_corr.py`: Basic regression flow
- `test_multiclass.py`: Classification with multiple scenarios

**Integration Tests**: Test full workflows
- `test_invariant.py`: Era-splitting with synthetic data
- `numerai_test.py`: Real-world data pipeline

**Benchmark Tests**: Verify performance
- Compare against LightGBM/XGBoost
- Track training/inference times

### Writing Good Tests

**Structure**:
```python
def test_feature_name():
    # 1. Setup
    X, y = generate_data()
    
    # 2. Execute
    model = WarpGBM(objective='...', ...)
    model.fit(X, y)
    preds = model.predict(X)
    
    # 3. Assert
    assert accuracy > threshold
    assert preds.shape == expected_shape
    assert model.num_classes == expected_classes
```

**Coverage Checklist**:
- ✅ Normal cases
- ✅ Edge cases (single class, empty data, etc.)
- ✅ Different data types (float32, int8, strings)
- ✅ With/without eval sets
- ✅ With/without era_id
- ✅ Error handling

### Test Data

```python
# Classification
from sklearn.datasets import make_classification, make_blobs

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=3,
    n_informative=15,
    random_state=42
)

# Regression
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=5000,
    n_features=50,
    noise=0.1,
    random_state=42
)

# Well-separated (for sanity checks)
X, y = make_blobs(n_samples=1000, centers=5, random_state=42)
```

---

## 🔥 GPU/CUDA Details

### Kernel Architecture

**1. Binner** (`binner.cu`):
- Input: Raw feature values, bin edges
- Output: Binned indices (int8)
- Parallelization: One thread per sample
- Used: Once at fit time, once at predict time

**2. Histogram** (`histogram_kernel.cu`):
- `histogram_level`: histograms `[slots, F, E, B, C]` of the nodes of a level from row segments
  (`order`, `seg_start`, `seg_len`, `seg_slot`, `tile_offset`); block per (row tile, feature group,
  channel chunk); lanes of a warp share a row and take its channels; branch-free inner loop
- `node_sums`: `[nodes, C]` totals per node (leaf values, monotone bounds)
- `sibling_subtract`: `hist[sibling] = parent - hist[built]` for split parents
- Fixed-point scales are computed on the device (`_device_scale`), per tree, for tiles of
  `_tile_rows` rows (1024–8192, from eras × bins)
- Used: a few times per tree level (most expensive operation)

**3. Split Finding** (`best_split_kernel.cu`):
- `level_split`: block per (feature, node), thread per candidate (`2 * num_bins`, both NaN
  directions); loops over eras with float prefix sums in shared memory; writes per-era gain, era-mean
  gain, directional agreement, monotone flag
- `select_split`: block per node; max agreement over all candidates, then `gain > min_split_gain`,
  monotone mask, first maximal gain → `best[node]` (−1 = leaf), per-era gains of the best
- Gain summed over outputs; direction only if all outputs agree; `min_child_weight` on the count channel, `min_sum_hessian` per output

**4. Prediction** (`predict.cu`):
- Input: bins with explicit strides (row- or feature-major), packed forest `nodes [T, M, 4]`
  (feature or −1, split bin | nan_left << 8, left, right), `leaf [T, M, Lw]` (× learning rate),
  `out_col` for per-output trees, base score `[K]` or `[N, K]`, checkpoint tree counts
- Output: `[checkpoints, N, K]`; one thread per row, all trees and outputs, in-order plain additions
  (deterministic, bit-identical to the NumPy path `_predict_raw_staged_cpu`)
- The packed forest is built once per round (`_pack_round`) and cached (`_packed_forest`)

### Memory Management

**GPU Tensors** (stay on GPU throughout training):
- `self.bins_fm`: Binned feature data, feature-major `[F, N]` int8
- `self.scores`: Current raw scores `[N, K]` (all objectives)
- `self.Y_gpu`: Target labels
- `self.era_indices`: Era assignments
- Histograms (transient, per tree level; the previous level is kept for subtraction)
- CUDA graph of the tree (`_tree_graphs`, released at the end of `fit`)

**CPU → GPU Transfer**:
- Only during `fit()` initialization
- Uses PyTorch's efficient transfer

**GPU → CPU Transfer**:
- Only for final predictions
- Deleted immediately after `.cpu().numpy()`

### CUDA Tuning

Since v2.7.0 the kernels choose their launch configuration (the `threads_per_block` and
`rows_per_thread` parameters were removed). Measure with
`benchmarks/bench_small_wide.py` (`--profile` for a kernel breakdown; `WarpGBM._use_cuda_graphs =
False` makes individual kernels visible to the profiler). The histogram kernel is bound by
shared-memory atomics; blocks use at most 32 KB of shared memory so that several fit on an SM.

---

## 🎨 Common Patterns

### Pattern 1: No Objective Branching in the Engine

Objective-specific behaviour lives in `Objective` subclasses (`grad_hess`, `init_score`,
`transform`, `prepare_targets`, `output_strategy`, `is_classifier`). The engine in `core.py` should
stay objective-agnostic; only label encoding for classifiers and metric selection look at
`objective_.is_classifier`.

### Pattern 2: Multi-Output Is the General Case

Gradients, hessians, scores and histograms always carry an output dimension K (K = 1 for
single-output regression). Write new engine features for `[N, K]` tensors.

### Pattern 3: Backward Compatibility

Always maintain defaults:

```python
def __init__(self, objective='regression', ...):  # Default = existing behavior
    # New features opt-in, not opt-out
```

### Pattern 4: Error Handling

```python
# Validate early
if self.objective == "binary" and self.num_classes != 2:
    raise ValueError(f"binary objective requires exactly 2 classes, got {self.num_classes}")

# Informative errors
if not isinstance(X, np.ndarray):
    raise TypeError(f"X must be numpy array, got {type(X)}")
```

### Pattern 5: Progress Logging

```python
print(f"🌲 Round {i+1}/{self.n_estimators} | Train Loss: {loss:.6f}")
```

Use emoji for visual distinction. Keep format consistent.

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce `num_bins`
- Process data in chunks
- Use `torch.cuda.empty_cache()` between operations
- For classification: Process classes sequentially, not in parallel

**2. Mismatched Shapes**

```
RuntimeError: shape '[5000]' is invalid for input of size 10000
```

**Debug**:
```python
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"scores shape: {self.scores.shape}")   # always [N, K]
```

**Common cause**: A custom objective returned grad/hess that do not broadcast to `[N, K]`
(the engine raises a `ValueError` naming the shapes).

**3. Label Encoding Issues**

```
ValueError: y contains previously unseen labels
```

**Solution**: Ensure `label_encoder` is fitted on train data and used on eval/test.

**4. Slow Training**

**Check**:
- Are you using pre-binned data but not getting speedup? (Pass `prebinned=True`; check `model.prebinned_`)
- Is `colsample_bytree` too small? (More sampling = more randomness = slower convergence)
- Is `max_depth` too large? (Exponential growth in tree size)

**Profile**:
```python
import time
start = time.time()
model.fit(X, y)
print(f"Training time: {time.time() - start:.2f}s")
```

**5. Tests Failing After Changes**

```bash
# Run specific failing test with verbose output
pytest tests/test_name.py::test_function -v -s

# Check if you broke regression
pytest tests/test_fit_predict_corr.py -v
```

**Common causes**:
- Changed default parameter values
- Modified tree structure format
- Broke GPU memory cleanup (use `del` and `gc.collect()`)

---

## 📊 Performance Tips

### For Developers

1. **Minimize GPU ↔ CPU transfers**: Keep tensors on GPU as long as possible
2. **Reuse buffers**: Don't allocate new histogram tensors every node
3. **Profile with PyTorch profiler**: Identify bottlenecks
4. **Use `@torch.no_grad()`**: During inference to save memory

### For Users (Document in README)

1. **Pre-bin your data**: Massive speedup if applicable
2. **Tune `num_bins`**: Sweet spot is usually 16-32 for most datasets
3. **Use `colsample_bytree < 1`**: Can speed up training AND improve generalization
4. **Early stopping**: Don't overtrain
5. **Batch predictions**: For large inference, predict in chunks

---

## 🚀 Future Directions

### Planned Features

**High Priority**:
- [ ] Multi-GPU support (data parallelism)
- [ ] SHAP values on GPU
- [ ] ONNX export for deployment

**Medium Priority**:
- [x] Monotonic constraints
- [ ] Feature interaction constraints
- [x] Custom loss functions (user-defined gradients) — see docs/CUSTOM_OBJECTIVES.md
- [ ] Learning rate scheduling

**Research**:
- [ ] Distributed training (across nodes)
- [ ] Neural network leaf functions
- [ ] Automatic hyperparameter tuning

### Design Considerations

**Multi-GPU**:
- Split histograms across GPUs
- AllReduce for gradient aggregation
- Requires NCCL integration

**SHAP**:
- Compute TreeSHAP on GPU in parallel
- Store tree structures in GPU-friendly format
- Challenge: Recursive traversal on GPU

**ONNX**:
- Export tree structures to ONNX TreeEnsemble operator
- Problem: Era-splitting logic not standard
- Solution: Bake era checks into tree structure

---

## 📚 Additional Resources

### Papers
- [XGBoost Paper](https://arxiv.org/abs/1603.02754): Foundation of histogram-based boosting
- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree): GOSS and EFB algorithms
- [Era Splitting](https://arxiv.org/abs/2309.14496): DES algorithm

### Code References
- LightGBM source: Understanding histogram optimizations
- XGBoost CUDA code: GPU kernel patterns
- PyTorch extensions: Building CUDA extensions

### Community
- Open GitHub issues for bugs
- Discussions for feature requests
- PRs welcome (follow testing guidelines)

---

## 🎓 Learning Path for New Contributors

1. **Week 1**: Read this guide + README, run all examples
2. **Week 2**: Study `core.py`, understand training loop
3. **Week 3**: Read CUDA kernel code, understand GPU operations
4. **Week 4**: Write a test, fix a bug, or add a small feature
5. **Week 5+**: Design and implement major features

---

## ✅ Pre-Commit Checklist

Before pushing code:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code follows existing style
- [ ] New features have tests
- [ ] Docstrings added for public methods
- [ ] README updated if API changed
- [ ] No print statements left in code (use for debugging only)
- [ ] No hard-coded paths or magic numbers
- [ ] Memory cleanup (del tensors, gc.collect())
- [ ] Commit message is descriptive

---

## 🤝 Questions?

If you're an AI agent working on this codebase and get stuck:

1. Re-read relevant sections of this guide
2. Check existing tests for usage patterns
3. Search GitHub issues for similar problems
4. Ask the user for clarification on requirements
5. Propose multiple solutions with tradeoffs

**Remember**: Backward compatibility is sacred. Don't break existing functionality.

---

**Happy coding! 🚀**

*Last updated: October 2025*


