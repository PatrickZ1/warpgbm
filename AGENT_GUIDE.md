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
│    ├─ Build histograms (CUDA)       │
│    ├─ Find best splits (CUDA)       │
│    ├─ Grow tree recursively         │
│    └─ Update predictions            │
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
- Training orchestration
- Objective-specific logic
- Tree data structures

**CUDA Layer** (`warpgbm/cuda/`):
- `binner.cu`: Feature quantization
- `histogram_kernel.cu`: Gradient/hessian accumulation
- `best_split_kernel.cu`: Split evaluation
- `predict.cu`: Tree traversal for inference
- `node_kernel.cpp`: Python bindings

**Utilities** (`warpgbm/metrics.py`):
- Loss functions
- Evaluation metrics
- Gradient computations

---

## 📁 Code Organization

```
warpgbm/
├── __init__.py              # Package exports
├── core.py                  # Main WarpGBM class (1000+ lines)
│   ├── __init__()           # Parameter validation
│   ├── fit()                # Entry point for training
│   ├── _fit_regression()    # Regression-specific training
│   ├── _fit_classification()# Classification-specific training
│   ├── grow_forest()        # Regression boosting loop
│   ├── grow_forest_multiclass() # Classification boosting loop
│   ├── grow_tree()          # Recursive tree building
│   ├── predict()            # Main prediction interface
│   ├── predict_proba()      # Classification probabilities
│   └── [20+ helper methods]
│
├── metrics.py               # Loss functions and metrics
│   ├── rmsle_torch()
│   ├── softmax()
│   ├── log_loss_torch()
│   └── accuracy_torch()
│
└── cuda/
    ├── node_kernel.cpp      # Python/CUDA bridge
    ├── binner.cu            # Quantile binning
    ├── histogram_kernel.cu  # Histogram accumulation
    ├── best_split_kernel.cu # Split finding (with DES)
    └── predict.cu           # Tree traversal

tests/
├── test_fit_predict_corr.py     # Basic regression
├── test_invariant.py            # Era-splitting tests
├── test_multiclass.py           # Classification suite
├── numerai_test.py              # Real-world data
├── numerai_invariant_test.py    # Era splitting + real data
└── full_numerai_test.py         # Full dataset test

examples/
└── Spiral Dataset.ipynb     # OOD generalization demo
```

---

## 🧠 Key Concepts

### 1. Objective Types

WarpGBM supports three objectives:

**Regression** (`objective='regression'`):
- Output: Continuous values
- Loss: MSE (mean squared error)
- Gradients: `residual = y_true - y_pred`
- Hessians: Constant (1.0)

**Binary Classification** (`objective='binary'`):
- Output: Class labels {0, 1} + probabilities
- Loss: Binary cross-entropy via softmax
- Trees: 2 trees per round (one per class)

**Multiclass** (`objective='multiclass'`):
- Output: Class labels + K-dimensional probabilities
- Loss: Categorical cross-entropy
- Trees: K trees per round (one per class)
- Gradients: `g_k = p_k - y_k` (softmax gradient)
- Hessians: `h_k = p_k * (1 - p_k)` (diagonal approximation)

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
    'feature': 5,           # Which feature to split on
    'bin': 3,               # Split threshold (in bin space)
    'left': {...},          # Left subtree (recursive)
    'right': {...}          # Right subtree (recursive)
}

# Leaf node:
{
    'leaf_value': 0.234,    # Prediction value
    'samples': 150          # Number of samples in leaf
}
```

**Multiclass**: `forest[round]` is a list of K trees (one per class).

**Regression**: `forest[round]` is a single tree.

### 4. Binning

WarpGBM quantizes continuous features into discrete bins for histogram-based training.

**Automatic Binning** (default):
- Computes quantile-based bin edges per feature
- Uses CUDA kernel for fast assignment
- Stored in `self.bin_edges`

**Pre-binned Data**:
- If input is integer with `max(X) < num_bins`, skips binning
- Huge speedup for pre-quantized data (e.g., Numerai)

### 5. Histograms

For each feature bin, accumulate:
- Gradient sum: `G = Σ gradient_i`
- Hessian sum: `H = Σ hessian_i`

**Shape**: `(num_eras, num_features, num_bins)`

**CUDA Kernel**: `histogram_kernel.cu` parallelizes over samples.

**Subtraction Trick**: For left/right child, compute smaller histogram and subtract from parent.

### 6. Multiclass Implementation

**Key Insight**: Train K independent trees per round using the SAME infrastructure as regression.

**Per Round**:
1. Compute softmax probabilities: `p = softmax(F)` where F is raw scores
2. Compute gradients: `g[:,k] = p[:,k] - y_onehot[:,k]`
3. Compute hessians: `h[:,k] = p[:,k] * (1 - p[:,k])`
4. For class k:
   - Set `self.residual = -g[:,k]`
   - Call `grow_tree()` with `class_k` parameter
   - Update `F[:,k]` (raw scores)

**No CUDA Changes Needed**: Reuse histogram/split kernels by passing per-class g/h.

### 7. Feature Importance

**Gain-Based Importance**: WarpGBM tracks the total gain (loss reduction) attributable to each feature across all splits.

**Implementation**:
- `feature_importance_`: Array of shape `(n_features,)` with total gain per feature
- `per_era_feature_importance_`: Array of shape `(n_eras, n_features)` with per-era gains

**Accumulation Logic** (in `grow_tree()`):
```python
# When a split is accepted:
global_feature_idx = self.feat_indices_tree[local_feature].item()
per_era_gains = self.per_era_gain[:, local_feature, best_bin]  # [num_eras]

for era_idx in range(self.num_eras):
    self.per_era_feature_importance_[era_idx, global_feature_idx] += per_era_gains[era_idx].item()
```

**Aggregation** (after training):
```python
# In grow_forest() and grow_forest_multiclass():
self.feature_importance_ = self.per_era_feature_importance_.sum(axis=0)
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

### Critical Implementation Detail: Gradient Restoration

**The Challenge**: When continuing training, we must restore `self.gradients` (the accumulated predictions) from the existing forest. Otherwise, new trees would be trained as if no previous trees exist.

**Solution** (in `_fit_regression` and `_fit_classification`):

```python
if self.warm_start and self._is_fitted and self._trees_trained > 0:
    # Restore predictions from existing forest
    self.gradients = torch.zeros_like(self.Y_gpu) + self.base_prediction
    
    # For regression: add predictions from each existing tree
    for tree in self.forest[:self._trees_trained]:
        if tree:
            leaf_updates = self._compute_tree_predictions(tree, self.bin_indices)
            self.gradients += leaf_updates
    
    # For classification: same but per-class
    for round_trees in self.forest[:self._trees_trained]:
        for class_k, tree in enumerate(round_trees):
            if tree:
                leaf_updates = self._compute_tree_predictions(tree, self.bin_indices)
                self.gradients[:, class_k] += leaf_updates
```

**Helper Method**: `_compute_tree_predictions(tree, bin_indices)`
- Traverses tree structure recursively
- Assigns leaf values to samples based on splits
- Returns tensor of predictions for all samples

### Save/Load Implementation

**save_model()**:
- Pickles all model state (hyperparameters + trained state)
- Includes: `forest`, `bin_edges`, `base_prediction`, `label_encoder`, feature importances, `_trees_trained`, etc.
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
- **Works with all objectives**: Regression, binary, multiclass

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
- `test_numerai_minimal_v5.py`: Integration test (slow, requires download)

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

Let's walk through adding a new objective type (e.g., Poisson regression):

**1. Update validation**:
```python
# In _validate_hyperparams()
if kwargs["objective"] not in ["regression", "multiclass", "binary", "poisson"]:
    raise ValueError(...)
```

**2. Add gradient/hessian computation**:
```python
# In core.py or metrics.py
def _compute_poisson_gradients_hessians(self, y_true):
    """Compute gradients for Poisson loss"""
    y_pred = torch.exp(self.gradients)
    gradients = y_pred - y_true
    hessians = y_pred
    return gradients, hessians
```

**3. Add fit path**:
```python
# In fit()
elif self.objective == "poisson":
    return self._fit_poisson(X, y, era_id, ...)
```

**4. Implement training logic**:
```python
def _fit_poisson(self, X, y, ...):
    # Similar to _fit_regression but with Poisson gradients
    self.bin_indices, ... = self.preprocess_gpu_data(X, y, era_id)
    # ... setup ...
    with torch.no_grad():
        self.grow_forest_poisson()  # Or reuse grow_forest()
    return self
```

**5. Add metric**:
```python
# In metrics.py
def poisson_loss_torch(y_true, y_pred):
    return torch.mean(y_pred - y_true * torch.log(y_pred + 1e-8))
```

**6. Add tests**:
```python
# tests/test_poisson.py
def test_poisson_regression():
    X, y = make_poisson_data()
    model = WarpGBM(objective='poisson', ...)
    model.fit(X, y)
    assert model.predict(X).mean() > 0
```

### Adding New Metrics

```python
# 1. Add to metrics.py
def my_custom_metric(y_true, y_pred):
    return torch.mean(...)

# 2. Update validation in core.py
valid_metrics = ["mse", "corr", "rmsle", "logloss", "accuracy", "custom"]

# 3. Add to get_eval_metric()
if self.eval_metric == "custom":
    return my_custom_metric(y_true, y_pred).item()
```

### Adding New Regularization

Example: L1 regularization (already in constructor, needs implementation):

```python
# In grow_tree(), update leaf value computation:
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
- Input: Binned data, gradients, hessians, sample indices, era indices
- Output: Gradient/hessian sums per (era, feature, bin)
- Parallelization: Block per feature, threads accumulate samples
- Used: Once per tree node (most expensive operation)

**3. Split Finding** (`best_split_kernel.cu`):
- Input: Histograms
- Output: Per-era gains and split directions
- Includes: DES logic (directional consistency check)
- Parallelization: Thread per (feature, bin) pair
- Used: Once per tree node

**4. Prediction** (`predict.cu`):
- Input: Binned data, flattened tree structures
- Output: Predictions
- Parallelization: One thread per (sample, tree) pair
- Used: At inference time

### Memory Management

**GPU Tensors** (stay on GPU throughout training):
- `self.bin_indices`: Binned feature data
- `self.gradients`: Current predictions (regression) or raw scores (classification)
- `self.Y_gpu`: Target labels
- `self.era_indices`: Era assignments
- Histograms (transient, recomputed per node)

**CPU → GPU Transfer**:
- Only during `fit()` initialization
- Uses PyTorch's efficient transfer

**GPU → CPU Transfer**:
- Only for final predictions
- Deleted immediately after `.cpu().numpy()`

### CUDA Tuning

**Key Parameters**:
- `threads_per_block`: 32-64 for most GPUs (multiple of warp size)
- `rows_per_thread`: 4-8 for most datasets
- Bigger values → less thread divergence, more registers

**Guidelines**:
- For small datasets (< 10K rows): Use smaller `rows_per_thread`
- For large datasets (> 1M rows): Increase `threads_per_block` to 128
- For many features (> 1K): Histogram kernel benefits from more parallelism

---

## 🎨 Common Patterns

### Pattern 1: Objective Branching

When adding features that differ by objective:

```python
def fit(self, X, y, ...):
    # Validate first
    early_stopping_rounds = self.validate_fit_params(...)
    
    # Branch by objective
    if self.objective in ["multiclass", "binary"]:
        # Classification path
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        return self._fit_classification(X, y_encoded, ...)
    else:
        # Regression path (default)
        return self._fit_regression(X, y, ...)
```

### Pattern 2: Reusing CUDA Kernels

Don't modify CUDA kernels for new features. Instead:

```python
# For multiclass: call histogram kernel K times
for class_k in range(self.num_classes):
    self.residual = -grads[:, class_k]  # Set per-class gradients
    hist_g, hist_h = self.compute_histograms(...)  # Reuse existing kernel
    # ... process ...
```

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
print(f"gradients shape: {self.gradients.shape}")
```

**Common cause**: Forgot to handle 2D gradients for multiclass.

**3. Label Encoding Issues**

```
ValueError: y contains previously unseen labels
```

**Solution**: Ensure `label_encoder` is fitted on train data and used on eval/test.

**4. Slow Training**

**Check**:
- Are you using pre-binned data but not getting speedup? (Check if it's actually integers < num_bins)
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
- [ ] Custom loss functions (user-defined gradients)
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


