![warpgbm](https://github.com/user-attachments/assets/dee9de16-091b-49c1-a8fa-2b4ab6891184)

# WarpGBM ⚡

> **Neural-speed gradient boosting. GPU-native. Distribution-aware. Production-ready.**

WarpGBM is a high-performance, GPU-accelerated Gradient Boosted Decision Tree (GBDT) library engineered from silicon up with PyTorch and custom CUDA kernels. Built for speed demons and researchers who refuse to compromise.

## Changes
This version is a (allowed under the GNU GPL license) fork/modification of the original WarpGBM implementation (https://github.com/jefferythewind/warpgbm).
In this version, we have made the following changes:
- Added support for monotonic constraints (with value-bound propagation, also per output for multi-output models)
- Added support for sample weights (1-D, or 2-D per output)
- Added support for multi-regression (vector targets, NaN = missing target)
- **Custom loss functions on the GPU**: write the per-sample loss as a torch expression; gradients and hessians come from autograd (or supply them explicitly). Works with multi-output targets, sample weights, missing targets, eras, monotonic constraints and warm starts. See [docs/CUSTOM_OBJECTIVES.md](docs/CUSTOM_OBJECTIVES.md)
- **Missing feature values**: NaN features get their own bin with a learned default direction
- **Early stopping with rollback**: the best evaluated round is recorded (`best_iteration_`, `best_score_`); predictions, feature importances and in-sample `train_predictions_` use the best model; higher-is-better metrics, custom metrics with per-row auxiliary data (`eval_aux` / `train_aux`)
- Deterministic training (fixed-point histograms), a global-memory fallback for many eras, and several bug fixes (multi-output split gain, `min_child_weight` with classification, class priors at prediction, weighted base score)

A full review of all changes, the bugs that were fixed and the behaviour changes is in [docs/FORK_CHANGES.md](docs/FORK_CHANGES.md).

## 🎯 What Sets WarpGBM Apart

**Regression + Classification Unified**  
Train on continuous targets or multiclass labels with the same blazing-fast infrastructure.

**Invariant Learning (DES Algorithm)**  
The only open-source GBDT that natively learns signals stable across shifting distributions. Powered by **[Directional Era-Splitting](https://arxiv.org/abs/2309.14496)** — because your data doesn't live in a vacuum.

**GPU-Accelerated Everything**  
Custom CUDA kernels for binning, histograms, splits, and inference. No compromises, no CPU bottlenecks.

**Scikit-Learn Compatible**  
Drop-in replacement. Same API you know, 10x the speed you need.

---

## 🚀 Quick Start

### Installation

```bash
# Latest from GitHub (recommended)
pip install git+https://github.com/jefferythewind/warpgbm.git

# Stable from PyPI
pip install warpgbm
```

**Prerequisites:** PyTorch with CUDA support ([install guide](https://pytorch.org/get-started/locally/))

### Regression in 5 Lines

```python
from warpgbm import WarpGBM
import numpy as np

model = WarpGBM(objective='regression', max_depth=5, n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Classification in 5 Lines

```python
from warpgbm import WarpGBM

model = WarpGBM(objective='multiclass', max_depth=5, n_estimators=50)
model.fit(X_train, y_train)  # y can be integers, strings, whatever
probabilities = model.predict_proba(X_test)
labels = model.predict(X_test)
```

---

## 🎮 Features

### Core Engine
- ⚡ **GPU-native CUDA kernels** for histogram building, split finding, binning, and prediction
- 🎯 **Multi-objective support**: regression, multi-output regression, binary, multiclass classification
- 🧩 **Custom objectives** — your own GPU loss; gradients and hessians via autograd or explicit
- 🕳️ **Missing values** — NaN features get their own bin with a learned direction; NaN targets are masked
- 📊 **Pre-binned data optimization** — skip binning if your data's already quantized
- 🔥 **Mixed precision support** — `float32` or `int8` inputs
- 🎲 **Stochastic features** — `colsample_bytree` for regularization

### Intelligence
- 🧠 **Invariant learning via DES** — identifies signals that generalize across time/regimes/environments
- 📈 **Smart initialization** — class priors for classification, mean for regression
- 🎯 **Automatic label encoding** — handles strings, integers, whatever you throw at it
- 🔍 **Feature importance** — gain-based importance with unique per-era tracking

### Training Utilities
- ✅ **Early stopping** with validation sets; rolls back to the best round (`best_iteration_`)
- 📊 **Rich metrics**: MSE, RMSLE, correlation, log loss, accuracy
- 🔍 **Progress tracking** with loss curves
- 🎚️ **Regularization** — L2 leaf penalties, min split gain, min child weight
- 💾 **Warm start & checkpointing** — save/load models, incremental training

---

## ⚔️ Benchmarks

### Synthetic Data: 1M Rows × 1K Features (Google Colab L4 GPU)

```
   WarpGBM:   corr = 0.8882, train = 17.4s, infer = 3.2s  ⚡
   XGBoost:   corr = 0.8877, train = 33.2s, infer = 8.0s
  LightGBM:   corr = 0.8604, train = 29.8s, infer = 1.6s
  CatBoost:   corr = 0.8935, train = 392.1s, infer = 379.2s
```

**2× faster than XGBoost. 23× faster than CatBoost.**

[→ Run the benchmark yourself](https://colab.research.google.com/drive/16U1kbYlD5HibGbnF5NGsjChZ1p1IA2pK?usp=sharing)

### Multiclass Classification: 3.5K Samples, 3 Classes, 50 Rounds

```
Training:   2.13s
Inference:  0.37s
Accuracy:   75.3%
```

**Production-ready multiclass at neural network speeds.**

---

## 📖 Examples

### Regression: Beat LightGBM on Your Laptop

```python
import numpy as np
from sklearn.datasets import make_regression
from warpgbm import WarpGBM

# Generate data
X, y = make_regression(n_samples=100_000, n_features=500, random_state=42)
X, y = X.astype(np.float32), y.astype(np.float32)

# Train
model = WarpGBM(
    objective='regression',
    max_depth=5, 
    n_estimators=100, 
    learning_rate=0.01,
    num_bins=32
)
model.fit(X, y)

# Predict
preds = model.predict(X)
print(f"Correlation: {np.corrcoef(preds, y)[0,1]:.4f}")
```

### Classification: Multiclass with Early Stopping

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from warpgbm import WarpGBM

# 5-class problem
X, y = make_classification(
    n_samples=10_000, 
    n_features=50,
    n_classes=5, 
    n_informative=30
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = WarpGBM(
    objective='multiclass',
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    num_bins=32
)

model.fit(
    X_train, y_train,
    X_eval=X_val, y_eval=y_val,
    eval_every_n_trees=10,
    early_stopping_rounds=5,
    eval_metric='logloss'
)

# Predictions use the best evaluated round (model.best_iteration_) after early stopping
probs = model.predict_proba(X_val)  # shape: (n_samples, n_classes)
labels = model.predict(X_val)        # class labels
print(model.best_iteration_, model.best_score_)
```

Evaluation runs every `eval_every_n_trees` rounds and always at the last round. Training stops once
`early_stopping_rounds` evaluations in a row did not improve on the best one, and the model then
uses the trees up to `best_iteration_` (all trees are kept; `predict(X, num_trees=...)` selects any
other prefix, `model.truncate()` drops the rest). Metrics where higher is better (`'accuracy'`,
`'corr'`) are maximized.

#### Custom metrics with auxiliary data

A callable metric is minimized; wrap it in `Metric(fn, higher_is_better=True)` to maximize it.
Per-row data that is not part of the targets reaches the metric through `aux`:

```python
import numpy as np, torch
from warpgbm import WarpGBM, Metric

# Train MSE on y / sigma, early-stop on Pearson(pred * sigma, raw y clipped at +-10 sigma)
def comp_corr(y_true, pred, weight, mask, aux):      # GPU tensors; aux: [n_eval, 2]
    y_raw, sigma = aux[:, 0], aux[:, 1]
    a = pred * sigma
    b = torch.clamp(y_raw, -10 * sigma, 10 * sigma)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))

model = WarpGBM(n_estimators=1000, learning_rate=0.05)
model.fit(
    X, y / sigma,
    X_eval=X_val, y_eval=y_val / sigma_val,
    eval_aux=np.column_stack([y_val, sigma_val]),   # (n_eval,) or (n_eval, A)
    train_aux=np.column_stack([y, sigma]),          # optional: same metric on the training set
    eval_metric=Metric(comp_corr, higher_is_better=True, name="comp_corr"),
    eval_every_n_trees=10, early_stopping_rounds=20,
)
model.train_predictions_   # in-sample raw scores of the best model, (n_samples, n_outputs)
```

### Invariant Learning: Distribution-Robust Signals

```python
# Your data spans multiple time periods/regimes/environments
# Pass era_id to learn only signals that work across ALL eras

model = WarpGBM(
    objective='regression',
    max_depth=8,
    n_estimators=100
)

model.fit(
    X, y, 
    era_id=era_labels  # Array marking which era each sample belongs to
)

# Now your model ignores spurious correlations that don't generalize!
```

### Monotonic Constraints

Constrain model behavior per feature using `1, 0, -1`:

- `1`: prediction is monotonic increasing w.r.t. that feature
- `0`: no constraint
- `-1`: prediction is monotonic decreasing

You can pass constraints as either a dense vector (`len == n_features`) or sparse index dict.

```python
# Dense vector form
model = WarpGBM(
    objective='regression',
    monotonic_constraints=[1, 0, -1, 0],
)

# Sparse dict form
model = WarpGBM(
    objective='regression',
    monotonic_constraints={0: 1, 2: -1},
)

# With eras: constraints are enforced per era during split selection
model.fit(X, y, era_id=era_labels)

# Without eras: constraints are enforced globally
model.fit(X, y)
```

### Missing Values

NaN feature values are put into a dedicated bin. Every split learns whether missing values go left or
right (`node["nan_left"]`), and missing values seen only at prediction time follow the right branch.
NaN *targets* are treated as missing and ignored (per output for multi-output targets).

```python
X[some_rows, 3] = np.nan          # no imputation needed
model = WarpGBM(objective='regression').fit(X, y)
```

### Custom Loss Functions (GPU)

```python
import torch
from warpgbm import WarpGBM, Objective, CustomObjective

# 1) a per-sample loss: gradient and hessian come from autograd
def log_cosh(y, pred):                             # y, pred: [n_samples, n_outputs]
    r = pred - y
    return r + torch.nn.functional.softplus(-2 * r) - 0.693147

model = WarpGBM(objective=log_cosh, n_estimators=300)

# 2) with options: constant hessian, initial score, link function, one tree per output, ...
q = 0.9
pinball = CustomObjective(
    loss_fn=lambda y, p: torch.maximum(q * (y - p), (q - 1) * (y - p)),
    hessian="ones",
    init_fn=lambda y, weight: torch.quantile(y[:, 0], q).reshape(1),
)

# 3) a reusable class, optionally with explicit derivatives
class PseudoHuber(Objective):
    def loss(self, y, pred):
        return torch.sqrt(1 + (pred - y) ** 2) - 1

    def grad_hess(self, y, pred):                  # optional; default is autograd of loss
        r = pred - y
        s = 1 + r ** 2
        return r / s.sqrt(), 1 / (s * s.sqrt())

model.fit(X, Y, sample_weight=W, era_id=eras)     # Y may be multi-output and contain NaN
```

The engine applies sample weights, missing-target masks, era splitting, monotonic constraints and
early stopping for you; the loss is written without them. See **[docs/CUSTOM_OBJECTIVES.md](docs/CUSTOM_OBJECTIVES.md)** for the full guide.

### Feature Importance: Understand Your Model

```python
from warpgbm import WarpGBM
from sklearn.datasets import load_iris

# Train a model
iris = load_iris()
X, y = iris.data, iris.target

model = WarpGBM(objective='multiclass', max_depth=5, n_estimators=100)
model.fit(X, y)

# Get feature importance (normalized)
importances = model.get_feature_importance()
for name, imp in zip(iris.feature_names, importances):
    print(f"{name}: {imp:.4f}")

# Output:
# sepal length (cm): 0.0002
# sepal width (cm): 0.0007
# petal length (cm): 0.1997
# petal width (cm): 0.7994
```

**Per-Era Feature Importance (Unique to WarpGBM!)**

When training with `era_id`, see which features are stable across environments:

```python
# Train with eras
model.fit(X, y, era_id=era_labels)

# Get per-era importance: shape (n_eras, n_features)
per_era_imp = model.get_per_era_feature_importance()

# Identify invariant features (high importance across ALL eras)
invariant_features = per_era_imp.min(axis=0) > threshold
```

### Warm Start: Incremental Training & Checkpointing

Train a model in stages, save checkpoints, and resume training later:

```python
from warpgbm import WarpGBM
import numpy as np

# Train 50 trees
model = WarpGBM(
    objective='regression',
    n_estimators=50,
    max_depth=5,
    learning_rate=0.1,
    warm_start=True  # Enable incremental training
)
model.fit(X, y)
predictions_50 = model.predict(X_test)

# Save checkpoint
model.save_model('checkpoint_50.pkl')

# Continue training for 50 more trees (total: 100)
model.n_estimators = 100
model.fit(X, y)  # Adds 50 trees on top of existing 50
predictions_100 = model.predict(X_test)

# Or load and continue training later
model_loaded = WarpGBM()
model_loaded.load_model('checkpoint_50.pkl')
model_loaded.warm_start = True
model_loaded.n_estimators = 100
model_loaded.fit(X, y)  # Resumes from 50 → 100 trees
```

**Use Cases:**
- **Hyperparameter tuning**: Train to 50 trees, evaluate, decide if you need 100 or 200
- **Checkpointing**: Save progress during long training runs
- **Iterative development**: Add more trees without retraining from scratch
- **Production updates**: Retrain models incrementally as new data arrives

### Pre-binned Data: Maximum Speed (Numerai Example)

```python
import pandas as pd
from numerapi import NumerAPI
from warpgbm import WarpGBM

# Download Numerai data (already quantized to integers)
napi = NumerAPI()
napi.download_dataset('v5.0/train.parquet', 'train.parquet')
train = pd.read_parquet('train.parquet')

features = [f for f in train.columns if 'feature' in f]
X = train[features].astype('int8').values
y = train['target'].values

# WarpGBM detects pre-binned data and skips binning
model = WarpGBM(max_depth=5, n_estimators=100, num_bins=20)
model.fit(X, y)  # Blazing fast!
```

**Result: 13× faster than LightGBM on Numerai data (49s vs 643s)**

---

## 🧠 Invariant Learning: Why It Matters

Most ML models assume your training and test data come from the same distribution. **Reality check: they don't.**

- Stock prices shift with market regimes
- User behavior changes over time  
- Experimental data varies by batch/site/condition

**Traditional GBDT:** Learns any signal that correlates with the target, including fragile patterns that break OOD.

**WarpGBM with DES:** Explicitly tests if each split generalizes across ALL environments (eras). Only keeps robust signals.

### The Algorithm

For each potential split, compute gain separately in each era. Only accept splits where:
1. Gain is positive in ALL eras
2. Split direction is consistent across eras

This prevents overfitting to spurious correlations that only work in some time periods or environments.

### Visual Intuition

<img src="https://github.com/user-attachments/assets/2be11ef3-6f2e-4636-ab91-307a73add247" alt="Era Splitting Visualization" width="400"/>

**Left:** Standard training pools all data together — learns any signal that correlates.  
**Right:** Era-aware training demands signals work across all periods — learns robust features only.

### Research Foundation

- **Invariant Risk Minimization**: [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893)
- **Hard-to-Vary Explanations**: [Parascandolo et al., 2020](https://arxiv.org/abs/2009.00329)
- **Era Splitting for Trees**: [DeLise, 2023](https://arxiv.org/abs/2309.14496)

---

## 📚 API Reference

### Constructor Parameters

```python
WarpGBM(
    objective='regression',        # 'regression', 'multi_regression', 'binary', 'multiclass',
                                   # an Objective instance, or a per-sample loss function (y, pred) -> loss
    num_bins=10,                   # Histogram bins for feature quantization (+1 extra bin for NaN)
    max_depth=3,                   # Maximum tree depth
    learning_rate=0.1,             # Shrinkage rate (aka eta)
    n_estimators=100,              # Number of boosting rounds
    min_child_weight=20,           # Min sum of sample weights in a child node (~ row count)
    min_split_gain=0.0,            # Min loss reduction to split
    min_sum_hessian=1e-3,          # Min hessian sum per child for an output to contribute to the gain
    L2_reg=1e-6,                   # L2 leaf regularization
    colsample_bytree=1.0,          # Feature subsample ratio per tree
    random_state=None,             # Seed of the model's own generator (fit never touches global RNGs)
    warm_start=False,              # If True, continue training from existing trees
    monotonic_constraints=None,    # Optional: vector or dict {feature_idx: -1|0|1}
    threads_per_block=64,          # CUDA block size (tune for your GPU)
    rows_per_thread=4,             # Rows processed per thread
    device='cuda',                 # 'cuda' (current device) or 'cuda:N'; training requires CUDA
    verbose=True,                  # False silences all training / save / load messages
)
```

Models can be trained on different GPUs, or concurrently in several threads, without switching
the current device; with the same `random_state` the forests are identical to sequential fits:

```python
from concurrent.futures import ThreadPoolExecutor

def train(device):
    return WarpGBM(device=device, colsample_bytree=0.8, random_state=42, verbose=False).fit(X, y)

with ThreadPoolExecutor(2) as pool:
    model_a, model_b = pool.map(train, ["cuda:0", "cuda:1"])
```

### Training Methods

```python
model.fit(
    X,                              # Features: np.array shape (n_samples, n_features), NaN allowed
    y,                              # Target: (n_samples,) or (n_samples, n_outputs); NaN = missing target
    era_id=None,                    # Optional: era labels for invariant learning
    sample_weight=None,             # Optional: (n_samples,) or (n_samples, n_outputs); auto-rescaled to mean=1
                                    #   (strictly positive for 1-D targets, non-negative for 2-D targets)
    X_eval=None,                    # Optional: validation features
    y_eval=None,                    # Optional: validation targets
    sample_weight_eval=None,        # Optional: eval weights (same rules as sample_weight)
    eval_every_n_trees=None,        # Eval frequency (in rounds)
    early_stopping_rounds=None,     # Stop after N evals without improving on the best; then use best_iteration_
    eval_metric=None,               # None (objective default), 'mse', 'rmsle', 'corr', 'logloss', 'accuracy',
                                    #   'loss' (objective's loss), callable(y, pred, weight, mask[, aux]) -> float
                                    #   (minimized) or Metric(fn, higher_is_better=True)
    eval_aux=None,                  # Optional: (n_eval,) or (n_eval, A) per-row data passed to the metric as aux=
    train_aux=None,                 # Optional: same for the training set (training metric)
)
```

The last round is always evaluated. With `early_stopping_rounds`, predictions, feature importances
and `train_predictions_` default to the best round; without it, an eval set only monitors
(`best_iteration_` is still recorded, but all trees are used). A `warm_start` continuation appends
after all trained rounds; call `model.truncate()` first to continue from the best round. The
objective cannot change between warm-start fits.
```

### Prediction & Utility Methods

```python
# Regression: returns predicted values
predictions = model.predict(X)

# Classification: returns class labels (decoded)
labels = model.predict(X)

# Classification: returns class probabilities
probabilities = model.predict_proba(X)  # shape: (n_samples, n_classes)

# Raw scores before the objective's transform (e.g. logits)
raw = model.predict_raw(X)

# All prediction methods take num_trees (boosting rounds); None = default model
# (best_iteration_ after early stopping, else all trained rounds)
all_trees = model.predict(X, num_trees=model.num_trees_)

# Feature importance: gain-based (like LightGBM/XGBoost), of the default model
importances = model.get_feature_importance(normalize=True)  # sums to 1.0
raw_gains = model.get_feature_importance(normalize=False)   # raw gain values
first_100 = model.get_feature_importance(num_trees=100)      # any prefix of rounds

# Per-era importance (when era_id was used in training); stored for the best round and all rounds
per_era_imp = model.get_per_era_feature_importance(normalize=True)  # shape: (n_eras, n_features)

# Drop the rounds after best_iteration_ (or after num_trees)
model.truncate()

# Save and load models
model.save_model('checkpoint.pkl')  # Saves all model state
model_loaded = WarpGBM()
model_loaded.load_model('checkpoint.pkl')  # Restores everything
```

### Attributes

```python
model.classes_                    # Unique class labels (classification only)
model.num_classes                 # Number of classes (classification only)
model.forest                      # Trained tree structures
model.training_loss               # Training loss (objective loss) per round, if an eval set was given
model.evals_result_               # {"train": [...], "eval": [...]} eval metric per checkpoint (natural values)
model.eval_iterations_            # Round count of every checkpoint
model.eval_loss                   # Alias of evals_result_["eval"]
model.num_trees_                  # Number of trained boosting rounds
model.best_iteration_             # Number of rounds of the best evaluated model (None without eval set)
model.best_score_                 # Eval metric value at best_iteration_
model.train_predictions_          # In-sample raw scores of the default model, (n_samples, n_outputs); not saved
model.feature_importance_         # Feature importance of the default model (sum across eras)
model.per_era_feature_importance_ # Per-era feature importance of the default model (when era_id used)
```

---

## 🔧 Installation Details

### Linux / macOS (Recommended)

```bash
pip install git+https://github.com/jefferythewind/warpgbm.git
```

Compiles CUDA extensions using your local PyTorch + CUDA setup.

### Colab / Mismatched CUDA Versions

```bash
pip install warpgbm --no-build-isolation
```

### Windows

```bash
git clone https://github.com/jefferythewind/warpgbm.git
cd warpgbm
python setup.py bdist_wheel
pip install dist/warpgbm-*.whl
```

---

## 🎯 Use Cases

**Financial ML:** Learn signals that work across market regimes  
**Time Series:** Robust forecasting across distribution shifts  
**Scientific Research:** Models that generalize across experimental batches  
**High-Speed Inference:** Production systems with millisecond SLAs  
**Kaggle/Competitions:** GPU-accelerated hyperparameter tuning  
**Multiclass Problems:** Image classification fallback, text categorization, fraud detection

---

## 🚧 Roadmap

- [ ] Multi-GPU training support
- [ ] SHAP value computation on GPU
- [ ] Feature interaction constraints
- [x] Monotonic constraints
- [x] Custom loss functions
- [x] Missing value handling
- [ ] Distributed training
- [ ] ONNX export for deployment

---

## 🙏 Acknowledgements

Built on the shoulders of PyTorch, scikit-learn, LightGBM, XGBoost, and the CUDA ecosystem. Special thanks to the GBDT research community and all contributors.

---

## 📝 Version History

### v2.5.0 (Current)
- 🖥️ `device="cuda:N"` works from any thread without switching the current device; kernels run on the current stream of the tensors' device
- 🎲 Per-model random generator: concurrent fits are reproducible and `fit` no longer seeds the global torch / numpy RNGs
- 🔇 `verbose=False` silences all messages
- 🚫 Non-CUDA devices (`device="cpu"`) are rejected at construction; there is no CPU code path

### v2.4.0
- ⏪ Early stopping rolls back to the best round: `best_iteration_`, `best_score_`, `evals_result_`; `predict(..., num_trees=)`, `truncate()`
- 🔍 Feature importances follow the used trees (`num_trees=`; per-era at the best round and all rounds)
- 📏 `Metric(fn, higher_is_better=True)`; `accuracy` / `corr` are maximized and reported as natural values; per-row `eval_aux` / `train_aux` for callable metrics
- 🎯 `train_predictions_`: in-sample predictions of the final model
- ⚠️ The objective can no longer change between warm-start fits

### v2.3.0
- 🧩 Custom objective API (`Objective`, `CustomObjective`, or a plain loss function; autograd gradients and hessians)
- 🕳️ NaN feature bin with learned default direction
- 🎯 Multi-output regression, sample weights, monotonic constraints (with bound propagation)
- 🔁 Deterministic fixed-point histograms and prediction; global-memory fallback for many eras
- 🐛 Fixes: multi-output split gain, `min_child_weight` with classification, class priors at prediction, weighted base score
- 📚 [docs/FORK_CHANGES.md](docs/FORK_CHANGES.md), [docs/CUSTOM_OBJECTIVES.md](docs/CUSTOM_OBJECTIVES.md)

### v2.2.0
- 💾 **Warm start support** for incremental training (closes #14)
- 📦 `save_model()` and `load_model()` methods for checkpointing
- 🔄 Resume training from saved models with `warm_start=True`
- ✅ Comprehensive test suite for warm start and save/load functionality
- 📚 Updated documentation with warm start examples

### v2.1.1
- 🎲 **random_state parameter** for reproducible results (closes #12)
- 🔧 Controls randomness in feature subsampling (`colsample_bytree`)
- ✅ Comprehensive reproducibility tests

### v2.1.0
- 🔍 **Feature importance** with gain-based tracking and unique per-era analysis
- 📊 `get_feature_importance()` and `get_per_era_feature_importance()` methods
- ✅ Comprehensive test suite comparing with LightGBM
- 📚 Updated documentation with feature importance examples

### v2.0.0
- ✨ **Multiclass classification support** via softmax objective
- 🎯 Binary classification mode
- 📊 New metrics: log loss, accuracy
- 🏷️ Automatic label encoding (supports strings)
- 🔮 `predict_proba()` for probability outputs
- ✅ Comprehensive test suite for classification
- 🔒 Full backward compatibility with regression
- 🐛 Fixed unused variable issue (#8)
- 🧹 Removed unimplemented L1_reg parameter
- 📚 Major documentation overhaul with AGENT_GUIDE.md

### v1.0.0
- 🧠 Invariant learning via Directional Era-Splitting (DES)
- 🚀 VRAM optimizations
- 📈 Era-aware histogram computation

### v0.1.26
- 🐛 Memory bug fixes in prediction
- 📊 Added correlation eval metric

### v0.1.25
- 🎲 Feature subsampling (`colsample_bytree`)

### v0.1.23
- ⏹️ Early stopping support
- ✅ Validation set evaluation

### v0.1.21
- ⚡ CUDA prediction kernel (replaced vectorized Python)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🤝 Contributing

Pull requests welcome! See [AGENT_GUIDE.md](AGENT_GUIDE.md) for architecture details and development guidelines.

---

**Built with 🔥 by @jefferythewind**

*"Train smarter. Predict faster. Generalize better."*
