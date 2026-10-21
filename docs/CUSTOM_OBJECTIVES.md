# Custom objectives (loss functions)

WarpGBM trains every model, built-in or custom, through one engine that needs the
**gradient and hessian of the loss with respect to the raw model score**, computed on the GPU.
For a custom objective you usually write only the loss as a torch expression. PyTorch autograd
supplies the gradient and the hessian, and the engine handles sample weights, missing targets,
multiple outputs, eras (directional era splitting), monotonic constraints, NaN features, warm
starts, evaluation and early stopping.

```python
from warpgbm import WarpGBM, Objective, CustomObjective
```

| You have… | Use | Derivatives |
|---|---|---|
| a loss function `loss(y, pred)` | `WarpGBM(objective=loss)` | autograd |
| a loss function plus options (init score, link, hessian mode, one tree per output, …) | `CustomObjective(loss_fn, ...)` | autograd |
| known gradients / hessians | `CustomObjective(loss_fn, grad_hess_fn=...)`, or subclass `Objective` and override `grad_hess` | yours |
| a reusable, picklable loss with parameters | subclass `Objective` and implement `loss` | autograd |

---

## 1. Conventions

* **Per-sample losses.** The loss of a row may depend only on that row's targets and scores. It
  is returned **per entry** (`[N, K]`) or **per row** (`[N]`, for losses that couple the outputs
  of a row, e.g. a cross-entropy). The engine never sees a dataset-level reduction; returning a
  scalar raises an error.
* **Shapes.** Raw scores `pred` are `[N, K]` (K = number of outputs, 1 for ordinary regression).
  Targets `y` are `[N, T]` float32. For regression-like objectives T = K. For the built-in
  classifier T = 1 (encoded labels) and K = number of classes.
* **Missing targets.** `NaN`/`inf` targets are missing. Before your code sees `y`, missing
  entries are **filled with 0** so the math stays finite; their gradients and hessians are zeroed
  afterwards (with `torch.where`, so NaN-safe). Multi-output targets can have a different missing
  pattern per output.
* **Weights.** `sample_weight` is `[N]` or `[N, K]` and normalized to mean 1. You do **not** use
  it in your loss. The engine multiplies the gradient and hessian entry-wise by the weight, which
  is the same as training on `Σ_ik w_ik · loss_ik`.
* **Sign convention.** As in XGBoost/LightGBM: `grad = dL/dscore`, `hess = d²L/dscore²`. Leaves
  are Newton steps `-Σgrad / (Σhess + L2_reg)` over the rows in the leaf, clamped to the monotonic
  bounds if constraints are active.
* **Device.** Everything is a CUDA tensor; stay in torch to keep it on the GPU.

What the engine does each round:

```
grad, hess = objective.grad_hess(y, pred)      # default: autograd of objective.loss
grad, hess = grad * weight, hess * weight      # weight: [N, K], 0 for missing targets
grad, hess = where(mask, grad, 0), where(mask, hess, 0)
```

It then checks that everything is finite and raises `FloatingPointError` otherwise.

---

## 2. Loss functions with autograd

### A plain function

```python
import torch
from warpgbm import WarpGBM

def log_cosh(y, pred):                        # [N, K] -> per-entry loss [N, K]
    r = pred - y
    return r + torch.nn.functional.softplus(-2 * r) - 0.693147

model = WarpGBM(objective=log_cosh, n_estimators=200)
model.fit(X, Y, sample_weight=W, era_id=eras)  # Y: (n,) or (n, K), may contain NaN; W: (n,) or (n, K)
```

### `CustomObjective`

```python
from warpgbm import CustomObjective

q = 0.9
pinball = CustomObjective(
    loss_fn=lambda y, p: torch.maximum(q * (y - p), (q - 1) * (y - p)),
    hessian="ones",                                       # zero curvature -> first-order boosting
    init_fn=lambda y, weight: torch.quantile(y[:, 0], q).reshape(1),
)

poisson = CustomObjective(
    loss_fn=lambda y, s: torch.exp(s) - y * s,            # log link
    init_fn=lambda y, weight: torch.log(y.mean(dim=0)),
    transform_fn=torch.exp,                               # predict() returns the rate, predict_raw() the log-rate
    name="poisson",
)
```

| argument | default | meaning |
|---|---|---|
| `loss_fn(y, pred)` | — | per-sample loss, `[N, K]` or `[N]` |
| `grad_hess_fn(y, pred)` | `None` | explicit `(grad, hess)`, each `[N, K]` or broadcastable; replaces autograd |
| `hessian` | `"autograd"` | `"autograd"`: exact diagonal hessian. `"ones"`: constant 1 (first-order boosting) |
| `min_hessian` | `0.0` | autograd hessians are clamped to at least this value |
| `init_fn(y, weight)` | one weighted Newton step from score 0 | constant start score, shape `[K]`; `weight` is `[N, K]` (0 for missing targets) |
| `transform_fn(raw)` | identity | maps raw scores to predictions |
| `output_strategy` | `"vector"` | `"vector"`: one tree per round with K-dimensional leaves (shared splits, gain summed over outputs). `"per_output"`: K trees per round, one per output |
| `name` | `"custom"` | shown in log messages |

At least one of `loss_fn` / `grad_hess_fn` is required. `loss_fn` is also used for the training
log and as the default eval metric.

### How the derivatives are computed

The gradient is `autograd.grad(loss.sum(), pred)`. Rows are independent, so one more backward
pass per output column gives the **exact diagonal hessian** `d²loss / dpred_k²` for every row,
also for losses that couple the outputs of a row: a per-row softmax cross-entropy gives exactly
`p(1 - p)`. The cost is 1 + K backward passes per round (2 for a single output).

Losses with zero or near-zero curvature (L1, pinball, log-cosh at large residuals) give Newton
leaves that are unbounded. Use `hessian="ones"`, or `min_hessian=...` together with `L2_reg`.

---

## 3. Subclassing `Objective`

A subclass is the reusable, picklable form of a custom loss. Implement `loss` and gradients come
from autograd; override `grad_hess` as well if you have cheaper or more stable closed forms.

```python
import torch
from warpgbm import Objective

class PseudoHuber(Objective):
    name = "pseudo_huber"

    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, y, pred):
        return self.delta**2 * (torch.sqrt(1 + ((pred - y) / self.delta) ** 2) - 1)

    # optional: explicit derivatives instead of autograd
    def grad_hess(self, y, pred):
        r = pred - y
        scale = 1.0 + (r / self.delta) ** 2
        return r / torch.sqrt(scale), 1.0 / (scale * torch.sqrt(scale))

    # optional: constant start score, shape [K]
    def init_score(self, y, weight):
        return y.median(dim=0).values

model = WarpGBM(objective=PseudoHuber(1.0), L2_reg=1.0, n_estimators=200)
```

| method / attribute | default |
|---|---|
| `loss(y, pred)` | none; `[N, K]` or `[N]`. Needed for autograd and for the `"loss"` metric |
| `grad_hess(y, pred)` | autograd of `loss` |
| `init_score(y, weight)` → `[K]` | one weighted Newton step from score 0 |
| `transform(raw)` | identity |
| `output_strategy` | `"vector"` |
| `hessian`, `min_hessian` | `"autograd"`, `0.0` (autograd path) |
| `prepare_targets(y_np)` → `[N, T]` | float32, 1-D → one column |
| `num_outputs(y)` | `T` |
| `default_metric` | `"loss"` (falls back to `"mse"` without `loss`) |

`save_model` pickles the objective with the model. Define custom classes and loss functions at
module level (no lambdas) if you want to `load_model` them in another process.

---

## 4. Evaluation metrics and early stopping

`fit(..., eval_metric=...)` accepts:

* built-in strings: `"mse"`, `"corr"`, `"rmsle"` (single output), `"logloss"`, `"accuracy"`
  (classification), and `"loss"` (weighted mean of the objective's `loss`). `"accuracy"` and
  `"corr"` are maximized, the others minimized; all are reported as natural values;
* `None` (default): the objective's `default_metric`. That is `"mse"` for regression,
  `"logloss"` for classification, and `"loss"` for custom objectives that implement `loss`;
* a **callable** `metric(y_true, pred, weight, mask) -> float`, which is **minimized**. `pred` is
  the transformed prediction (what `predict` returns); all inputs are GPU tensors, 1-D for
  single-output regression and 2-D otherwise;
* `Metric(fn, higher_is_better=True, name="...")` to **maximize** a callable (and name it in the
  log).

```python
from warpgbm import Metric

def mae(y_true, pred, weight, mask):
    w = weight * mask
    return float((torch.abs(y_true - pred) * w).sum() / w.sum())

model.fit(X, Y, X_eval=Xv, y_eval=Yv, eval_every_n_trees=10, early_stopping_rounds=5, eval_metric=mae)
```

### Auxiliary per-row data

Metrics that need per-row data besides the targets (raw targets, scales, groups, …) get it through
`fit(..., eval_aux=A_eval, train_aux=A_train)`: numpy arrays of shape `[n]` or `[n, A]`, passed to
the callable as `aux=` (a float32 GPU tensor of the same shape). The callable must have an `aux`
parameter (or `**kwargs`). If `aux` has no default and `train_aux` is not given, the metric is
not computed on the training set.

```python
# trained on y / sigma; early-stop on Pearson(pred * sigma, raw y clipped at +-10 sigma)
def comp_corr(y_true, pred, weight, mask, aux):
    y_raw, sigma = aux[:, 0], aux[:, 1]
    a, b = pred * sigma, torch.clamp(y_raw, -10 * sigma, 10 * sigma)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))

model.fit(X, y / sigma, X_eval=Xv, y_eval=yv / sigma_v, eval_aux=np.column_stack([yv, sigma_v]),
          eval_metric=Metric(comp_corr, higher_is_better=True), eval_every_n_trees=10,
          early_stopping_rounds=20)
```

### Best iteration and rollback

* The metric is evaluated every `eval_every_n_trees` rounds and at the last round. The results are
  in `evals_result_ = {"train": [...], "eval": [...]}` (the eval metric on both sets) with the
  round counts in `eval_iterations_`; `eval_loss` is an alias of `evals_result_["eval"]`, and
  `training_loss` keeps the objective loss per round.
* `best_iteration_` (number of rounds, 1-based) and `best_score_` record the best evaluation
  (strictly better wins, so ties keep the earlier round; NaN never wins).
* With `early_stopping_rounds=r`, training stops after `r` evaluations in a row without
  improvement, and the **default model is the best one**: `predict`, `predict_raw`,
  `predict_proba`, `get_feature_importance`, `get_per_era_feature_importance` and
  `train_predictions_` use the first `best_iteration_` rounds. Without `early_stopping_rounds` the
  eval set only monitors and all trees are used.
* All trained trees are kept: `predict(X, num_trees=n)` (and the other methods) select any prefix
  of rounds, and `truncate(num_trees=None)` drops the rounds after `best_iteration_`.
* `get_feature_importance(num_trees=n)` is exact for any `n` (each split stores its gain). Per-era
  importances are stored for `best_iteration_` and for all rounds only.
* `train_predictions_`: in-sample raw scores of the default model, `[n_samples, n_outputs]`
  (equal to `predict_raw(X_train)` up to float rounding). Not saved by `save_model`.

---

## 5. Warm starts and persistence

* `warm_start=True` continues from the existing trees, always **after all trained rounds** (the
  bin edges of the first fit are reused). To continue from the best round of an early-stopped
  fit, call `model.truncate()` first. A continuation tracks its best round over its own
  evaluations; without an eval set the model uses all trees.
* The objective cannot change between warm-start fits (this raises a `ValueError`).
* `save_model` pickles the objective instance (see §3) and keeps `best_iteration_`,
  `best_score_` and the eval history, so a loaded model predicts with the same default model.

---

## 6. Interaction with other features

| feature | behaviour with custom objectives |
|---|---|
| sample weights | `[N]` or `[N, K]`, normalized to mean 1, applied to grad/hess by the engine; 0 for missing targets |
| `min_child_weight` | minimum **sum of row weights** in a child (≈ row count when unweighted); independent of the hessian scale |
| `min_sum_hessian` | per output: an output whose child hessian sum is below this does not contribute to the split gain |
| eras / DES | per-era histograms of the gradients; the gain is averaged over eras and split directions must agree across eras |
| monotonic constraints | value bounds are propagated to the children and leaves are clamped, for every output (vector leaves) or per class tree |
| NaN features | own bin with a learned default direction, independent of the objective |
| determinism | histograms are accumulated in fixed point and predictions sum the trees in a fixed order, so with `random_state` set, results are bit-reproducible run to run |
