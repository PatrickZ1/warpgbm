# Custom objectives (loss functions)

WarpGBM trains every model, built-in or custom, through one engine that only needs
**gradients and hessians of the loss with respect to the raw model score**, computed on the GPU.
The engine takes care of sample weights, missing targets, eras (directional era splitting),
monotonic constraints, NaN features, warm starts, evaluation and early stopping. So a custom
objective is usually just the math of the loss.

```python
from warpgbm import WarpGBM, Objective, GradHessObjective, AutogradObjective, DatasetLossObjective, losses
```

There are four ways to define a loss:

| You have… | Use | Derivatives |
|---|---|---|
| a per-sample loss with known grad / hess | subclass `Objective` or `GradHessObjective(fn)` | you write them |
| a per-sample loss as a torch expression | `AutogradObjective(loss_fn)` | autograd (diagonal hessian via a 2nd pass) |
| a **dataset-level** loss (correlation, per-era statistics, …), possibly plus per-sample terms | `DatasetLossObjective(loss_fn)` | autograd gradient + curvature estimate |
| just a function `(y, pred, ctx) -> (grad, hess)` | pass it directly: `WarpGBM(objective=fn)` | you write them |

---

## 1. Conventions

* **Shapes.** Raw scores `pred` are always `[N, K]` (K = number of outputs, 1 for ordinary
  regression). Targets `y` are `[N, T]` float32. For regression-like objectives T = K. For the
  built-in classifier T = 1 (encoded labels) and K = number of classes.
* **Missing targets.** `NaN`/`inf` targets are treated as missing. Before your code sees `y`,
  missing entries are **filled with 0**, so the math stays finite. Their gradients and hessians are
  then zeroed with `torch.where` (NaN-safe). `ctx.mask` tells you which entries are observed.
* **Sign convention.** Same as XGBoost/LightGBM: `grad = dL/dscore`, `hess = d²L/dscore²`.
  Leaves are Newton steps `-Σgrad / (Σhess + L2_reg)` (summed over the rows in the leaf), clamped
  to the monotonic bounds if constraints are active.
* **Device.** Everything is a CUDA tensor; stay in torch to keep it on the GPU.

### The context object

Every call receives an `ObjectiveContext`:

| field | shape | meaning |
|---|---|---|
| `ctx.weight` | `[N, K]` | normalized sample weight (mean 1), **already 0 where the target is missing** |
| `ctx.mask` | `[N, K]` bool | observed score entries |
| `ctx.target_mask` | `[N, T]` bool | observed target entries |
| `ctx.era_index` | `[N]` int32 | era of each row, 0..E-1 (all 0 without `era_id`) |
| `ctx.num_outputs` | int | K |
| `ctx.iteration` | int | current boosting round (0 in `init_score` and evaluation) |
| `ctx.row_weight` | `[N]` | mean of `ctx.weight` over outputs (used for `min_child_weight`) |

### What the engine does with your output

For per-sample objectives (`apply_weights = True`, the default):

```
grad, hess = objective.grad_hess(y, pred, ctx)   # unweighted, [N, K] or broadcastable
grad, hess = grad * ctx.weight, hess * ctx.weight
grad, hess = where(ctx.mask, grad, 0), where(ctx.mask, hess, 0)
```

It then checks that everything is finite and raises `FloatingPointError` otherwise. Objectives
with `apply_weights = False` must apply the weights themselves (masked entries are still zeroed).

---

## 2. Per-sample losses

### Subclassing `Objective`

```python
import torch
from warpgbm import Objective

class PseudoHuber(Objective):
    name = "pseudo_huber"

    def __init__(self, delta=1.0):
        self.delta = delta

    def grad_hess(self, y, pred, ctx):
        r = pred - y
        scale = 1.0 + (r / self.delta) ** 2
        return r / torch.sqrt(scale), 1.0 / (scale * torch.sqrt(scale))

    def loss(self, y, pred, ctx):            # optional: training log + default eval metric
        return self.delta**2 * (torch.sqrt(1 + ((pred - y) / self.delta) ** 2) - 1)

    def init_score(self, y, ctx):            # optional: constant start score, shape [K]
        return y.median(dim=0).values

model = WarpGBM(objective=PseudoHuber(1.0), L2_reg=1.0, n_estimators=200)
model.fit(X, y, sample_weight=w, era_id=eras)
```

Hooks you can override:

| method / attribute | default |
|---|---|
| `grad_hess(y, pred, ctx)` | **required** |
| `loss(y, pred, ctx)` | none; return `[N, K]` (element-wise), `[N]` (per row) or a scalar |
| `init_score(y, ctx)` → `[K]` | one weighted Newton step from score 0 |
| `transform(raw)` | identity (e.g. return `torch.exp(raw)` for a log link) |
| `prepare_targets(y_np)` → `[N, T]` | float32, 1-D → one column |
| `num_outputs(y)` | `T` |
| `output_strategy` | `"vector"`: one tree per round with K-dimensional leaves (shared splits, gain summed over outputs). `"per_output"`: K trees per round, one per score column |
| `apply_weights` | `True` |
| `default_metric` | `"loss"` (falls back to `"mse"` if `loss` is not implemented) |

Custom objectives are pickled together with the model by `save_model`, so define the class at
module level so that it can be imported again by `load_model`.

### Plain functions: `GradHessObjective`

```python
# Poisson regression with a log link
poisson = GradHessObjective(
    lambda y, s, ctx: (torch.exp(s) - y, torch.exp(s)),
    init_fn=lambda y, ctx: torch.log(y.mean(dim=0)),
    transform_fn=torch.exp,          # predict() returns the rate, predict_raw() the log-rate
    name="poisson",
)
```

### Autograd: `AutogradObjective`

```python
log_cosh = AutogradObjective(lambda y, p: p - y + torch.nn.functional.softplus(-2 * (p - y)) - 0.693147)

q = 0.9
pinball = AutogradObjective(
    lambda y, p: torch.maximum(q * (y - p), (q - 1) * (y - p)),
    hessian="ones",                                   # zero curvature -> first-order boosting
    init_fn=lambda y, ctx: torch.quantile(y[:, 0], q).reshape(1),
)
```

`loss_fn(y, pred)` must be **element-wise** (`[N, K]` or `[N]`). The diagonal hessian comes from
a second autograd pass over `sum(grad)`. This is exact for losses that are separable per entry;
for losses that couple the outputs of a row it gives the row sums of the hessian. Use
`hessian="ones"` for losses with zero or near-zero curvature (L1, pinball, log-cosh at large
residuals), and/or `min_hessian=...` together with `L2_reg` to keep Newton leaves bounded.

---

## 3. Dataset-level losses: `DatasetLossObjective`

Use this for losses that are **not a sum over rows**, such as Pearson correlation over the whole
dataset, per-era correlations, or a mix like `-corr + λ·MSE`.

```python
def loss_fn(y, pred, ctx) -> scalar tensor
```

Write the loss as a weighted **average** over the observed entries. The helpers in
`warpgbm.losses` do this and accept `weight`, `mask` and an optional `group` (e.g. eras).

### Example: multi-output Pearson correlation over the full dataset

```python
from warpgbm import WarpGBM, DatasetLossObjective, losses

def pearson_mse(y, pred, ctx):
    # y, pred: [N, K]; ctx.mask hides NaN targets per output; ctx.weight may be 2-D
    corr = losses.pearson_corr(pred, y, weight=ctx.weight, mask=ctx.mask)      # [K], full dataset
    mse = losses.weighted_mse(pred, y, weight=ctx.weight, mask=ctx.mask)       # keeps the scale fixed
    return -corr.mean() + 0.1 * mse

model = WarpGBM(objective=DatasetLossObjective(pearson_mse), n_estimators=300, max_depth=4, num_bins=32)
model.fit(
    X, Y,                         # Y: (n_samples, K), may contain NaN
    sample_weight=W,              # (n_samples,) or (n_samples, K)
    era_id=eras,                  # optional: directional era splitting during tree growth
)
P = model.predict(X_test)         # (n_samples, K)
```

* **Full dataset vs per era.** `pearson_corr(..., group=None)` correlates over all rows, one value
  per output. With `group=ctx.era_index` it averages per-era correlations, weighted by era weight
  (`group_weighting="equal"` gives every era the same weight, like Numerai's mean per-era corr).
  `ctx.era_index` comes from `fit(..., era_id=...)`; without `era_id` it is all zeros and a
  per-era loss silently becomes the pooled one.
* **Eras for the loss vs eras for the trees.** Passing `era_id` also turns on directional era
  splitting (DES) in the split finder, which is applied to whatever gradients your loss produces.
  Pass `WarpGBM(era_splitting=False)` to give the eras to the loss only (pooled split finding).
* **Multi-output.** `pearson_corr` handles each output column separately with its own mask and
  weights, so outputs with different scales, offsets and missing patterns are fine. With the
  default `output_strategy="vector"` all outputs share one tree per round (vector leaves). Pass
  `output_strategy="per_output"` for one tree per output.

### How the gradients are scaled

The gradient of the averaged loss is multiplied by the total effective weight
`ctx.weight.sum()`. This brings it to the per-sample scale of the engine, so `L2_reg`,
`min_sum_hessian` and `learning_rate` keep their usual meaning. For example,
`0.5 * losses.weighted_mse` gives exactly the gradients of the built-in squared error. Weights
are not applied a second time (`apply_weights = False`).

### Hessian options

| `hessian=` | meaning | when |
|---|---|---|
| `"auto"` (default) | `h = c_k · ctx.weight`, where the curvature `c_k` of output k is measured along the gradient direction with one finite-difference Hessian-vector product per round. Needs first derivatives only. Exact for an MSE term | correlations and other scale-invariant losses, whose curvature grows like 1/std(pred)² |
| `"ones"` | `h = ctx.weight` | first-order boosting; the learning rate alone sets the step size |
| `"exact"` | autograd row-sum trick | separable parts are exact; shift-invariant parts (correlations) contribute 0 |
| callable `fn(y, pred, ctx)` | your own diagonal hessian of the averaged loss | |

In our experiments (multi-output Pearson + MSE, with NaN targets, 2-D weights, with and without
eras), `"auto"` matched the out-of-sample correlation of a plain MSE model. `"ones"` and
`"exact"` were noticeably worse at the same learning rate. `"auto"` uses the absolute value of the
measured curvature, so a negative curvature (e.g. while a correlation is still negative) gives a
descent step instead of flipping its sign.

### Warm-up: why correlation losses don't start from a constant prediction

A correlation is undefined for constant predictions, and its gradient grows like 1/std(pred).
Boosting starts from a constant score, so `DatasetLossObjective` uses squared-error gradients for
the first `warmup_rounds` rounds (default **10**, read from `ctx.iteration`) before switching to
your loss. Without the warm-up, the first trees take huge steps and the predictions blow up
(std ~1e6 in our tests). Set `warmup_rounds=0` for losses that are fine at constant predictions.
The warm-up requires T == K. As an alternative, you can warm up explicitly and switch objectives
(see §5).

`warmup_target` chooses what the warm-up regresses on:

| `warmup_target=` | warm-up regresses on | use for |
|---|---|---|
| `"raw"` (default) | the targets | losses with an MSE term on the raw targets |
| `"standardized"` | the weighted, standardized targets per output (plus the current mean prediction) | pooled correlation losses, targets with very different scales |
| `"era_standardized"` | the targets standardized within each era | per-era correlation losses |

For pure correlation losses the standardized warm-ups were clearly better than `"raw"`: a raw
warm-up spends its rounds on era offsets and on the largest-scale output, which a correlation
does not reward. `losses.standardize(x, weight, mask, group)` is available for your own losses.

The default `init_fn` is the weighted mean of the targets per output.

### λ and the prediction scale

Correlation is scale-invariant. The MSE term pins the scale: at the optimum the scale is the
least-squares scale for any λ > 0. During training, a small λ lets the correlation curvature
dominate, so the scale grows slowly. If you care about calibrated magnitudes, use a larger λ or
more trees.

### Correlation losses: math, assumptions and what to expect

Notation: weights normalized to mean 1, `σ_p` the weighted std of the prediction, `z^y`, `z^p` the
weighted standardized target and prediction, `ρ` their correlation (all per output, and per era
for a per-era loss).

* **Gradient.** For `L = -ρ`, after the rescaling above, `g_i = -(w_i / σ_p) (z^y_i - ρ z^p_i)`.
  It sums to zero and is orthogonal to `p`: the trees never learn a shift or a rescaling of the
  current prediction, only new directions. For a per-era loss this holds within every era, so
  era-level offsets and scales have no effect on training.
* **Curvature.** The Hessian is zero along the shift and scale directions and is `ρ / σ_p² · w`
  on the directions orthogonal to them, which is what `hessian="auto"` measures. The Newton step
  `Σ(-g) / Σh` along a tree's direction matches the exact line search for the correlation, so
  `learning_rate` keeps its usual meaning. Because `g ∝ 1/σ_p` and `h ∝ 1/σ_p²`, each output's
  split gain `G²/H` does not depend on that output's scale, so with multiple outputs every target
  gets an equal say in the shared trees. Under multi-output MSE the largest-scale target dominates.
* **Assumptions.** Predictions must not be constant (hence the warm-up). The diagonal hessian
  `c_k · w` is an approximation: it is exact for directions orthogonal to `1` and `p`, and trees
  are not restricted to those directions.
* **Relation to MSE.** Maximizing the pooled correlation is equivalent to minimizing MSE after an
  affine recalibration of the predictions. So **a correlation loss is not expected to beat MSE on
  suitably standardized targets** (standardized per output, and per era for a per-era metric).
  Its advantage is that it does that standardization implicitly and exactly, for the current
  predictions, including NaN targets and 2-D weights. MSE on standardized targets also keeps
  re-fitting the shrunken part of the current prediction, which acts as a mild regularizer.
  The correlation gradient projects that direction out, so correlation losses tend to peak
  earlier and need early stopping on a validation set.

Results of an untuned study (synthetic data, 240 eras, train/validation/test split by era,
checkpoint chosen on validation, 2 seeds, 400 trees, depth 4, learning rate 0.05; metric: per-era
corr where eras matter, otherwise pooled; mean corr over outputs):

| scenario | MSE | MSE, standardized targets | −corr (standardized warm-up) | −corr + 0.5·MSE(centred pred, standardized y) |
|---|---|---|---|---|
| single output, homoscedastic | 0.331 | 0.331 | 0.322 | 0.328 |
| single output, large era offsets | 0.117 | 0.321 | 0.320 | 0.320 |
| 3 outputs, scales 10 / 1 / 0.1 | 0.286 | 0.298 | 0.289 | 0.292 |
| 3 outputs, era offsets + different scales | 0.163 | 0.293 | 0.288 | 0.290 |

In short, the correlation loss beats MSE by a wide margin whenever offsets or scales differ
between eras or outputs, and it ties (within ~0.01) with MSE on targets standardized the same
way. DES works with correlation gradients like with any other gradients. On data without
era-unstable features it costs a little in-sample fit, as it does for MSE.

### Cost

A dataset objective with `"auto"` evaluates `loss_fn` and its gradient twice per round, which is
cheap for correlation-type losses (a few reductions on the GPU).

---

## 4. Evaluation metrics

`fit(..., eval_metric=...)` accepts:

* built-in strings: `"mse"`, `"corr"`, `"rmsle"` (single output), `"logloss"`, `"accuracy"`
  (classification), and `"loss"` (the objective's own `loss`; works for scalar dataset losses);
* `None` (default): the objective's `default_metric`. That is `"mse"` for regression,
  `"logloss"` for classification, and `"loss"` for custom objectives that implement `loss`;
* a **callable**, `metric(y_true, pred, weight, mask) -> float`, where **lower is better** (early
  stopping minimizes it). `pred` is the transformed prediction (what `predict` returns). Inputs
  are 1-D for single-output regression and 2-D otherwise. If the callable takes an `era`
  argument, it also receives the eval-set era index. Pass the eval eras with
  `fit(..., era_id_eval=...)`:

```python
def one_minus_per_era_corr(y_true, pred, weight, mask, era):
    return 1.0 - float(losses.pearson_corr(pred, y_true, weight=weight, mask=mask, group=era).mean())

model.fit(X, Y, era_id=eras, X_eval=Xv, y_eval=Yv, era_id_eval=eras_v,
          eval_every_n_trees=10, early_stopping_rounds=5, eval_metric=one_minus_per_era_corr)
```

---

## 5. Warm starts, objective switching, persistence

* `warm_start=True` continues from the existing trees (training scores are rebuilt from the
  forest; the bin edges of the first fit are reused).
* You may **switch the objective** between warm-start fits, as long as the score layout is
  compatible (same `output_strategy`, same classifier/regressor type, same K). For example,
  pre-train with MSE and fine-tune with a custom loss:

  ```python
  model = WarpGBM(objective="multi_regression", n_estimators=50, warm_start=True)
  model.fit(X, Y, era_id=eras)
  model.objective = DatasetLossObjective(pearson_mse, warmup_rounds=0)
  model.n_estimators = 300
  model.fit(X, Y, era_id=eras)
  ```
* `save_model` pickles the objective instance. Define custom classes and loss functions at module
  level (no lambdas) if you want to `load_model` them in another process.

---

## 6. Interaction with other features

| feature | behaviour with custom objectives |
|---|---|
| sample weights | `[N]` or `[N, T]`, normalized to mean 1; `ctx.weight` is 0 for missing targets |
| `min_child_weight` | minimum **sum of row weights** in a child (≈ row count when unweighted); independent of the hessian scale |
| `min_sum_hessian` | per output: an output whose child hessian sum is below this does not contribute to the split gain |
| eras / DES | per-era histograms of your gradients; the gain is averaged over eras and split directions must agree across eras. `era_splitting=False` keeps the eras in `ctx.era_index` but splits on pooled histograms |
| monotonic constraints | value bounds are propagated to the children and leaves are clamped, for every output (vector leaves) or per class tree |
| NaN features | own bin with a learned default direction, independent of the objective |
| determinism | histograms are accumulated in fixed point and predictions sum the trees in a fixed order, so with `random_state` set, results are bit-reproducible run to run |
