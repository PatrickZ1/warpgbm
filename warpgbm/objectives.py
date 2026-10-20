# warpgbm/objectives.py
"""
Objective (loss function) interface for WarpGBM.

Every objective - built-in or user-defined - works on GPU tensors and follows the
XGBoost / LightGBM convention:

    grad = dL / d score        hess = d^2 L / d score^2

where ``score`` is the raw (untransformed) model output of shape [N, K].

The training engine takes care of sample weights, missing (NaN) targets, eras,
monotonic constraints, warm starts and evaluation. A custom objective therefore only
has to implement the element-wise math. See docs/CUSTOM_OBJECTIVES.md for a guide.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from warpgbm.metrics import softmax

VECTOR = "vector"
PER_OUTPUT = "per_output"
_STRATEGIES = (VECTOR, PER_OUTPUT)


@dataclass
class ObjectiveContext:
    """
    Side information handed to every objective call.

    Attributes:
        weight:      [N, K] float32, effective weight per score entry
                     (normalized sample weight, already zero where the target is missing).
        mask:        [N, K] bool, True where the score entry has an observed target.
        target_mask: [N, T] bool, True where the target entry is observed (T = target columns).
        era_index:   [N] int32, era index (0..E-1) of every row.
        num_outputs: K, the number of raw score columns.
        iteration:   index of the current boosting round (0 during init_score / evaluation).
    """

    weight: Tensor
    mask: Tensor
    target_mask: Tensor
    era_index: Tensor
    num_outputs: int
    iteration: int = 0

    @property
    def row_weight(self) -> Tensor:
        """[N] mean effective weight per row."""
        return self.weight.mean(dim=1)


class Objective:
    """
    Base class for all objectives.

    Subclasses must implement :meth:`grad_hess`. Everything else has sensible defaults.

    Class attributes:
        name:            identifier used in log messages.
        output_strategy: "vector"     -> one tree per round with K-dimensional leaves
                                         (shared split structure across outputs);
                         "per_output" -> K trees per round, one per score column.
        apply_weights:   if True (default), the engine multiplies grad/hess by ``ctx.weight``
                         and zeroes entries with missing targets. Set to False for losses that
                         are not separable per row (e.g. correlation), which then must use
                         ``ctx.weight`` / ``ctx.mask`` themselves. Masked entries are always
                         zeroed by the engine.
        is_classifier:   True for objectives whose targets are class labels.
        default_metric:  eval metric used when ``fit(eval_metric=None)``; "loss" uses :meth:`loss`.
    """

    name = "custom"
    output_strategy = VECTOR
    apply_weights = True
    is_classifier = False
    default_metric = "loss"

    # ─── Targets ───
    def prepare_targets(self, y: np.ndarray) -> np.ndarray:
        """Convert user targets to a float32 array of shape [N, T]. NaN marks a missing target."""
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def num_outputs(self, y: np.ndarray) -> int:
        """Number of raw score columns K for targets ``y`` of shape [N, T]."""
        return y.shape[1]

    def score_mask(self, target_mask: Tensor, num_outputs: int) -> Tensor:
        """Map the target mask [N, T] to a score mask [N, K]."""
        if target_mask.shape[1] == num_outputs:
            return target_mask
        row_valid = target_mask.all(dim=1, keepdim=True)
        return row_valid.expand(-1, num_outputs)

    # ─── Optimization ───
    def init_score(self, y: Tensor, ctx: ObjectiveContext) -> Tensor:
        """
        Constant initial score [K]. Default: one weighted Newton step from score 0.
        """
        pred0 = torch.zeros(
            (y.shape[0], ctx.num_outputs), device=y.device, dtype=torch.float32
        )
        g, h = compute_grad_hess(self, y, pred0, ctx)
        g_sum = g.sum(dim=0)
        h_sum = h.sum(dim=0)
        return torch.where(
            h_sum > 0, -g_sum / torch.clamp(h_sum, min=1e-12), torch.zeros_like(g_sum)
        )

    def grad_hess(
        self, y: Tensor, pred: Tensor, ctx: ObjectiveContext
    ) -> Tuple[Tensor, Tensor]:
        """
        Gradient and hessian w.r.t. the raw score.

        Args:
            y:    [N, T] float32 targets (missing entries are filled with 0).
            pred: [N, K] float32 raw scores.
            ctx:  ObjectiveContext.

        Returns:
            (grad, hess), each [N, K] (or broadcastable to it).
        """
        raise NotImplementedError

    def loss(self, y: Tensor, pred: Tensor, ctx: ObjectiveContext) -> Tensor:
        """Element-wise loss [N, K] (or per-row [N]). Optional; used for logging/eval."""
        raise NotImplementedError

    def transform(self, raw: Tensor) -> Tensor:
        """Map raw scores to predictions (e.g. sigmoid / softmax). Default: identity."""
        return raw

    def has_loss(self) -> bool:
        return type(self).loss is not Objective.loss

    def __repr__(self):
        return f"{type(self).__name__}()"


# ─── Engine helpers ───────────────────────────────────────────────────────────


def compute_grad_hess(
    objective: Objective, y: Tensor, pred: Tensor, ctx: ObjectiveContext
) -> Tuple[Tensor, Tensor]:
    """Call ``objective.grad_hess`` and apply weights/masking. Returns contiguous [N, K] tensors."""
    g, h = objective.grad_hess(y, pred, ctx)
    shape = pred.shape
    g = torch.as_tensor(g, device=pred.device, dtype=torch.float32)
    h = torch.as_tensor(h, device=pred.device, dtype=torch.float32)
    try:
        g = g.expand(shape) if g.shape != shape else g
        h = h.expand(shape) if h.shape != shape else h
    except RuntimeError as e:
        raise ValueError(
            f"Objective {objective!r} returned grad/hess of shape {tuple(g.shape)}/{tuple(h.shape)}, "
            f"expected {tuple(shape)}."
        ) from e

    if objective.apply_weights:
        g = g * ctx.weight
        h = h * ctx.weight

    # torch.where (not multiplication) so NaN/inf at masked positions cannot leak through
    zero = torch.zeros((), device=pred.device, dtype=torch.float32)
    g = torch.where(ctx.mask, g, zero).contiguous()
    h = torch.where(ctx.mask, h, zero).contiguous()

    if not bool(torch.isfinite(g).all() & torch.isfinite(h).all()):
        raise FloatingPointError(
            f"Objective {objective!r} produced non-finite gradients or hessians."
        )
    return g, h


def weighted_loss(
    objective: Objective, y: Tensor, pred: Tensor, ctx: ObjectiveContext
) -> float:
    """Weighted mean of ``objective.loss`` over observed entries (a scalar loss is returned as is)."""
    loss = torch.as_tensor(objective.loss(y, pred, ctx), device=pred.device)
    if loss.ndim == 0:
        return float(loss)
    if loss.ndim == 1:
        w = ctx.row_weight
        valid = ctx.mask.any(dim=1)
    else:
        w = ctx.weight
        valid = ctx.mask
        if loss.shape != w.shape:
            loss = loss.expand(w.shape)
    zero = torch.zeros((), device=pred.device, dtype=torch.float32)
    loss = torch.where(valid, loss.to(torch.float32), zero)
    w = torch.where(valid, w, zero)
    denom = w.sum()
    if denom <= 0:
        return float("nan")
    return float((loss * w).sum() / denom)


# ─── Built-in objectives ─────────────────────────────────────────────────────


class SquaredError(Objective):
    """L(y, s) = 0.5 * (s - y)^2. Used for 'regression' and 'multi_regression'."""

    name = "regression"
    output_strategy = VECTOR
    default_metric = "mse"

    def init_score(self, y, ctx):
        w = ctx.weight
        w_sum = w.sum(dim=0)
        return torch.where(
            w_sum > 0,
            (y * w).sum(dim=0) / torch.clamp(w_sum, min=1e-12),
            torch.zeros_like(w_sum),
        )

    def grad_hess(self, y, pred, ctx):
        return pred - y, torch.ones_like(pred)

    def loss(self, y, pred, ctx):
        return (pred - y) ** 2


class Softmax(Objective):
    """
    Multiclass cross-entropy with softmax link. Used for 'multiclass' and 'binary'.
    Targets are encoded labels 0..K-1 (one column); scores have K columns.
    Uses one tree per class per round and the diagonal hessian p * (1 - p).
    """

    name = "multiclass"
    output_strategy = PER_OUTPUT
    is_classifier = True
    default_metric = "logloss"

    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = num_classes

    def num_outputs(self, y):
        if self.num_classes is None:
            return int(np.nanmax(y)) + 1
        return self.num_classes

    def _onehot(self, y, k):
        return torch.nn.functional.one_hot(y[:, 0].long(), k).to(torch.float32)

    def init_score(self, y, ctx):
        k = ctx.num_outputs
        w = ctx.row_weight
        counts = (self._onehot(y, k) * w[:, None]).sum(dim=0)
        priors = counts / torch.clamp(w.sum(), min=1e-12)
        return torch.where(
            priors > 0, torch.log(priors + 1e-10), torch.zeros_like(priors)
        )

    def grad_hess(self, y, pred, ctx):
        probs = softmax(pred, dim=1)
        grad = probs - self._onehot(y, pred.shape[1])
        hess = torch.clamp(probs * (1.0 - probs), min=1e-6)
        return grad, hess

    def loss(self, y, pred, ctx):
        probs = torch.clamp(softmax(pred, dim=1), 1e-15, 1 - 1e-15)
        return -torch.log(probs.gather(1, y.long()))[:, 0]

    def transform(self, raw):
        return softmax(raw, dim=1)


# ─── Convenience wrappers for custom losses ──────────────────────────────────


class GradHessObjective(Objective):
    """
    Wrap plain functions into an Objective.

    Args:
        grad_hess_fn: fn(y, pred, ctx) -> (grad, hess) on GPU tensors.
        loss_fn:      optional fn(y, pred, ctx) -> element-wise loss (for logging / eval).
        init_fn:      optional fn(y, ctx) -> [K] initial score. Default: one Newton step from 0.
        transform_fn: optional fn(raw) -> prediction.
        output_strategy, apply_weights, name: see :class:`Objective`.
    """

    def __init__(
        self,
        grad_hess_fn: Callable,
        loss_fn: Optional[Callable] = None,
        init_fn: Optional[Callable] = None,
        transform_fn: Optional[Callable] = None,
        output_strategy: str = VECTOR,
        apply_weights: bool = True,
        name: str = "custom",
    ):
        if output_strategy not in _STRATEGIES:
            raise ValueError(f"output_strategy must be one of {_STRATEGIES}.")
        self.grad_hess_fn = grad_hess_fn
        self.loss_fn = loss_fn
        self.init_fn = init_fn
        self.transform_fn = transform_fn
        self.output_strategy = output_strategy
        self.apply_weights = apply_weights
        self.name = name

    def grad_hess(self, y, pred, ctx):
        return self.grad_hess_fn(y, pred, ctx)

    def loss(self, y, pred, ctx):
        if self.loss_fn is None:
            raise NotImplementedError
        return self.loss_fn(y, pred, ctx)

    def has_loss(self):
        return self.loss_fn is not None

    def init_score(self, y, ctx):
        if self.init_fn is not None:
            return torch.as_tensor(
                self.init_fn(y, ctx), device=y.device, dtype=torch.float32
            ).reshape(-1)
        return super().init_score(y, ctx)

    def transform(self, raw):
        return raw if self.transform_fn is None else self.transform_fn(raw)

    def __repr__(self):
        return f"GradHessObjective(name={self.name!r})"


class AutogradObjective(Objective):
    """
    Build an objective from a differentiable torch loss; derivatives come from autograd.

    Args:
        loss_fn:  fn(y, pred) -> element-wise loss of shape [N, K] (or [N]).
                  ``y`` is [N, T] with missing entries filled with 0, ``pred`` is [N, K].
        hessian:  "exact" -> diagonal hessian via a second autograd pass (exact for losses
                             that are separable per score entry; for losses that couple the
                             outputs of a row it yields the row-sums of the hessian),
                  "ones"  -> constant hessian of 1 (first-order boosting; useful for losses
                             with zero curvature such as L1 / pinball).
        min_hessian:     hessians are clamped to at least this value.
        init_fn, transform_fn, output_strategy, name: see :class:`GradHessObjective`.
    """

    def __init__(
        self,
        loss_fn: Callable,
        hessian: str = "exact",
        min_hessian: float = 0.0,
        init_fn: Optional[Callable] = None,
        transform_fn: Optional[Callable] = None,
        output_strategy: str = VECTOR,
        name: str = "autograd",
    ):
        if hessian not in ("exact", "ones"):
            raise ValueError("hessian must be 'exact' or 'ones'.")
        if output_strategy not in _STRATEGIES:
            raise ValueError(f"output_strategy must be one of {_STRATEGIES}.")
        self.loss_fn = loss_fn
        self.hessian = hessian
        self.min_hessian = float(min_hessian)
        self.init_fn = init_fn
        self.transform_fn = transform_fn
        self.output_strategy = output_strategy
        self.name = name

    def grad_hess(self, y, pred, ctx):
        with torch.enable_grad():
            p = pred.detach().clone().requires_grad_(True)
            loss = self.loss_fn(y, p)
            need_hess = self.hessian == "exact"
            (g,) = torch.autograd.grad(loss.sum(), p, create_graph=need_hess)
            if need_hess and g.requires_grad:
                (h,) = torch.autograd.grad(g.sum(), p, allow_unused=True)
                if h is None:
                    h = torch.zeros_like(p)
            elif need_hess:
                h = torch.zeros_like(p)
            else:
                h = torch.ones_like(p)
        h = h.detach()
        if self.min_hessian > 0:
            h = torch.clamp(h, min=self.min_hessian)
        return g.detach(), h

    def loss(self, y, pred, ctx):
        return self.loss_fn(y, pred)

    def init_score(self, y, ctx):
        if self.init_fn is not None:
            return torch.as_tensor(
                self.init_fn(y, ctx), device=y.device, dtype=torch.float32
            ).reshape(-1)
        return super().init_score(y, ctx)

    def transform(self, raw):
        return raw if self.transform_fn is None else self.transform_fn(raw)

    def __repr__(self):
        return f"AutogradObjective(name={self.name!r}, hessian={self.hessian!r})"


class DatasetLossObjective(Objective):
    """
    Objective from a differentiable **dataset-level** loss such as a (per-era) correlation,
    optionally combined with separable terms (e.g. ``-pearson + lambda * mse``).

    Args:
        loss_fn: fn(y, pred, ctx) -> scalar tensor. Write it as a weighted *average* over the
                 observed entries (use ``ctx.weight``, ``ctx.mask`` and ``ctx.era_index``; the
                 helpers in :mod:`warpgbm.losses` do this). y is [N, T] with missing entries
                 filled with 0, pred is [N, K].
        hessian: how to obtain the diagonal hessian used for Newton leaf values:
                 "auto"  -> (default) h = c_k * ctx.weight, where the curvature c_k of output k
                            is measured along the gradient direction with one finite-difference
                            Hessian-vector product per round. Needs first derivatives only and
                            adapts to scale-invariant losses (correlations), whose curvature
                            grows like 1 / std(pred)^2; exact (2 * lambda) for an MSE term.
                 "ones"  -> unit curvature per observed entry, i.e. h = ctx.weight. The learning
                            rate alone then sets the step size.
                 "exact" -> autograd row-sum trick (exact for separable parts such as an MSE
                            term; zero for shift-invariant parts such as correlations; requires
                            twice-differentiable losses, i.e. not soft_rank).
                 callable fn(y, pred, ctx) -> [N, K] hessian on the averaged loss scale.
        min_hessian: hessians are clamped to at least this value (after rescaling).
        init_fn: fn(y, ctx) -> [K] initial score. Default: weighted mean of the targets
                 (if T == K), else 0.
        warmup_rounds: number of initial boosting rounds (``ctx.iteration < warmup_rounds``)
                 that use squared-error gradients instead of ``loss_fn``. Scale-invariant losses
                 such as correlations are undefined for constant predictions (their gradient
                 grows like 1 / std(pred)), so they need non-degenerate predictions to start
                 from. Requires targets with one column per score (T == K); set to 0 to disable.
        transform_fn, output_strategy, name: see :class:`GradHessObjective`.

    Scaling: gradients (and "exact"/callable hessians) of the averaged loss are multiplied by
    the total effective weight ``ctx.weight.sum()``. ``0.5 * losses.weighted_mse`` written this
    way thus gets exactly the gradients of the per-sample built-in, and ``L2_reg``,
    ``min_sum_hessian`` and ``learning_rate`` keep their usual meaning. Weights are *not* applied again by the engine
    (``apply_weights = False``); masked entries are still zeroed.
    """

    apply_weights = False

    def __init__(
        self,
        loss_fn: Callable,
        hessian="auto",
        min_hessian: float = 0.0,
        init_fn: Optional[Callable] = None,
        transform_fn: Optional[Callable] = None,
        output_strategy: str = VECTOR,
        name: str = "dataset_loss",
        curvature_floor: float = 1e-6,
        warmup_rounds: int = 10,
    ):
        if not callable(hessian) and hessian not in ("auto", "ones", "exact"):
            raise ValueError("hessian must be 'auto', 'ones', 'exact' or a callable.")
        if output_strategy not in _STRATEGIES:
            raise ValueError(f"output_strategy must be one of {_STRATEGIES}.")
        self.loss_fn = loss_fn
        self.hessian = hessian
        self.min_hessian = float(min_hessian)
        self.init_fn = init_fn
        self.transform_fn = transform_fn
        self.output_strategy = output_strategy
        self.name = name
        self.curvature_floor = float(curvature_floor)
        self.warmup_rounds = int(warmup_rounds)

    def _loss_grad(self, y, pred, ctx, create_graph=False):
        with torch.enable_grad():
            p = pred.detach().clone().requires_grad_(True)
            loss = self.loss_fn(y, p, ctx)
            if loss.ndim != 0:
                raise ValueError(
                    f"DatasetLossObjective loss_fn must return a scalar, got shape {tuple(loss.shape)}."
                )
            (g,) = torch.autograd.grad(loss, p, create_graph=create_graph)
        return p, g

    def _auto_curvature(self, y, pred, g, ctx, scale):
        """Per-output curvature along the gradient direction (finite-difference HVP)."""
        mask = ctx.mask.to(pred.dtype)
        v = g.detach() * mask
        v_rms = torch.sqrt((v * v).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0))
        v = v / v_rms.clamp(min=1e-30)
        mean_p = (pred * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)
        spread = torch.sqrt((((pred - mean_p) * mask) ** 2).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0))
        delta = 1e-3 * torch.clamp(spread, min=1e-3)
        _, g2 = self._loss_grad(y, pred + delta * v, ctx)
        hv = (g2.detach() - g.detach()) / delta
        numer = (v * hv).sum(dim=0) * scale
        denom = (ctx.weight * v * v).sum(dim=0).clamp(min=1e-30)
        return torch.clamp(numer / denom, min=self.curvature_floor)

    def grad_hess(self, y, pred, ctx):
        if ctx.iteration < self.warmup_rounds:
            if y.shape[1] != pred.shape[1]:
                raise ValueError(
                    "warmup_rounds > 0 needs one target column per score column; "
                    "pass warmup_rounds=0 or an init_fn with non-degenerate predictions."
                )
            # squared-error warm-up (weights are applied here since apply_weights=False)
            return (pred - y) * ctx.weight, ctx.weight

        # The loss is an average: rescale to the per-sample ("sum") scale of the engine
        scale = ctx.weight.sum()
        exact = self.hessian == "exact"
        p, g = self._loss_grad(y, pred, ctx, create_graph=exact)

        if exact:
            h = None
            if g.requires_grad:
                with torch.enable_grad():
                    (h,) = torch.autograd.grad(g.sum(), p, allow_unused=True)
            h = torch.zeros_like(pred) if h is None else h.detach() * scale
        elif self.hessian == "auto":
            h = self._auto_curvature(y, pred, g, ctx, scale) * ctx.weight
        elif self.hessian == "ones":
            h = ctx.weight
        else:
            h = torch.as_tensor(self.hessian(y, pred, ctx), device=pred.device) * scale

        g = g.detach() * scale
        if self.min_hessian > 0:
            h = torch.clamp(h, min=self.min_hessian)
        return g, h

    def loss(self, y, pred, ctx):
        return self.loss_fn(y, pred, ctx)

    def init_score(self, y, ctx):
        if self.init_fn is not None:
            return torch.as_tensor(
                self.init_fn(y, ctx), device=y.device, dtype=torch.float32
            ).reshape(-1)
        if y.shape[1] == ctx.num_outputs:
            return SquaredError().init_score(y, ctx)
        return torch.zeros(ctx.num_outputs, device=y.device)

    def transform(self, raw):
        return raw if self.transform_fn is None else self.transform_fn(raw)

    def __repr__(self):
        return f"DatasetLossObjective(name={self.name!r}, hessian={self.hessian!r})"


BUILTIN_OBJECTIVES = ("regression", "multi_regression", "multiclass", "binary")


def resolve_objective(objective) -> Objective:
    """Turn the ``objective`` hyperparameter into an Objective instance."""
    if isinstance(objective, Objective):
        return objective
    if objective in ("regression", "multi_regression"):
        obj = SquaredError()
        obj.name = objective
        return obj
    if objective in ("multiclass", "binary"):
        obj = Softmax()
        obj.name = objective
        return obj
    if callable(objective):
        return GradHessObjective(objective)
    raise ValueError(
        "objective must be 'regression', 'multi_regression', 'binary', 'multiclass', "
        f"an Objective instance or a callable (y, pred, ctx) -> (grad, hess); got {objective!r}."
    )
