# warpgbm/objectives.py
"""
Objective (loss function) interface for WarpGBM.

An objective works on GPU tensors and follows the XGBoost / LightGBM convention

    grad = dL / d score        hess = d^2 L / d score^2

where ``score`` is the raw (untransformed) model output of shape [N, K].

The simplest custom objective is a differentiable per-sample loss ``loss(y, pred)``;
gradients and the diagonal hessian then come from PyTorch autograd. The training engine
applies sample weights, masks missing (NaN) targets and handles eras, monotonic constraints,
warm starts and evaluation. See docs/CUSTOM_OBJECTIVES.md for a guide.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from warpgbm.metrics import softmax

VECTOR = "vector"
PER_OUTPUT = "per_output"
_STRATEGIES = (VECTOR, PER_OUTPUT)
_HESSIANS = ("autograd", "ones")


class Objective:
    """
    Base class for all objectives.

    Implement :meth:`loss` (gradient and hessian then come from autograd) and/or
    :meth:`grad_hess` (explicit derivatives). Everything else has sensible defaults.

    Losses are per sample: the loss of a row may only depend on that row's targets and scores.
    The engine multiplies grad/hess by the sample weights and zeroes entries whose target is
    missing, so neither ``loss`` nor ``grad_hess`` has to deal with weights or NaNs.

    Class attributes:
        name:            identifier used in log messages.
        output_strategy: "vector"     -> one tree per round with K-dimensional leaves
                                         (shared split structure across outputs);
                         "per_output" -> K trees per round, one per score column.
        hessian:         used by the autograd ``grad_hess``: "autograd" (exact diagonal hessian)
                         or "ones" (constant 1, i.e. first-order boosting; for losses with zero
                         curvature such as L1 / pinball).
        min_hessian:     autograd hessians are clamped to at least this value.
        is_classifier:   True for objectives whose targets are class labels.
        default_metric:  eval metric used when ``fit(eval_metric=None)``; "loss" uses :meth:`loss`.
    """

    name = "custom"
    output_strategy = VECTOR
    hessian = "autograd"
    min_hessian = 0.0
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

    # ─── Loss and derivatives ───
    def loss(self, y: Tensor, pred: Tensor) -> Tensor:
        """
        Per-sample loss, differentiable with torch.

        Args:
            y:    [N, T] float32 targets (missing entries are filled with 0).
            pred: [N, K] float32 raw scores.

        Returns:
            [N, K] loss per entry, or [N] loss per row (for losses that couple the outputs of a row).
        """
        raise NotImplementedError

    def grad_hess(self, y: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Gradient and diagonal hessian w.r.t. the raw scores, each [N, K] (or broadcastable).
        Default: autograd of :meth:`loss`.
        """
        if not self.has_loss():
            raise NotImplementedError(
                f"{type(self).__name__} must implement loss() or grad_hess()."
            )
        return autograd_grad_hess(
            self.loss, y, pred, hessian=self.hessian, min_hessian=self.min_hessian
        )

    def init_score(self, y: Tensor, weight: Tensor) -> Tensor:
        """
        Constant initial score [K]. Default: one weighted Newton step from score 0.

        ``weight`` is [N, K], the normalized sample weights (0 where the target is missing).
        """
        pred0 = torch.zeros(weight.shape, device=y.device, dtype=torch.float32)
        g, h = compute_grad_hess(self, y, pred0, weight, weight > 0)
        g_sum = g.sum(dim=0)
        h_sum = h.sum(dim=0)
        return torch.where(
            h_sum > 0, -g_sum / torch.clamp(h_sum, min=1e-12), torch.zeros_like(g_sum)
        )

    def transform(self, raw: Tensor) -> Tensor:
        """Map raw scores to predictions (e.g. sigmoid / softmax). Default: identity."""
        return raw

    def has_loss(self) -> bool:
        return type(self).loss is not Objective.loss

    def _hessian_is_one(self) -> bool:
        """
        True if :meth:`grad_hess` always returns a hessian of 1 (before weighting). The engine then
        reads the hessian sums from the row-weight histogram instead of accumulating them
        (a performance hint; False is always safe).
        """
        return (
            type(self).grad_hess is Objective.grad_hess
            and self.hessian == "ones"
            and self.min_hessian <= 1.0
        )

    def __repr__(self):
        return f"{type(self).__name__}()"


class CustomObjective(Objective):
    """
    Objective from plain functions.

    Args:
        loss_fn:      fn(y, pred) -> per-sample loss ([N, K] or [N]); gradients and hessians
                      come from autograd unless ``grad_hess_fn`` is given.
        grad_hess_fn: optional fn(y, pred) -> (grad, hess) with explicit derivatives.
        hessian:      "autograd" (exact diagonal) or "ones" (constant 1); autograd path only.
        min_hessian:  autograd hessians are clamped to at least this value.
        init_fn:      optional fn(y, weight) -> [K] initial score. Default: one Newton step from 0.
        transform_fn: optional fn(raw) -> prediction.
        output_strategy: "vector" (default) or "per_output", see :class:`Objective`.
        name:         identifier used in log messages.

    At least one of ``loss_fn`` / ``grad_hess_fn`` is required. Without ``loss_fn`` the
    ``"loss"`` eval metric is unavailable.
    """

    def __init__(
        self,
        loss_fn: Optional[Callable] = None,
        grad_hess_fn: Optional[Callable] = None,
        hessian: str = "autograd",
        min_hessian: float = 0.0,
        init_fn: Optional[Callable] = None,
        transform_fn: Optional[Callable] = None,
        output_strategy: str = VECTOR,
        name: str = "custom",
    ):
        if loss_fn is None and grad_hess_fn is None:
            raise ValueError("CustomObjective needs loss_fn and/or grad_hess_fn.")
        if hessian not in _HESSIANS:
            raise ValueError(f"hessian must be one of {_HESSIANS}.")
        if output_strategy not in _STRATEGIES:
            raise ValueError(f"output_strategy must be one of {_STRATEGIES}.")
        self.loss_fn = loss_fn
        self.grad_hess_fn = grad_hess_fn
        self.hessian = hessian
        self.min_hessian = float(min_hessian)
        self.init_fn = init_fn
        self.transform_fn = transform_fn
        self.output_strategy = output_strategy
        self.name = name

    def loss(self, y, pred):
        if self.loss_fn is None:
            raise NotImplementedError
        return self.loss_fn(y, pred)

    def has_loss(self):
        return self.loss_fn is not None

    def grad_hess(self, y, pred):
        if self.grad_hess_fn is not None:
            return self.grad_hess_fn(y, pred)
        return super().grad_hess(y, pred)

    def init_score(self, y, weight):
        if self.init_fn is not None:
            return torch.as_tensor(
                self.init_fn(y, weight), device=y.device, dtype=torch.float32
            ).reshape(-1)
        return super().init_score(y, weight)

    def transform(self, raw):
        return raw if self.transform_fn is None else self.transform_fn(raw)

    def _hessian_is_one(self):
        return self.grad_hess_fn is None and self.hessian == "ones" and self.min_hessian <= 1.0

    def __repr__(self):
        return f"CustomObjective(name={self.name!r})"


# ─── Engine helpers ───────────────────────────────────────────────────────────


def autograd_grad_hess(
    loss_fn: Callable,
    y: Tensor,
    pred: Tensor,
    hessian: str = "autograd",
    min_hessian: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Gradient and diagonal hessian of a per-sample loss via autograd.

    Rows are independent, so the diagonal is exact: one backward pass per score column
    (a single one for K = 1) gives d^2 L / d score_k^2 for every row.
    """
    n, k = pred.shape
    with torch.enable_grad():
        p = pred.detach().clone().requires_grad_(True)
        loss = loss_fn(y, p)
        if not torch.is_tensor(loss) or loss.shape not in ((n,), (n, k)):
            shape = tuple(loss.shape) if torch.is_tensor(loss) else type(loss).__name__
            raise ValueError(
                f"loss must return a per-sample tensor of shape ({n},) or ({n}, {k}), got {shape}."
            )
        need_hess = hessian == "autograd"
        (g,) = torch.autograd.grad(loss.sum(), p, create_graph=need_hess)
        if not need_hess:
            h = torch.ones_like(p)
        elif not g.requires_grad:  # loss is linear in pred
            h = torch.zeros_like(p)
        else:
            cols = []
            for j in range(k):
                (hj,) = torch.autograd.grad(
                    g[:, j].sum(), p, retain_graph=j < k - 1, allow_unused=True
                )
                cols.append(torch.zeros(n, device=p.device) if hj is None else hj[:, j])
            h = torch.stack(cols, dim=1)
    h = h.detach()
    if min_hessian > 0:
        h = torch.clamp(h, min=min_hessian)
    return g.detach(), h


def compute_grad_hess(
    objective: Objective,
    y: Tensor,
    pred: Tensor,
    weight: Tensor,
    mask: Tensor,
    check_finite: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Call ``objective.grad_hess``, apply weights and the target mask. Returns contiguous [N, K].

    check_finite=False skips the finiteness check (a host sync); the caller must check instead.
    """
    g, h = objective.grad_hess(y, pred)
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

    # torch.where (not multiplication) so NaN/inf at masked positions cannot leak through
    zero = torch.zeros((), device=pred.device, dtype=torch.float32)
    g = torch.where(mask, g * weight, zero).contiguous()
    h = torch.where(mask, h * weight, zero).contiguous()

    if check_finite and not bool(torch.isfinite(g).all() & torch.isfinite(h).all()):
        raise FloatingPointError(
            f"Objective {objective!r} produced non-finite gradients or hessians."
        )
    return g, h


def weighted_loss(
    objective: Objective, y: Tensor, pred: Tensor, weight: Tensor, mask: Tensor
) -> float:
    """Weighted mean of ``objective.loss`` over the observed entries."""
    loss = torch.as_tensor(objective.loss(y, pred), device=pred.device)
    if loss.ndim == 1:
        w = weight.mean(dim=1)
        valid = mask.any(dim=1)
    else:
        w = weight
        valid = mask
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

    def init_score(self, y, weight):
        w_sum = weight.sum(dim=0)
        return torch.where(
            w_sum > 0,
            (y * weight).sum(dim=0) / torch.clamp(w_sum, min=1e-12),
            torch.zeros_like(w_sum),
        )

    def grad_hess(self, y, pred):
        return pred - y, torch.ones_like(pred)

    def _hessian_is_one(self):
        return type(self).grad_hess is SquaredError.grad_hess

    def loss(self, y, pred):
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

    def init_score(self, y, weight):
        k = weight.shape[1]
        w = weight.mean(dim=1)
        counts = (self._onehot(y, k) * w[:, None]).sum(dim=0)
        priors = counts / torch.clamp(w.sum(), min=1e-12)
        return torch.where(
            priors > 0, torch.log(priors + 1e-10), torch.zeros_like(priors)
        )

    def grad_hess(self, y, pred):
        probs = softmax(pred, dim=1)
        grad = probs - self._onehot(y, pred.shape[1])
        hess = torch.clamp(probs * (1.0 - probs), min=1e-6)
        return grad, hess

    def loss(self, y, pred):
        probs = torch.clamp(softmax(pred, dim=1), 1e-15, 1 - 1e-15)
        return -torch.log(probs.gather(1, y.long()))[:, 0]

    def transform(self, raw):
        return softmax(raw, dim=1)


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
        return CustomObjective(loss_fn=objective)
    raise ValueError(
        "objective must be 'regression', 'multi_regression', 'binary', 'multiclass', "
        f"an Objective instance or a per-sample loss function (y, pred) -> loss; got {objective!r}."
    )
