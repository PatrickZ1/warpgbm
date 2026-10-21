# warpgbm/losses.py
"""
GPU building blocks for dataset-level (non-separable) losses and metrics.

All functions work on torch tensors of shape [N] or [N, K] (K = outputs, handled column-wise),
accept optional per-entry ``weight`` / ``mask`` tensors and an optional ``group`` tensor [N]
(e.g. ``ctx.era_index``). With ``group`` the statistic is computed within every group and then
averaged over groups, which is how per-era metrics such as Numerai's per-era correlation work.

Typical use inside a :class:`warpgbm.objectives.DatasetLossObjective`::

    from warpgbm import losses

    def pearson_mse(y, pred, ctx):
        corr = losses.pearson_corr(pred, y, weight=ctx.weight, mask=ctx.mask)  # [K]
        mse = losses.weighted_mse(pred, y, weight=ctx.weight, mask=ctx.mask)
        return -corr.mean() + 0.1 * mse
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def _as_2d(x: Tensor) -> Tensor:
    return x.unsqueeze(1) if x.ndim == 1 else x


def _prepare(x: Tensor, weight: Optional[Tensor], mask: Optional[Tensor]):
    x2 = _as_2d(x)
    if mask is None:
        mask2 = torch.ones_like(x2, dtype=torch.bool)
    else:
        mask2 = _as_2d(mask).expand_as(x2)
    if weight is None:
        w2 = torch.ones_like(x2)
    else:
        w2 = _as_2d(weight).to(x2.dtype).expand_as(x2)
    w2 = torch.where(mask2, w2, torch.zeros((), device=x2.device, dtype=x2.dtype))
    return x2, w2, mask2


def _group_index(group: Optional[Tensor], n: int, device) -> Tuple[Tensor, int]:
    if group is None:
        return torch.zeros(n, dtype=torch.long, device=device), 1
    g = group.long()
    return g, int(g.max().item()) + 1 if g.numel() else 1


# ─── Separable helpers ───────────────────────────────────────────────────────


def weighted_mse(pred: Tensor, target: Tensor, weight=None, mask=None) -> Tensor:
    """Weighted mean squared error over observed entries (scalar)."""
    p, w, m = _prepare(pred, weight, mask)
    t = _as_2d(target).to(p.dtype)
    err2 = torch.where(m, (p - t) ** 2, torch.zeros((), device=p.device, dtype=p.dtype))
    return (w * err2).sum() / w.sum().clamp(min=1e-12)


def standardize(
    x: Tensor,
    weight: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    group: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> Tensor:
    """
    Weighted standardization (zero mean, unit variance) per column, within each group if
    ``group`` is given. Masked entries are returned as 0. Differentiable.
    """
    x2, w, m = _prepare(x, weight, mask)
    g, num_groups = _group_index(group, x2.shape[0], x2.device)
    zero = torch.zeros((), device=x2.device, dtype=x2.dtype)
    x2 = torch.where(m, x2, zero)

    def gsum(v):
        return torch.zeros(num_groups, x2.shape[1], device=v.device, dtype=v.dtype).index_add(0, g, v)

    w_safe = gsum(w).clamp(min=1e-12)
    centered = x2 - (gsum(w * x2) / w_safe)[g]
    std = torch.sqrt(gsum(w * centered * centered) / w_safe + eps)
    out = torch.where(m, centered / std[g], zero)
    return out[:, 0] if x.ndim == 1 else out


# ─── Correlation ─────────────────────────────────────────────────────────────


def pearson_corr(
    a: Tensor,
    b: Tensor,
    weight: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    group: Optional[Tensor] = None,
    eps: float = 1e-12,
    group_weighting: str = "weight",
) -> Tensor:
    """
    Weighted Pearson correlation between ``a`` and ``b``, differentiable in both.

    Returns a tensor of shape [K] (scalar for 1-D inputs): the mean over groups of the per-group
    correlation. ``group_weighting="weight"`` weights groups by their total weight,
    ``"equal"`` gives every non-empty group the same weight (Numerai-style mean per-era corr).
    Groups whose variance is zero contribute a correlation of 0.
    """
    if group_weighting not in ("weight", "equal"):
        raise ValueError("group_weighting must be 'weight' or 'equal'.")
    a2, w, m = _prepare(a, weight, mask)
    b2 = _as_2d(b).to(a2.dtype).expand_as(a2)
    zero = torch.zeros((), device=a2.device, dtype=a2.dtype)
    a2 = torch.where(m, a2, zero)
    b2 = torch.where(m, b2, zero)

    g, num_groups = _group_index(group, a2.shape[0], a2.device)
    k = a2.shape[1]

    def gsum(x):
        return torch.zeros(num_groups, k, device=x.device, dtype=x.dtype).index_add(0, g, x)

    w_g = gsum(w)
    w_safe = w_g.clamp(min=1e-12)
    mean_a = gsum(w * a2) / w_safe
    mean_b = gsum(w * b2) / w_safe
    ca = a2 - mean_a[g]
    cb = b2 - mean_b[g]
    cov = gsum(w * ca * cb) / w_safe
    var_a = gsum(w * ca * ca) / w_safe
    var_b = gsum(w * cb * cb) / w_safe
    corr = cov / torch.sqrt(var_a * var_b + eps)

    gw = w_g if group_weighting == "weight" else (w_g > 0).to(w_g.dtype)
    group_weight = gw / gw.sum(dim=0, keepdim=True).clamp(min=1e-12)
    out = (group_weight * corr).sum(dim=0)
    return out[0] if a.ndim == 1 else out
