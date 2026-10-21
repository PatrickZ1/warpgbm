"""
Quantile binning of feature matrices, the same code WarpGBM.fit and predict use.

Public contract for binned data: int8 bins 0 … num_bins-1 for values and ``MISSING_BIN`` (-1)
for NaN. Internally, WarpGBM stores missing values in the extra bin ``num_bins`` (``nan_bin``);
that index is a label, not an ordering, and never reaches the caller.
"""

import numpy as np
import torch

from warpgbm.cuda import _get_node_kernel

MISSING_BIN = -1  # bin index of a missing (NaN) value in pre-binned data
MIN_NUM_BINS = 2
MAX_NUM_BINS = 127  # value bins + the internal NaN bin must fit int8

# nanquantile sorts a copy of its input (values + int64 indices); the edges are computed for
# blocks of features with at most this many elements to bound that workspace
_QUANTILE_BLOCK_NUMEL = 1 << 24


def _check_device(device):
    try:
        device_type = torch.device(device).type
    except (RuntimeError, TypeError) as e:
        raise ValueError(f"Invalid device {device!r}.") from e
    if device_type != "cuda":
        raise ValueError(f"WarpGBM requires a CUDA device ('cuda' or 'cuda:N'), got {device!r}.")


def _check_num_bins(num_bins):
    if isinstance(num_bins, bool) or not isinstance(num_bins, (int, np.integer)):
        raise TypeError(f"num_bins must be an integer, got {type(num_bins)}.")
    if not MIN_NUM_BINS <= num_bins <= MAX_NUM_BINS:
        raise ValueError(f"num_bins must be between {MIN_NUM_BINS} and {MAX_NUM_BINS} inclusive.")


def _to_float_gpu(X, device, name="X"):
    """X as a contiguous float32 [N, F] tensor on device."""
    if torch.is_tensor(X):
        X_t = X.to(device=device, dtype=torch.float32)
    else:
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    if X_t.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {tuple(X_t.shape)}.")
    return X_t.contiguous()


def _quantile_edges(X, num_bins):
    """
    Bin edges [F, num_bins - 1] of X [N, F] (float32, on the GPU): the interior quantiles
    1/num_bins … (num_bins-1)/num_bins of every column, NaNs excluded; 0 for all-NaN columns.
    """
    n, f = X.shape
    q = torch.linspace(0, 1, num_bins + 1, device=X.device, dtype=torch.float32)[1:-1]
    edges = torch.empty((f, num_bins - 1), dtype=torch.float32, device=X.device)
    block = max(1, _QUANTILE_BLOCK_NUMEL // max(n, 1))
    nan = torch.tensor(float("nan"), dtype=torch.float32, device=X.device)
    for start in range(0, f, block):
        stop = min(start + block, f)
        X_block = X[:, start:stop]
        # The CUDA sort orders NaNs with the sign bit set (e.g. numpy's 0/0) before all numbers,
        # so nanquantile would count them as values; canonical NaNs sort last and are excluded
        X_block = torch.where(torch.isnan(X_block), nan, X_block)
        edges[start:stop] = torch.nanquantile(X_block, q, dim=0).T
    return torch.nan_to_num_(edges, nan=0.0)


def _bin_matrix(X, edges, nan_bin):
    """int8 bins [N, F] of X [N, F] (float32, on the GPU): number of edges <= x, NaN -> nan_bin."""
    bins = torch.empty(X.shape, dtype=torch.int8, device=X.device)
    _get_node_kernel().bin_matrix(X, edges.contiguous(), bins, nan_bin)
    return bins


def fit_bins(X, num_bins, device="cuda"):
    """
    Quantile bin edges of the columns of X, exactly as ``WarpGBM.fit`` computes them.

    Parameters
    ----------
    X : array-like or torch.Tensor of shape (n_samples, n_features)
        Raw feature values; NaN marks a missing value and is excluded from the quantiles.
    num_bins : int
        Number of value bins (2 … 127), the ``num_bins`` of the models that will use the bins.
    device : str, default "cuda"
        CUDA device for the computation and the returned edges.

    Returns
    -------
    torch.Tensor of shape (n_features, num_bins - 1), float32, on ``device``
        Sorted interior quantiles of every column; 0 for all-NaN columns.
    """
    _check_num_bins(num_bins)
    _check_device(device)
    with torch.no_grad():
        X_t = _to_float_gpu(X, device)
        if X_t.shape[0] == 0:
            raise ValueError("X must contain at least one row.")
        return _quantile_edges(X_t, int(num_bins))


def apply_bins(X, edges, device=None):
    """
    Bin X with edges from ``fit_bins``: bin = number of edges <= x, ``MISSING_BIN`` (-1) for NaN.

    The result is what ``WarpGBM(prebinned=True)`` takes, and the same bins ``WarpGBM.fit``
    assigns internally (where missing values use the internal NaN bin instead of -1).

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor of shape (n_samples, n_features)
        Raw feature values.
    edges : torch.Tensor or array-like of shape (n_features, num_bins - 1)
        Sorted bin edges per feature, e.g. from ``fit_bins``.
    device : str or None, default None
        CUDA device for the computation; defaults to the device of ``edges`` (or "cuda").

    Returns
    -------
    int8 bins of shape (n_samples, n_features), values in [-1, num_bins - 1]: a numpy array for a
    numpy input, a tensor on ``device`` for a tensor input.
    """
    if device is None:
        device = edges.device if torch.is_tensor(edges) and edges.is_cuda else "cuda"
    _check_device(device)
    with torch.no_grad():
        edges_t = torch.as_tensor(edges, dtype=torch.float32).to(device).contiguous()
        if edges_t.ndim != 2:
            raise ValueError(f"edges must be 2-dimensional, got shape {tuple(edges_t.shape)}.")
        if not MIN_NUM_BINS <= edges_t.shape[1] + 1 <= MAX_NUM_BINS:
            raise ValueError(
                f"edges must have between {MIN_NUM_BINS - 1} and {MAX_NUM_BINS - 1} columns "
                f"(num_bins - 1), got {edges_t.shape[1]}."
            )
        if edges_t.shape[1] > 1:
            unsorted = (edges_t[:, 1:] < edges_t[:, :-1]).any()
        else:
            unsorted = torch.zeros((), dtype=torch.bool, device=device)
        if bool(unsorted | torch.isnan(edges_t).any()):
            raise ValueError("edges must be free of NaN and sorted along each row.")

        X_t = _to_float_gpu(X, device)
        if X_t.shape[1] != edges_t.shape[0]:
            raise ValueError(
                f"X has {X_t.shape[1]} features, but edges has {edges_t.shape[0]} rows."
            )
        bins = _bin_matrix(X_t, edges_t, MISSING_BIN)
    return bins if torch.is_tensor(X) else bins.cpu().numpy()


def _bin_range_error(X, num_bins, name):
    """ValueError naming the first column of X with a value outside [MISSING_BIN, num_bins - 1]."""
    bad = (X < MISSING_BIN) | (X >= num_bins)
    col = int(np.flatnonzero(bad.any(axis=0))[0])
    values = X[bad[:, col], col]
    return ValueError(
        f"{name} is not valid pre-binned input: column {col} has values outside "
        f"[{MISSING_BIN}, {num_bins - 1}] (e.g. {values[0]}, min {X[:, col].min()}, "
        f"max {X[:, col].max()}). Pre-binned input holds bin indices 0 … num_bins-1 and "
        f"{MISSING_BIN} for missing values (see warpgbm.apply_bins)."
    )


def _check_prebinned_array(X, name):
    if not isinstance(X, np.ndarray) or not np.issubdtype(X.dtype, np.integer):
        dtype = X.dtype if isinstance(X, np.ndarray) else type(X).__name__
        raise TypeError(
            f"{name} must be an integer numpy array of bin indices for a pre-binned model "
            f"(prebinned=True), got {dtype}. Bin raw features with warpgbm.apply_bins, or "
            "use prebinned=False."
        )
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {X.shape}")


def _prebinned_to_gpu(X, num_bins, device, name="X"):
    """
    Validate pre-binned input (integers in [MISSING_BIN, num_bins - 1]) and upload it as the
    internal int8 bins [N, F]: MISSING_BIN becomes the NaN bin num_bins.
    """
    _check_prebinned_array(X, name)
    if X.size == 0:
        return torch.empty(X.shape, dtype=torch.int8, device=device)

    if X.dtype != np.int8:
        # Check before narrowing: int8 would wrap out-of-range values into range
        if X.min() < MISSING_BIN or X.max() >= num_bins:
            raise _bin_range_error(X, num_bins, name)
        bins = torch.from_numpy(np.ascontiguousarray(X, dtype=np.int8)).to(device)
    else:
        bins = torch.from_numpy(np.ascontiguousarray(X)).to(device)
        lo, hi = torch.stack(torch.aminmax(bins)).tolist()
        if lo < MISSING_BIN or hi >= num_bins:
            raise _bin_range_error(X, num_bins, name)
    return bins.masked_fill_(bins == MISSING_BIN, num_bins).contiguous()


def _prebinned_to_cpu(X, num_bins, name="X"):
    """CPU counterpart of _prebinned_to_gpu: validated internal int8 bins [N, F] as numpy."""
    _check_prebinned_array(X, name)
    if X.size and (X.min() < MISSING_BIN or X.max() >= num_bins):
        raise _bin_range_error(X, num_bins, name)
    bins = np.array(X, dtype=np.int8)
    bins[bins == MISSING_BIN] = num_bins
    return bins


def _bin_matrix_cpu(X, edges, nan_bin):
    """
    CPU counterpart of _bin_matrix for small inputs: int8 bins [N, F] of X [N, F] with edges
    [F, E] (numpy float32). Same rule as the CUDA binner: number of edges <= x, NaN -> nan_bin.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
    with np.errstate(invalid="ignore"):
        bins = (X[:, :, None] >= edges[None, :, :]).sum(axis=2, dtype=np.int8)
    bins[np.isnan(X)] = nan_bin
    return bins
