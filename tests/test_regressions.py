"""
Regression tests for bugs found while reviewing the fork changes (see docs/FORK_CHANGES.md).
Each test is labelled with the issue letter used in that document.
"""

import numpy as np
import pytest
import torch
from sklearn.datasets import load_iris

from warpgbm import WarpGBM
from warpgbm.core import WarpGBM as _WarpGBM, _get_node_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_A_multiclass_default_min_child_weight_still_splits():
    # min_child_weight is a sample-weight sum, not a hessian sum (softmax hessians are <= 0.25)
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float32)
    model = WarpGBM(objective="multiclass", n_estimators=20, max_depth=3, random_state=0)
    model.fit(X, y)
    assert (model.predict(X) == y).mean() > 0.9
    assert model.get_feature_importance().sum() > 0


def test_A_min_child_weight_counts_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    # No child can hold more than the full data set -> never split
    model = WarpGBM(objective="binary", n_estimators=3, max_depth=2, min_child_weight=401, random_state=0)
    model.fit(X, y)
    assert all("leaf_value" in tree for round_trees in model.forest for tree in round_trees)


def test_B_multi_output_opposite_sign_targets():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5)).astype(np.float32)
    f = 2.0 * X[:, 0]
    Y = np.column_stack([f, -f]).astype(np.float32)

    model = WarpGBM(objective="multi_regression", n_estimators=50, max_depth=3, random_state=0)
    model.fit(X, Y)
    mse = np.mean((model.predict(X) - Y) ** 2, axis=0)
    assert np.all(mse < 0.1 * Y.var(axis=0)), mse


def _min_slice_diff(model, columns, grid, slice_values, n_features):
    worst = np.inf
    for v in slice_values:
        X_grid = np.zeros((len(grid), n_features), dtype=np.float32)
        X_grid[:, 0] = grid
        X_grid[:, 1] = v
        preds = model.predict(X_grid)
        preds = preds.reshape(len(grid), -1)
        for k in columns:
            worst = min(worst, np.diff(preds[:, k]).min())
    return worst


def test_C_monotonic_with_feature_interaction():
    rng = np.random.default_rng(1)
    n = 20000
    x0 = rng.uniform(-1, 1, n)
    x1 = rng.uniform(-1, 1, n)
    # Left of x0=0 a narrow x1 region is high, right of it everything is medium:
    # without bound propagation the x1 split under the left branch breaks monotonicity.
    y = np.where(x0 < 0, 10.0 * (x1 > 0.8), 5.0).astype(np.float32)
    X = np.column_stack([x0, x1]).astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=30,
        max_depth=3,
        num_bins=32,
        monotonic_constraints=[1, 0],
        random_state=0,
    )
    model.fit(X, y)
    worst = _min_slice_diff(model, [0], np.linspace(-1, 1, 200), np.linspace(-1, 1, 21), 2)
    assert worst >= -1e-5, worst


def test_C_monotonic_decreasing_with_interaction():
    rng = np.random.default_rng(2)
    n = 20000
    x0 = rng.uniform(-1, 1, n)
    x1 = rng.uniform(-1, 1, n)
    y = np.where(x0 > 0, 10.0 * (x1 > 0.8), 5.0).astype(np.float32)
    X = np.column_stack([x0, x1]).astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=30,
        max_depth=3,
        num_bins=32,
        monotonic_constraints={0: -1},
        random_state=0,
    )
    model.fit(X, y)
    for v in np.linspace(-1, 1, 21):
        grid = np.linspace(-1, 1, 200)
        preds = model.predict(np.column_stack([grid, np.full_like(grid, v)]).astype(np.float32))
        assert np.diff(preds).max() <= 1e-5


def test_D_monotonic_per_output_for_multi_regression():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 3)).astype(np.float32)
    y0 = X[:, 0] + np.where(X[:, 1] > 0, 3.0, 0.0) * (X[:, 0] < 0)
    y1 = 0.5 * X[:, 0] + X[:, 2]
    Y = np.column_stack([y0, y1]).astype(np.float32)

    model = WarpGBM(
        objective="multi_regression",
        n_estimators=30,
        max_depth=3,
        num_bins=32,
        monotonic_constraints=[1, 0, 0],
        random_state=0,
    )
    model.fit(X, Y)
    worst = _min_slice_diff(model, [0, 1], np.linspace(-2, 2, 200), [-1.0, 0.0, 1.0], 3)
    assert worst >= -1e-5, worst


def test_E_weighted_base_prediction():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3000, 3)).astype(np.float32)
    y = rng.normal(size=3000).astype(np.float32)
    w = np.where(y > 0, 10.0, 1.0).astype(np.float32)

    model = WarpGBM(objective="regression", n_estimators=1, max_depth=1, random_state=0)
    model.fit(X, y, sample_weight=w)
    assert abs(model.base_prediction - np.average(y, weights=w)) < 1e-4


@pytest.mark.parametrize("objective", ["binary", "multiclass"])
def test_F_classification_predict_uses_training_prior(objective):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    y = (rng.random(3000) < 0.1).astype(int)
    if objective == "multiclass":
        y = y + (rng.random(3000) < 0.3).astype(int) * (1 - y) * 2  # classes 0/1/2

    model = WarpGBM(objective=objective, n_estimators=1, max_depth=1, min_child_weight=1, random_state=0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    for k in range(proba.shape[1]):
        assert abs(proba[:, k].mean() - (y == k).mean()) < 0.02


def _reference_histograms(bins_fm, values, rows, feat_idx, era, E, B):
    """[F, E, B, C] float64 sums of values [N, C] over rows, per feature/era/bin."""
    out = torch.zeros(len(feat_idx), E, B, values.shape[1], dtype=torch.float64, device=values.device)
    v = values[rows].double()
    e = era[rows].long()
    for j, f in enumerate(feat_idx.long().tolist()):
        flat = e * B + bins_fm[f, rows].long()
        out[j].view(E * B, -1).index_add_(0, flat, v)
    return out


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize(
    "num_eras, K, Kh",
    [(3, 3, 3), (3, 20, 0), (500, 3, 3)],  # 500 eras exceed shared memory -> global fallback
)
def test_G_histogram_kernel_matches_reference(num_eras, K, Kh, deterministic):
    torch.manual_seed(0)
    N, F_master, B = 5000, 6, 33
    dev = "cuda"
    bins_fm = torch.randint(0, B, (F_master, N), dtype=torch.int8, device=dev)
    grad = torch.randn(N, K, device=dev)
    hess = torch.rand(N, Kh, device=dev)
    row_weight = torch.rand(N, device=dev)
    era = torch.randint(0, num_eras, (N,), dtype=torch.int32, device=dev)
    feat_idx = torch.tensor([4, 0, 2], dtype=torch.int32, device=dev)
    # Two segments (nodes) of sorted rows, written to slots 1 and 0
    perm = torch.randperm(N, device=dev)
    seg_a, seg_b = perm[:1500].sort().values, perm[1500:4000].sort().values
    order = torch.cat([seg_a, seg_b]).to(torch.int32)

    model = _WarpGBM()
    model._tile_rows_ = T = 1024
    scales = torch.stack([model._device_scale(t) for t in (grad, hess, row_weight)])
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
    tiles = [(len(seg_a) + T - 1) // T, (len(seg_b) + T - 1) // T]
    C = K + Kh + 1
    hist = torch.zeros(
        (2, 3, num_eras, B, C), device=dev, dtype=torch.int64 if deterministic else torch.float32
    )
    _get_node_kernel().histogram_level(
        bins_fm, grad, hess, row_weight, order,
        i32([0, len(seg_a)]), i32([len(seg_a), len(seg_b)]), i32([1, 0]),
        i32([0, tiles[0], tiles[0] + tiles[1]]), feat_idx, era, hist, scales,
        sum(tiles) + 2, T,
    )
    if deterministic:
        hist = _WarpGBM._acc_to_float(hist, scales, K, Kh)
    values = torch.cat([grad, hess, row_weight[:, None]], dim=1)
    for slot, rows in ((1, seg_a), (0, seg_b)):
        ref = _reference_histograms(bins_fm, values, rows.long(), feat_idx, era, num_eras, B)
        torch.testing.assert_close(hist[slot].double(), ref, rtol=1e-4, atol=1e-4)


def test_G_many_eras_training_learns():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20000, 5)).astype(np.float32)
    y = X[:, 0].copy()
    era = rng.integers(0, 400, 20000).astype(np.int32)

    model = WarpGBM(objective="regression", n_estimators=10, max_depth=3, num_bins=32, random_state=0)
    model.fit(X, y, era_id=era)
    assert np.mean((model.predict(X) - y) ** 2) < 0.5 * y.var()


def test_nan_targets_single_output_regression_are_masked():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    y = (2.0 * X[:, 0]).astype(np.float32)
    y_nan = y.copy()
    y_nan[rng.random(2000) < 0.2] = np.nan

    model = WarpGBM(objective="regression", n_estimators=40, max_depth=3, random_state=0)
    model.fit(X, y_nan)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert np.mean((preds - y) ** 2) < 0.1 * y.var()


def _to_legacy_state(state):
    """Rewrite a saved state into the format written before the objective refactor."""
    import copy

    legacy = copy.deepcopy(state)
    for key in ("objective_", "min_sum_hessian", "_targets_1d", "num_outputs"):
        legacy.pop(key, None)

    def convert(node):
        if "leaf_value" in node:
            return
        node["feature"] = torch.tensor(node["feature"], device="cuda")  # old trees held tensors
        node.pop("nan_left", None)
        convert(node["left"])
        convert(node["right"])

    for entry in legacy["forest"]:
        for tree in entry if isinstance(entry, list) else [entry]:
            if tree:
                convert(tree)
    return legacy


@pytest.mark.parametrize("objective", ["regression", "multiclass"])
def test_models_saved_in_the_legacy_format_still_load(objective, tmp_path):
    import pickle

    rng = np.random.default_rng(8)
    X = rng.normal(size=(1000, 4)).astype(np.float32)
    y = X[:, 0] + 0.1 * rng.normal(size=1000).astype(np.float32)
    if objective == "multiclass":
        y = np.digitize(y, [-0.5, 0.5])

    model = WarpGBM(objective=objective, n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    path = tmp_path / "model.pkl"
    model.save_model(str(path))
    with open(path, "rb") as f:
        state = pickle.load(f)
    legacy = _to_legacy_state(state)
    if objective == "multiclass":
        legacy["base_prediction"] = None  # the old multiclass code did not store its prior
    with open(path, "wb") as f:
        pickle.dump(legacy, f)

    loaded = WarpGBM().load_model(str(path))
    if objective == "regression":
        np.testing.assert_allclose(loaded.predict(X), model.predict(X), rtol=1e-5, atol=1e-5)
    else:
        proba = loaded.predict_proba(X)
        assert proba.shape == (1000, 3) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (loaded.predict(X) == y).mean() > 0.8


def test_predictions_are_bit_reproducible():
    # the predict kernel sums trees in a fixed order (it used float atomics across trees)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5000, 8)).astype(np.float32)
    y = (X[:, 0] + np.sin(X[:, 1]) + rng.normal(size=5000)).astype(np.float32)
    model = WarpGBM(n_estimators=200, max_depth=5, random_state=0).fit(X, y)
    first = model.predict(X)
    for _ in range(3):
        np.testing.assert_array_equal(model.predict(X.copy()), first)
