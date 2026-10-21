"""NaN feature values are routed through a dedicated bin with a learned default direction."""

import numpy as np
import pytest
import torch

from warpgbm import WarpGBM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _split_nodes(node):
    if "leaf_value" in node:
        return []
    return [node] + _split_nodes(node["left"]) + _split_nodes(node["right"])


def _nan_data(seed, n=4000, nan_frac=0.2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    nan_rows = rng.random(n) < nan_frac
    return X, nan_rows


def test_bin_edges_finite_and_nan_bin_index():
    X, nan_rows = _nan_data(0)
    X[nan_rows, 0] = np.nan
    y = np.nan_to_num(X[:, 0]).astype(np.float32)

    model = WarpGBM(objective="regression", n_estimators=5, max_depth=2, num_bins=16, random_state=0)
    model.fit(X, y)

    edges = model.bin_edges.cpu().numpy()
    assert np.isfinite(edges).all()
    bins = model.bin_inference_data(X).cpu().numpy()
    assert (bins[nan_rows, 0] == model.nan_bin).all()
    assert (bins[~nan_rows, 0] < model.num_bins).all()


def test_informative_missingness_is_learned():
    X, nan_rows = _nan_data(1)
    y = X[:, 0].copy()
    y[nan_rows] = 5.0  # missing rows have their own target level
    X[nan_rows, 0] = np.nan

    model = WarpGBM(objective="regression", n_estimators=60, max_depth=3, num_bins=32, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert abs(preds[nan_rows].mean() - 5.0) < 0.3
    assert np.mean((preds[~nan_rows] - y[~nan_rows]) ** 2) < 0.1


@pytest.mark.parametrize("nan_target, expect_left", [(0.0, True), (1.0, False)])
def test_nan_default_direction_is_learned(nan_target, expect_left):
    X, nan_rows = _nan_data(2)
    y = (X[:, 0] > 0).astype(np.float32)
    y[nan_rows] = nan_target  # NaN rows behave like the low (left) or high (right) side
    X[nan_rows, 0] = np.nan

    model = WarpGBM(
        objective="regression", n_estimators=1, max_depth=1, num_bins=16,
        learning_rate=1.0, random_state=0,
    )
    model.fit(X, y)
    root = model.forest[0]
    assert root["feature"] == 0
    assert root["nan_left"] is expect_left
    preds = model.predict(X)
    assert abs(preds[nan_rows].mean() - nan_target) < 0.05


def test_nan_only_at_predict_time_routes_right():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 3)).astype(np.float32)
    y = (2.0 * X[:, 0] - X[:, 1]).astype(np.float32)

    model = WarpGBM(objective="regression", n_estimators=30, max_depth=3, random_state=0)
    model.fit(X, y)
    assert not any(node["nan_left"] for tree in model.forest for node in _split_nodes(tree))

    X_test = X[:200].copy()
    X_big = X_test.copy()
    X_test[:, 0] = np.nan
    X_big[:, 0] = 1e30  # NaN without training NaNs follows the right branch, like +inf
    np.testing.assert_allclose(model.predict(X_test), model.predict(X_big), rtol=1e-6, atol=1e-6)


def test_all_nan_column_is_harmless():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, 3)).astype(np.float32)
    X[:, 2] = np.nan
    y = X[:, 0].copy()

    model = WarpGBM(objective="regression", n_estimators=20, max_depth=3, random_state=0)
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()
    assert model.get_feature_importance()[2] == 0.0


def test_nan_features_classification_and_multi_output():
    X, nan_rows = _nan_data(5)
    y_cls = (X[:, 0] > 0).astype(int)
    y_cls[nan_rows] = 1
    Y = np.column_stack([X[:, 0], -X[:, 0]]).astype(np.float32)
    Y[nan_rows] = 3.0
    X[nan_rows, 0] = np.nan

    clf = WarpGBM(objective="binary", n_estimators=20, max_depth=3, random_state=0)
    clf.fit(X, y_cls)
    assert (clf.predict(X) == y_cls).mean() > 0.9

    reg = WarpGBM(objective="multi_regression", n_estimators=40, max_depth=3, random_state=0)
    reg.fit(X, Y)
    preds = reg.predict(X)
    assert np.all(np.abs(preds[nan_rows].mean(axis=0) - 3.0) < 0.3)


def test_prebinned_integer_input_unaffected():
    rng = np.random.default_rng(6)
    X = rng.integers(0, 8, size=(2000, 3)).astype(np.int32)
    y = (X[:, 0] >= 4).astype(np.float32)

    model = WarpGBM(objective="regression", n_estimators=20, max_depth=2, num_bins=8, random_state=0)
    model.fit(X, y)
    assert np.mean((model.predict(X) - y) ** 2) < 0.01


def _signed_nan_data(seed=7, n=10_000, f=3):
    """Same data with +NaN and with sign-bit NaN (as numpy's 0/0 gives) in the same positions."""
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(size=(n, f)).astype(np.float32)
    X_pos[: int(0.3 * n), 0] = np.nan
    X_pos[rng.random(n) < 0.1, 1] = np.nan
    X_pos[:, 2] = np.nan  # all-NaN column
    X_neg = X_pos.copy()
    X_neg[np.isnan(X_neg)] = -np.float32(np.nan)
    assert np.signbit(X_neg[np.isnan(X_neg)]).all()
    assert not np.signbit(X_pos[np.isnan(X_pos)]).any()
    return X_pos, X_neg


def test_sign_of_nan_does_not_change_edges():
    from warpgbm import fit_bins

    X_pos, X_neg = _signed_nan_data()
    num_bins = 16
    edges_pos = fit_bins(X_pos, num_bins).cpu().numpy()
    edges_neg = fit_bins(X_neg, num_bins).cpu().numpy()
    np.testing.assert_array_equal(edges_neg, edges_pos)

    q = np.linspace(0, 1, num_bins + 1)[1:-1]
    for f in range(2):
        np.testing.assert_allclose(edges_neg[f], np.nanquantile(X_neg[:, f], q), rtol=1e-5, atol=1e-6)
    assert (edges_neg[2] == 0).all()
    assert (np.diff(edges_neg, axis=1) >= 0).all()


def test_sign_of_nan_does_not_change_fit():
    X_pos, X_neg = _signed_nan_data(8)
    y = (np.nan_to_num(X_pos[:, 0]) + np.isnan(X_pos[:, 0]) - np.nan_to_num(X_pos[:, 1])).astype(np.float32)

    def fit(X):
        return WarpGBM(objective="regression", n_estimators=10, max_depth=3, num_bins=16,
                       random_state=0, verbose=False).fit(X, y)

    pos, neg = fit(X_pos), fit(X_neg)
    assert torch.equal(neg.bin_edges, pos.bin_edges)
    assert neg.forest == pos.forest
    np.testing.assert_array_equal(neg.predict(X_neg), pos.predict(X_pos))
