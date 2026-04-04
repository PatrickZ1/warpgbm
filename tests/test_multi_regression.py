import time

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from warpgbm import WarpGBM


def _find_any_leaf(node):
    if "leaf_value" in node:
        return node["leaf_value"]
    left = _find_any_leaf(node["left"])
    if left is not None:
        return left
    return _find_any_leaf(node["right"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multi_regression_fit_predict_shapes_and_vector_leaves():
    X, y = make_regression(
        n_samples=600,
        n_features=14,
        n_targets=3,
        noise=0.3,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = WarpGBM(
        objective="multi_regression",
        n_estimators=35,
        max_depth=4,
        learning_rate=0.1,
        num_bins=32,
        random_state=42,
        device="cuda",
    )
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape
    assert model.num_outputs == y.shape[1]

    leaf_value = _find_any_leaf(model.forest[0])
    assert isinstance(leaf_value, list)
    assert len(leaf_value) == y.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multi_regression_nan_targets_match_explicit_masked_weighting():
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_targets=3,
        noise=0.2,
        random_state=7,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    rng = np.random.default_rng(7)
    nan_mask = rng.random(y.shape) < 0.25
    y_nan = y.copy()
    y_nan[nan_mask] = np.nan

    y_filled = np.nan_to_num(y_nan, nan=0.0).astype(np.float32)
    sample_weight_mask = (~nan_mask).astype(np.float32)

    params = dict(
        objective="multi_regression",
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        num_bins=32,
        min_child_weight=1,
        random_state=7,
        device="cuda",
    )

    model_nan = WarpGBM(**params)
    model_nan.fit(X, y_nan)
    preds_nan = model_nan.predict(X)

    model_mask = WarpGBM(**params)
    model_mask.fit(X, y_filled, sample_weight=sample_weight_mask)
    preds_mask = model_mask.predict(X)

    assert np.isfinite(preds_nan).all()
    assert np.isfinite(preds_mask).all()

    observed_mask = ~nan_mask
    mse_nan = np.mean((preds_nan[observed_mask] - y[observed_mask]) ** 2)
    mse_mask = np.mean((preds_mask[observed_mask] - y[observed_mask]) ** 2)
    mse_ratio = mse_nan / max(mse_mask, 1e-8)

    # NaN handling via internal mask should behave similarly to explicit mask weights.
    assert 0.8 < mse_ratio < 1.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multi_regression_uniform_1d_vs_2d_sample_weight_equivalence():
    X, y = make_regression(
        n_samples=400,
        n_features=12,
        n_targets=2,
        noise=0.15,
        random_state=17,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    w_1d = np.full(X.shape[0], 2.5, dtype=np.float32)
    w_2d = np.repeat(w_1d[:, None], y.shape[1], axis=1)

    params = dict(
        objective="multi_regression",
        n_estimators=25,
        max_depth=3,
        learning_rate=0.1,
        random_state=17,
        device="cuda",
    )

    model_1d = WarpGBM(**params)
    model_1d.fit(X, y, sample_weight=w_1d)
    pred_1d = model_1d.predict(X)

    model_2d = WarpGBM(**params)
    model_2d.fit(X, y, sample_weight=w_2d)
    pred_2d = model_2d.predict(X)

    np.testing.assert_allclose(pred_1d, pred_2d, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multi_regression_vs_independent_models_parity():
    rng = np.random.default_rng(123)
    n_samples = 1200
    n_features = 16

    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    latent = 2.0 * X[:, 0] - 1.4 * X[:, 1] + 0.6 * X[:, 2]

    y0 = latent + 0.15 * rng.normal(size=n_samples)
    y1 = 0.8 * latent + 0.7 * X[:, 3] + 0.15 * rng.normal(size=n_samples)
    y2 = -0.5 * latent + 0.4 * X[:, 4] + 0.15 * rng.normal(size=n_samples)
    y = np.column_stack([y0, y1, y2]).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=123,
    )

    common = dict(
        n_estimators=40,
        max_depth=4,
        learning_rate=0.1,
        num_bins=32,
        random_state=123,
        device="cuda",
    )

    start = time.time()
    multi_model = WarpGBM(objective="multi_regression", **common)
    multi_model.fit(X_train, y_train)
    multi_fit_time = time.time() - start

    start = time.time()
    multi_preds = multi_model.predict(X_test)
    multi_pred_time = time.time() - start

    independent_preds = []
    baseline_fit_start = time.time()
    for target_idx in range(y_train.shape[1]):
        model_k = WarpGBM(objective="regression", **common)
        model_k.fit(X_train, y_train[:, target_idx])
        independent_preds.append(model_k.predict(X_test))
    baseline_fit_time = time.time() - baseline_fit_start

    baseline_pred_start = time.time()
    independent_preds = np.column_stack(independent_preds)
    baseline_pred_time = time.time() - baseline_pred_start

    multi_mse = np.mean((multi_preds - y_test) ** 2, axis=0)
    baseline_mse = np.mean((independent_preds - y_test) ** 2, axis=0)
    mse_ratio = multi_mse / np.maximum(baseline_mse, 1e-8)

    print("\nVector-leaf multi-output vs independent models")
    print(
        f"fit_time_multi={multi_fit_time:.4f}s fit_time_baseline={baseline_fit_time:.4f}s"
    )
    print(
        f"pred_time_multi={multi_pred_time:.4f}s pred_time_baseline={baseline_pred_time:.4f}s"
    )
    print(f"multi_mse={multi_mse}")
    print(f"baseline_mse={baseline_mse}")
    print(f"mse_ratio={mse_ratio}")

    # Shared-split vector model should remain close to one-model-per-target baseline.
    assert np.all(mse_ratio < 1.6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multi_regression_warm_start_equivalent_to_full_training():
    X, y = make_regression(
        n_samples=500,
        n_features=12,
        n_targets=3,
        noise=0.2,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model_full = WarpGBM(
        objective="multi_regression",
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=31,
        device="cuda",
    )
    model_full.fit(X, y)
    preds_full = model_full.predict(X)

    model_warm = WarpGBM(
        objective="multi_regression",
        n_estimators=25,
        max_depth=4,
        learning_rate=0.1,
        random_state=31,
        warm_start=True,
        device="cuda",
    )
    model_warm.fit(X, y)
    model_warm.n_estimators = 50
    model_warm.fit(X, y)
    preds_warm = model_warm.predict(X)

    np.testing.assert_allclose(preds_full, preds_warm, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_multi_regression_predict_chunking_consistency():
    X, y = make_regression(
        n_samples=700,
        n_features=9,
        n_targets=2,
        noise=0.25,
        random_state=87,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = WarpGBM(
        objective="multi_regression",
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        random_state=87,
        device="cuda",
    )
    model.fit(X, y)

    full_preds = model.predict(X)
    split = X.shape[0] // 2
    chunked_preds = np.vstack([model.predict(X[:split]), model.predict(X[split:])])

    np.testing.assert_allclose(full_preds, chunked_preds, rtol=1e-5, atol=1e-4)
