import numpy as np
import pytest
from sklearn.datasets import make_classification

from warpgbm import WarpGBM


def _build_regression_data(n_samples=600, seed=42):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, size=n_samples).astype(np.float32)
    x1 = rng.normal(0.0, 0.3, size=n_samples).astype(np.float32)
    X = np.column_stack([x0, x1]).astype(np.float32)
    return X, x0


def _assert_non_decreasing(values, atol=1e-6):
    diffs = np.diff(values)
    assert np.all(diffs >= -atol), f"Expected non-decreasing sequence, min diff={diffs.min()}"


def _assert_non_increasing(values, atol=1e-6):
    diffs = np.diff(values)
    assert np.all(diffs <= atol), f"Expected non-increasing sequence, max diff={diffs.max()}"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_monotonic_increasing_regression_without_eras():
    X, x0 = _build_regression_data(seed=7)
    y = (3.0 * x0 + 0.05 * np.random.default_rng(7).normal(size=len(x0))).astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=40,
        max_depth=3,
        learning_rate=0.1,
        num_bins=32,
        monotonic_constraints=[1, 0],
        random_state=7,
        device="cuda",
    )
    model.fit(X, y)

    grid = np.linspace(-2.0, 2.0, 200, dtype=np.float32)
    X_grid = np.column_stack([grid, np.zeros_like(grid)]).astype(np.float32)
    preds = model.predict(X_grid)
    _assert_non_decreasing(preds, atol=1e-5)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_monotonic_decreasing_regression_without_eras():
    X, x0 = _build_regression_data(seed=9)
    y = (-3.0 * x0 + 0.05 * np.random.default_rng(9).normal(size=len(x0))).astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=40,
        max_depth=3,
        learning_rate=0.1,
        num_bins=32,
        monotonic_constraints=[-1, 0],
        random_state=9,
        # device="cpu",
    )
    model.fit(X, y)

    grid = np.linspace(-2.0, 2.0, 200, dtype=np.float32)
    X_grid = np.column_stack([grid, np.zeros_like(grid)]).astype(np.float32)
    preds = model.predict(X_grid)
    _assert_non_increasing(preds, atol=1e-5)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_monotonic_constraints_vector_and_dict_equivalent():
    X, x0 = _build_regression_data(seed=21)
    y = (2.4 * x0 + 0.04 * np.random.default_rng(21).normal(size=len(x0))).astype(np.float32)

    common_params = dict(
        objective="regression",
        n_estimators=35,
        max_depth=3,
        learning_rate=0.1,
        num_bins=24,
        random_state=21,
        # device="cpu",
    )

    model_vector = WarpGBM(monotonic_constraints=[1, 0], **common_params)
    model_vector.fit(X, y)
    pred_vector = model_vector.predict(X)

    model_dict = WarpGBM(monotonic_constraints={0: 1}, **common_params)
    model_dict.fit(X, y)
    pred_dict = model_dict.predict(X)

    np.testing.assert_allclose(pred_vector, pred_dict, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_monotonic_constraints_with_eras_regression():
    X, x0 = _build_regression_data(seed=33)
    era = (x0 > 0).astype(np.int32)
    noise = 0.03 * np.random.default_rng(33).normal(size=len(x0)).astype(np.float32)
    y = np.where(era == 0, 2.0 * x0, 0.8 * x0).astype(np.float32) + noise

    model = WarpGBM(
        objective="regression",
        n_estimators=45,
        max_depth=3,
        learning_rate=0.08,
        num_bins=32,
        monotonic_constraints={0: 1},
        random_state=33,
        device="cuda",
    )
    model.fit(X, y, era_id=era)

    grid = np.linspace(-2.0, 2.0, 240, dtype=np.float32)
    X_grid = np.column_stack([grid, np.zeros_like(grid)]).astype(np.float32)
    preds = model.predict(X_grid)
    _assert_non_decreasing(preds, atol=1e-5)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_monotonic_constraints_per_era_strict_blocks_conflicting_signal():
    rng = np.random.default_rng(90)
    n = 800
    x0 = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
    X = x0.reshape(-1, 1)
    era = np.zeros(n, dtype=np.int32)
    era[n // 2 :] = 1

    y = np.empty(n, dtype=np.float32)
    y[: n // 2] = 2.0 * x0[: n // 2]
    y[n // 2 :] = -2.0 * x0[n // 2 :]
    y += 0.01 * rng.normal(size=n).astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        num_bins=32,
        monotonic_constraints=[1],
        random_state=90,
        device="cuda",
    )
    model.fit(X, y, era_id=era)

    assert "leaf_value" in model.forest[0], "Conflicting eras under strict per-era monotonicity should block splitting."


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA is required")
def test_monotonic_constraints_binary_smoke_with_and_without_eras():
    X, y = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        n_redundant=2,
        random_state=17,
    )
    X = X.astype(np.float32)
    era = (X[:, 0] > np.median(X[:, 0])).astype(np.int32)

    model_no_era = WarpGBM(
        objective="binary",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        monotonic_constraints={0: 1, 1: 0},
        random_state=17,
        device="cuda",
    )
    model_no_era.fit(X, y)
    proba_no_era = model_no_era.predict_proba(X)

    model_era = WarpGBM(
        objective="binary",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        monotonic_constraints={0: 1, 1: 0},
        random_state=17,
        device="cuda",
    )
    model_era.fit(X, y, era_id=era)
    proba_era = model_era.predict_proba(X)

    assert proba_no_era.shape == (X.shape[0], 2)
    assert proba_era.shape == (X.shape[0], 2)
    assert np.isfinite(proba_no_era).all()
    assert np.isfinite(proba_era).all()


def test_monotonic_constraints_validation_vector_length():
    X = np.random.default_rng(4).normal(size=(100, 3)).astype(np.float32)
    y = np.random.default_rng(5).normal(size=100).astype(np.float32)

    model = WarpGBM(objective="regression", n_estimators=5, max_depth=2, monotonic_constraints=[1, 0])

    with pytest.raises(ValueError, match="length must equal n_features"):
        model.fit(X, y)


def test_monotonic_constraints_validation_invalid_values():
    X = np.random.default_rng(6).normal(size=(100, 3)).astype(np.float32)
    y = np.random.default_rng(7).normal(size=100).astype(np.float32)

    model = WarpGBM(
        objective="regression",
        n_estimators=5,
        max_depth=2,
        monotonic_constraints=[1, 2, 0],
    )

    with pytest.raises(ValueError, match="values must be one of"):
        model.fit(X, y)


def test_monotonic_constraints_validation_dict_keys():
    X = np.random.default_rng(8).normal(size=(120, 3)).astype(np.float32)
    y = np.random.default_rng(9).normal(size=120).astype(np.float32)

    out_of_range = WarpGBM(
        objective="regression",
        n_estimators=5,
        max_depth=2,
        monotonic_constraints={5: 1},
    )
    with pytest.raises(ValueError, match="out of range"):
        out_of_range.fit(X, y)

    wrong_key_type = WarpGBM(
        objective="regression",
        n_estimators=5,
        max_depth=2,
        monotonic_constraints={"0": 1},
    )
    with pytest.raises(TypeError, match="integer feature indices"):
        wrong_key_type.fit(X, y)
