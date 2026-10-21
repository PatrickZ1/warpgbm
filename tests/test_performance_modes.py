"""
v2.7.0: deterministic switch, level-wise training engine (CUDA graphs, memory chunking),
packed-forest prediction (staged, CPU path).
"""
import pickle

import numpy as np
import pytest
import torch

import warpgbm.core as core
from warpgbm import WarpGBM, apply_bins, fit_bins


def _data(n=3000, f=30, k=4, seed=0, nan=0.03):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float32)
    X[rng.random((n, f)) < nan] = np.nan
    signal = np.nan_to_num(X[:, :k]) + 0.5 * np.nan_to_num(X[:, k : 2 * k]) ** 2
    Y = (signal + 0.3 * rng.standard_normal((n, k))).astype(np.float32)
    era = (np.arange(n) % 3).astype(np.int32)
    return X, Y, era


def _params(**kw):
    p = dict(
        objective="multi_regression",
        num_bins=16,
        max_depth=5,
        n_estimators=25,
        min_child_weight=10,
        colsample_bytree=0.7,
        random_state=3,
        verbose=False,
    )
    p.update(kw)
    return p


def _fit(X, Y, era, **kw):
    return WarpGBM(**_params(**kw)).fit(X, Y, era_id=era)


@pytest.fixture
def no_cuda_graphs(monkeypatch):
    monkeypatch.setattr(WarpGBM, "_use_cuda_graphs", False)


# ─── deterministic switch ───


def test_deterministic_parameter_validation_and_persistence(tmp_path):
    with pytest.raises(TypeError):
        WarpGBM(deterministic="yes")
    X, Y, era = _data(n=500)
    model = _fit(X, Y, era, deterministic=False, n_estimators=3)
    assert model.get_params()["deterministic"] is False
    path = tmp_path / "m.pkl"
    model.save_model(str(path))
    assert WarpGBM(verbose=False).load_model(str(path)).deterministic is False

    # Models saved before v2.7 load as deterministic
    with open(path, "rb") as f:
        state = pickle.load(f)
    del state["deterministic"]
    with open(path, "wb") as f:
        pickle.dump(state, f)
    loaded = WarpGBM(verbose=False, deterministic=False).load_model(str(path))
    assert loaded.deterministic is True
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(objective="regression"),
        dict(monotonic_constraints={0: 1, 1: -1}),
        dict(objective="multiclass", n_estimators=8),
    ],
)
def test_deterministic_fits_are_identical(kw):
    X, Y, era = _data()
    y = Y
    if kw.get("objective") == "regression":
        y = Y[:, 0]
    elif kw.get("objective") == "multiclass":
        y = np.digitize(np.nan_to_num(X[:, 0]), [-0.5, 0.5])
    fits = [WarpGBM(**_params(**kw)).fit(X, y, era_id=era) for _ in range(3)]
    for other in fits[1:]:
        assert other.forest == fits[0].forest
        np.testing.assert_array_equal(other.predict_raw(X), fits[0].predict_raw(X))
        np.testing.assert_array_equal(other.feature_importance_, fits[0].feature_importance_)


def test_cuda_graph_replay_matches_eager(monkeypatch):
    X, Y, era = _data()
    graphs = _fit(X, Y, era)
    monkeypatch.setattr(WarpGBM, "_use_cuda_graphs", False)
    eager = _fit(X, Y, era)
    assert graphs.forest == eager.forest
    np.testing.assert_array_equal(graphs.train_predictions_, eager.train_predictions_)


def test_fast_mode_matches_deterministic_quality():
    X, Y, era = _data(n=4000)
    det = _fit(X[:3000], Y[:3000], era[:3000], n_estimators=40)
    fast = _fit(X[:3000], Y[:3000], era[:3000], n_estimators=40, deterministic=False)
    mse = lambda m: float(np.mean((m.predict(X[3000:]) - Y[3000:]) ** 2))
    assert abs(mse(fast) - mse(det)) < 0.02 * mse(det)
    # Same algorithm: the predictions differ by float rounding (and rare near-tie splits) only
    assert np.mean(np.abs(fast.predict(X) - det.predict(X))) < 1e-3


def test_level_chunking_without_subtraction_matches(monkeypatch, no_cuda_graphs):
    # A tiny histogram budget processes every level node by node without histogram subtraction.
    # Fixed-point histograms are exact, so the forest is identical.
    X, Y, era = _data()
    reference = _fit(X, Y, era)
    monkeypatch.setattr(core, "_HIST_MEMORY_FRACTION", 1e-9)
    chunked = _fit(X, Y, era)
    assert chunked.forest == reference.forest


def test_unit_hessian_shortcut_matches_full_hessian():
    # Squared error without missing targets reads the hessian sums from the row-weight channel;
    # an equivalent custom loss (autograd hessian = 1) accumulates them
    X, Y, era = _data()
    builtin = _fit(X, Y, era)
    custom = _fit(X, Y, era, objective=lambda y, p: 0.5 * (p - y) ** 2)
    assert builtin._targets_1d is False
    np.testing.assert_allclose(custom.predict(X), builtin.predict(X), atol=1e-4)


def test_non_finite_gradients_raise():
    X, Y, era = _data(n=500)

    def bad_loss(y, p):
        return torch.sqrt(p - y)  # NaN gradients for p < y

    with pytest.raises(FloatingPointError):
        _fit(X, Y, era, objective=bad_loss, n_estimators=3)


# ─── prediction ───


def _models():
    X, Y, era = _data()
    yield "multi", X, _fit(X, Y, era)
    yield "regression", X, _fit(X, Y[:, 0], era, objective="regression")
    ycls = np.digitize(np.nan_to_num(X[:, 0]), [-0.5, 0.5])
    yield "multiclass", X, _fit(X, ycls, era, objective="multiclass", n_estimators=10)
    bins = apply_bins(X, fit_bins(X, 16))
    yield "prebinned", bins, _fit(bins, Y, era, prebinned=True)


@pytest.mark.parametrize("name, X, model", list(_models()), ids=lambda v: v if isinstance(v, str) else "")
def test_staged_and_cpu_predictions_are_exact(name, X, model):
    n = model.num_trees_
    rounds = [n, 0, 3, 3, n // 2]
    staged = model.staged_predict_raw(X, rounds)
    staged_cpu = model.staged_predict_raw(X[:50], rounds, device="cpu")
    for p, r in enumerate(rounds):
        gpu = model.predict_raw(X, num_trees=r)
        np.testing.assert_array_equal(staged[p], gpu)
        np.testing.assert_array_equal(staged_cpu[p], gpu[:50])
        np.testing.assert_array_equal(model.predict_raw(X[:7], num_trees=r, device="cpu"), gpu[:7])
    if model._is_classifier():
        # Raw scores are identical; the softmax itself may round differently on the CPU
        np.testing.assert_allclose(
            model.predict_proba(X[:9], device="cpu"), model.predict_proba(X[:9]), atol=1e-6
        )
        labels = model.staged_predict(X[:20], [2, n])
        np.testing.assert_array_equal(labels[1], model.predict(X[:20]))
        proba = model.staged_predict_proba(X[:20], [2, n], device="cpu")
        np.testing.assert_allclose(proba[1], model.predict_proba(X[:20]), atol=1e-6)
    else:
        np.testing.assert_array_equal(model.predict(X[:9], device="cpu"), model.predict(X[:9]))
        np.testing.assert_array_equal(model.staged_predict(X[:9], [2, n])[1], model.predict(X[:9]))


def test_cpu_prediction_with_nan_only_at_predict_time():
    X, Y, era = _data(nan=0.0)
    model = _fit(X, Y, era)
    Xp = X[:20].copy()
    Xp[::3, :5] = np.nan
    np.testing.assert_array_equal(model.predict(Xp, device="cpu"), model.predict(Xp))


def test_prediction_device_validation():
    X, Y, era = _data(n=500)
    model = _fit(X, Y, era, n_estimators=2)
    with pytest.raises(ValueError):
        model.predict(X, device="cuda:0")
    with pytest.raises(TypeError):
        model.staged_predict(X, 2)


def test_staged_after_truncate_and_eval_scores_equal_predictions():
    X, Y, era = _data(n=4000)
    model = WarpGBM(**_params(n_estimators=30)).fit(
        X[:3000], Y[:3000], era_id=era[:3000],
        X_eval=X[3000:], y_eval=Y[3000:], eval_every_n_trees=5,
    )
    # The eval scores are accumulated round by round with the prediction kernel's additions
    for r, value in zip(model.eval_iterations_, model.evals_result_["eval"]):
        pred = model.predict(X[3000:], num_trees=r)
        mse = float(np.mean((pred - Y[3000:]) ** 2))
        assert abs(mse - value) <= 1e-6 * max(1.0, abs(value))
    full = model.predict_raw(X, num_trees=12)
    model.truncate(12)
    np.testing.assert_array_equal(model.predict_raw(X), full)
    np.testing.assert_array_equal(model.staged_predict_raw(X, [12])[0], full)


def test_early_stopping_with_pipelined_rounds():
    # Round i + 1 is already on the GPU when round i triggers the stop; the model must still be
    # the best round, with consistent training predictions
    X, Y, era = _data(n=4000)
    model = WarpGBM(**_params(n_estimators=300, learning_rate=0.5)).fit(
        X[:3000], Y[:3000], era_id=era[:3000],
        X_eval=X[3000:], y_eval=Y[3000:], eval_every_n_trees=1, early_stopping_rounds=3,
    )
    assert model.num_trees_ < 300
    assert model.num_trees_ == model.best_iteration_ + 3
    best = model.predict_raw(X[:3000])
    np.testing.assert_allclose(model.train_predictions_, best, atol=1e-5)
