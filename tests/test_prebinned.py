"""Explicit pre-binned mode, the public binning functions and the training-loss recording switch."""

import warnings

import numpy as np
import pytest
import torch

import warpgbm.binning as binning
from warpgbm import MISSING_BIN, WarpGBM, apply_bins, fit_bins

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

NUM_BINS = 16


def _data(seed=0, n=3000, f=4, nan_frac=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    X[rng.random(X.shape) < nan_frac] = np.nan
    y = (np.nan_to_num(X[:, 0]) - 0.5 * np.nan_to_num(X[:, 1]) + np.isnan(X[:, 2])).astype(np.float32)
    return X, y


def _model(**kwargs):
    params = dict(num_bins=NUM_BINS, n_estimators=15, max_depth=3, random_state=0, verbose=False)
    params.update(kwargs)
    return WarpGBM(**params)


def _internal(bins, num_bins=NUM_BINS):
    """Public bins (-1 = missing) -> WarpGBM's internal bins (NaN bin = num_bins)."""
    return np.where(bins == MISSING_BIN, num_bins, bins)


# ─── fit_bins / apply_bins ───


def test_fit_bins_and_apply_bins_match_fit():
    X, y = _data()
    model = _model().fit(X, y)

    edges = fit_bins(X, NUM_BINS)
    assert edges.shape == (X.shape[1], NUM_BINS - 1)
    assert torch.equal(edges, model.bin_edges)

    bins = apply_bins(X, edges)
    assert isinstance(bins, np.ndarray) and bins.dtype == np.int8
    assert ((bins == MISSING_BIN) == np.isnan(X)).all()
    assert bins.min() >= MISSING_BIN and bins.max() <= NUM_BINS - 1
    np.testing.assert_array_equal(_internal(bins), model.bin_inference_data(X).cpu().numpy())


def test_apply_bins_counts_edges_le_x():
    X, _ = _data(1)
    edges = fit_bins(X, NUM_BINS)
    X_t = torch.as_tensor(X, device="cuda")
    expected = torch.searchsorted(edges, X_t.T.contiguous(), right=True).T
    expected = torch.where(torch.isnan(X_t), MISSING_BIN, expected).to(torch.int8)
    bins = apply_bins(X_t, edges)
    assert torch.is_tensor(bins) and bins.is_cuda
    assert torch.equal(bins, expected)


def test_fit_bins_feature_blocks_match(monkeypatch):
    X, _ = _data(2, f=7)
    full = fit_bins(X, NUM_BINS)
    per_column = torch.stack(
        [
            torch.nanquantile(
                torch.as_tensor(X[:, f], device="cuda"),
                torch.linspace(0, 1, NUM_BINS + 1, device="cuda")[1:-1],
            )
            for f in range(X.shape[1])
        ]
    )
    assert torch.equal(full, torch.nan_to_num(per_column, nan=0.0))
    monkeypatch.setattr(binning, "_QUANTILE_BLOCK_NUMEL", 2 * X.shape[0])  # 2 features per block
    assert torch.equal(fit_bins(X, NUM_BINS), full)


def test_apply_bins_validation():
    X, _ = _data()
    edges = fit_bins(X, NUM_BINS)
    with pytest.raises(ValueError, match="features"):
        apply_bins(X[:, :2], edges)
    with pytest.raises(ValueError, match="sorted"):
        apply_bins(X, edges.flip(1))
    with pytest.raises(ValueError, match="columns"):
        apply_bins(X, torch.zeros((X.shape[1], 200), device="cuda"))
    with pytest.raises(ValueError, match="num_bins"):
        fit_bins(X, 1)
    with pytest.raises(ValueError, match="CUDA"):
        fit_bins(X, NUM_BINS, device="cpu")


# ─── prebinned=True ───


def test_prebinned_model_equals_raw_model():
    """Bins from apply_bins (with -1) train exactly the model that the raw features train."""
    X, y = _data(3)
    X_tr, X_ev, y_tr, y_ev = X[:2400], X[2400:], y[:2400], y[2400:]
    fit_kwargs = dict(y_eval=y_ev, eval_every_n_trees=3, early_stopping_rounds=3)

    raw = _model(colsample_bytree=0.75).fit(X_tr, y_tr, X_eval=X_ev, **fit_kwargs)

    edges = fit_bins(X_tr, NUM_BINS)
    B_tr, B_ev = apply_bins(X_tr, edges), apply_bins(X_ev, edges)
    assert (B_tr == MISSING_BIN).any()
    pre = _model(colsample_bytree=0.75, prebinned=True).fit(B_tr, y_tr, X_eval=B_ev, **fit_kwargs)

    assert pre.prebinned_ and pre.bin_edges is None
    assert pre.forest == raw.forest
    assert pre.evals_result_ == raw.evals_result_
    assert pre.best_iteration_ == raw.best_iteration_
    np.testing.assert_array_equal(pre.predict(B_ev), raw.predict(X_ev))
    np.testing.assert_array_equal(pre.predict_raw(B_ev), raw.predict_raw(X_ev))
    np.testing.assert_array_equal(pre.train_predictions_, raw.train_predictions_)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64, np.uint8])
def test_prebinned_integer_dtypes(dtype):
    X, y = _data(4, nan_frac=0.0 if dtype == np.uint8 else 0.15)
    bins = apply_bins(X, fit_bins(X, NUM_BINS))
    ref = _model(prebinned=True).fit(bins, y)
    model = _model(prebinned=True).fit(bins.astype(dtype), y)
    assert model.forest == ref.forest
    np.testing.assert_array_equal(model.predict(bins.astype(dtype)), ref.predict(bins))


def test_prebinned_classifier_predict_proba():
    X, _ = _data(5)
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int) + 2 * np.isnan(X[:, 1])
    edges = fit_bins(X, NUM_BINS)
    raw = _model(objective="multiclass").fit(X, y)
    pre = _model(objective="multiclass", prebinned=True).fit(apply_bins(X, edges), y)
    bins = apply_bins(X, edges)
    np.testing.assert_array_equal(pre.predict_proba(bins), raw.predict_proba(X))
    np.testing.assert_array_equal(pre.predict(bins), raw.predict(X))


def test_prebinned_rejects_floats():
    X, y = _data(6)
    bins = apply_bins(X, fit_bins(X, NUM_BINS))
    with pytest.raises(TypeError, match="integer"):
        _model(prebinned=True).fit(np.nan_to_num(X), y)

    model = _model(prebinned=True)
    with pytest.raises(TypeError, match="X_eval"):
        model.fit(bins, y, X_eval=bins.astype(np.float32), y_eval=y, eval_every_n_trees=5)

    model = _model(prebinned=True).fit(bins, y)
    for method in (model.predict, model.predict_raw):
        with pytest.raises(TypeError, match="integer"):
            method(bins.astype(np.float32))


@pytest.mark.parametrize(
    "bad_value, dtype",
    [(NUM_BINS, np.int8), (-2, np.int8), (NUM_BINS, np.int64), (300, np.int16), (-129, np.int32)],
)
def test_prebinned_rejects_out_of_range(bad_value, dtype):
    X, y = _data(7)
    bins = apply_bins(X, fit_bins(X, NUM_BINS)).astype(dtype)
    bad = bins.copy()
    bad[10, 2] = bad_value  # e.g. 300 would wrap to 44 when narrowed to int8
    with pytest.raises(ValueError, match="column 2"):
        _model(prebinned=True).fit(bad, y)
    model = _model(prebinned=True).fit(bins, y)
    with pytest.raises(ValueError, match="column 2"):
        model.predict(bad)


def test_prebinned_feature_count_checked():
    X, y = _data(8)
    bins = apply_bins(X, fit_bins(X, NUM_BINS))
    model = _model(prebinned=True).fit(bins, y)
    with pytest.raises(ValueError, match="shape"):
        model.predict(bins[:, :3])


def test_prebinned_save_load(tmp_path):
    X, y = _data(9)
    bins = apply_bins(X, fit_bins(X, NUM_BINS))
    model = _model(prebinned=True).fit(bins, y)
    path = tmp_path / "model.pkl"
    model.save_model(path)

    loaded = WarpGBM(verbose=False).load_model(path)
    assert loaded.prebinned is True and loaded.prebinned_ is True and loaded.bin_edges is None
    np.testing.assert_array_equal(loaded.predict(bins), model.predict(bins))
    with pytest.raises(TypeError, match="integer"):
        loaded.predict(X)


def test_prebinned_warm_start():
    X, y = _data(10)
    bins = apply_bins(X, fit_bins(X, NUM_BINS))
    full = _model(prebinned=True, n_estimators=20).fit(bins, y)

    model = _model(prebinned=True, n_estimators=10, warm_start=True).fit(bins, y)
    model.n_estimators = 20
    model.fit(bins, y)
    assert model.num_trees_ == 20
    np.testing.assert_allclose(model.predict(bins), full.predict(bins), rtol=1e-5, atol=1e-5)

    model.n_estimators = 25
    model.prebinned = False
    with pytest.raises(ValueError, match="prebinned"):
        model.fit(X, y)


# ─── prebinned=None (auto) / False ───


def test_auto_detection_is_recorded():
    X, y = _data(11, nan_frac=0.0)
    bins = apply_bins(X, fit_bins(X, NUM_BINS))
    model = _model().fit(bins, y)
    assert model.prebinned_ and model.bin_edges is None
    # The decision is made at fit time: float input to the auto-detected model raises
    with pytest.raises(TypeError, match="integer"):
        model.predict(bins.astype(np.float32))


def test_auto_detection_warns_on_missing_bins():
    X, y = _data(12)
    bins = apply_bins(X, fit_bins(X, NUM_BINS))  # contains -1
    with pytest.warns(UserWarning, match="prebinned=True"):
        model = _model().fit(bins, y)
    assert not model.prebinned_ and model.bin_edges is not None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = _model(prebinned=False).fit(bins, y)
    assert not explicit.prebinned_


def test_raw_model_bins_small_integers_with_its_edges():
    """No predict-time heuristic: small integer input to a raw-feature model is binned with its edges."""
    rng = np.random.default_rng(13)
    X = rng.normal(scale=3.0, size=(3000, 3)).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)
    model = _model(prebinned=False).fit(X, y)

    X_int = np.clip(np.round(X[:200]), 0, NUM_BINS - 1).astype(np.int8)
    np.testing.assert_array_equal(model.predict(X_int), model.predict(X_int.astype(np.float32)))


def test_invalid_prebinned_param():
    with pytest.raises(TypeError, match="prebinned"):
        WarpGBM(prebinned="yes")


# ─── record_training_loss ───


def _fit_with_eval(record, n_estimators=13, every=4):
    X, y = _data(14)
    model = _model(record_training_loss=record, n_estimators=n_estimators)
    return model.fit(
        X[:2400], y[:2400], X_eval=X[2400:], y_eval=y[2400:], eval_every_n_trees=every
    )


def test_record_training_loss_modes():
    every_round = _fit_with_eval("all")
    at_evals = _fit_with_eval("eval")
    none = _fit_with_eval("none")

    assert len(every_round.training_loss) == 13
    assert at_evals.eval_iterations_ == [1, 5, 9, 13]
    expected = [every_round.training_loss[i - 1] for i in at_evals.eval_iterations_]
    np.testing.assert_allclose(at_evals.training_loss, expected, rtol=1e-6)
    assert none.training_loss == []

    # Training is unaffected
    assert at_evals.forest == every_round.forest == none.forest
    assert at_evals.evals_result_ == every_round.evals_result_ == none.evals_result_


def test_record_training_loss_with_aux_metric():
    """A metric that needs aux has no training score; the loss is computed at checkpoints."""

    def metric(y_true, pred, weight, mask, aux):
        return float(((pred - aux) ** 2).mean())

    X, y = _data(15)
    model = _model(record_training_loss="eval").fit(
        X[:2400], y[:2400], X_eval=X[2400:], y_eval=y[2400:], eval_every_n_trees=5,
        eval_metric=metric, eval_aux=y[2400:],
    )
    assert model.evals_result_["train"] == []
    assert model.eval_iterations_ == [1, 6, 11, 15]
    assert len(model.training_loss) == 4 and all(np.isfinite(model.training_loss))


def test_invalid_record_training_loss():
    with pytest.raises(ValueError, match="record_training_loss"):
        WarpGBM(record_training_loss="sometimes")
