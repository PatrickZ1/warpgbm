"""Best-iteration tracking, rollback, num_trees, importances, metric aux data, train predictions."""

import os
import tempfile

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from warpgbm import Metric, WarpGBM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

COMMON = dict(max_depth=3, num_bins=16, learning_rate=0.2, random_state=0)


def _regression(seed=0, n_train=2000, n_eval=700, n_targets=1):
    X, y = make_regression(
        n_samples=n_train + n_eval, n_features=8, n_targets=n_targets, noise=5.0, random_state=seed
    )
    X, y = X.astype(np.float32), y.astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


class Scripted:
    """Metric returning preset values on the eval set (the train set has a different size)."""

    def __init__(self, values, n_eval):
        self.values, self.n_eval, self.calls = list(values), n_eval, 0

    def __call__(self, y_true, pred, weight, mask):
        if len(y_true) != self.n_eval:
            return 0.0
        value = self.values[self.calls]
        self.calls += 1
        return value


def _fit_scripted(values, metric_wrapper=None, early_stopping_rounds=2, n_estimators=None, **kw):
    X, y, Xe, ye = _regression()
    fn = Scripted(values, len(ye))
    metric = fn if metric_wrapper is None else metric_wrapper(fn)
    model = WarpGBM(n_estimators=n_estimators or len(values), **dict(COMMON, **kw))
    model.fit(
        X, y, X_eval=Xe, y_eval=ye, eval_every_n_trees=1,
        early_stopping_rounds=early_stopping_rounds, eval_metric=metric,
    )
    return model, (X, y, Xe, ye)


# ─── best iteration / early stopping ─────────────────────────────────────────


def test_minimized_metric_best_iteration_and_stopping_rule():
    # best 3.0 at round 3; then two evals without improvement -> stop after round 5
    model, _ = _fit_scripted([5.0, 4.0, 3.0, 3.5, 3.2, 1.0, 0.5])
    assert model.best_iteration_ == 3
    assert model.best_score_ == 3.0
    assert model._trees_trained == model.num_trees_ == 5
    assert model.eval_iterations_ == [1, 2, 3, 4, 5]
    assert model.evals_result_["eval"] == [5.0, 4.0, 3.0, 3.5, 3.2]
    assert model.eval_loss is model.evals_result_["eval"]


def test_maximized_metric_with_metric_wrapper():
    wrap = lambda fn: Metric(fn, higher_is_better=True, name="score")
    model, _ = _fit_scripted([0.1, 0.3, 0.2, 0.25, 0.1, 0.9], wrap, early_stopping_rounds=3)
    assert (model.best_iteration_, model.best_score_, model._trees_trained) == (2, 0.3, 5)
    assert model.eval_metric_name_ == "score" and model.eval_metric_higher_is_better_

    # a later improvement resets the patience; ties keep the earlier round
    model, _ = _fit_scripted([0.1, 0.3, 0.2, 0.3, 0.5, 0.4], wrap, early_stopping_rounds=3)
    assert (model.best_iteration_, model.best_score_, model._trees_trained) == (5, 0.5, 6)
    model, _ = _fit_scripted([0.3, 0.3, 0.3], wrap, early_stopping_rounds=5)
    assert model.best_iteration_ == 1


def test_nan_scores_never_become_best():
    model, _ = _fit_scripted([float("nan"), 2.0, float("nan"), 3.0], early_stopping_rounds=5)
    assert model.best_iteration_ == 2


def test_default_predictions_use_best_iteration_after_early_stopping():
    model, (X, y, Xe, ye) = _fit_scripted([5.0, 4.0, 3.0, 3.5, 3.2, 1.0])
    best, total = model.best_iteration_, model._trees_trained
    np.testing.assert_array_equal(model.predict(Xe), model.predict(Xe, num_trees=best))
    np.testing.assert_array_equal(model.predict_raw(Xe), model.predict_raw(Xe, num_trees=best))
    assert not np.allclose(model.predict(Xe), model.predict(Xe, num_trees=total))
    np.testing.assert_allclose(model.predict(Xe, num_trees=0), model.base_prediction)


def test_eval_set_without_early_stopping_keeps_all_trees():
    X, y, Xe, ye = _regression(1)
    model = WarpGBM(n_estimators=30, **COMMON).fit(X, y, X_eval=Xe, y_eval=ye, eval_every_n_trees=4)
    assert model.best_iteration_ is not None
    assert model.eval_iterations_[-1] == 30  # the last round is always evaluated
    np.testing.assert_array_equal(model.predict(Xe), model.predict(Xe, num_trees=30))


def test_natural_values_and_direction_of_builtin_metrics():
    X, y = make_classification(n_samples=3000, n_features=10, n_informative=4, random_state=0)
    X = X.astype(np.float32)
    Xt, yt, Xe, ye = X[:2000], y[:2000], X[2000:], y[2000:]
    for metric, best_fn in [("accuracy", max), ("logloss", min)]:
        model = WarpGBM(objective="binary", n_estimators=40, **COMMON)
        model.fit(Xt, yt, X_eval=Xe, y_eval=ye, eval_every_n_trees=3,
                  early_stopping_rounds=50, eval_metric=metric)
        scores = model.evals_result_["eval"]
        assert model.best_score_ == best_fn(scores)
        assert model.best_iteration_ == model.eval_iterations_[scores.index(best_fn(scores))]
        np.testing.assert_array_equal(
            model.predict_proba(Xe), model.predict_proba(Xe, num_trees=model.best_iteration_)
        )
    model = WarpGBM(objective="binary", n_estimators=10, **COMMON)
    model.fit(Xt, yt, X_eval=Xe, y_eval=ye, eval_every_n_trees=1, eval_metric="accuracy")
    assert 0.5 < model.evals_result_["eval"][-1] <= 1.0  # accuracy, not 1 - accuracy

    Xr, yr, Xre, yre = _regression(2)
    model = WarpGBM(n_estimators=20, **COMMON)
    model.fit(Xr, yr, X_eval=Xre, y_eval=yre, eval_every_n_trees=1, eval_metric="corr")
    assert model.evals_result_["eval"][-1] > 0.5
    assert model.best_score_ == max(model.evals_result_["eval"])


def test_num_trees_validation():
    model, (X, y, Xe, ye) = _fit_scripted([3.0, 2.0, 1.0])
    for bad in (-1, 4):
        with pytest.raises(ValueError, match="num_trees"):
            model.predict(Xe, num_trees=bad)
    with pytest.raises(TypeError, match="num_trees"):
        model.predict(Xe, num_trees=1.5)


# ─── feature importance ──────────────────────────────────────────────────────


def _importance_by_walk(model, n):
    totals = np.zeros(model.num_features)
    stack = [t for t in model.forest[:n] if t]
    while stack:
        node = stack.pop()
        if "leaf_value" not in node:
            totals[node["feature"]] += node["gain"]
            stack += [node["left"], node["right"]]
    return totals


def test_feature_importance_follows_the_used_trees():
    X, y, Xe, ye = _regression(3)
    era = np.repeat(np.arange(4), len(y) // 4).astype(np.int32)
    values = [5.0, 4.0, 3.0] + [3.5] * 10
    model = WarpGBM(n_estimators=13, **COMMON)
    model.fit(X, y, era_id=era, X_eval=Xe, y_eval=ye, eval_every_n_trees=1,
              early_stopping_rounds=10, eval_metric=Scripted(values, len(ye)))
    best, total = model.best_iteration_, model._trees_trained
    assert (best, total) == (3, 13)

    imp_best = model.get_feature_importance(normalize=False)
    np.testing.assert_allclose(imp_best, _importance_by_walk(model, best), rtol=1e-5)
    np.testing.assert_allclose(imp_best, model.get_per_era_feature_importance(normalize=False).sum(0), rtol=1e-4)
    np.testing.assert_allclose(model.feature_importance_, imp_best)
    imp_all = model.get_feature_importance(normalize=False, num_trees=total)
    np.testing.assert_allclose(
        imp_all, model.get_per_era_feature_importance(normalize=False, num_trees=total).sum(0), rtol=1e-4
    )
    assert imp_all.sum() > imp_best.sum()
    np.testing.assert_allclose(model.get_feature_importance(num_trees=5, normalize=False), _importance_by_walk(model, 5), rtol=1e-5)
    assert model.get_per_era_feature_importance().shape == (4, X.shape[1])
    with pytest.raises(ValueError, match="Per-era feature importance"):
        model.get_per_era_feature_importance(num_trees=5)

    # trees without stored gains (older models): only the all-trees snapshot is available
    for node_tree in model.forest[:total]:
        stack = [node_tree]
        while stack:
            node = stack.pop()
            if "leaf_value" not in node:
                node.pop("gain")
                stack += [node["left"], node["right"]]
    np.testing.assert_allclose(model.get_feature_importance(num_trees=total, normalize=False), imp_all, rtol=1e-4)
    with pytest.raises(ValueError, match="per-split gains"):
        model.get_feature_importance(num_trees=5)


# ─── aux data for callable metrics ───────────────────────────────────────────


def test_eval_and_train_aux_reach_the_metric():
    X, y, Xe, ye = _regression(4)
    eval_aux = np.column_stack([ye, np.arange(len(ye))]).astype(np.float32)
    train_aux = np.arange(len(y), dtype=np.float32)
    seen = []

    def metric(y_true, pred, weight, mask, aux):
        seen.append((len(y_true), tuple(aux.shape), aux.device.type, float(aux.reshape(len(y_true), -1)[-1, -1])))
        return float(torch.mean((pred - y_true) ** 2))

    model = WarpGBM(n_estimators=4, **COMMON)
    model.fit(X, y, X_eval=Xe, y_eval=ye, eval_every_n_trees=2, eval_metric=metric,
              eval_aux=eval_aux, train_aux=train_aux)
    assert (len(ye), (len(ye), 2), "cuda", len(ye) - 1) in seen
    assert (len(y), (len(y),), "cuda", len(y) - 1) in seen
    assert len(model.evals_result_["train"]) == len(model.evals_result_["eval"]) == 3

    # required aux without train_aux: the train metric is skipped
    model.fit(X, y, X_eval=Xe, y_eval=ye, eval_every_n_trees=2, eval_metric=metric, eval_aux=eval_aux)
    assert model.evals_result_["train"] == [] and len(model.evals_result_["eval"]) == 3

    # optional aux: called without it where none is given
    def optional(y_true, pred, weight, mask, aux=None):
        return 0.0 if aux is None else 1.0

    model.fit(X, y, X_eval=Xe, y_eval=ye, eval_every_n_trees=2, eval_metric=optional, eval_aux=eval_aux)
    assert set(model.evals_result_["train"]) == {0.0} and set(model.evals_result_["eval"]) == {1.0}


def test_aux_validation():
    X, y, Xe, ye = _regression(5, n_train=300, n_eval=100)
    aux = np.zeros(len(ye), dtype=np.float32)
    fit = lambda **kw: WarpGBM(n_estimators=2, **COMMON).fit(X, y, eval_every_n_trees=1, **kw)
    no_aux = lambda y_true, pred, weight, mask: 0.0
    with_aux = lambda y_true, pred, weight, mask, aux: 0.0
    with pytest.raises(ValueError, match="callable eval_metric"):
        fit(X_eval=Xe, y_eval=ye, eval_metric="mse", eval_aux=aux)
    with pytest.raises(ValueError, match="does not accept"):
        fit(X_eval=Xe, y_eval=ye, eval_metric=no_aux, eval_aux=aux)
    with pytest.raises(ValueError, match="same number of rows"):
        fit(X_eval=Xe, y_eval=ye, eval_metric=with_aux, eval_aux=aux[:-1])
    with pytest.raises(ValueError, match="train_aux"):
        WarpGBM(n_estimators=2, **COMMON).fit(X, y, eval_metric=with_aux, train_aux=np.zeros(len(y)))
    with pytest.raises(TypeError, match="numpy array"):
        fit(X_eval=Xe, y_eval=ye, eval_metric=with_aux, eval_aux=[0.0] * len(ye))


def test_early_stopping_on_competition_metric_with_scaled_targets():
    # train MSE on y / sigma, early-stop on Pearson(pred * sigma, clip(y_raw, +-10 sigma))
    rng = np.random.default_rng(6)
    n = 6000
    X = rng.normal(size=(n, 6)).astype(np.float32)
    sigma = np.exp(rng.normal(size=n)).astype(np.float32)
    y_raw = (sigma * (X[:, 0] + 0.5 * np.tanh(X[:, 1]) + rng.standard_t(3, size=n))).astype(np.float32)
    tr, ev = slice(0, 4000), slice(4000, n)

    def comp_corr(y_true, pred, weight, mask, aux):
        y, s = aux[:, 0], aux[:, 1]
        a, b = pred * s, torch.clamp(y, -10 * s, 10 * s)
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))

    model = WarpGBM(n_estimators=300, **dict(COMMON, learning_rate=0.3))
    model.fit(
        X[tr], y_raw[tr] / sigma[tr], X_eval=X[ev], y_eval=y_raw[ev] / sigma[ev],
        eval_aux=np.column_stack([y_raw[ev], sigma[ev]]), eval_every_n_trees=5,
        early_stopping_rounds=4, eval_metric=Metric(comp_corr, higher_is_better=True),
    )
    assert model.best_score_ == max(model.evals_result_["eval"]) > 0.2
    assert model.evals_result_["train"] == []  # needs aux, no train_aux given
    assert model._trees_trained < 300 and model.best_iteration_ < model._trees_trained
    pred = model.predict(X[ev]) * sigma[ev]
    clipped = np.clip(y_raw[ev], -10 * sigma[ev], 10 * sigma[ev])
    np.testing.assert_allclose(np.corrcoef(pred, clipped)[0, 1], model.best_score_, atol=1e-4)


# ─── train predictions ───────────────────────────────────────────────────────


@pytest.mark.parametrize("n_targets", [1, 3])
def test_train_predictions_match_predict_raw(n_targets):
    X, y, Xe, ye = _regression(7, n_targets=n_targets)
    model = WarpGBM(objective="regression" if n_targets == 1 else "multi_regression",
                    n_estimators=60, **dict(COMMON, learning_rate=0.5))
    model.fit(X, y, X_eval=Xe, y_eval=ye, eval_every_n_trees=2, early_stopping_rounds=3)
    assert model.train_predictions_.shape == (len(y), n_targets)
    expected = model.predict_raw(X).reshape(len(y), -1)
    scale = np.abs(expected).max()
    np.testing.assert_allclose(model.train_predictions_, expected, atol=1e-5 * scale)

    model.fit(X, y)  # no eval set: all trees
    np.testing.assert_allclose(model.train_predictions_, model.predict_raw(X).reshape(len(y), -1), atol=1e-5 * scale)


# ─── persistence / warm start ────────────────────────────────────────────────


def test_save_load_keeps_best_iteration_and_default_model():
    model, (X, y, Xe, ye) = _fit_scripted([5.0, 4.0, 3.0, 3.5, 3.2])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        path = tmp.name
    try:
        model.save_model(path)
        loaded = WarpGBM().load_model(path)
    finally:
        os.remove(path)
    assert (loaded.best_iteration_, loaded.best_score_) == (3, 3.0)
    assert loaded.evals_result_ == model.evals_result_ and loaded.eval_iterations_ == model.eval_iterations_
    np.testing.assert_array_equal(loaded.predict(Xe), model.predict(Xe))
    np.testing.assert_allclose(loaded.get_feature_importance(), model.get_feature_importance())
    np.testing.assert_allclose(loaded.get_per_era_feature_importance(), model.get_per_era_feature_importance())


def test_warm_start_continues_after_all_trees_or_after_truncate():
    model, (X, y, Xe, ye) = _fit_scripted([5.0, 4.0, 3.0, 3.5, 3.2], warm_start=True)
    assert (model.best_iteration_, model._trees_trained) == (3, 5)
    first_rounds = [model.forest[r] for r in range(5)]

    # continue after all trained trees; without an eval set the model is all trees
    model.n_estimators = 8
    model.fit(X, y)
    assert model._trees_trained == 8 and model.best_iteration_ is None
    assert all(model.forest[r] is first_rounds[r] for r in range(5))
    np.testing.assert_array_equal(model.predict(Xe), model.predict(Xe, num_trees=8))

    # truncate to the best round, then continue from there
    model2, _ = _fit_scripted([5.0, 4.0, 3.0, 3.5, 3.2], warm_start=True)
    best_pred = model2.predict(Xe)
    model2.truncate()
    assert model2._trees_trained == 3 and all(not t for t in model2.forest[3:])
    np.testing.assert_array_equal(model2.predict(Xe), best_pred)
    assert model2.get_per_era_feature_importance().shape == (1, X.shape[1])
    model2.n_estimators = 6
    model2.fit(X, y)
    assert model2._trees_trained == 6
    assert all(model2.forest[r] for r in range(6))

    with pytest.raises(ValueError, match="best_iteration_"):
        WarpGBM(n_estimators=2, **COMMON).fit(X, y).truncate()
