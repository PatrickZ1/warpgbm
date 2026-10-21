"""Custom objectives: autograd losses, explicit grad/hess, multi-output, NaN targets, weights, eras."""

import os
import tempfile

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression

from warpgbm import CustomObjective, Objective, WarpGBM
from warpgbm.objectives import Softmax, autograd_grad_hess

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

DEV = "cuda"


class MyMSE(Objective):
    """Squared error with explicit derivatives (module level -> picklable)."""

    name = "my_mse"

    def grad_hess(self, y, pred):
        return pred - y, torch.ones_like(pred)

    def loss(self, y, pred):
        return (pred - y) ** 2


class AutogradMSE(Objective):
    """Squared error from the loss alone; grad/hess come from autograd."""

    def loss(self, y, pred):
        return 0.5 * (pred - y) ** 2


class PseudoHuber(Objective):
    name = "pseudo_huber"

    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, y, pred):
        return self.delta**2 * (torch.sqrt(1.0 + ((pred - y) / self.delta) ** 2) - 1.0)

    def init_score(self, y, weight):
        return y.median(dim=0).values


def half_mse(y, pred):
    return 0.5 * (pred - y) ** 2


def log_cosh(y, pred):
    r = pred - y
    return r + torch.nn.functional.softplus(-2.0 * r) - np.log(2.0)


def _regression_data(seed=0, n=2000, n_targets=1):
    X, y = make_regression(n_samples=n, n_features=8, n_targets=n_targets, noise=0.5, random_state=seed)
    return X.astype(np.float32), y.astype(np.float32)


COMMON = dict(n_estimators=30, max_depth=4, num_bins=32, learning_rate=0.1, random_state=0)


# ─── autograd derivatives ────────────────────────────────────────────────────


def test_autograd_hessian_is_the_exact_diagonal_also_for_row_coupled_losses():
    # softmax cross-entropy couples the outputs of a row: the autograd diagonal must be p (1 - p)
    torch.manual_seed(0)
    pred = torch.randn(500, 4, device=DEV)
    y = torch.randint(0, 4, (500, 1), device=DEV).float()

    def ce(y, p):  # per-row loss, shape [N]
        return torch.nn.functional.cross_entropy(p, y[:, 0].long(), reduction="none")

    g, h = autograd_grad_hess(ce, y, pred)
    g_ref, h_ref = Softmax().grad_hess(y, pred)
    torch.testing.assert_close(g, g_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(h, h_ref, rtol=1e-5, atol=1e-6)

    g1, h1 = autograd_grad_hess(ce, y, pred, hessian="ones")
    torch.testing.assert_close(g1, g_ref, rtol=1e-5, atol=1e-6)
    assert (h1 == 1).all()


def test_autograd_rejects_non_per_sample_losses():
    X, y = _regression_data(13, n=300)
    with pytest.raises(ValueError, match="per-sample"):
        WarpGBM(objective=lambda y, p: ((p - y) ** 2).mean(), n_estimators=2).fit(X, y)


# ─── engine integration ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "objective",
    [MyMSE(), AutogradMSE(), CustomObjective(half_mse), half_mse],
    ids=["explicit", "autograd_subclass", "custom_objective", "plain_function"],
)
def test_custom_mse_matches_builtin_regression(objective):
    X, y = _regression_data()
    builtin = WarpGBM(objective="regression", **COMMON).fit(X, y)
    custom = WarpGBM(objective=objective, **COMMON).fit(X, y)
    assert custom.predict(X).shape == (X.shape[0],)
    np.testing.assert_allclose(custom.predict(X), builtin.predict(X), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("objective", [MyMSE(), AutogradMSE()], ids=["explicit", "autograd"])
def test_multi_output_custom_with_nan_targets_and_2d_weights_matches_builtin(objective):
    X, Y = _regression_data(5, n_targets=3)
    rng = np.random.default_rng(5)
    Y[rng.random(Y.shape) < 0.25] = np.nan
    W = rng.uniform(0.5, 2.0, size=Y.shape).astype(np.float32)

    builtin = WarpGBM(objective="multi_regression", **COMMON).fit(X, Y, sample_weight=W)
    custom = WarpGBM(objective=objective, **COMMON).fit(X, Y, sample_weight=W)

    preds = custom.predict(X)
    assert preds.shape == Y.shape
    assert np.isfinite(preds).all()
    np.testing.assert_allclose(preds, builtin.predict(X), rtol=1e-4, atol=1e-3)


def test_pseudo_huber_is_robust_to_outliers():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, 4)).astype(np.float32)
    clean = (3.0 * X[:, 0]).astype(np.float32)
    y = clean + 0.1 * rng.normal(size=4000).astype(np.float32)
    outliers = rng.random(4000) < 0.05
    y[outliers] += 200.0

    params = dict(n_estimators=80, max_depth=3, num_bins=32, learning_rate=0.2, random_state=0)
    mse_model = WarpGBM(objective="regression", **params).fit(X, y)
    huber_model = WarpGBM(objective=PseudoHuber(delta=1.0), L2_reg=1.0, **params).fit(X, y)

    err_mse = np.median(np.abs(mse_model.predict(X) - clean))
    err_huber = np.median(np.abs(huber_model.predict(X) - clean))
    assert err_huber < 0.5 * err_mse


def test_quantile_objective_with_constant_hessian():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6000, 3)).astype(np.float32)
    y = (X[:, 0] + (1.0 + np.abs(X[:, 1])) * rng.normal(size=6000)).astype(np.float32)
    q = 0.9

    def pinball(y, p):
        r = y - p
        return torch.maximum(q * r, (q - 1.0) * r)

    obj = CustomObjective(
        pinball,
        hessian="ones",
        init_fn=lambda y, weight: torch.quantile(y[:, 0], q).reshape(1),
    )
    model = WarpGBM(objective=obj, n_estimators=100, max_depth=3, num_bins=32, learning_rate=0.5, random_state=0)
    model.fit(X, y)
    coverage = np.mean(y <= model.predict(X))
    assert 0.85 < coverage < 0.95

    loss_model = np.mean(np.maximum(q * (y - model.predict(X)), (q - 1) * (y - model.predict(X))))
    const = np.quantile(y, q)
    loss_const = np.mean(np.maximum(q * (y - const), (q - 1) * (y - const)))
    assert loss_model < 0.9 * loss_const


def test_autograd_handles_nan_targets_and_weights():
    X, Y = _regression_data(6, n_targets=2)
    Y = Y / Y.std(axis=0)  # log-cosh has ~zero curvature for huge residuals
    Y_nan = Y.copy()
    Y_nan[np.random.default_rng(6).random(Y.shape) < 0.3] = np.nan
    W = np.random.default_rng(7).uniform(0.5, 2.0, size=len(Y)).astype(np.float32)

    # NaN targets are filled with 0 before the loss sees them and masked afterwards
    model = WarpGBM(objective=log_cosh, L2_reg=1.0, **dict(COMMON, n_estimators=60)).fit(X, Y_nan, sample_weight=W)
    preds = model.predict(X)
    observed = np.isfinite(Y_nan)
    assert np.isfinite(preds).all()
    assert np.mean((preds[observed] - Y[observed]) ** 2) < 0.2 * np.nanvar(Y)


def test_per_output_strategy_matches_independent_models():
    X, Y = _regression_data(7, n_targets=2)
    model = WarpGBM(objective=CustomObjective(half_mse, output_strategy="per_output"), **COMMON).fit(X, Y)
    assert len(model.forest) == COMMON["n_estimators"]
    assert len(model.forest[0]) == 2

    preds = model.predict(X)
    for k in range(2):
        single = WarpGBM(objective="regression", **COMMON).fit(X, Y[:, k])
        np.testing.assert_allclose(preds[:, k], single.predict(X), rtol=1e-4, atol=1e-3)


def test_custom_objective_with_eras_matches_builtin():
    X, y = _regression_data(8)
    era = (np.arange(len(y)) % 5).astype(np.int32)
    builtin = WarpGBM(objective="regression", **COMMON).fit(X, y, era_id=era)
    custom = WarpGBM(objective=AutogradMSE(), **COMMON).fit(X, y, era_id=era)
    np.testing.assert_allclose(custom.predict(X), builtin.predict(X), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("explicit", [True, False])
def test_poisson_with_transform_and_init(explicit):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(3000, 3)).astype(np.float32)
    rate = np.exp(0.5 * X[:, 0])
    y = rng.poisson(rate).astype(np.float32)

    # Poisson regression with log link: L = exp(s) - y s, grad = exp(s) - y, hess = exp(s)
    obj = CustomObjective(
        loss_fn=lambda y, p: torch.exp(p) - y * p,
        grad_hess_fn=(lambda y, p: (torch.exp(p) - y, torch.exp(p))) if explicit else None,
        init_fn=lambda y, weight: torch.log(y.mean(dim=0)),
        transform_fn=torch.exp,
        name="poisson",
    )
    model = WarpGBM(objective=obj, n_estimators=60, max_depth=3, num_bins=32, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert (pred > 0).all()
    assert np.corrcoef(np.log(pred), 0.5 * X[:, 0])[0, 1] > 0.9
    np.testing.assert_allclose(np.log(pred), model.predict_raw(X), rtol=1e-5, atol=1e-5)


def test_callable_eval_metric_and_loss_metric():
    X, y = _regression_data(11)
    X_tr, X_ev, y_tr, y_ev = X[:1500], X[1500:], y[:1500], y[1500:]

    def mae(y_true, pred, weight, mask):
        return float((torch.abs(y_true - pred) * weight * mask).sum() / (weight * mask).sum())

    # 31 trees: evaluations run after trees 1, 6, ..., 31, so the last one sees the final model
    model = WarpGBM(objective=MyMSE(), **dict(COMMON, n_estimators=31))
    model.fit(X_tr, y_tr, X_eval=X_ev, y_eval=y_ev, eval_every_n_trees=5, eval_metric=mae)
    assert len(model.eval_loss) == 7
    np.testing.assert_allclose(model.eval_loss[-1], np.mean(np.abs(model.predict(X_ev) - y_ev)), rtol=1e-3)
    assert model.eval_loss[-1] < model.eval_loss[0]

    # Default metric of a custom objective is its loss
    model.fit(X_tr, y_tr, X_eval=X_ev, y_eval=y_ev, eval_every_n_trees=5)
    assert model.eval_metric == "loss"
    np.testing.assert_allclose(model.eval_loss[-1], np.mean((model.predict(X_ev) - y_ev) ** 2), rtol=1e-3)


# ─── warm start / persistence ────────────────────────────────────────────────


def test_custom_objective_warm_start_and_save_load():
    X, y = _regression_data(10)
    params = dict(COMMON, n_estimators=40)

    full = WarpGBM(objective=MyMSE(), **params).fit(X, y)

    warm = WarpGBM(objective=MyMSE(), **dict(params, n_estimators=20, warm_start=True))
    warm.fit(X, y)
    warm.n_estimators = 40
    warm.fit(X, y)
    np.testing.assert_allclose(warm.predict(X), full.predict(X), rtol=1e-4, atol=1e-3)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        path = tmp.name
    try:
        full.save_model(path)
        loaded = WarpGBM().load_model(path)
        assert isinstance(loaded.objective_, MyMSE)
        np.testing.assert_allclose(loaded.predict(X), full.predict(X), rtol=1e-4, atol=1e-4)
    finally:
        os.remove(path)


def test_warm_start_keeps_the_objective_and_rejects_switching():
    X, Y = _regression_data(9, n_targets=2)
    Y = Y / Y.std(axis=0)
    model = WarpGBM(objective=log_cosh, warm_start=True, **dict(COMMON, n_estimators=20))
    model.fit(X, Y)
    model.n_estimators = 40
    model.fit(X, Y)  # same (module-level) loss function: continues
    assert model._trees_trained == 40
    assert isinstance(model.objective_, CustomObjective)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        path = tmp.name
    try:
        model.save_model(path)
        loaded = WarpGBM().load_model(path)
        np.testing.assert_allclose(loaded.predict(X), model.predict(X), rtol=1e-4, atol=1e-4)
        loaded.warm_start = True
        loaded.n_estimators = 50
        loaded.fit(X, Y)
        assert loaded._trees_trained == 50
    finally:
        os.remove(path)

    # changing the objective while continuing is not allowed
    model.objective = "multi_regression"
    model.n_estimators = 60
    with pytest.raises(ValueError, match="objective cannot be changed"):
        model.fit(X, Y)


def test_objective_errors_are_reported():
    X, y = _regression_data(13, n=300)

    nan_obj = CustomObjective(grad_hess_fn=lambda y, p: (p * float("nan"), torch.ones_like(p)))
    with pytest.raises(FloatingPointError):
        WarpGBM(objective=nan_obj, n_estimators=2).fit(X, y)

    bad_shape = CustomObjective(grad_hess_fn=lambda y, p: (torch.zeros(3), torch.ones(3)))
    with pytest.raises(ValueError, match="expected"):
        WarpGBM(objective=bad_shape, n_estimators=2).fit(X, y)

    no_loss = CustomObjective(grad_hess_fn=lambda y, p: (p - y, torch.ones_like(p)))
    with pytest.raises(ValueError, match="loss"):
        WarpGBM(objective=no_loss, n_estimators=2).fit(
            X, y, X_eval=X, y_eval=y, eval_every_n_trees=1, eval_metric="loss"
        )

    with pytest.raises(ValueError, match="loss_fn"):
        CustomObjective()
    with pytest.raises(ValueError, match="hessian"):
        CustomObjective(half_mse, hessian="exact")

    class Empty(Objective):
        pass

    with pytest.raises(NotImplementedError, match="loss"):
        WarpGBM(objective=Empty(), n_estimators=2).fit(X, y)
