"""Dataset-level objectives: full-dataset (multi-output) Pearson correlation, eras, warm starts."""

import os
import tempfile

import numpy as np
import pytest
import torch

from warpgbm import AutogradObjective, DatasetLossObjective, Objective, WarpGBM, losses

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

DEV = "cuda"


def pearson_mse(y, pred, ctx):
    """-mean_k pearson_k(pred, y) + 0.1 * MSE over the full dataset (module level -> picklable)."""
    corr = losses.pearson_corr(pred, y, weight=ctx.weight, mask=ctx.mask)  # [K]
    return -corr.mean() + 0.1 * losses.weighted_mse(pred, y, weight=ctx.weight, mask=ctx.mask)


def half_mse(y, pred, ctx):
    return 0.5 * losses.weighted_mse(pred, y, weight=ctx.weight, mask=ctx.mask)


def _multi_output_data(seed=0, n_eras=60, per_era=250, nan_frac=0.2):
    rng = np.random.default_rng(seed)
    era = np.repeat(np.arange(n_eras), per_era).astype(np.int32)
    n = len(era)
    X = rng.normal(size=(n, 10)).astype(np.float32)
    base = 0.5 * X[:, 0] + 0.3 * np.tanh(2 * X[:, 1])
    Y = np.column_stack(
        [
            base + 0.4 * X[:, 2] + rng.normal(size=n),
            -base + 0.3 * X[:, 3] ** 2 + rng.normal(size=n),
            5.0 + 3.0 * (0.5 * X[:, 4] + rng.normal(size=n)),  # different offset / scale
        ]
    ).astype(np.float32)
    Y[rng.random(Y.shape) < nan_frac] = np.nan
    W = rng.uniform(0.5, 2.0, size=Y.shape).astype(np.float32)
    return X, Y, W, era


def _corr_per_output(P, Y):
    out = []
    for k in range(Y.shape[1]):
        m = np.isfinite(Y[:, k])
        out.append(np.corrcoef(P[m, k], Y[m, k])[0, 1])
    return np.array(out)


COMMON = dict(n_estimators=80, max_depth=4, num_bins=32, learning_rate=0.1, random_state=0)


# ─── helpers ──────────────────────────────────────────────────────────────────


def _weighted_corr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return cov / np.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w))


def test_pearson_corr_matches_numpy_multi_output_masked_weighted_and_grouped():
    rng = np.random.default_rng(1)
    n, k = 4000, 3
    a = rng.normal(size=(n, k))
    b = a * np.array([1.0, -0.5, 0.1]) + rng.normal(size=(n, k))
    w = rng.uniform(0.2, 3.0, size=(n, k))
    mask = rng.random((n, k)) > 0.25
    group = rng.integers(0, 5, size=n)

    t = lambda x: torch.tensor(x, device=DEV)
    full = losses.pearson_corr(t(a), t(b), weight=t(w), mask=t(mask)).cpu().numpy()
    grouped = losses.pearson_corr(t(a), t(b), weight=t(w), mask=t(mask), group=t(group)).cpu().numpy()

    for j in range(k):
        m = mask[:, j]
        assert abs(full[j] - _weighted_corr(a[m, j], b[m, j], w[m, j])) < 1e-10
        corrs, totals = [], []
        for g in range(5):
            mg = m & (group == g)
            corrs.append(_weighted_corr(a[mg, j], b[mg, j], w[mg, j]))
            totals.append(w[mg, j].sum())
        assert abs(grouped[j] - np.average(corrs, weights=totals)) < 1e-10


def test_weighted_mse_helper():
    rng = np.random.default_rng(2)
    p, y, w = rng.normal(size=(500, 2)), rng.normal(size=(500, 2)), rng.uniform(0.1, 2, size=(500, 2))
    mask = rng.random((500, 2)) > 0.3
    t = lambda x: torch.tensor(x, device=DEV)
    got = losses.weighted_mse(t(p), t(y), weight=t(w), mask=t(mask)).item()
    assert abs(got - np.sum(w * mask * (p - y) ** 2) / np.sum(w * mask)) < 1e-10


# ─── engine integration ──────────────────────────────────────────────────────


@pytest.mark.parametrize("hessian", ["exact", "auto"])
def test_dataset_level_mse_matches_builtin_multi_regression(hessian):
    # An averaged 0.5 * MSE gets exactly the per-sample built-in gradients after rescaling,
    # including NaN targets and 2-D weights.
    X, Y, W, _ = _multi_output_data(3)
    builtin = WarpGBM(objective="multi_regression", **COMMON).fit(X, Y, sample_weight=W)
    custom = WarpGBM(
        objective=DatasetLossObjective(half_mse, hessian=hessian, warmup_rounds=0), **COMMON
    ).fit(X, Y, sample_weight=W)
    np.testing.assert_allclose(custom.predict(X), builtin.predict(X), rtol=1e-3, atol=1e-3)


def test_dataset_objective_gradients_are_scaled_masked_and_not_reweighted():
    X, Y, W, era = _multi_output_data(4, n_eras=4, per_era=100)
    model = WarpGBM(objective=DatasetLossObjective(pearson_mse, warmup_rounds=0), n_estimators=1)
    model.objective_ = model._resolve_objective()
    model.num_outputs = 3
    y_prepared = model.objective_.prepare_targets(Y)
    w = model._normalize_sample_weight(W)
    era_t = torch.tensor(era, device=DEV, dtype=torch.int32)
    Y_gpu, ctx = model._build_context(y_prepared, w, era_t)
    pred = torch.randn(Y_gpu.shape, device=DEV)

    from warpgbm.objectives import compute_grad_hess

    g, h = compute_grad_hess(model.objective_, Y_gpu, pred, ctx)

    p = pred.clone().requires_grad_(True)
    (g_ref,) = torch.autograd.grad(pearson_mse(Y_gpu, p, ctx), p)
    g_ref = g_ref * ctx.weight.sum()
    torch.testing.assert_close(g, torch.where(ctx.mask, g_ref, torch.zeros_like(g_ref)))
    assert (g[~ctx.mask] == 0).all() and (h[~ctx.mask] == 0).all()
    assert (h[ctx.mask] > 0).all()


def test_multi_output_pearson_from_scratch_is_stable_and_competitive():
    X, Y, W, era = _multi_output_data(5)
    train, test = era < 40, era >= 40

    mse = WarpGBM(objective="multi_regression", **COMMON).fit(X[train], Y[train], sample_weight=W[train])
    corr_mse = _corr_per_output(mse.predict(X[test]), Y[test])

    model = WarpGBM(objective=DatasetLossObjective(pearson_mse), **COMMON)
    model.fit(X[train], Y[train], sample_weight=W[train])
    P = model.predict(X[test])

    assert P.shape == Y[test].shape and np.isfinite(P).all()
    # the MSE term keeps the scale in check (no blow-up from the scale-invariant correlation)
    assert np.all(P.std(axis=0) < 3.0 * np.nanstd(Y, axis=0))
    corr = _corr_per_output(P, Y[test])
    assert np.all(corr > 0.9 * corr_mse), (corr, corr_mse)


def test_multi_output_pearson_with_era_splitting_and_era_eval_metric():
    X, Y, W, era = _multi_output_data(6)
    train, test = era < 40, era >= 40

    def per_era_corr_metric(y_true, pred, weight, mask, era):
        # lower is better -> 1 - mean per-era correlation
        return 1.0 - float(losses.pearson_corr(pred, y_true, weight=weight, mask=mask, group=era).mean())

    model = WarpGBM(objective=DatasetLossObjective(pearson_mse), **COMMON)
    model.fit(
        X[train], Y[train], sample_weight=W[train], era_id=era[train],
        X_eval=X[test], y_eval=Y[test], era_id_eval=era[test], sample_weight_eval=W[test],
        eval_every_n_trees=10, eval_metric=per_era_corr_metric,
    )
    assert model.num_eras == 40
    assert model.get_per_era_feature_importance().shape == (40, X.shape[1])
    assert len(model.eval_loss) == COMMON["n_estimators"] // 10
    assert all(np.isfinite(model.eval_loss))
    assert model.eval_loss[-1] < model.eval_loss[0]
    assert np.all(_corr_per_output(model.predict(X[test]), Y[test]) > 0.2)

    # default eval metric of a dataset objective is its (scalar) loss
    model.fit(X[train], Y[train], era_id=era[train], X_eval=X[test], y_eval=Y[test], eval_every_n_trees=20)
    assert model.eval_metric == "loss"
    assert all(np.isfinite(model.eval_loss))


def test_eval_set_eras_reach_the_eval_context():
    X, Y, W, era = _multi_output_data(7, n_eras=6, per_era=100)
    seen = {}

    def metric(y_true, pred, weight, mask, era):
        seen["eras"] = int(era.max().item()) + 1
        return 0.0

    model = WarpGBM(objective="multi_regression", n_estimators=2, max_depth=2)
    model.fit(X, Y, X_eval=X, y_eval=Y, era_id_eval=era, eval_every_n_trees=1, eval_metric=metric)
    assert seen["eras"] == 6

    with pytest.raises(ValueError, match="era_id_eval must have same length"):
        model.fit(X, Y, X_eval=X, y_eval=Y, era_id_eval=era[:-1], eval_every_n_trees=1)


class IterationRecorder(Objective):
    name = "recorder"

    def __init__(self):
        self.iterations = []

    def grad_hess(self, y, pred, ctx):
        self.iterations.append(ctx.iteration)
        return pred - y, torch.ones_like(pred)


def test_context_iteration_and_warmup_rounds():
    X, Y, W, _ = _multi_output_data(8, n_eras=4, per_era=200, nan_frac=0.0)
    rec = IterationRecorder()
    WarpGBM(objective=rec, n_estimators=5, max_depth=2).fit(X, Y)
    assert rec.iterations[-5:] == [0, 1, 2, 3, 4]

    # During warm-up the dataset objective follows squared error exactly
    params = dict(n_estimators=3, max_depth=3, random_state=0)
    warm = WarpGBM(objective=DatasetLossObjective(pearson_mse, warmup_rounds=3), **params).fit(X, Y)
    mse = WarpGBM(objective="multi_regression", **params).fit(X, Y)
    np.testing.assert_allclose(warm.predict(X), mse.predict(X), rtol=1e-4, atol=1e-4)


def test_warm_start_switching_objective_and_save_load():
    X, Y, W, era = _multi_output_data(9)
    model = WarpGBM(objective="multi_regression", warm_start=True, **dict(COMMON, n_estimators=20))
    model.fit(X, Y, sample_weight=W, era_id=era)

    model.objective = DatasetLossObjective(pearson_mse, warmup_rounds=0)
    model.n_estimators = 60
    model.fit(X, Y, sample_weight=W, era_id=era)
    assert model._trees_trained == 60
    assert isinstance(model.objective_, DatasetLossObjective)
    assert np.isfinite(model.predict(X)).all()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        path = tmp.name
    try:
        model.save_model(path)
        loaded = WarpGBM().load_model(path)
        np.testing.assert_allclose(loaded.predict(X), model.predict(X), rtol=1e-4, atol=1e-4)
        # continue training the loaded model with its stored dataset objective
        loaded.warm_start = True
        loaded.n_estimators = 70
        loaded.fit(X, Y, sample_weight=W, era_id=era)
        assert loaded._trees_trained == 70
    finally:
        os.remove(path)

    # incompatible switch: vector-leaf forest cannot continue with one tree per output
    model.objective = AutogradObjective(lambda y, p: (p - y) ** 2, output_strategy="per_output")
    model.n_estimators = 80
    with pytest.raises(ValueError, match="output_strategy"):
        model.fit(X, Y)


def test_dataset_objective_rejects_non_scalar_loss():
    X, Y, W, _ = _multi_output_data(10, n_eras=2, per_era=100)
    bad = DatasetLossObjective(lambda y, p, ctx: (p - y) ** 2, warmup_rounds=0)
    with pytest.raises(ValueError, match="scalar"):
        WarpGBM(objective=bad, n_estimators=2).fit(X, np.nan_to_num(Y))


# ─── standardization, group weighting, era handling, warm-up targets, curvature ──


def test_standardize_matches_numpy_masked_weighted_and_grouped():
    rng = np.random.default_rng(11)
    n, k = 3000, 2
    x = 3.0 + 2.0 * rng.normal(size=(n, k))
    w = rng.uniform(0.2, 3.0, size=(n, k))
    mask = rng.random((n, k)) > 0.2
    group = rng.integers(0, 4, size=n)
    t = lambda a: torch.tensor(a, device=DEV)
    z = losses.standardize(t(x), weight=t(w), mask=t(mask), group=t(group)).cpu().numpy()
    for j in range(k):
        assert (z[~mask[:, j], j] == 0).all()
        for g in range(4):
            m = mask[:, j] & (group == g)
            mu = np.average(x[m, j], weights=w[m, j])
            sd = np.sqrt(np.average((x[m, j] - mu) ** 2, weights=w[m, j]))
            np.testing.assert_allclose(z[m, j], (x[m, j] - mu) / sd, rtol=1e-8, atol=1e-8)
    # 1-D in -> 1-D out
    assert losses.standardize(t(x[:, 0])).shape == (n,)


def test_pearson_corr_equal_group_weighting():
    rng = np.random.default_rng(12)
    sizes = [50, 500, 2000]
    group = np.repeat(np.arange(3), sizes)
    a = rng.normal(size=group.size)
    b = a * np.array([0.1, 0.5, 2.0])[group] + rng.normal(size=group.size)
    t = lambda x: torch.tensor(x, device=DEV)
    equal = losses.pearson_corr(t(a), t(b), group=t(group), group_weighting="equal").item()
    ref = np.mean([np.corrcoef(a[group == g], b[group == g])[0, 1] for g in range(3)])
    assert abs(equal - ref) < 1e-10
    with pytest.raises(ValueError, match="group_weighting"):
        losses.pearson_corr(t(a), t(b), group_weighting="size")


class EraRecorder(DatasetLossObjective):
    def __init__(self):
        super().__init__(half_mse, warmup_rounds=0)
        self.num_eras_seen = None

    def grad_hess(self, y, pred, ctx):
        self.num_eras_seen = int(ctx.era_index.max().item()) + 1
        return super().grad_hess(y, pred, ctx)


def test_era_splitting_false_keeps_eras_for_the_loss_only():
    X, Y, W, era = _multi_output_data(13, n_eras=8, per_era=200, nan_frac=0.0)
    y = Y[:, 0]
    rec = EraRecorder()
    model = WarpGBM(objective=rec, era_splitting=False, **COMMON).fit(X, y, era_id=era)
    assert rec.num_eras_seen == 8       # the loss sees the eras ...
    assert model.num_eras == 1          # ... the split finder does not
    no_eras = WarpGBM(objective="regression", **COMMON).fit(X, y)
    pooled = WarpGBM(objective="regression", era_splitting=False, **COMMON).fit(X, y, era_id=era)
    np.testing.assert_array_equal(pooled.predict(X), no_eras.predict(X))
    with pytest.raises(TypeError, match="era_splitting"):
        WarpGBM(era_splitting="yes")


@pytest.mark.parametrize("target", ["standardized", "era_standardized"])
def test_standardized_warmup_targets(target):
    # During warm-up the objective regresses on (per-era) standardized targets on top of the
    # mean prediction, i.e. it behaves like MSE trained on the standardized targets.
    X, Y, W, era = _multi_output_data(14, n_eras=6, per_era=300, nan_frac=0.0)
    Y[:, 2] *= 100.0
    params = dict(n_estimators=3, max_depth=3, random_state=0, L2_reg=0.0)
    obj = DatasetLossObjective(pearson_mse, warmup_rounds=3, warmup_target=target)
    warm = WarpGBM(objective=obj, **params).fit(X, Y, era_id=era)

    if target == "standardized":
        Z = (Y - Y.mean(0)) / Y.std(0)
    else:
        Z = np.empty_like(Y)
        for e in np.unique(era):
            i = era == e
            Z[i] = (Y[i] - Y[i].mean(0)) / Y[i].std(0)
    ref = WarpGBM(objective="multi_regression", **params).fit(X, Z, era_id=era)
    expected = ref.predict(X) - Z.mean(0) + Y.mean(0)
    np.testing.assert_allclose(warm.predict(X), expected, rtol=1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="warmup_target"):
        DatasetLossObjective(pearson_mse, warmup_target="ranks")


def neg_corr(y, pred, ctx):
    return -losses.pearson_corr(pred, y, weight=ctx.weight, mask=ctx.mask).mean()


def test_auto_curvature_matches_theory_and_stays_positive_for_negative_correlation():
    # For L = -corr the curvature along the gradient is rho / var(pred) (sum scale);
    # with rho < 0 the auto hessian uses |c| so the Newton step remains a descent step.
    X, Y, W, era = _multi_output_data(15, n_eras=4, per_era=500, nan_frac=0.0)
    model = WarpGBM(objective=DatasetLossObjective(neg_corr, warmup_rounds=0), n_estimators=1)
    model.objective_ = model._resolve_objective()
    model.num_outputs = 3
    era_t = torch.zeros(len(Y), device=DEV, dtype=torch.int32)
    Y_gpu, ctx = model._build_context(model.objective_.prepare_targets(Y), None, era_t)
    from warpgbm.objectives import compute_grad_hess

    for sign in (1.0, -1.0):
        pred = sign * Y_gpu + 2.0 * torch.randn_like(Y_gpu)
        g, h = compute_grad_hess(model.objective_, Y_gpu, pred, ctx)
        rho = losses.pearson_corr(pred, Y_gpu)
        c = h[0] / ctx.weight[0]
        torch.testing.assert_close(c, rho.abs() / pred.var(dim=0, unbiased=False), rtol=2e-2, atol=0)
        # one Newton step along -g/h increases the correlation
        stepped = pred - 0.5 * g / h
        assert (losses.pearson_corr(stepped, Y_gpu) > rho).all()
