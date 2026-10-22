"""
Warm starts with new hyperparameters, round slicing and inactive outputs (v2.8.0):
per-round learning rates / depths (round_learning_rates_, round_max_depths_), slice_rounds(),
and vector trees grown on the outputs with targets only.
"""
import copy
import pickle

import numpy as np
import pytest

from warpgbm import WarpGBM

K = 6
NAN_COLS = [0, 2, 3, 5]
ACTIVE_COLS = [1, 4]
NUM_BINS = 16


def _data(prebinned, n=2400, f=8, seed=0):
    rng = np.random.default_rng(seed)
    Xf = rng.normal(size=(n, f)).astype(np.float32)
    W = rng.normal(size=(f, K)).astype(np.float32)
    Y = (np.tanh(Xf) @ W + 0.3 * rng.normal(size=(n, K))).astype(np.float32)
    era = np.repeat(np.arange(4), n // 4).astype(np.int32)
    if prebinned:
        X = np.clip(((Xf + 3) / 6 * NUM_BINS).astype(np.int32), 0, NUM_BINS - 1)
        X[rng.random(X.shape) < 0.02] = -1
    else:
        X = Xf
        X[rng.random(X.shape) < 0.02] = np.nan
    return X, Y, era


def _model(prebinned, **kw):
    params = dict(
        objective="multi_regression",
        num_bins=NUM_BINS,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=40,
        min_child_weight=10,
        L2_reg=1.0,
        random_state=3,
        prebinned=prebinned,
        deterministic=True,
        verbose=False,
    )
    params.update(kw)
    return WarpGBM(**params)


def _rows(n, seed=1):
    return np.sort(np.random.default_rng(seed).choice(n, size=n * 2 // 3, replace=False))


def _subset(X, Y, era):
    """Two thirds of the rows, the targets of NAN_COLS missing."""
    rows = _rows(len(X))
    Ys = Y[rows].copy()
    Ys[:, NAN_COLS] = np.nan
    return X[rows], Ys, era[rows]


def _warm_copy(model, n_new=30, **changes):
    cp = copy.deepcopy(model)
    cp.warm_start = True
    for name, value in changes.items():
        setattr(cp, name, value)
    cp.n_estimators = model.num_trees_ + n_new
    return cp


def _same_tree(a, b, atol=0.0):
    if "leaf_value" in a:
        return "leaf_value" in b and np.allclose(a["leaf_value"], b["leaf_value"], rtol=0, atol=atol)
    return (
        "leaf_value" not in b
        and (a["feature"], a["bin"], a["nan_left"]) == (b["feature"], b["bin"], b["nan_left"])
        and _same_tree(a["left"], b["left"], atol)
        and _same_tree(a["right"], b["right"], atol)
    )


@pytest.fixture(scope="module", params=[False, True], ids=["raw", "prebinned"])
def fitted(request):
    prebinned = request.param
    X, Y, era = _data(prebinned)
    model = _model(prebinned)
    model.fit(X, Y, era_id=era)
    return prebinned, X, Y, era, model


# ─── F1 / F2: warm start with new hyperparameters ───


@pytest.mark.parametrize("depth_change", [2, -2])
def test_warm_start_new_hyperparams_keeps_inherited_rounds(fitted, depth_change):
    prebinned, X, Y, era, model = fitted
    n = model.num_trees_
    ref_gpu = model.predict(X)
    ref_cpu = model.predict(X, device="cpu")

    cp = _warm_copy(
        model,
        learning_rate=0.03,
        max_depth=model.max_depth + depth_change,
        colsample_bytree=0.7,
        min_child_weight=4,
        L2_reg=0.5,
    )
    Xs, Ys, es = _subset(X, Y, era)
    cp.fit(Xs, Ys, era_id=es)

    # The inherited rounds predict exactly as in the parent
    np.testing.assert_array_equal(cp.predict(X, num_trees=n), ref_gpu)
    np.testing.assert_array_equal(cp.predict(X, num_trees=n, device="cpu"), ref_cpu)
    # The new trees have leaf 0 at the outputs without targets
    full_gpu, full_cpu = cp.predict(X), cp.predict(X, device="cpu")
    np.testing.assert_array_equal(full_gpu[:, NAN_COLS], ref_gpu[:, NAN_COLS])
    np.testing.assert_array_equal(full_cpu[:, NAN_COLS], ref_cpu[:, NAN_COLS])
    assert not np.allclose(full_gpu[:, ACTIVE_COLS], ref_gpu[:, ACTIVE_COLS])
    np.testing.assert_allclose(full_cpu, full_gpu, rtol=0, atol=1e-6)

    assert cp.round_learning_rates_ == [0.1] * n + [0.03] * 30
    assert cp.round_max_depths_ == [4] * n + [4 + depth_change] * 30
    new_depths = [cp._tree_depth(t) for t in cp.forest[n : n + 30]]
    assert max(new_depths) <= 4 + depth_change
    if depth_change > 0:
        assert max(new_depths) > 4
    # Training-time scores used the per-round rates as well
    np.testing.assert_allclose(cp.train_predictions_, cp.predict(Xs), rtol=0, atol=1e-5)
    # The parent is untouched
    np.testing.assert_array_equal(model.predict(X), ref_gpu)
    assert len(model.round_learning_rates_) == n


def test_changing_learning_rate_does_not_rescale_existing_rounds(fitted):
    _, X, _, _, model = fitted
    cp = copy.deepcopy(model)
    ref = cp.predict(X)
    cp.learning_rate = 0.5
    cp.max_depth = 2
    np.testing.assert_array_equal(cp.predict(X), ref)
    np.testing.assert_array_equal(cp.predict(X, device="cpu"), model.predict(X, device="cpu"))


def test_two_learning_rates_equal_parts(fitted):
    prebinned, X, Y, era, _ = fitted
    n1, n2 = 15, 20
    m = _model(prebinned, n_estimators=n1, learning_rate=0.2, warm_start=True)
    m.fit(X, Y, era_id=era)
    p1 = m.predict_raw(X)
    m.learning_rate = 0.05
    m.n_estimators = n1 + n2
    m.fit(X, Y, era_id=era)

    np.testing.assert_array_equal(m.predict_raw(X, num_trees=n1), p1)
    assert m.round_learning_rates_ == [0.2] * n1 + [0.05] * n2
    # Packed leaves are the tree leaves times their round's rate
    nodes, leaf, _, round_end = m._packed_forest(n1 + n2)
    for r in (0, n1 - 1, n1, n1 + n2 - 1):
        t = int(round_end[r]) - 1
        values = np.asarray(_leaf_values(m.forest[r]), dtype=np.float32)
        used = 2 * len(values) - 1  # nodes of the tree, in pre-order
        packed = leaf[t][:used][nodes[t][:used, 0] < 0]
        np.testing.assert_array_equal(packed, values * np.float32(m.round_learning_rates_[r]))
    # The second part as a slice
    part = m.slice_rounds(n1)
    np.testing.assert_allclose(part.predict_raw(X), m.predict_raw(X) - p1, rtol=0, atol=1e-5)
    # Training-time scores of the second fit started from the first part at rate 0.2
    np.testing.assert_allclose(m.train_predictions_, m.predict_raw(X), rtol=0, atol=1e-5)


def _leaf_values(tree):
    """Leaf values in pre-order (the order of the packed node ids)."""
    stack, out = [tree], []
    while stack:
        node = stack.pop()
        if "leaf_value" in node:
            out.append(node["leaf_value"])
        else:
            stack.extend((node["right"], node["left"]))
    return out


# ─── F3: slice_rounds ───


@pytest.mark.parametrize("device", [None, "cpu"])
def test_slice_rounds_predictions(fitted, device):
    _, X, Y, era, model = fitted
    cp = _warm_copy(model, learning_rate=0.05, max_depth=3)
    Xs, Ys, es = _subset(X, Y, era)
    cp.fit(Xs, Ys, era_id=es)
    k = model.num_trees_
    part = cp.slice_rounds(k)
    expected = cp.predict(X, device=device) - cp.predict(X, num_trees=k, device=device)
    np.testing.assert_allclose(part.predict(X, device=device), expected, rtol=0, atol=1e-5)
    # parent + slice == copy
    np.testing.assert_allclose(
        model.predict(X, device=device) + part.predict(X, device=device),
        cp.predict(X, device=device),
        rtol=0,
        atol=1e-5,
    )
    # stop=
    mid = cp.slice_rounds(5, 25)
    expected = cp.predict(X, num_trees=25, device=device) - cp.predict(X, num_trees=5, device=device)
    np.testing.assert_allclose(mid.predict(X, device=device), expected, rtol=0, atol=1e-5)
    assert mid.round_learning_rates_ == cp.round_learning_rates_[5:25]


def test_slice_rounds_memory_and_independence(fitted):
    _, X, Y, era, model = fitted
    cp = _warm_copy(model, n_new=60)
    Xs, Ys, es = _subset(X, Y, era)
    cp.fit(Xs, Ys, era_id=es)
    n_cp = cp.num_trees_
    ref = cp.predict(X)
    k = model.num_trees_
    part = cp.slice_rounds(k)

    assert part.num_trees_ == n_cp - k
    assert len(part.forest) == n_cp - k
    assert part.round_learning_rates_ == cp.round_learning_rates_[k:]
    assert part.round_max_depths_ == cp.round_max_depths_[k:]
    assert part.train_predictions_ is None
    np.testing.assert_array_equal(part._base_score_np(), np.zeros(K, dtype=np.float32))
    assert part.feature_importance_ is not None and part.feature_importance_.shape == (X.shape[1],)
    size_full = len(pickle.dumps(cp))
    size_part = len(pickle.dumps(part))
    assert size_part < size_full * 0.75 * (n_cp - k) / n_cp

    # The source model is unchanged and the slice owns its trees
    assert cp.num_trees_ == n_cp and len(cp.round_learning_rates_) == n_cp
    np.testing.assert_array_equal(cp.predict(X), ref)
    assert part.forest[0] is not cp.forest[k]
    part.forest[0]["left"] = {"leaf_value": [9.0] * K, "samples": 1}
    np.testing.assert_array_equal(cp.predict(X), ref)


def test_slice_rounds_cannot_warm_start(fitted):
    _, X, Y, era, model = fitted
    part = model.slice_rounds(10)
    part.warm_start = True
    part.n_estimators = 100
    with pytest.raises(ValueError, match="slice_rounds"):
        part.fit(X, Y, era_id=era)
    with pytest.raises(ValueError):
        model.slice_rounds(5, model.num_trees_ + 1)
    with pytest.raises(ValueError):
        model.slice_rounds(10, 5)


def test_slice_rounds_default_stop_after_rollback():
    X, Y, era = _data(False)
    model = _model(False, n_estimators=200, learning_rate=0.3)
    model.fit(
        X[:1800], Y[:1800], era_id=era[:1800], X_eval=X[1800:], y_eval=Y[1800:],
        eval_every_n_trees=1, early_stopping_rounds=5,
    )
    best = model.best_iteration_
    assert best < model.num_trees_
    part = model.slice_rounds(3)
    assert part.num_trees_ == best - 3
    np.testing.assert_allclose(
        model.predict(X, num_trees=3) + part.predict(X), model.predict(X), rtol=0, atol=1e-5
    )


def test_slice_rounds_single_output_and_per_output():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2).astype(np.float32)
    reg = WarpGBM(n_estimators=30, max_depth=3, verbose=False).fit(X, y)
    part = reg.slice_rounds(10)
    assert part.base_prediction == 0.0
    np.testing.assert_allclose(
        part.predict(X), reg.predict(X) - reg.predict(X, num_trees=10), rtol=0, atol=1e-5
    )
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    clf = WarpGBM(objective="multiclass", n_estimators=20, max_depth=3, verbose=False)
    clf.fit(X, labels)
    part = clf.slice_rounds(5)
    assert len(part.forest) == 15
    np.testing.assert_allclose(
        part.predict_raw(X), clf.predict_raw(X) - clf.predict_raw(X, num_trees=5), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        part.predict_raw(X, device="cpu"), part.predict_raw(X), rtol=0, atol=1e-6
    )


# ─── Persistence and truncate ───


def test_save_load_keeps_round_meta(fitted, tmp_path):
    _, X, Y, era, model = fitted
    cp = _warm_copy(model, learning_rate=0.02, max_depth=6)
    Xs, Ys, es = _subset(X, Y, era)
    cp.fit(Xs, Ys, era_id=es)
    path = tmp_path / "m.pkl"
    cp.save_model(str(path))
    loaded = WarpGBM(verbose=False).load_model(str(path))
    assert loaded.round_learning_rates_ == cp.round_learning_rates_
    assert loaded.round_max_depths_ == cp.round_max_depths_
    np.testing.assert_array_equal(loaded.predict(X), cp.predict(X))
    np.testing.assert_array_equal(loaded.predict(X, device="cpu"), cp.predict(X, device="cpu"))


def test_load_v27_model_and_pickle(fitted, tmp_path):
    _, X, _, _, model = fitted
    path = tmp_path / "m.pkl"
    model.save_model(str(path))
    with open(path, "rb") as f:
        state = pickle.load(f)
    for key in ("round_learning_rates_", "round_max_depths_", "_is_slice"):
        del state[key]
    with open(path, "wb") as f:
        pickle.dump(state, f)
    loaded = WarpGBM(verbose=False).load_model(str(path))
    n = model.num_trees_
    assert loaded.round_learning_rates_ == [model.learning_rate] * n
    # Depths come from the stored trees
    assert loaded.round_max_depths_ == [model._tree_depth(t) for t in model.forest[:n]]
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    # An instance pickled by v2.7 (attributes missing, class defaults None)
    old = copy.deepcopy(model)
    del old.__dict__["round_learning_rates_"], old.__dict__["round_max_depths_"]
    old._invalidate_prediction_cache()
    old = pickle.loads(pickle.dumps(old))
    np.testing.assert_array_equal(old.predict(X), model.predict(X))
    assert len(old.round_learning_rates_) == n


def test_truncate_then_new_learning_rate(fitted):
    _, X, Y, era, model = fitted
    k = 25
    ref = model.predict(X, num_trees=k)
    cp = copy.deepcopy(model).truncate(k)
    assert len(cp.round_learning_rates_) == k and len(cp.round_max_depths_) == k
    cp.warm_start = True
    cp.learning_rate = 0.25
    cp.max_depth = 2
    cp.n_estimators = k + 10
    cp.fit(X, Y, era_id=era)
    np.testing.assert_array_equal(cp.predict(X, num_trees=k), ref)
    assert cp.round_learning_rates_ == [0.1] * k + [0.25] * 10
    assert cp.round_max_depths_ == [4] * k + [2] * 10


# ─── F4: hyperparameters are validated at fit ───


def test_invalid_hyperparams_set_on_fitted_model_raise(fitted):
    _, X, Y, era, model = fitted
    cp = _warm_copy(model, learning_rate=0.0)
    with pytest.raises(ValueError, match="learning_rate"):
        cp.fit(X, Y, era_id=era)
    cp = _warm_copy(model, max_depth=np.int64(3), learning_rate=np.float32(0.05))
    cp.fit(X, Y, era_id=era)  # numpy scalars are accepted
    assert cp.round_max_depths_[-1] == 3


# ─── F5: outputs without targets ───


def test_inactive_outputs_give_identical_trees(fitted):
    _, X, Y, era, model = fitted
    n = model.num_trees_
    Xs, Ys, es = _subset(X, Y, era)
    Ys[:, ACTIVE_COLS[1]] = np.nan  # 5 of 6 outputs without targets
    fast = _warm_copy(model, learning_rate=0.05)
    slow = _warm_copy(model, learning_rate=0.05)
    slow._skip_inactive_outputs = False
    fast.fit(Xs, Ys, era_id=es)
    slow.fit(Xs, Ys, era_id=es)

    for r in range(n, n + 30):
        assert _same_tree(fast.forest[r], slow.forest[r], atol=1e-6), r
        for value in _leaf_values(fast.forest[r]):
            assert len(value) == K
            assert all(value[c] == 0.0 for c in range(K) if c != ACTIVE_COLS[0])
    np.testing.assert_array_equal(fast.predict(X), slow.predict(X))
    np.testing.assert_array_equal(fast.train_predictions_, slow.train_predictions_)


def test_active_outputs_argument_matches_nan_targets(fitted):
    _, X, Y, era, model = fitted
    Xs, Ys, es = _subset(X, Y, era)
    by_nan = _warm_copy(model)
    by_nan.fit(Xs, Ys, era_id=es)
    by_arg = _warm_copy(model)
    Y_full = Y[_rows(len(X))]  # the same rows with all targets present
    by_arg.fit(Xs, Y_full, era_id=es, active_outputs=ACTIVE_COLS)
    np.testing.assert_array_equal(by_arg.predict(X), by_nan.predict(X))
    mask = np.zeros(K, dtype=bool)
    mask[ACTIVE_COLS] = True
    by_mask = _warm_copy(model)
    by_mask.fit(
        Xs, Y_full, era_id=es, active_outputs=mask,
        X_eval=X[:300], y_eval=Y[:300], eval_every_n_trees=5,
    )
    np.testing.assert_array_equal(by_mask.predict(X), by_nan.predict(X))
    with pytest.raises(ValueError):
        _warm_copy(model).fit(Xs, Y_full, era_id=es, active_outputs=[0, K])


def test_fresh_fit_with_inactive_outputs():
    X, Y, era = _data(False)
    Y = Y.copy()
    Y[:, NAN_COLS] = np.nan
    fast = _model(False, n_estimators=20)
    slow = _model(False, n_estimators=20)
    slow._skip_inactive_outputs = False
    fast.fit(X, Y, era_id=era)
    slow.fit(X, Y, era_id=era)
    np.testing.assert_array_equal(fast.predict(X), slow.predict(X))
    assert np.all(fast.predict(X)[:, NAN_COLS] == 0.0)
