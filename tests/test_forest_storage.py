"""
v2.9 forest storage: compact blocks of rounds, shared by copies / forks / slices, the CUDA and
NumPy predictors on them, prediction-cache release, fork, init_scores, predict_many.

Predictions are checked bit for bit against the v2.8 predictor, reimplemented here (padded
per-tree layout, leaves pre-multiplied by the learning rate, sequential float32 sums: v2.8's
NumPy path, documented to equal its CUDA kernel).
"""
import copy
import pickle

import numpy as np
import pytest
import torch

import warpgbm
from warpgbm import WarpGBM
from warpgbm.objectives import PER_OUTPUT

K = 6


# ─── v2.8 reference predictor ───


def _v28_pack_tree(tree, max_nodes, leaf_width, learning_rate):
    nodes = np.zeros((max_nodes, 4), dtype=np.int32)
    nodes[:, 0] = -1
    leaf = np.zeros((max_nodes, leaf_width), dtype=np.float32)
    counter = 0
    stack = [(tree, None, 0)]
    while stack:
        node, parent, side = stack.pop()
        i = counter
        counter += 1
        if parent is not None:
            nodes[parent, side] = i
        if "leaf_value" in node:
            value = node["leaf_value"]
            if isinstance(value, (list, tuple, np.ndarray)):
                leaf[i] = np.asarray(value, dtype=np.float32)[:leaf_width]
            else:
                leaf[i] = value
        else:
            nodes[i, 0] = int(node["feature"])
            nodes[i, 1] = int(node["bin"]) | (int(bool(node.get("nan_left", False))) << 8)
            stack.append((node["right"], i, 3))
            stack.append((node["left"], i, 2))
    leaf *= np.float32(learning_rate)
    return nodes, leaf


def _v28_predict_raw(model, X, num_trees):
    """Raw scores [N, K] of the first num_trees rounds as v2.8 computed them."""
    per_output = model._output_strategy() == PER_OUTPUT
    Kout = model.num_outputs
    leaf_width = 1 if per_output else Kout
    M = 2 ** (max(model.round_max_depths_, default=model.max_depth) + 1)
    nodes, leaves, out_col = [], [], []
    for r, round_trees in enumerate(model.forest[:num_trees]):
        trees = round_trees if per_output else [round_trees]
        for k, tree in enumerate(trees):
            if tree:
                n, lf = _v28_pack_tree(tree, M, leaf_width, model.round_learning_rates_[r])
                nodes.append(n)
                leaves.append(lf)
                out_col.append(k)
    bins = model._bin_inference_cpu(X)
    base = model._base_score_np()
    n_rows = bins.shape[0]
    if not nodes:
        return np.broadcast_to(base, (n_rows, Kout)).copy()
    nodes, leaves = np.stack(nodes), np.stack(leaves)
    T = nodes.shape[0]
    offset = (np.arange(T, dtype=np.int64) * M)[:, None]
    feature = nodes[..., 0].reshape(-1)
    split_bin = (nodes[..., 1] & 0xFF).reshape(-1)
    nan_left = ((nodes[..., 1] >> 8) & 1).astype(bool).reshape(-1)
    left = (nodes[..., 2] + offset).reshape(-1)
    right = (nodes[..., 3] + offset).reshape(-1)
    leaf = leaves.reshape(T * M, leaf_width)
    F = bins.shape[1]
    flat_bins = bins.reshape(-1).astype(np.int64)
    row_offset = (np.arange(n_rows, dtype=np.int64) * F)[:, None]
    node = np.broadcast_to(np.arange(T, dtype=np.int64) * M, (n_rows, T)).copy()
    for _ in range(M.bit_length() - 1):
        f = feature[node]
        is_leaf = f < 0
        if is_leaf.all():
            break
        x = flat_bins[row_offset + np.maximum(f, 0)]
        go_left = np.where(x == model.nan_bin, nan_left[node], x <= split_bin[node])
        node = np.where(is_leaf, node, np.where(go_left, left[node], right[node]))
    values = leaf[node]
    if per_output:
        dense = np.zeros((n_rows, T, Kout), dtype=np.float32)
        dense[:, np.arange(T), np.asarray(out_col)] = values[..., 0]
        values = dense
    cum = np.add.accumulate(values, axis=1)
    return base + cum[:, -1]


def _check_exact(model, X, rounds=None):
    """GPU and CPU raw scores of every round count in rounds equal the v2.8 predictor exactly."""
    rounds = [model.num_trees_] if rounds is None else rounds
    staged_gpu = model.staged_predict_raw(X, rounds)
    staged_cpu = model.staged_predict_raw(X, rounds, device="cpu")
    for p, r in enumerate(rounds):
        ref = _v28_predict_raw(model, X, r).reshape(staged_gpu[p].shape)
        np.testing.assert_array_equal(staged_gpu[p], ref, err_msg=f"GPU, {r} rounds")
        np.testing.assert_array_equal(staged_cpu[p], ref, err_msg=f"CPU, {r} rounds")


# ─── Data and models ───


def _data(n=2400, f=10, k=K, seed=0, nan=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    if nan:
        X[rng.random(X.shape) < 0.05] = np.nan
    W = rng.normal(size=(f, k)).astype(np.float32)
    Y = (np.tanh(np.nan_to_num(X)) @ W + 0.5 * rng.normal(size=(n, k))).astype(np.float32)
    era = (np.arange(n) * 4 // n).astype(np.int32)
    return X, Y, era


def _multi(k=K, **kw):
    params = dict(objective="multi_regression", n_estimators=25, max_depth=4, num_bins=16,
                  learning_rate=0.2, random_state=0, verbose=False)
    params.update(kw)
    return WarpGBM(**params)


def _only(Y, cols):
    Ys = np.full_like(Y, np.nan)
    Ys[:, cols] = Y[:, cols]
    return Ys


@pytest.fixture(scope="module")
def chain():
    """Parent (dense K) -> fork on 2 outputs (deeper, new rate) -> fork of that on 1 output."""
    X, Y, era = _data()
    parent = _multi().fit(X, Y, era_id=era)
    mid = parent.fork(n_new=15, max_depth=6, learning_rate=0.05)
    mid.fit(X, _only(Y, [1, 4]), era_id=era)
    leaf = mid.fork(n_new=10, max_depth=2, learning_rate=0.3)
    leaf.fit(X, _only(Y, [4]), era_id=era)
    return X, Y, era, parent, mid, leaf


# ─── Exactness against v2.8 ───


def test_dense_sparse_and_deep_rounds_are_exact(chain):
    X, _, _, parent, mid, leaf = chain
    _check_exact(parent, X, [0, 1, 7, 25])
    # Three blocks: checkpoints at, inside and across block boundaries
    _check_exact(leaf, X, [0, 3, 24, 25, 26, 33, 40, 41, 45, 50])
    assert len(leaf._forest_blocks()) == 3
    assert [b.num_rounds for b in leaf._forest_blocks()] == [25, 15, 10]


def test_single_output_and_raw_feature_nan_bins_are_exact():
    X, Y, era = _data(k=1)
    model = WarpGBM(n_estimators=30, max_depth=5, num_bins=24, random_state=1, verbose=False)
    model.fit(X, Y[:, 0], era_id=era)
    _check_exact(model, X, [0, 10, 30])
    assert model.predict(X).shape == (len(X),)


def test_per_output_multiclass_is_exact():
    X, Y, _ = _data(k=1, seed=3)
    labels = np.digitize(Y[:, 0], np.quantile(Y[:, 0], [0.3, 0.7]))
    clf = WarpGBM(objective="multiclass", n_estimators=12, max_depth=3, random_state=0, verbose=False)
    clf.fit(X, labels)
    _check_exact(clf, X, [0, 5, 12])
    # every per-output tree stores its one column
    (block,) = clf._forest_blocks()
    assert set(block.n_cols.tolist()) <= {0, 1}
    assert clf.round_active_outputs_[0] == (0, 1, 2)


def test_more_than_32_outputs_use_the_wide_kernel_exactly():
    X, Y, era = _data(k=40, seed=4)
    model = _multi(k=40, n_estimators=8, max_depth=3).fit(X, Y, era_id=era)
    cp = model.fork(n_new=6)
    cp.fit(X, _only(Y, [3, 35]), era_id=era)
    _check_exact(cp, X, [0, 4, 8, 11, 14])
    np.testing.assert_array_equal(
        warpgbm.predict_many([model, cp], X, raw=True),
        np.stack([model.predict_raw(X), cp.predict_raw(X)]),
    )


def test_truncate_inside_a_block_and_continue_is_exact(chain):
    X, Y, era, _, mid, _ = chain
    cp = copy.deepcopy(mid).truncate(31)
    assert [b.num_rounds for b in cp._forest_blocks()] == [25, 6]
    _check_exact(cp, X, [25, 31])
    cp.n_estimators = 36
    cp.fit(X, Y, era_id=era)
    _check_exact(cp, X, [31, 36])


def test_many_small_warm_starts_extend_one_block():
    X, Y, era = _data()
    model = _multi(n_estimators=5, warm_start=True).fit(X, Y, era_id=era)
    for total in range(8, 60, 3):
        model.n_estimators = total
        model.fit(X, Y, era_id=era)
    assert len(model._forest_blocks()) == 1
    _check_exact(model, X, [0, 5, 29, model.num_trees_])


# ─── Storage, view, copies ───


def test_forest_view_round_trip_and_read_only(chain):
    X, _, _, parent, mid, _ = chain
    view = mid.forest
    assert len(view) == mid.num_trees_
    assert view == list(view) and view[-1] == view[len(view) - 1]
    tree = view[30]
    assert set(tree) == {"feature", "bin", "nan_left", "left", "right", "gain"}
    # Editing the returned dicts does not change the model
    before = mid.predict(X)
    tree["left"] = {"leaf_value": [9.0] * K, "samples": 1}
    assert mid.forest[30] != tree
    np.testing.assert_array_equal(mid.predict(X), before)
    with pytest.raises(ValueError):
        mid._forest_blocks()[0].values[0] = 1.0
    # Tree dicts -> blocks -> tree dicts is the identity
    rebuilt = copy.deepcopy(mid)
    rebuilt._set_dict_forest(list(view))
    assert rebuilt.forest == view
    np.testing.assert_array_equal(rebuilt.predict(X), before)


def test_deepcopy_fork_and_slice_share_blocks(chain):
    X, _, _, parent, mid, leaf = chain
    pb = parent._forest_blocks()
    assert mid._forest_blocks()[: len(pb)] == pb and all(
        a is b for a, b in zip(mid._forest_blocks(), pb)
    )
    cp = copy.deepcopy(leaf)
    assert all(a is b for a, b in zip(cp._forest_blocks(), leaf._forest_blocks()))
    part = leaf.slice_rounds(mid.num_trees_)
    assert part._forest_blocks()[0] is leaf._forest_blocks()[-1]
    assert part.bin_edges is leaf.bin_edges
    np.testing.assert_allclose(mid.predict(X) + part.predict(X), leaf.predict(X), rtol=0, atol=1e-5)
    # A pickle of several models stores a shared block once
    both = pickle.dumps([leaf, part])
    assert len(both) < len(pickle.dumps(leaf)) + 0.5 * len(pickle.dumps(part))
    a, b = pickle.loads(both)
    assert b._forest_blocks()[0] is a._forest_blocks()[-1]
    np.testing.assert_array_equal(a.predict(X), leaf.predict(X))


def test_fork_leaves_the_parent_unchanged_and_equals_the_deepcopy_path():
    X, Y, era = _data(seed=5)
    parent = _multi().fit(X, Y, era_id=era)
    ref = parent.predict(X)
    via_fork = parent.fork(n_new=12, learning_rate=0.07, max_depth=5)
    via_fork.fit(X, _only(Y, [2]), era_id=era)
    old = copy.deepcopy(parent)
    old.warm_start, old.n_estimators, old.learning_rate, old.max_depth = True, 37, 0.07, 5
    old.fit(X, _only(Y, [2]), era_id=era)
    assert via_fork.forest == old.forest
    np.testing.assert_array_equal(via_fork.predict(X), old.predict(X))
    np.testing.assert_array_equal(parent.predict(X), ref)
    assert parent.num_trees_ == 25 and parent.n_estimators == 25
    assert via_fork.round_active_outputs_[-1] == (2,)
    # Own memory of the fork: its new block only
    own = via_fork._forest_blocks()[len(parent._forest_blocks()):]
    assert len(own) == 1 and own[0].num_rounds == 12
    with pytest.raises(ValueError):
        parent.fork(n_new=-1)
    with pytest.raises(ValueError):
        parent.fork(not_a_param=1)
    with pytest.raises(ValueError, match="slice_rounds"):
        via_fork.slice_rounds(25).fork(n_new=1)


def test_mixed_depth_rounds_store_only_their_nodes():
    # R2: rounds of depth 4 and 7 in one model
    X, Y, era = _data(n=4000, seed=6)
    model = _multi(n_estimators=20, min_child_weight=1).fit(X, Y, era_id=era)
    deep = model.fork(n_new=10, max_depth=7)
    deep.fit(X, Y, era_id=era)
    nodes = sum(b.feature.size for b in deep._forest_blocks())
    assert nodes <= 20 * (2**5 - 1) + 10 * (2**8 - 1)
    _check_exact(deep, X, [20, 30])
    # Device copy of the trees: the size of the stored trees, not 2^(7+1) slots per tree
    deep.release_prediction_cache()
    deep.predict(X)
    report = deep.memory_report()
    assert report["device_cache_bytes"] < 2 * report["forest_bytes"]
    assert report["device_cache_bytes"] < 30 * 2**8 * (16 + 4 * K) / 4


def test_compact_trees_are_much_smaller_than_the_v28_caches():
    # R3: v2.8 kept per tree (depth 4, K = 20) two copies of [32, 4] int32 + [32, 20] float32
    # plus a flat copy; the blocks hold the actual nodes once
    X, Y, era = _data(n=3000, k=20, seed=7)
    model = _multi(k=20, n_estimators=40).fit(X, Y, era_id=era)
    cp = model.fork(n_new=40)
    cp.fit(X, _only(Y, [5]), era_id=era)
    part = cp.slice_rounds(40)
    v28_per_tree = 2 * (32 * 4 * 4 + 32 * 20 * 4) + 32 * 25
    assert model.memory_report()["forest_bytes"] / 40 < 0.5 * v28_per_tree
    # Leaves of a 1-output copy store one column
    assert part.memory_report()["forest_bytes"] / 40 < 0.1 * v28_per_tree
    assert set(part._forest_blocks()[0].n_cols.tolist()) <= {0, 1}


# ─── R1: releasing the prediction caches ───


def test_release_returns_gpu_memory_to_the_pre_predict_level():
    X, Y, era = _data(seed=8)
    model = _multi(n_estimators=40).fit(X, Y, era_id=era)
    model.release_prediction_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    ref = model.predict(X)
    assert model.memory_report()["device_cache_bytes"] > 0
    assert torch.cuda.memory_allocated() > before
    model.release_prediction_cache()
    assert torch.cuda.memory_allocated() == before
    np.testing.assert_array_equal(model.predict(X), ref)
    model.release_prediction_cache()
    # cache=False never keeps a device copy
    np.testing.assert_array_equal(model.predict(X, cache=False), ref)
    np.testing.assert_array_equal(model.staged_predict(X, [40], cache=False)[0], ref)
    assert torch.cuda.memory_allocated() == before
    assert model.memory_report()["device_cache_bytes"] == 0
    # The v2.8 private name still works
    model.predict(X)
    model._invalidate_prediction_cache()
    assert torch.cuda.memory_allocated() == before


def test_training_tensors_are_released_after_fit():
    X, Y, era = _data(seed=9)
    model = _multi(colsample_bytree=0.5, monotonic_constraints=[1] + [0] * 9).fit(X, Y, era_id=era)
    assert model.feature_indices is None and model.feat_indices_tree is None
    assert model.monotonic_required_split_direction_gpu_ is None
    assert not model.unique_eras.is_cuda
    assert model.memory_report()["other_device_bytes"] <= model.base_prediction.numel() * 4
    # ... and rebuilt by a warm start
    model.warm_start, model.n_estimators = True, 30
    model.fit(X, Y, era_id=era)
    assert model.num_trees_ == 30


# ─── R5 / R7 ───


def test_init_scores_give_the_same_trees():
    X, Y, era = _data(seed=10)
    parent = _multi().fit(X, Y, era_id=era)
    rows = np.isfinite(Y[:, 3]) & (np.arange(len(Y)) % 3 != 0)
    raw = parent.predict_raw(X)  # once for several copies
    a = parent.fork(n_new=10)
    a.fit(X[rows], _only(Y[rows], [3]), era_id=era[rows])
    b = parent.fork(n_new=10)
    b.fit(X[rows], _only(Y[rows], [3]), era_id=era[rows], init_scores=raw[rows])
    c = parent.fork(n_new=10)
    c.fit(X[rows], _only(Y[rows], [3]), era_id=era[rows], init_scores=torch.from_numpy(raw[rows]).cuda())
    assert a.forest == b.forest == c.forest
    np.testing.assert_array_equal(a.train_predictions_, b.train_predictions_)
    with pytest.raises(ValueError, match="shape"):
        parent.fork(n_new=1).fit(X, Y, era_id=era, init_scores=raw[:10])
    with pytest.raises(ValueError, match="warm_start"):
        _multi().fit(X, Y, era_id=era, init_scores=raw)


def test_keep_train_predictions_false():
    X, Y, era = _data(seed=11)
    model = _multi().fit(X, Y, era_id=era, keep_train_predictions=False)
    assert model.train_predictions_ is None
    assert model.memory_report()["train_predictions_bytes"] == 0
    kept = _multi().fit(X, Y, era_id=era)
    assert kept.forest == model.forest and kept.train_predictions_.shape == Y.shape


@pytest.mark.parametrize("device", [None, "cpu"])
def test_predict_many_equals_predict_and_shares_prefixes(chain, device, monkeypatch):
    X, _, _, parent, mid, leaf = chain
    sibling = mid.fork(n_new=4)
    sibling.fit(X, _only(chain[1], [1]), era_id=chain[2])
    part = leaf.slice_rounds(mid.num_trees_)
    models = [parent, mid, leaf, sibling, part]
    expected = np.stack([m.predict(X, device=device) for m in models])
    raw_expected = np.stack([m.predict_raw(X, device=device) for m in models])
    if device is None:
        import warpgbm.core as core

        calls = []
        original = core._run_block

        def counted(*args, **kwargs):
            calls.append(args[2])
            return original(*args, **kwargs)

        monkeypatch.setattr(core, "_run_block", counted)
    np.testing.assert_array_equal(warpgbm.predict_many(models, X, device=device), expected)
    if device is None:
        # parent block once, mid's block once (leaf and sibling continue from it), leaf's,
        # sibling's own block, part = leaf's own block from zero sums
        assert len(calls) == 5
    np.testing.assert_array_equal(
        warpgbm.predict_many(models, X, device=device, raw=True), raw_expected
    )
    rounds = [3, 30, 41, 40, 7]
    np.testing.assert_array_equal(
        warpgbm.predict_many(models, X, num_trees=rounds, device=device, raw=True),
        np.stack([m.predict_raw(X, num_trees=r, device=device) for m, r in zip(models, rounds)]),
    )


def test_predict_many_validation_and_raw_feature_groups():
    X, Y, era = _data(seed=12)
    a = _multi().fit(X, Y, era_id=era)
    b = _multi(random_state=1).fit(X[:1200], Y[:1200], era_id=era[:1200])  # other bin edges
    np.testing.assert_array_equal(
        warpgbm.predict_many([a, b], X), np.stack([a.predict(X), b.predict(X)])
    )
    with pytest.raises(ValueError):
        warpgbm.predict_many([], X)
    with pytest.raises(ValueError):
        warpgbm.predict_many([a, _multi(k=2).fit(X, Y[:, :2])], X)
    with pytest.raises(ValueError):
        warpgbm.predict_many([a, b], X, num_trees=[1])
