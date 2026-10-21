import numpy as np
from sklearn.datasets import make_regression

from warpgbm import WarpGBM


def _build_regression_model(X, y, n_estimators=40):
    model = WarpGBM(
        objective="regression",
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        random_state=123,
    )
    model.fit(X, y)
    return model


def test_predict_full_equals_two_subsets():
    X, y = make_regression(
        n_samples=800,
        n_features=16,
        noise=0.2,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = _build_regression_model(X, y, n_estimators=30)

    full_preds = model.predict(X)
    split_idx = X.shape[0] // 2
    chunk_preds = np.concatenate(
        [
            model.predict(X[:split_idx]),
            model.predict(X[split_idx:]),
        ]
    )

    np.testing.assert_allclose(full_preds, chunk_preds, rtol=1e-5, atol=1e-4)


def test_flatten_cache_reused_and_invalidated_on_retraining():
    X, y = make_regression(
        n_samples=500,
        n_features=12,
        noise=0.2,
        random_state=7,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = _build_regression_model(X, y, n_estimators=20)

    # Every round is flattened (packed) once; later predictions reuse it
    call_counter = {"calls": 0}
    original_pack = model._pack_tree

    def counted_pack(tree, max_nodes, leaf_width):
        call_counter["calls"] += 1
        return original_pack(tree, max_nodes, leaf_width)

    model._pack_tree = counted_pack

    model.predict(X[:200])
    assert call_counter["calls"] == 20

    model.predict(X[200:400])
    model.predict(X[:50], num_trees=7)
    model.staged_predict(X[:50], [3, 9, 20])
    assert call_counter["calls"] == 20

    # Warm start: only the new rounds are packed
    model.warm_start = True
    model.n_estimators = 30
    model.fit(X, y)
    model.predict(X[:200])
    assert call_counter["calls"] == 30

    # A fresh fit replaces the forest: everything is packed again
    model.warm_start = False
    model.n_estimators = 15
    model.fit(X, y)
    model.predict(X[:200])
    assert call_counter["calls"] == 45

    # truncate keeps the packed rounds it retains
    model.truncate(10)
    model.predict(X[:200])
    assert call_counter["calls"] == 45
