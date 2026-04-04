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

    call_counter = {"calls": 0}
    original_flatten = model.flatten_tree

    def counted_flatten(tree, max_nodes):
        call_counter["calls"] += 1
        return original_flatten(tree, max_nodes)

    model.flatten_tree = counted_flatten

    model.predict(X[:200])
    first_predict_calls = call_counter["calls"]
    assert first_predict_calls > 0

    model.predict(X[200:400])
    assert call_counter["calls"] == first_predict_calls

    model.warm_start = True
    model.n_estimators = 30
    model.fit(X, y)
    model.predict(X[:200])
    after_continue_calls = call_counter["calls"]
    assert after_continue_calls > first_predict_calls

    model.warm_start = False
    model.n_estimators = 15
    model.fit(X, y)
    model.predict(X[:200])
    assert call_counter["calls"] > after_continue_calls
