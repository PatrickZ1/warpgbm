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


def test_device_copies_cached_per_block_and_released(monkeypatch):
    from warpgbm.forest import Block, _device_key

    X, y = make_regression(
        n_samples=500,
        n_features=12,
        noise=0.2,
        random_state=7,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = _build_regression_model(X, y, n_estimators=20)

    # The device copy of a block is uploaded once; later predictions reuse it
    uploads = []
    original = Block.device_arrays

    def counted(self, device, cache=True):
        if _device_key(device) not in self._device:
            uploads.append(self.num_trees)
        return original(self, device, cache)

    monkeypatch.setattr(Block, "device_arrays", counted)

    model.predict(X[:200])
    assert uploads == [20]
    model.predict(X[200:400])
    model.predict(X[:50], num_trees=7)
    model.staged_predict(X[:50], [3, 9, 20])
    assert uploads == [20]

    # Warm start: the training scores reuse the cached copy; the small block is extended
    model.warm_start = True
    model.n_estimators = 30
    model.fit(X, y)
    assert uploads == [20] and len(model._forest_blocks()) == 1
    model.predict(X[:200])
    assert uploads == [20, 30]

    # A fresh fit replaces the forest
    model.warm_start = False
    model.n_estimators = 15
    model.fit(X, y)
    ref10 = model.predict(X[:200], num_trees=10)
    assert uploads == [20, 30, 15]

    # truncate copies the block part it keeps
    model.truncate(10)
    np.testing.assert_array_equal(model.predict(X[:200]), ref10)
    assert uploads == [20, 30, 15, 10]

    # release drops the device copies; cache=False never stores one
    model.release_prediction_cache()
    assert model.memory_report()["device_cache_bytes"] == 0
    np.testing.assert_array_equal(model.predict(X[:200], cache=False), ref10)
    assert model.memory_report()["device_cache_bytes"] == 0
    assert uploads == [20, 30, 15, 10, 10]
