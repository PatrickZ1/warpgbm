import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from warpgbm import WarpGBM


def test_sample_weight_all_ones_equivalent_regression():
    X, y = make_regression(n_samples=300, n_features=12, noise=0.1, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    w = np.ones(X.shape[0], dtype=np.float32) * 3.0

    model_unweighted = WarpGBM(
        objective="regression",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        colsample_bytree=0.8,
        device="cuda",
    )
    model_unweighted.fit(X, y)
    pred_unweighted = model_unweighted.predict(X)

    model_weighted = WarpGBM(
        objective="regression",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        colsample_bytree=0.8,
        device="cuda",
    )
    model_weighted.fit(X, y, sample_weight=w)
    pred_weighted = model_weighted.predict(X)

    np.testing.assert_allclose(pred_unweighted, pred_weighted, rtol=1e-3, atol=1e-3)


def test_sample_weight_all_ones_equivalent_binary_classification():
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=6,
        n_redundant=4,
        n_classes=2,
        random_state=7,
    )
    X = X.astype(np.float32)
    w = np.ones(X.shape[0], dtype=np.float32)

    model_unweighted = WarpGBM(
        objective="binary",
        n_estimators=25,
        max_depth=3,
        learning_rate=0.1,
        random_state=7,
        colsample_bytree=0.8,
        device="cuda",
    )
    model_unweighted.fit(X, y)
    proba_unweighted = model_unweighted.predict_proba(X)

    model_weighted = WarpGBM(
        objective="binary",
        n_estimators=25,
        max_depth=3,
        learning_rate=0.1,
        random_state=7,
        colsample_bytree=0.8,
        device="cuda",
    )
    model_weighted.fit(X, y, sample_weight=w)
    proba_weighted = model_weighted.predict_proba(X)

    np.testing.assert_allclose(proba_unweighted, proba_weighted, rtol=1e-3, atol=1e-3)

def test_sample_weight_eval_for_multiclass_accuracy():
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )
    X = X.astype(np.float32)

    w_train = np.ones(X.shape[0], dtype=np.float32)
    w_train[y == 0] = 3.0

    w_eval = np.ones(X.shape[0], dtype=np.float32)
    w_eval[y == 2] = 5.0

    model = WarpGBM(
        objective="multiclass",
        n_estimators=15,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        device="cuda",
    )

    model.fit(
        X,
        y,
        sample_weight=w_train,
        X_eval=X,
        y_eval=y,
        sample_weight_eval=w_eval,
        eval_every_n_trees=5,
        early_stopping_rounds=3,
        eval_metric="accuracy",
    )

    assert len(model.training_loss) > 0
    assert len(model.eval_loss) > 0
    assert all(np.isfinite(model.eval_loss))


def test_sample_weight_validation_errors():
    X, y = make_regression(n_samples=100, n_features=6, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = WarpGBM(objective="regression", n_estimators=5, max_depth=2)

    with pytest.raises(ValueError, match="strictly positive"):
        bad_w = np.ones(X.shape[0], dtype=np.float32)
        bad_w[0] = 0.0
        model.fit(X, y, sample_weight=bad_w)

    with pytest.raises(ValueError, match="same length"):
        wrong_len = np.ones(X.shape[0] - 1, dtype=np.float32)
        model.fit(X, y, sample_weight=wrong_len)

    with pytest.raises(ValueError, match="sample_weight_eval can only be used"):
        eval_w = np.ones(X.shape[0], dtype=np.float32)
        model.fit(X, y, sample_weight_eval=eval_w)


def test_weight_effect_noise_regression_tracks_weighted_target_mean():
    rng = np.random.default_rng(123)
    n = 400

    # Pure noise features, random binary targets in {-1, +1}
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)

    w_pos_heavy = np.ones(n, dtype=np.float32)
    w_pos_heavy[y > 0] = 8.0

    w_neg_heavy = np.ones(n, dtype=np.float32)
    w_neg_heavy[y < 0] = 8.0

    params = dict(
        objective="regression",
        n_estimators=80,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=n + 1,  # prevent splits -> constant model behavior
        random_state=123,
    )

    model_pos = WarpGBM(**params)
    model_pos.fit(X, y, sample_weight=w_pos_heavy)
    pred_pos = model_pos.predict(X)

    model_neg = WarpGBM(**params)
    model_neg.fit(X, y, sample_weight=w_neg_heavy)
    pred_neg = model_neg.predict(X)

    expected_pos = np.average(y, weights=w_pos_heavy)
    expected_neg = np.average(y, weights=w_neg_heavy)

    assert pred_pos.mean() > pred_neg.mean()
    assert abs(pred_pos.mean() - expected_pos) < 0.15
    assert abs(pred_neg.mean() - expected_neg) < 0.15


def test_weight_effect_noise_binary_classification_shifts_class_probability():
    rng = np.random.default_rng(321)
    n = 500

    # Pure noise features, random binary labels
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.int32)

    w_one_heavy = np.ones(n, dtype=np.float32)
    w_one_heavy[y == 1] = 10.0

    w_zero_heavy = np.ones(n, dtype=np.float32)
    w_zero_heavy[y == 0] = 10.0

    params = dict(
        objective="binary",
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=n + 1,  # prevent splits -> global class bias learning
        random_state=321,
    )

    model_one = WarpGBM(**params)
    model_one.fit(X, y, sample_weight=w_one_heavy)
    p1_one = model_one.predict_proba(X)[:, 1].mean()

    model_zero = WarpGBM(**params)
    model_zero.fit(X, y, sample_weight=w_zero_heavy)
    p1_zero = model_zero.predict_proba(X)[:, 1].mean()

    expected_one = np.average(y, weights=w_one_heavy)
    expected_zero = np.average(y, weights=w_zero_heavy)

    assert p1_one > p1_zero
    assert abs(p1_one - expected_one) < 0.20
    assert abs(p1_zero - expected_zero) < 0.20
