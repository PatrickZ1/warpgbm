import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from warpgbm import WarpGBM


def test_sample_weight_all_ones_equivalent_regression():
    X, y = make_regression(n_samples=300, n_features=12, noise=0.1, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    w = np.ones(X.shape[0], dtype=np.float32)

    model_unweighted = WarpGBM(
        objective="regression",
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        colsample_bytree=0.8,
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
    )
    model_weighted.fit(X, y, sample_weight=w)
    pred_weighted = model_weighted.predict(X)

    np.testing.assert_allclose(pred_unweighted, pred_weighted, rtol=1e-3, atol=1e-3)


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
