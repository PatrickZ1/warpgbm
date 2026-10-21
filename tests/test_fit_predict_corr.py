import numpy as np
from warpgbm import WarpGBM
from sklearn.datasets import make_regression
import time
from sklearn.metrics import mean_squared_error


def test_fit_predictpytee_correlation():
    np.random.seed(42)
    N = 100_000
    F = 1000
    X, y = make_regression(n_samples=N, n_features=F, noise=0.1, random_state=42)
    era = np.zeros(N, dtype=np.int32)
    corrs = []
    mses = []

    model = WarpGBM(
        max_depth=10,
        num_bins=10,
        n_estimators=100,
        learning_rate=1,
        colsample_bytree=1.0,
    )

    start_fit = time.time()
    model.fit(
        X,
        y,
        era_id=era,
        X_eval=X,
        y_eval=y,
        eval_every_n_trees=10,
        early_stopping_rounds=1,
        eval_metric="corr",
    )
    fit_time = time.time() - start_fit
    print(f"  Fit time:     {fit_time:.3f} seconds")

    start_pred = time.time()
    preds = model.predict(X)
    pred_time = time.time() - start_pred
    print(f"  Predict time: {pred_time:.3f} seconds")

    corr = np.corrcoef(preds, y)[0, 1]
    mse = mean_squared_error(preds, y)
    print(f"  Correlation:  {corr:.4f}")
    print(f"  MSE:  {mse:.4f}")

    assert (corr > 0.9), f"In-sample correlation too low: {corrs}"
    assert (mse < 2), f"In-sample mse too high: {mses}"
