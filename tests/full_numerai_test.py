from numerapi import NumerAPI
import pandas as pd
import numpy as np
from warpgbm import WarpGBM
import time
from sklearn.metrics import mean_squared_error


def predict_in_chunks(model, X, chunk_size=100_000):
    preds = []
    for i in range(0, X.shape[0], chunk_size):
        X_chunk = X[i : i + chunk_size]
        preds.append(model.predict(X_chunk))
    return np.concatenate(preds)


def test_numerai_data():
    napi = NumerAPI()
    napi.download_dataset("v5.0/train.parquet", "numerai_train.parquet")
    napi.download_dataset("v5.0/validation.parquet", "numerai_validation.parquet")

    data = pd.concat([
        pd.read_parquet("numerai_train.parquet"),
        pd.read_parquet("numerai_validation.parquet")
    ])
    features = [f for f in list(data) if "feature" in f]
    target = "target"
    data = data.loc[data[ target].isna() == False ]

    X = data[features].astype("int8").values[:]
    y = data[target].values

    model = WarpGBM(
        max_depth=3,
        num_bins=5,
        n_estimators=10,
        learning_rate=1,
        colsample_bytree=0.8,
    )

    start_fit = time.time()
    model.fit(
        X,
        y,
        # era_id=era,
        # X_eval=X,
        # y_eval=y,
        # eval_every_n_trees=10,
        # early_stopping_rounds=1,
    )
    fit_time = time.time() - start_fit
    print(f"  Fit time:     {fit_time:.3f} seconds")

    start_pred = time.time()
    preds = predict_in_chunks(model, X, chunk_size=500_000)
    pred_time = time.time() - start_pred
    print(f"  Predict time: {pred_time:.3f} seconds")

    corr = np.corrcoef(preds, y)[0, 1]
    mse = mean_squared_error(preds, y)
    print(f"  Correlation:  {corr:.4f}")
    print(f"  MSE:  {mse:.4f}")

    # assert corr > 0.68, f"In-sample correlation too low: {corr}"
    # assert mse < 0.03, f"In-sample mse too high: {mse}"
