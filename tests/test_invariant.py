import numpy as np
from warpgbm import WarpGBM
import time

import os
import requests

def download_file_if_missing(url, local_dir):
    filename = os.path.basename(url)
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"✅ {filename} already exists, skipping download.")
        return

    # Convert GitHub blob URL to raw URL
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    print(f"⬇️  Downloading {filename}...")
    response = requests.get(raw_url)
    response.raise_for_status()

    os.makedirs(local_dir, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Saved to {local_path}")

# === Usage ===

urls = [
    "https://github.com/jefferythewind/era-splitting-notebook-examples/blob/main/Synthetic%20Memorization%20Data%20Set/X_train.npy",
    "https://github.com/jefferythewind/era-splitting-notebook-examples/blob/main/Synthetic%20Memorization%20Data%20Set/y_train.npy",
    "https://github.com/jefferythewind/era-splitting-notebook-examples/blob/main/Synthetic%20Memorization%20Data%20Set/X_test.npy",
    "https://github.com/jefferythewind/era-splitting-notebook-examples/blob/main/Synthetic%20Memorization%20Data%20Set/y_test.npy",
    "https://github.com/jefferythewind/era-splitting-notebook-examples/blob/main/Synthetic%20Memorization%20Data%20Set/X_eras.npy",
]


local_folder = "./synthetic_data"

for url in urls:
    download_file_if_missing(url, local_folder)

def test_fit_predictpytee_correlation():
    import numpy as np
    import os
    from warpgbm import WarpGBM
    from sklearn.metrics import mean_squared_error
    import time

    # Load the real dataset from local .npy files
    data_dir = "./synthetic_data"
    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))
    # era = np.zeros(X.shape[0], dtype=np.int32)  # one era for default GBDT equivalence
    era = np.load(os.path.join(data_dir, "X_eras.npy"))

    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    model = WarpGBM(
        max_depth=10,
        num_bins=127,
        n_estimators=50,
        learning_rate=1,
        colsample_bytree=0.9,
        min_child_weight=4,
        # learning_rate=1 makes the result sensitive to the column draws (test MSE 0.008-0.09
        # across seeds); fix the seed so the test is deterministic
        random_state=0,
    )

    start_fit = time.time()
    model.fit(
        X,
        y,
        era_id=era,
        X_eval=X_test,
        y_eval=y_test,
        eval_every_n_trees=10,
        early_stopping_rounds=100,
        eval_metric="corr",
    )
    fit_time = time.time() - start_fit
    print(f"  Fit time:     {fit_time:.3f} seconds")

    start_pred = time.time()
    preds = model.predict(X_test)
    pred_time = time.time() - start_pred
    print(f"  Predict time: {pred_time:.3f} seconds")

    corr = np.corrcoef(preds, y_test)[0, 1]
    mse = mean_squared_error(preds, y_test)
    print(f"  Correlation:  {corr:.4f}")
    print(f"  MSE:          {mse:.4f}")

    assert corr > 0.95, f"Out-of-sample correlation too low: {corr:.4f}"
    assert mse < 0.02, f"Out-of-sample MSE too high: {mse:.4f}"

