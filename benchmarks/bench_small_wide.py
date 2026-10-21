"""
Benchmark of training and prediction on small, wide data (the hedge workload):
10k rows x 1000 features, K = 20 outputs (multi_regression), a few eras.

    python benchmarks/bench_small_wide.py              # variant grid, both modes
    python benchmarks/bench_small_wide.py --quick      # baseline config only
    python benchmarks/bench_small_wide.py --profile    # CUDA kernel breakdown of the baseline
    python benchmarks/bench_small_wide.py --large      # also depth 7 / 64 bins / 8 eras / colsample 1

Prints ms per tree for every variant, single-row predict latency and the cost of staged
prediction over 16 checkpoints. Works against older versions too (no `deterministic` there).
"""
import argparse
import inspect
import time

import numpy as np
import torch

from warpgbm import WarpGBM

BASE = dict(
    objective="multi_regression",
    num_bins=16,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=50,
    L2_reg=0.3,
    colsample_bytree=0.4,
    random_state=0,
    verbose=False,
)

VARIANTS = [
    ("baseline", {}),
    ("depth 7", dict(max_depth=7)),
    ("64 bins", dict(num_bins=64)),
    ("8 eras", dict(_eras=8)),
    ("colsample 1.0", dict(colsample_bytree=1.0)),
    ("K = 1", dict(_k=1)),
]
# Needs several GB of GPU memory
LARGE = [("depth 7, 64 bins, 8 eras, cs 1.0", dict(max_depth=7, num_bins=64, _eras=8, colsample_bytree=1.0))]

_HAS_DETERMINISTIC = "deterministic" in inspect.signature(WarpGBM.__init__).parameters


def make_data(n=10_000, f=1000, k=20, eras=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float32)
    X[rng.random((n, f)) < 0.02] = np.nan
    signal = np.nan_to_num(X[:, :k]) * 0.5 + 0.3 * np.nan_to_num(X[:, k : 2 * k]) ** 2
    Y = (signal + rng.standard_normal((n, k))).astype(np.float32)
    if k == 1:
        Y = Y[:, 0]
    era = np.repeat(np.arange(eras), n // eras + 1)[:n].astype(np.int32)
    return X, Y, era


def params(overrides, deterministic):
    p = dict(BASE)
    p.update({key: v for key, v in overrides.items() if not key.startswith("_")})
    if p.get("objective") == "multi_regression" and overrides.get("_k", 20) == 1:
        p["objective"] = "regression"
    if _HAS_DETERMINISTIC:
        p["deterministic"] = deterministic
    return p


def time_fit(X, Y, era, p, n_trees):
    # Warm-up fit on a small slice (kernel loading, caching allocator)
    WarpGBM(**{**p, "n_estimators": 2}).fit(X[:500], Y[:500], era_id=era[:500])
    torch.cuda.synchronize()
    t = time.perf_counter()
    model = WarpGBM(**{**p, "n_estimators": n_trees}).fit(X, Y, era_id=era)
    torch.cuda.synchronize()
    return model, (time.perf_counter() - t) / n_trees * 1000


def time_call(fn, repeats):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeats * 1000


def bench_prediction(model, X):
    x1 = X[:1]
    print(f"  single-row predict (GPU):         {time_call(lambda: model.predict(x1), 200):8.3f} ms")
    if "device" in inspect.signature(model.predict).parameters:
        print(f"  single-row predict (CPU):         {time_call(lambda: model.predict(x1, device='cpu'), 200):8.3f} ms")
    rounds = list(range(5, model.num_trees_ + 1, 5))[:16]
    Xv = X[:2000]
    if hasattr(model, "staged_predict"):
        t = time_call(lambda: model.staged_predict(Xv, rounds), 5)
        print(f"  staged predict, {len(rounds)} checkpoints:     {t:8.3f} ms")

    def loop():
        model._invalidate_prediction_cache()
        for c in rounds:
            model.predict(Xv, num_trees=c)

    print(f"  predict at {len(rounds)} checkpoints (cold):  {time_call(loop, 2):8.3f} ms")


def profile_baseline(X, Y, era, deterministic):
    from torch.profiler import ProfilerActivity, profile

    p = params({}, deterministic)
    WarpGBM(**{**p, "n_estimators": 2}).fit(X[:500], Y[:500], era_id=era[:500])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        WarpGBM(**{**p, "n_estimators": 10}).fit(X, Y, era_id=era)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="baseline config only")
    parser.add_argument("--profile", action="store_true", help="kernel breakdown of the baseline")
    parser.add_argument("--large", action="store_true", help="add the memory-heavy variant")
    parser.add_argument("--trees", type=int, default=30)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}, deterministic switch: {_HAS_DETERMINISTIC}")
    modes = [True, False] if _HAS_DETERMINISTIC else [True]
    data = {}

    def get_data(eras, k):
        if (eras, k) not in data:
            data[(eras, k)] = make_data(eras=eras, k=k)
        return data[(eras, k)]

    if args.profile:
        X, Y, era = get_data(3, 20)
        for det in modes:
            print(f"\n── profile, deterministic={det} ──")
            profile_baseline(X, Y, era, det)
        return

    variants = VARIANTS[:1] if args.quick else VARIANTS + (LARGE if args.large else [])
    print(f"\n{'variant':<36}" + "".join(f"{'det=' + str(m):>14}" for m in modes) + "   (ms / tree)")
    baseline_models = {}
    for name, overrides in variants:
        X, Y, era = get_data(overrides.get("_eras", 3), overrides.get("_k", 20))
        n_trees = args.trees if overrides.get("max_depth", 4) < 7 else max(5, args.trees // 3)
        row = f"{name:<36}"
        for det in modes:
            model, ms = time_fit(X, Y, era, params(overrides, det), n_trees)
            row += f"{ms:14.1f}"
            if name == "baseline":
                baseline_models[det] = model
        print(row, flush=True)

    X, Y, era = get_data(3, 20)
    for det, _ in baseline_models.items():
        model, _ = time_fit(X, Y, era, params({}, det), 80)
        print(f"\n── prediction, baseline model with 80 trees, deterministic={det} ──")
        bench_prediction(model, X)


if __name__ == "__main__":
    main()
