"""
Benchmark of warm-started copies trained on a subset of the outputs (the hedge chain): a K = 20
multi_regression parent, then copies whose targets are NaN except for 4 outputs (structure
level, 500 rounds) or 1 output (output level, 300 rounds). 12k rows x 1000 features, depth 4,
64 bins, 8 eras.

    python benchmarks/bench_inactive_outputs.py            # skip inactive outputs vs all K
    python benchmarks/bench_inactive_outputs.py --legacy   # current code only (e.g. v2.7.1)

Run with PYTHONPATH=<repo>, else it imports the installed package.
"""
import argparse
import copy
import time

import numpy as np
import torch

from warpgbm import WarpGBM

N, F, K, ERAS = 12_000, 1000, 20, 8
PARAMS = dict(
    objective="multi_regression",
    num_bins=64,
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=50,
    L2_reg=0.3,
    colsample_bytree=0.4,
    random_state=0,
    verbose=False,
)


def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, F)).astype(np.float32)
    W = rng.normal(size=(F, K)).astype(np.float32) / np.sqrt(F)
    Y = (np.tanh(X) @ W + rng.normal(size=(N, K))).astype(np.float32)
    era = np.repeat(np.arange(ERAS), N // ERAS).astype(np.int32)
    return X, Y, era


def time_copy(parent, X, Y, era, active, rounds, skip):
    cp = copy.deepcopy(parent)
    cp.warm_start = True
    cp.n_estimators = parent.num_trees_ + rounds
    if not skip:
        cp._skip_inactive_outputs = False
    Ys = np.full_like(Y, np.nan)
    Ys[:, active] = Y[:, active]
    torch.cuda.synchronize()
    t = time.perf_counter()
    cp.fit(X, Ys, era_id=era)
    torch.cuda.synchronize()
    return time.perf_counter() - t, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", action="store_true", help="only the default code path")
    ap.add_argument("--parent-rounds", type=int, default=100)
    args = ap.parse_args()

    X, Y, era = data()
    parent = WarpGBM(n_estimators=args.parent_rounds, **PARAMS)
    parent.fit(X, Y, era_id=era)
    cases = [("structure copy (4 of 20, 500 rounds)", [0, 5, 10, 15], 500),
             ("output copy (1 of 20, 300 rounds)", [7], 300)]
    modes = [("default", True)] if args.legacy else [("all K", False), ("active only", True)]
    for name, active, rounds in cases:
        time_copy(parent, X, Y, era, active, 20, True)  # warm-up (kernels, allocator)
        results = {}
        for mode, skip in modes:
            results[mode], cp = time_copy(parent, X, Y, era, active, rounds, skip)
        line = " | ".join(f"{m}: {s:6.2f} s ({1000 * s / rounds:5.2f} ms/round)" for m, s in results.items())
        if not args.legacy:
            line += f" | speed-up {results['all K'] / results['active only']:.1f}x"
        print(f"{name:40s} {line}")


if __name__ == "__main__":
    main()
