"""
Benchmark of a hedge level chain (hedge_ts_v2 hedge/train.py::fit_chain): a K = 20 multi_regression
code model, 5 warm-started structure copies (targets NaN except 4 outputs) and 20 output copies of
those (1 output), prebinned, then every copy sliced to its own rounds (LevelChain.compact).
12k rows x 1863 features, 64 bins, 8 eras, hedge's default level parameters.

    python benchmarks/bench_chain.py                    # depth 4 everywhere
    python benchmarks/bench_chain.py --copy-depth 7     # copies grown at depth 7
    python benchmarks/bench_chain.py --small            # 200 / 150 / 100 rounds
    python benchmarks/bench_chain.py --save a.npz       # store the predictions ...
    python benchmarks/bench_chain.py --compare a.npz b.npz   # ... and compare two runs bitwise

Prints fit / copy / predict times, GPU memory after predicting with all 26 full models, the bytes
held per model (forest + caches, CPU and GPU) and the CPU latency of a 4-row chain prediction.
Works against v2.8 (deepcopy path) and v2.9 (fork, predict_many, release_prediction_cache).
Run with PYTHONPATH=<repo>, else it imports the installed package.
"""
import argparse
import copy
import gc
import sys
import time

import numpy as np
import torch

import warpgbm
from warpgbm import WarpGBM, apply_bins, fit_bins

N, N_VA, F, K, ERAS, BINS = 12_000, 4_000, 1863, 20, 8, 64
STRUCTURE_COLS = [list(range(4 * s, 4 * s + 4)) for s in range(5)]
CODE = dict(
    objective="multi_regression", num_bins=BINS, max_depth=4, learning_rate=0.0297,
    min_child_weight=29.8, L2_reg=0.236, colsample_bytree=0.0279, min_split_gain=0.12,
    min_sum_hessian=23.9, prebinned=True, random_state=0, verbose=False,
)
LEVEL_ATTRS = ("learning_rate", "max_depth", "min_child_weight", "min_split_gain", "min_sum_hessian",
               "L2_reg", "colsample_bytree")
NEW = hasattr(WarpGBM, "fork")


def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N + N_VA, F)).astype(np.float32)
    X[rng.random(X.shape) < 0.02] = np.nan
    W = rng.normal(size=(40, K)).astype(np.float32) / np.sqrt(40)
    Y = (np.tanh(np.nan_to_num(X[:, :40])) @ W + 2.0 * rng.normal(size=(N + N_VA, K))).astype(np.float32)
    Y[rng.random(Y.shape) < 0.1] = np.nan
    edges = fit_bins(X[:N], BINS, "cuda")
    B = apply_bins(X, edges)
    era = (np.arange(N) * ERAS // N).astype(np.int32)
    return B[:N], Y[:N], era, B[N:]


def specialize(parent, cols, B, Y, era, n_new, scale, depth):
    rows = np.isfinite(Y[:, cols]).any(axis=1)
    Ys = np.full_like(Y[rows], np.nan)
    Ys[:, cols] = Y[rows][:, cols]
    params = dict(min_child_weight=CODE["min_child_weight"] * scale,
                  min_split_gain=CODE["min_split_gain"] * scale, max_depth=depth)
    t0 = time.perf_counter()
    if NEW:
        m = parent.fork(n_new=n_new, **params)
    else:
        m = copy.deepcopy(parent)
        for a, v in params.items():
            setattr(m, a, v)
        m.warm_start = True
        m.n_estimators = parent.num_trees_ + n_new
    t1 = time.perf_counter()
    kw = {"keep_train_predictions": False} if NEW else {}
    m.fit(B[rows], Ys, era_id=era[rows], **kw)
    torch.cuda.synchronize()
    return m, t1 - t0, time.perf_counter() - t1


def rss_mb():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024


def held_bytes(model):
    """(cpu, gpu) bytes reachable from the model: arrays / tensors / Python objects (approx.)."""
    seen, buffers, cpu, gpu = set(), set(), 0, 0
    stack = [model.__dict__]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            base = o
            while isinstance(base.base, np.ndarray):
                base = base.base
            if id(base) not in buffers:
                buffers.add(id(base))
                cpu += base.nbytes
        elif torch.is_tensor(o):
            key = (o.untyped_storage().data_ptr(), str(o.device))
            if key not in buffers:
                buffers.add(key)
                n = o.untyped_storage().nbytes()
                if o.is_cuda:
                    gpu += n
                else:
                    cpu += n
        elif isinstance(o, dict):
            cpu += sys.getsizeof(o)
            stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            cpu += sys.getsizeof(o)
            stack.extend(o)
        elif isinstance(o, (float, int, bool, str)) or o is None:
            cpu += sys.getsizeof(o)
        elif hasattr(o, "__dict__") and type(o).__module__.startswith("warpgbm"):
            stack.append(o.__dict__)
    return cpu, gpu


def release(m):
    if hasattr(m, "release_prediction_cache"):
        m.release_prediction_cache()
    else:
        m._invalidate_prediction_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--copy-depth", type=int, default=4)
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        a, b = (np.load(p) for p in args.compare)
        for k in sorted(set(a) & set(b)):
            same = np.array_equal(a[k], b[k])
            diff = float(np.nanmax(np.abs(a[k].astype(np.float64) - b[k]))) if a[k].size else 0.0
            print(f"{k:16s} {'identical' if same else 'DIFFERENT'}  max |diff| {diff:.3g}")
        return

    n_code, n_struct, n_out = (200, 150, 100) if args.small else (600, 500, 300)
    print(f"warpgbm {getattr(warpgbm, '__file__', '?')} | {'v2.9 API' if NEW else 'v2.8 API'} | "
          f"rounds {n_code}/{n_struct}/{n_out}, copy depth {args.copy_depth}")
    B, Y, era, B_va = data()
    torch.cuda.synchronize()
    gc.collect()
    rss0 = rss_mb()

    t0 = time.perf_counter()
    code = WarpGBM(**CODE, n_estimators=n_code)
    code.fit(B, Y, era_id=era, **({"keep_train_predictions": False} if NEW else {}))
    torch.cuda.synchronize()
    t_code = time.perf_counter() - t0

    structs, outs, tc, tf = [], [], {"s": 0.0, "o": 0.0}, {"s": 0.0, "o": 0.0}
    for cols in STRUCTURE_COLS:
        m, c, f = specialize(code, cols, B, Y, era, n_struct, 4 / 20, args.copy_depth)
        structs.append(m)
        tc["s"] += c
        tf["s"] += f
    for s, cols in zip(structs, STRUCTURE_COLS):
        for o in cols:
            m, c, f = specialize(s, [o], B, Y, era, n_out, 1 / 20, args.copy_depth)
            outs.append(m)
            tc["o"] += c
            tf["o"] += f
    print(f"fit: code {t_code:.2f}s | 5 structure copies: copy {tc['s']:.2f}s fit {tf['s']:.2f}s | "
          f"20 output copies: copy {tc['o']:.2f}s fit {tf['o']:.2f}s | chain {t_code + sum(tc.values()) + sum(tf.values()):.2f}s")

    models = [code, *structs, *outs]
    saved = {}
    # GPU predictions of the 26 full models on the validation rows (tuning: FoldResult.predict)
    for m in models:
        release(m)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    preds = [m.predict(B_va) for m in models]
    torch.cuda.synchronize()
    t_pred = time.perf_counter() - t0
    held = torch.cuda.memory_allocated() - base_mem
    peak = torch.cuda.max_memory_allocated() - base_mem
    saved["full_gpu"] = np.stack(preds)
    line = f"GPU predict 26 full models x {N_VA} rows: {1e3 * t_pred:.0f} ms, caches held {held / 2**20:.1f} MB, peak +{peak / 2**20:.1f} MB"
    t0 = time.perf_counter()
    preds = [m.predict(B_va) for m in models]
    torch.cuda.synchronize()
    line += f" | warm {1e3 * (time.perf_counter() - t0):.0f} ms"
    if NEW:
        for m in models:
            m.release_prediction_cache()
        t0 = time.perf_counter()
        many = warpgbm.predict_many(models, B_va, cache=False)
        torch.cuda.synchronize()
        line += f" | predict_many {1e3 * (time.perf_counter() - t0):.0f} ms"
        assert np.array_equal(many, saved["full_gpu"]), "predict_many differs from predict"
    print(line)
    cpu_full = sum(held_bytes(m)[0] for m in models)
    for m in models:
        release(m)

    # Compaction: every copy keeps its own rounds only
    t0 = time.perf_counter()
    slices = [code] + [m.slice_rounds(n_code) for m in structs] + [m.slice_rounds(n_code + n_struct) for m in outs]
    t_slice = time.perf_counter() - t0
    del structs, outs, models, m
    gc.collect()
    saved["slices_gpu"] = np.stack([m.predict(B_va) for m in slices])
    for m in slices:
        release(m)

    # CPU inference of a few rows (hedge predict.predict_code_array): warm-up builds the CPU caches
    x4 = B_va[:4]
    saved["slices_cpu"] = np.stack([m.predict(B_va, device="cpu") for m in slices])
    gc.collect()
    rss1 = rss_mb()
    reps = 30
    t0 = time.perf_counter()
    for _ in range(reps):
        for m in slices:
            m.predict(x4, device="cpu")
    t_cpu = (time.perf_counter() - t0) / reps
    line = f"slice_rounds 25 copies {1e3 * t_slice:.0f} ms | CPU 4-row chain predict (26 calls) {1e3 * t_cpu:.2f} ms"
    if NEW:
        t0 = time.perf_counter()
        for _ in range(reps):
            warpgbm.predict_many(slices, x4, device="cpu")
        line += f" | predict_many cpu {1e3 * (time.perf_counter() - t0) / reps:.2f} ms"
    print(line)
    cpu_sl = [held_bytes(m) for m in slices]
    trees = [m.num_trees_ for m in slices]
    print(f"bytes held: 26 full models after predict {cpu_full / 2**20:.1f} MB CPU | 26 compacted models "
          f"(after CPU predict) {sum(c for c, _ in cpu_sl) / 2**20:.2f} MB CPU, {sum(g for _, g in cpu_sl) / 2**20:.2f} MB GPU "
          f"= {sum(c for c, _ in cpu_sl) / sum(trees):.0f} B/tree | RSS +{rss1 - rss0:.0f} MB")
    if NEW:
        rep = slices[-1].memory_report()
        print("memory_report(output slice):", {k: round(v / 1024, 1) for k, v in rep.items()}, "KB")
    if args.save:
        np.savez(args.save, **saved)
        print("saved", args.save)


if __name__ == "__main__":
    main()
