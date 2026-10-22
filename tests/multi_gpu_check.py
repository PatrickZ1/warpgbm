"""
Multi-GPU check (needs 2 GPUs, e.g. Kaggle T4 x2). Run from the repository root after
`TORCH_CUDA_ARCH_LIST=7.5 python setup.py build_ext --inplace`:

    python tests/multi_gpu_check.py

1. pytest tests/test_multi_gpu.py (the cuda:1 tests are skipped on one GPU)
2. the 2.7.0 failure case: prebinned multi_regression fitted on cuda:1 from a thread whose
   current device is cuda:0; cuda:1 == cuda:0 (bit-identical), concurrent == sequential
3. diagnosis: the same case without the device scope of the public methods, with
   CUDA_LAUNCH_BLOCKING=1, in a subprocess (it may crash), to localise the original fault
"""
import os
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from warpgbm import WarpGBM, core  # noqa: E402


def data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 64, (3000, 1681), dtype=np.int8)
    Y = rng.standard_normal((3000, 20)).astype(np.float32)
    return X, Y


def fit(device, X, Y):
    m = WarpGBM(objective="multi_regression", num_bins=64, max_depth=4, n_estimators=50,
                device=device, prebinned=True, verbose=False)
    m.fit(X, Y)
    torch.cuda.synchronize(device)
    return m


def in_threads(jobs):
    """[fn(*args) for fn, args in jobs], all running concurrently in fresh threads"""
    out, errors = [None] * len(jobs), []

    def run(i, fn, args):
        try:
            out[i] = fn(*args)
        except BaseException as e:
            traceback.print_exc()
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i, fn, args)) for i, (fn, args) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return out


def same(a, b, X):
    return (a.forest == b.forest
            and np.array_equal(a.train_predictions_, b.train_predictions_)
            and np.array_equal(a.predict(X), b.predict(X)))


def check():
    X, Y = data()
    t = time.time()
    m1, = in_threads([(fit, ("cuda:1", X, Y))])
    print(f"  fit on cuda:1 from a worker thread: OK ({time.time() - t:.1f} s)")
    m0, = in_threads([(fit, ("cuda:0", X, Y))])
    assert same(m0, m1, X), "cuda:1 model differs from cuda:0 model"
    p1, = in_threads([(m1.predict, (X,))])
    assert np.array_equal(p1, m1.predict(X, device="cpu")), "GPU predict on cuda:1 != CPU predict"
    print("  cuda:1 == cuda:0 (trees, train predictions, GPU predictions); cuda:1 GPU == CPU predict")
    c0, c1 = in_threads([(fit, ("cuda:0", X, Y)), (fit, ("cuda:1", X, Y))])
    assert same(c0, m0, X) and same(c1, m1, X), "concurrent fits differ from sequential ones"
    print("  concurrent fits on cuda:0 / cuda:1 == sequential fits")
    assert torch.cuda.current_device() == 0


def diagnose():
    """The failure case with the device scope removed (child process, CUDA_LAUNCH_BLOCKING=1)"""
    for name, value in list(vars(WarpGBM).items()):
        if callable(value) and not isinstance(value, staticmethod) and hasattr(value, "__wrapped__"):
            setattr(WarpGBM, name, value.__wrapped__)
    X, Y = data()
    in_threads([(fit, ("cuda:1", X, Y))])
    print("  without the device scope: fit on cuda:1 OK -> the event / capture-stream fixes suffice")


if __name__ == "__main__":
    if sys.argv[1:] == ["diagnose"]:
        diagnose()
        sys.exit(0)

    print(f"torch {torch.__version__}, {torch.cuda.device_count()} GPUs:",
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    assert torch.cuda.device_count() >= 2, "needs two GPUs"

    print("\n[1] pytest tests/test_multi_gpu.py", flush=True)
    rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "-rs", "tests/test_multi_gpu.py"])
    print(f"  pytest exit code {rc}")

    print("\n[2] 2.7.0 failure case (1681 features, 20 outputs)", flush=True)
    try:
        check()
        ok = True
    except Exception:
        traceback.print_exc()
        ok = False

    print("\n[3] diagnosis: without device scope, CUDA_LAUNCH_BLOCKING=1", flush=True)
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    diag = subprocess.run([sys.executable, __file__, "diagnose"], env=env,
                          capture_output=True, text=True)
    print(diag.stdout[-3000:], diag.stderr[-6000:], f"  exit code {diag.returncode}", sep="\n")

    print("\nSUMMARY: pytest", "OK" if rc == 0 else "FAILED", "| failure case", "OK" if ok else "FAILED",
          "| diagnosis exit code", diag.returncode)
