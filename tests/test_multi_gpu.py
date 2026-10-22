"""Explicit devices and concurrent fits in several threads (one GPU or two)"""
import threading

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression

from warpgbm import WarpGBM, apply_bins, fit_bins

two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")


def _data():
    X, y = make_regression(n_samples=2000, n_features=30, noise=1.0, random_state=0)
    X = X.astype(np.float32)
    X[::17, 3] = np.nan  # exercise the NaN bin
    era = np.arange(X.shape[0]) % 4
    return X, y.astype(np.float32), era


def _fit(device, random_state, X, y, era):
    model = WarpGBM(
        max_depth=4,
        n_estimators=40,
        num_bins=16,
        colsample_bytree=0.6,
        random_state=random_state,
        device=device,
        verbose=False,
    )
    model.fit(X, y, era_id=era, X_eval=X[:500], y_eval=y[:500], eval_every_n_trees=5)
    return model


def _assert_same_forest(a, b, X):
    assert np.array_equal(a.predict_raw(X), b.predict_raw(X))
    assert np.array_equal(a.feature_importance_, b.feature_importance_)
    assert a.evals_result_ == b.evals_result_


def _fit_concurrently(jobs, X, y, era):
    results = [None] * len(jobs)
    errors = []
    barrier = threading.Barrier(len(jobs))

    def run(i, device, seed):
        try:
            barrier.wait()
            results[i] = _fit(device, seed, X, y, era)
        except BaseException as e:  # re-raised in the main thread
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i, d, s)) for i, (d, s) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


@pytest.mark.parametrize(
    "devices",
    [("cuda:0", "cuda:0"), pytest.param(("cuda:0", "cuda:1"), marks=two_gpus)],
)
def test_concurrent_fits_match_sequential(devices):
    X, y, era = _data()
    # Different seeds: with a shared (global) generator the interleaved draws would mix up
    jobs = [(devices[0], 42), (devices[1], 7), (devices[0], 42), (devices[1], 7)]
    sequential = [_fit(d, s, X, y, era) for d, s in jobs[:2]]
    concurrent = _fit_concurrently(jobs, X, y, era)

    for i, model in enumerate(concurrent):
        _assert_same_forest(sequential[i % 2], model, X)
    # Sanity check: the seed matters
    assert not np.array_equal(sequential[0].predict_raw(X), sequential[1].predict_raw(X))


@two_gpus
def test_explicit_device_without_switching():
    """device="cuda:1" works while the current device is cuda:0 and leaves it unchanged"""
    X, y, era = _data()
    torch.cuda.set_device(0)

    explicit = _fit("cuda:1", 42, X, y, era)
    preds_explicit = explicit.predict_raw(X)
    assert torch.cuda.current_device() == 0

    with torch.cuda.device(1):
        switched = _fit("cuda:1", 42, X, y, era)
        preds_switched = switched.predict_raw(X)

    assert np.array_equal(preds_explicit, preds_switched)
    assert np.array_equal(explicit.feature_importance_, switched.feature_importance_)


def _in_thread(fn, *args):
    """fn(*args) in a new thread, whose current CUDA device is the default (cuda:0)"""
    out, errors = [], []

    def run():
        try:
            out.append(fn(*args))
        except BaseException as e:  # re-raised in the caller
            errors.append(e)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if errors:
        raise errors[0]
    return out[0]


def _fit_prebinned_multi(device, X, Y):
    model = WarpGBM(
        objective="multi_regression",
        num_bins=64,
        max_depth=4,
        n_estimators=50,
        device=device,
        prebinned=True,
        verbose=False,
    )
    model.fit(X, Y)
    torch.cuda.synchronize(device)
    return model


@two_gpus
def test_second_gpu_from_worker_thread_matches_first():
    """
    Regression test (2.7.0): a fit on cuda:1 from a thread whose current device is cuda:0 died
    with an illegal memory access at the first CUDA graph capture. The model must equal the one
    fitted on cuda:0 (trees and GPU / CPU predictions bit-identical).
    """
    rng = np.random.default_rng(0)
    X = rng.integers(0, 64, (3000, 300), dtype=np.int8)
    Y = rng.standard_normal((3000, 20)).astype(np.float32)

    m1 = _in_thread(_fit_prebinned_multi, "cuda:1", X, Y)
    m0 = _in_thread(_fit_prebinned_multi, "cuda:0", X, Y)
    assert m1.forest == m0.forest
    assert np.array_equal(m1.train_predictions_, m0.train_predictions_)
    p1 = _in_thread(m1.predict, X)
    assert np.array_equal(p1, m0.predict(X))
    assert np.array_equal(p1, m1.predict(X, device="cpu"))
    staged = _in_thread(m1.staged_predict, X, [10, 50])
    assert np.array_equal(staged, m0.staged_predict(X, [10, 50]))
    assert torch.cuda.current_device() == 0


@two_gpus
def test_second_gpu_eval_and_classifier():
    """Eval sets, early stopping and a classifier on cuda:1 equal the same fits on cuda:0"""
    X, y, era = _data()
    labels = (y > np.median(y)).astype(int) + (y > np.quantile(y, 0.8)).astype(int)
    fits = {}
    for device in ("cuda:0", "cuda:1"):
        reg = _fit(device, 42, X, y, era)
        clf = WarpGBM(
            objective="multiclass", max_depth=3, n_estimators=30, num_bins=16,
            device=device, random_state=1, verbose=False,
        )
        clf.fit(X, labels, X_eval=X[:500], y_eval=labels[:500], eval_every_n_trees=3,
                early_stopping_rounds=4)
        fits[device] = (reg, clf)
    (reg0, clf0), (reg1, clf1) = fits["cuda:0"], fits["cuda:1"]
    _assert_same_forest(reg0, reg1, X)
    assert clf0.forest == clf1.forest
    assert clf0.best_iteration_ == clf1.best_iteration_
    assert np.array_equal(clf0.predict_proba(X), clf1.predict_proba(X))
    assert torch.cuda.current_device() == 0


@two_gpus
def test_binning_on_second_gpu():
    X, _, _ = _data()
    edges0 = fit_bins(X, 32, device="cuda:0")
    edges1 = fit_bins(X, 32, device="cuda:1")
    assert edges1.device == torch.device("cuda:1")
    assert torch.equal(edges0.cpu(), edges1.cpu())
    assert np.array_equal(apply_bins(X, edges0), apply_bins(X, edges1))
    assert apply_bins(torch.from_numpy(X).to("cuda:1"), edges1).device == torch.device("cuda:1")
    assert torch.cuda.current_device() == 0


def test_verbose_false_is_silent(capsys, tmp_path):
    X, y, era = _data()
    model = _fit("cuda", 1, X, y, era)
    model.predict(X)
    model.predict(np.nan_to_num(X).clip(0, 3).astype(np.int8))  # pre-binned path
    path = tmp_path / "model.pkl"
    model.save_model(path)
    WarpGBM(verbose=False).load_model(path)
    # Warm start that has nothing left to train
    model.warm_start = True
    model.fit(X, y)
    assert capsys.readouterr().out == ""

    model.verbose = True
    model.save_model(path)
    assert "Model saved" in capsys.readouterr().out


@pytest.mark.parametrize("device", ["cpu", "meta", "not-a-device"])
def test_non_cuda_device_rejected(device):
    """There is no CPU code path; non-CUDA devices fail at construction"""
    with pytest.raises(ValueError):
        WarpGBM(device=device)


def test_cuda_device_accepted_without_allocation():
    for device in ("cuda", "cuda:1", 0, torch.device("cuda", 0)):  # an int is a CUDA index
        WarpGBM(device=device)
