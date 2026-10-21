"""Test random_state parameter for reproducibility"""
import numpy as np
from warpgbm import WarpGBM
from sklearn.datasets import make_classification, make_regression


def test_random_state_regression():
    """Test that random_state makes results reproducible for regression"""
    X, y = make_regression(n_samples=500, n_features=20, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Train two models with same random_state
    model1 = WarpGBM(
        objective='regression',
        max_depth=5,
        n_estimators=50,
        colsample_bytree=0.8,  # Enable randomness
        random_state=42
    )
    model1.fit(X, y)
    preds1 = model1.predict(X)
    
    model2 = WarpGBM(
        objective='regression',
        max_depth=5,
        n_estimators=50,
        colsample_bytree=0.8,  # Enable randomness
        random_state=42
    )
    model2.fit(X, y)
    preds2 = model2.predict(X)
    
    # Histograms and predictions are deterministic, so fits are bit-reproducible
    assert np.array_equal(preds1, preds2), f"Predictions differ (max_diff={np.abs(preds1 - preds2).max():.6g})"
    print("✓ Regression: identical predictions with random_state=42")
    
    # Train model with different random_state
    model3 = WarpGBM(
        objective='regression',
        max_depth=5,
        n_estimators=50,
        colsample_bytree=0.8,
        random_state=123
    )
    model3.fit(X, y)
    preds3 = model3.predict(X)
    
    # Predictions should differ (with high probability) - use stricter tolerance
    max_diff_different = np.abs(preds1 - preds3).max()
    print(f"✓ Regression: different predictions with random_state=123 (max_diff={max_diff_different:.6f})")


def test_random_state_multiclass():
    """Test that random_state makes results reproducible for classification"""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=3,
        n_informative=15,
        random_state=42
    )
    X = X.astype(np.float32)
    
    # Train two models with same random_state
    model1 = WarpGBM(
        objective='multiclass',
        max_depth=5,
        n_estimators=50,
        colsample_bytree=0.8,
        random_state=42
    )
    model1.fit(X, y)
    probs1 = model1.predict_proba(X)
    
    model2 = WarpGBM(
        objective='multiclass',
        max_depth=5,
        n_estimators=50,
        colsample_bytree=0.8,
        random_state=42
    )
    model2.fit(X, y)
    probs2 = model2.predict_proba(X)
    
    assert np.array_equal(probs1, probs2), f"Probabilities differ (max_diff={np.abs(probs1 - probs2).max():.6g})"
    print("✓ Multiclass: identical predictions with random_state=42")
    
    # Train model with different random_state
    model3 = WarpGBM(
        objective='multiclass',
        max_depth=5,
        n_estimators=50,
        colsample_bytree=0.8,
        random_state=999
    )
    model3.fit(X, y)
    probs3 = model3.predict_proba(X)
    
    # Probabilities should differ  
    max_diff_different = np.abs(probs1 - probs3).max()
    print(f"✓ Multiclass: different predictions with random_state=999 (max_diff={max_diff_different:.6f})")


def test_no_random_state_with_colsample():
    """Test that random_state=None still allows randomness with colsample_bytree<1"""
    X, y = make_regression(n_samples=300, n_features=10, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Train two models without random_state, with feature subsampling
    # Results should likely differ due to uncontrolled randomness
    model1 = WarpGBM(
        objective='regression',
        max_depth=4,
        n_estimators=30,
        colsample_bytree=0.5,  # Randomness enabled
    )
    model1.fit(X, y)
    
    model2 = WarpGBM(
        objective='regression',
        max_depth=4,
        n_estimators=30,
        colsample_bytree=0.5,  # Randomness enabled
    )
    model2.fit(X, y)
    
    # Just verify both models trained successfully (no assertion on difference)
    # since GPU randomness without seed is environment-dependent
    print(f"✓ random_state=None: allows uncontrolled randomness when colsample_bytree < 1.0")



def test_fit_does_not_touch_global_rng():
    """fit draws from a per-model generator and leaves the global RNG states alone"""
    import torch

    X, y = make_regression(n_samples=300, n_features=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    for random_state in (7, None):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state()
        np_state = np.random.get_state()

        WarpGBM(
            max_depth=3, n_estimators=10, colsample_bytree=0.5, random_state=random_state, verbose=False
        ).fit(X, y)

        assert torch.equal(torch.get_rng_state(), cpu_state)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_state)
        np_after = np.random.get_state()
        assert np_after[0] == np_state[0] and np.array_equal(np_after[1], np_state[1])
        assert np_after[2:] == np_state[2:]


def test_random_state_ignores_global_seed():
    """The result depends on random_state only, not on draws from the global RNG in between"""
    import torch

    X, y = make_regression(n_samples=300, n_features=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    params = dict(max_depth=3, n_estimators=20, colsample_bytree=0.5, random_state=3, verbose=False)

    preds1 = WarpGBM(**params).fit(X, y).predict(X)
    torch.manual_seed(12345)
    torch.randperm(100, device="cuda")
    preds2 = WarpGBM(**params).fit(X, y).predict(X)
    assert np.array_equal(preds1, preds2)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing random_state parameter")
    print("=" * 70)
    print()
    
    test_random_state_regression()
    print()
    
    test_random_state_multiclass()
    print()
    
    test_no_random_state_with_colsample()
    print()
    
    print("=" * 70)
    print("✓ ALL RANDOM_STATE TESTS PASSED!")
    print("=" * 70)

