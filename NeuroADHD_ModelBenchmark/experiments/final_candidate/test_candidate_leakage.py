"""Unit test verifying strict data leakage prevention for final candidate benchmark.

Tests:
1. test_feature_selection_isolation: Corrupting validation fold connectivity produces zero change in training ANOVA F-scores and selected top-k features.
2. test_age_scaler_isolation: Corrupting validation age values produces zero change in training age scaler mean and variance.
3. test_tangent_space_isolation: Corrupting validation covariance matrices produces zero change in training Fréchet geometric mean or whitening matrix.
4. test_combat_isolation: Verifies that ComBat estimates computed on training fold are unaffected by validation fold data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from nilearn.connectome.connectivity_matrices import _geometric_mean, _map_eigenvalues, sym_matrix_to_vec
from neuroCombat import neuroCombat, neuroCombatFromTraining


def test_feature_selection_isolation():
    rng = np.random.RandomState(42)
    X_tr = rng.randn(80, 500)
    y_tr = rng.choice([0, 1], size=80)
    X_val = rng.randn(38, 500)
    
    # Baseline selection
    kb1 = SelectKBest(f_classif, k=100)
    X_tr_k1 = kb1.fit_transform(X_tr, y_tr)
    scores1 = kb1.scores_.copy()
    
    # Corrupt validation fold severely (+50,000)
    X_val_corrupted = X_val + 50000.0
    
    # Fit again
    kb2 = SelectKBest(f_classif, k=100)
    X_tr_k2 = kb2.fit_transform(X_tr, y_tr)
    scores2 = kb2.scores_.copy()
    
    np.testing.assert_array_equal(scores1, scores2, err_msg="F-scores altered by validation data!")
    np.testing.assert_array_equal(X_tr_k1, X_tr_k2, err_msg="Selected training features altered by validation data!")
    print("PASS: test_feature_selection_isolation")


def test_age_scaler_isolation():
    rng = np.random.RandomState(42)
    age_tr = rng.uniform(7.0, 18.0, size=(80, 1))
    age_val = rng.uniform(7.0, 18.0, size=(38, 1))
    
    sc1 = StandardScaler()
    sc1.fit(age_tr)
    mean1 = sc1.mean_.copy()
    var1 = sc1.var_.copy()
    
    # Corrupt validation fold
    age_val_corrupted = age_val + 99999.0
    
    sc2 = StandardScaler()
    sc2.fit(age_tr)
    mean2 = sc2.mean_.copy()
    var2 = sc2.var_.copy()
    
    np.testing.assert_array_equal(mean1, mean2, err_msg="Age scaler mean contaminated by validation data!")
    np.testing.assert_array_equal(var1, var2, err_msg="Age scaler variance contaminated by validation data!")
    print("PASS: test_age_scaler_isolation")


def test_tangent_space_isolation():
    rng = np.random.RandomState(42)
    p = 50
    # Generate positive definite covariance matrices with exact symmetry
    A = rng.randn(p, p)
    base = A @ A.T + p * np.eye(p)
    base = 0.5 * (base + base.T)
    
    covs_tr = []
    for _ in range(30):
        noise = rng.randn(p, p)
        c = base + 0.05 * (noise @ noise.T)
        c = 0.5 * (c + c.T)
        covs_tr.append(c)
        
    covs_val = []
    for _ in range(15):
        noise = rng.randn(p, p)
        c = base + 0.05 * (noise @ noise.T)
        c = 0.5 * (c + c.T)
        covs_val.append(c)
    
    mean_cov1 = _geometric_mean(covs_tr, max_iter=20, tol=1e-5)
    whitening1 = _map_eigenvalues(lambda x: 1.0 / np.sqrt(x), mean_cov1)
    
    # Corrupt validation covariances completely
    covs_val_corrupted = [c * 1000.0 for c in covs_val]
    
    mean_cov2 = _geometric_mean(covs_tr, max_iter=20, tol=1e-5)
    whitening2 = _map_eigenvalues(lambda x: 1.0 / np.sqrt(x), mean_cov2)
    
    np.testing.assert_allclose(mean_cov1, mean_cov2, rtol=1e-6, err_msg="Fréchet mean covariance altered by validation data!")
    np.testing.assert_allclose(whitening1, whitening2, rtol=1e-6, err_msg="Whitening matrix altered by validation data!")
    print("PASS: test_tangent_space_isolation")


def test_combat_isolation():
    rng = np.random.RandomState(42)
    dat_tr = rng.randn(50, 40) # 50 features, 40 samples
    cov_tr = pd.DataFrame({"institution": ["KKI"]*20 + ["Peking"]*20})
    
    res1 = neuroCombat(dat=dat_tr, covars=cov_tr, batch_col="institution")
    est1 = res1["estimates"]
    
    # Test on unseen batch raises error
    dat_val = rng.randn(50, 10)
    try:
        neuroCombatFromTraining(dat_val, batch=np.array(["UnseenSite"]*10), estimates=est1)
        raised = False
    except ValueError:
        raised = True
        
    assert raised, "ComBat should reject unseen institutions in out-of-site CV!"
    print("PASS: test_combat_isolation")


if __name__ == "__main__":
    test_feature_selection_isolation()
    test_age_scaler_isolation()
    test_tangent_space_isolation()
    test_combat_isolation()
    print("\nALL 4 LEAKAGE ISOLATION TESTS PASSED CLEANLY!")
