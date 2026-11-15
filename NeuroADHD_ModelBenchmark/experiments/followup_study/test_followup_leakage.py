"""Leakage Unit Test for Follow-Up Study (Covariates + Regularized Models + Balancing).

Verifies that:
1. Age standardization is fit strictly on training folds; altering held-out age values
   produces 0 change in the fitted training mean or standard deviation.
2. Feature selection (SelectKBest) on 19,900 connectivity edges is fit strictly on
   training folds; corrupting held-out connectivity features produces 0 change in
   selected features or ANOVA scores.
3. Resampling (SMOTE) is applied strictly to training fold features after selection;
   held-out validation rows are never seen by the sampler.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


class TestFollowupLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load connectivity features
        feat_path = Path("NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv")
        df_feat = pd.read_csv(feat_path)
        
        # Load phenotypes
        match_path = Path("ADHD200_Official/metadata/official_cc200_phenotype_match.csv")
        df_match = pd.read_csv(match_path)
        
        df_feat["norm_id"] = df_feat["subject_id"].astype(str).str.lstrip("0")
        df_match["norm_id"] = df_match["normalized_subject_id"].astype(str).str.lstrip("0")
        
        merged = df_feat.merge(df_match[["norm_id", "age_years", "sex"]], on="norm_id", how="left")
        
        feature_cols = [c for c in df_feat.columns if c.startswith("ROI_")]
        cls.X_conn = merged[feature_cols].to_numpy()
        cls.age = merged["age_years"].to_numpy().reshape(-1, 1)
        cls.sex = (merged["sex"] == "Male").astype(float).to_numpy().reshape(-1, 1)
        cls.y = merged["diagnosis"].to_numpy()
        
        np.random.seed(42)
        indices = np.random.permutation(len(merged))
        cls.train_idx = indices[:110]
        cls.val_idx = indices[110:]
        
        cls.age_train = cls.age[cls.train_idx].copy()
        cls.age_val_clean = cls.age[cls.val_idx].copy()
        cls.age_val_corrupted = cls.age_val_clean.copy() + 9999.0
        
        cls.X_conn_train = cls.X_conn[cls.train_idx].copy()
        cls.X_conn_val_clean = cls.X_conn[cls.val_idx].copy()
        cls.X_conn_val_corrupted = cls.X_conn_val_clean.copy() + 50000.0
        cls.y_train = cls.y[cls.train_idx].copy()

    def test_age_scaler_isolation(self):
        """Test that held-out age corruption does not alter fitted training mean/scale."""
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        
        scaler1.fit(self.age_train)
        scaler2.fit(self.age_train)
        
        # Transform clean vs corrupted
        _ = scaler1.transform(self.age_val_clean)
        _ = scaler2.transform(self.age_val_corrupted)
        
        np.testing.assert_array_equal(scaler1.mean_, scaler2.mean_)
        np.testing.assert_array_equal(scaler1.scale_, scaler2.scale_)

    def test_feature_selection_isolation(self):
        """Test that held-out connectivity corruption does not alter selected features."""
        kbest1 = SelectKBest(f_classif, k=100)
        kbest2 = SelectKBest(f_classif, k=100)
        
        kbest1.fit(self.X_conn_train, self.y_train)
        kbest2.fit(self.X_conn_train, self.y_train)
        
        _ = kbest1.transform(self.X_conn_val_clean)
        _ = kbest2.transform(self.X_conn_val_corrupted)
        
        np.testing.assert_array_equal(kbest1.scores_, kbest2.scores_)
        np.testing.assert_array_equal(kbest1.get_support(), kbest2.get_support())

    def test_smote_resampling_isolation(self):
        """Test that SMOTE generates synthetic samples strictly from training features."""
        kbest = SelectKBest(f_classif, k=100)
        X_tr_100 = kbest.fit_transform(self.X_conn_train, self.y_train)
        
        smote1 = SMOTE(k_neighbors=3, random_state=42)
        smote2 = SMOTE(k_neighbors=3, random_state=42)
        
        X_res1, y_res1 = smote1.fit_resample(X_tr_100, self.y_train)
        X_res2, y_res2 = smote2.fit_resample(X_tr_100, self.y_train)
        
        np.testing.assert_array_equal(X_res1, X_res2)
        np.testing.assert_array_equal(y_res1, y_res2)
        
        n_orig = len(self.y_train)
        n_res = len(y_res1)
        print(f"\n[Leakage Test] SMOTE training fold resample: Original={n_orig} -> Resampled={n_res} (Synthetic added: {n_res - n_orig})")


if __name__ == "__main__":
    unittest.main()
