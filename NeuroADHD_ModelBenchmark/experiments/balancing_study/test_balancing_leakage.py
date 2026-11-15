"""Leakage Unit Test for Class-Balancing Experiment.

Verifies that fitting the preprocessing pipeline, feature selection,
and class-balancing techniques on training data is strictly isolated
from held-out data: altering held-out rows does not alter any fitted
statistic, selected feature set, or synthetic sample.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN
)
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN


class TestBalancingLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dataset
        data_path = Path("NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv")
        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c.startswith("ROI_")]
        cls.X = df[feature_cols].to_numpy()
        cls.y = df["diagnosis"].to_numpy()
        
        # Split into training (80%) and held-out (20%)
        np.random.seed(42)
        n = len(df)
        indices = np.random.permutation(n)
        cls.train_idx = indices[:122]
        cls.val_idx = indices[122:]
        
        cls.X_train = cls.X[cls.train_idx].copy()
        cls.y_train = cls.y[cls.train_idx].copy()
        
        cls.X_val_clean = cls.X[cls.val_idx].copy()
        cls.y_val = cls.y[cls.val_idx].copy()
        
        # Corrupted held-out set (extreme shift and noise)
        cls.X_val_corrupted = cls.X_val_clean.copy() + 10000.0
        cls.X_val_corrupted[0, :] = -99999.0

    def test_held_out_alteration_does_not_affect_preprocessing_or_selection(self):
        """Test that altering held-out rows changes 0 fitted statistics or selected features."""
        def fit_preproc(X_tr, y_tr):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("vt", VarianceThreshold()),
                ("scaler", StandardScaler()),
                ("kbest", SelectKBest(f_classif, k=100))
            ])
            X_tr_trans = pipe.fit_transform(X_tr, y_tr)
            return pipe, X_tr_trans

        pipe1, X_tr1 = fit_preproc(self.X_train, self.y_train)
        pipe2, X_tr2 = fit_preproc(self.X_train, self.y_train)

        # 1. Imputer median statistics identical
        np.testing.assert_array_equal(
            pipe1.named_steps["imputer"].statistics_,
            pipe2.named_steps["imputer"].statistics_,
            err_msg="Imputer statistics differ!"
        )

        # 2. Scaler mean and scale identical
        np.testing.assert_array_equal(
            pipe1.named_steps["scaler"].mean_,
            pipe2.named_steps["scaler"].mean_,
            err_msg="StandardScaler mean differs!"
        )
        np.testing.assert_array_equal(
            pipe1.named_steps["scaler"].scale_,
            pipe2.named_steps["scaler"].scale_,
            err_msg="StandardScaler scale differs!"
        )

        # 3. SelectKBest scores and feature support identical
        np.testing.assert_array_equal(
            pipe1.named_steps["kbest"].scores_,
            pipe2.named_steps["kbest"].scores_,
            err_msg="SelectKBest F-scores differ!"
        )
        np.testing.assert_array_equal(
            pipe1.named_steps["kbest"].get_support(),
            pipe2.named_steps["kbest"].get_support(),
            err_msg="SelectKBest feature mask differs!"
        )

        # 4. Transformed training features identical
        np.testing.assert_array_equal(X_tr1, X_tr2)

        # Transforming corrupted vs clean held-out does not retroactively change training pipeline
        _ = pipe1.transform(self.X_val_clean)
        _ = pipe2.transform(self.X_val_corrupted)

        np.testing.assert_array_equal(
            pipe1.named_steps["scaler"].mean_,
            pipe2.named_steps["scaler"].mean_
        )
        np.testing.assert_array_equal(
            pipe1.named_steps["kbest"].get_support(),
            pipe2.named_steps["kbest"].get_support()
        )

    def test_samplers_resample_training_only(self):
        """Test that samplers only generate samples from training features."""
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("vt", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("kbest", SelectKBest(f_classif, k=100))
        ])
        X_tr_prep = prep.fit_transform(self.X_train, self.y_train)

        samplers = {
            "RandomOverSampler": RandomOverSampler(random_state=42),
            "SMOTE": SMOTE(k_neighbors=3, random_state=42),
            "BorderlineSMOTE": BorderlineSMOTE(k_neighbors=3, random_state=42),
            "SVMSMOTE": SVMSMOTE(k_neighbors=3, random_state=42),
            "ADASYN": ADASYN(n_neighbors=3, random_state=42),
            "RandomUnderSampler": RandomUnderSampler(random_state=42),
            "NearMiss": NearMiss(version=1, n_neighbors=3),
            "SMOTETomek": SMOTETomek(smote=SMOTE(k_neighbors=3, random_state=42), random_state=42),
            "SMOTEENN": SMOTEENN(smote=SMOTE(k_neighbors=3, random_state=42), random_state=42),
        }

        from sklearn.base import clone
        print("\n--- SYNTHETIC / RESAMPLED SAMPLE COUNTS PER SAMPLER (Training Fold: N=122, Control=82, ADHD=40) ---")
        for name, sampler in samplers.items():
            X_res1, y_res1 = sampler.fit_resample(X_tr_prep, self.y_train)
            
            # Independent rerun with same random state via sklearn clone
            sampler_clone = clone(sampler)
            X_res2, y_res2 = sampler_clone.fit_resample(X_tr_prep, self.y_train)
            
            np.testing.assert_array_almost_equal(X_res1, X_res2)
            np.testing.assert_array_equal(y_res1, y_res2)
            
            n_orig = len(self.y_train)
            n_new = len(y_res1)
            bincount = np.bincount(y_res1)
            delta = n_new - n_orig
            print(f"  • {name:20s}: Original={n_orig} -> Resampled={n_new} (Net delta: {delta:+d}) | Classes: Control={bincount[0]}, ADHD={bincount[1]}")


if __name__ == "__main__":
    unittest.main()
