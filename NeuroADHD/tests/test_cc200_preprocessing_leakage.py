"""Synthetic checks that held-out partitions cannot affect fitted transforms."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.cc200_ml_preprocessing import fit_transform_splits  # noqa: E402

N_FEATURES = 19_900


def synthetic_partitions():
    rng = np.random.default_rng(742)
    y_train = np.array([0, 1] * 10, dtype=np.int64)
    train = rng.normal(size=(20, N_FEATURES))
    # Add a training-only signal; final column is constant and should be removed.
    train[:, 0] += y_train * 2.0
    train[:, -1] = 3.0
    train[0, 1] = np.nan
    validation = rng.normal(size=(4, N_FEATURES))
    test = rng.normal(size=(4, N_FEATURES))
    columns = [f"ROI_{i:05d}" for i in range(N_FEATURES)]
    return (
        pd.DataFrame(train, columns=columns),
        pd.DataFrame(validation, columns=columns),
        pd.DataFrame(test, columns=columns),
        y_train,
        [f"train-{i}" for i in range(20)],
        [f"validation-{i}" for i in range(4)],
        [f"test-{i}" for i in range(4)],
    )


class TestCC200PreprocessingLeakage(unittest.TestCase):
    def test_validation_and_test_cannot_change_training_fit_or_selected_features(self):
        X_train, X_validation, X_test, y_train, ids_train, ids_validation, ids_test = synthetic_partitions()
        first = fit_transform_splits(
            X_train, X_validation, X_test, y_train=y_train,
            subject_ids_train=ids_train, subject_ids_validation=ids_validation,
            subject_ids_test=ids_test, k=100,
        )

        # Change held-out values drastically while keeping the training partition fixed.
        changed_validation = X_validation.copy()
        changed_test = X_test.copy()
        changed_validation.loc[:, :] = 1e12
        changed_test.loc[:, :] = -1e12
        second = fit_transform_splits(
            X_train, changed_validation, changed_test, y_train=y_train,
            subject_ids_train=ids_train, subject_ids_validation=ids_validation,
            subject_ids_test=ids_test, k=100,
        )

        np.testing.assert_allclose(
            first.pipeline.named_steps["imputer"].statistics_,
            second.pipeline.named_steps["imputer"].statistics_,
        )
        np.testing.assert_allclose(first.pipeline.named_steps["scaler"].mean_, second.pipeline.named_steps["scaler"].mean_)
        np.testing.assert_array_equal(
            first.pipeline.named_steps["select"].get_support(),
            second.pipeline.named_steps["select"].get_support(),
        )
        self.assertEqual(first.selected_feature_names, second.selected_feature_names)
        np.testing.assert_allclose(first.X_train.to_numpy(), second.X_train.to_numpy())
        self.assertEqual(first.pipeline.named_steps["variance"].n_features_in_, N_FEATURES)
        self.assertEqual(first.X_train.shape, (20, 100))
        self.assertEqual(first.X_validation.shape, (4, 100))
        self.assertEqual(first.X_test.shape, (4, 100))
        self.assertEqual(first.subject_ids_train, tuple(ids_train))
        self.assertEqual(first.subject_ids_validation, tuple(ids_validation))
        self.assertEqual(first.subject_ids_test, tuple(ids_test))
        self.assertEqual(first.fitted_subject_ids, tuple(ids_train))
        self.assertTrue(all(np.isfinite(x.to_numpy()).all() for x in (first.X_train, first.X_validation, first.X_test)))


    def test_subject_overlap_is_rejected(self):
        X_train, X_validation, X_test, y_train, ids_train, ids_validation, ids_test = synthetic_partitions()
        ids_validation[0] = ids_train[0]
        with self.assertRaisesRegex(ValueError, "more than one partition"):
            fit_transform_splits(
                X_train, X_validation, X_test, y_train=y_train,
                subject_ids_train=ids_train, subject_ids_validation=ids_validation,
                subject_ids_test=ids_test, k=100,
            )


if __name__ == "__main__":
    unittest.main()
