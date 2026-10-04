"""Leakage-safe CC200 input and preprocessing helpers.

All learned transforms live in one sklearn Pipeline. ``fit_transform_splits``
fits that pipeline exactly once on X_train/y_train, then only calls transform on
validation and test. Subject IDs and sites remain external metadata; diagnosis is
the target and ROI-pair columns alone are passed to the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_TABLE = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"
EXPECTED_INPUT_FEATURES = 19_900
DEFAULT_SELECTED_FEATURES = 100


@dataclass
class PreprocessedSplits:
    """Transformed numeric matrices plus unchanged IDs and the fitted pipeline."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    subject_ids_train: tuple[str, ...]
    subject_ids_validation: tuple[str, ...]
    subject_ids_test: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    pipeline: Pipeline
    fitted_subject_ids: tuple[str, ...]


def load_cc200_xy(feature_table: str | Path = DEFAULT_FEATURE_TABLE):
    """Return X, y, and metadata separately; metadata never enters X."""
    table = pd.read_csv(feature_table, dtype={"subject_id": str, "site": str})
    required = {"subject_id", "site", "diagnosis"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")
    feature_columns = [column for column in table.columns if column.startswith("ROI_")]
    if len(feature_columns) != EXPECTED_INPUT_FEATURES:
        raise ValueError(f"Expected {EXPECTED_INPUT_FEATURES} CC200 edge features, found {len(feature_columns)}")
    non_feature_columns = set(table.columns).difference(feature_columns)
    if non_feature_columns != required:
        raise ValueError(
            "Feature table must contain only subject_id, site, diagnosis, and ROI_* columns; "
            f"unexpected columns: {sorted(non_feature_columns.difference(required))}"
        )
    if table["subject_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate subject IDs")
    if table[["subject_id", "site", "diagnosis"]].isna().any().any():
        raise ValueError("Subject metadata or diagnosis contains missing values")
    if not set(table["diagnosis"].unique()).issubset({0, 1}):
        raise ValueError("Diagnosis must be encoded as Control=0 and ADHD=1")
    X = table.loc[:, feature_columns].astype("float64")
    y = table["diagnosis"].astype("int64")
    metadata = table.loc[:, ["subject_id", "site"]].copy()
    # Subject IDs are the stable row keys used by the split manifests.
    subject_index = pd.Index(metadata["subject_id"].astype(str), name="subject_id")
    X.index = subject_index
    y.index = subject_index
    return X, y, metadata


def make_preprocessing_pipeline(k: int = DEFAULT_SELECTED_FEATURES) -> Pipeline:
    """Create an unfitted impute/filter/scale/select pipeline.

    The intended order is median imputation -> zero-variance filtering ->
    standardization -> ANOVA F-statistic SelectKBest. Every learned statistic
    is fitted by sklearn only when this pipeline is fit on a training fold.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
        ]
    )


def make_estimator_pipeline(estimator: Any, k: int = DEFAULT_SELECTED_FEATURES) -> Pipeline:
    """Wrap a future classifier; do not fit this here.

    For CV/tuning, pass this entire Pipeline to cross-validation/GridSearchCV so
    imputation, variance filtering, scaling, and selection are refit per fold.
    """
    return Pipeline(
        steps=[
            ("preprocessing", make_preprocessing_pipeline(k=k)),
            ("estimator", estimator),
        ]
    )


def _validate_feature_frame(X: pd.DataFrame, name: str) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame with named ROI feature columns")
    if X.shape[1] != EXPECTED_INPUT_FEATURES:
        raise ValueError(f"{name} must have {EXPECTED_INPUT_FEATURES} ROI-pair columns; got {X.shape[1]}")
    if not all(isinstance(c, str) and c.startswith("ROI_") for c in X.columns):
        raise ValueError(f"{name} must contain ROI-pair features only")


def _validate_ids(ids: Sequence[str], name: str, n_rows: int) -> tuple[str, ...]:
    values = tuple(str(value) for value in ids)
    if len(values) != n_rows:
        raise ValueError(f"{name} has {len(values)} IDs for {n_rows} rows")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate subject IDs")
    return values


def fit_transform_splits(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    y_train: Sequence[int],
    subject_ids_train: Sequence[str],
    subject_ids_validation: Sequence[str],
    subject_ids_test: Sequence[str],
    k: int = DEFAULT_SELECTED_FEATURES,
) -> PreprocessedSplits:
    """Fit learned preprocessing on train only; transform validation and test.

    Validation/test labels are deliberately not accepted. Feature columns must
    be in identical order in all partitions. IDs are checked for preservation,
    uniqueness, and cross-partition overlap, then returned unchanged.
    """
    matrices = ((X_train, "X_train"), (X_validation, "X_validation"), (X_test, "X_test"))
    for X, name in matrices:
        _validate_feature_frame(X, name)
    expected_columns = list(X_train.columns)
    for X, name in matrices[1:]:
        if list(X.columns) != expected_columns:
            raise ValueError(f"{name} feature names/order differ from X_train")

    ids_train = _validate_ids(subject_ids_train, "subject_ids_train", len(X_train))
    ids_validation = _validate_ids(subject_ids_validation, "subject_ids_validation", len(X_validation))
    ids_test = _validate_ids(subject_ids_test, "subject_ids_test", len(X_test))
    sets = {"train": set(ids_train), "validation": set(ids_validation), "test": set(ids_test)}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise ValueError("A subject ID occurs in more than one partition")

    y = np.asarray(y_train, dtype=np.int64)
    if y.ndim != 1 or len(y) != len(X_train):
        raise ValueError("y_train must be one-dimensional and match X_train rows")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y_train must contain both encoded classes 0 and 1")

    pipeline = make_preprocessing_pipeline(k=k)
    # The only fit call: neither validation nor test features/labels enter it.
    pipeline.fit(X_train, y)
    train_array = pipeline.transform(X_train)
    validation_array = pipeline.transform(X_validation)
    test_array = pipeline.transform(X_test)

    arrays = (train_array, validation_array, test_array)
    expected_output_features = k
    for name, array, expected_rows in zip(
        ("train", "validation", "test"), arrays, (len(X_train), len(X_validation), len(X_test))
    ):
        if array.shape != (expected_rows, expected_output_features):
            raise RuntimeError(f"Transformed {name} shape {array.shape}; expected {(expected_rows, expected_output_features)}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"Transformed {name} contains NaN or Inf values")

    variance_support = pipeline.named_steps["variance"].get_support()
    selection_support = pipeline.named_steps["select"].get_support()
    selected_feature_names = tuple(np.asarray(expected_columns)[variance_support][selection_support].tolist())
    if len(selected_feature_names) != k:
        raise RuntimeError(f"Expected {k} selected feature names; found {len(selected_feature_names)}")

    return PreprocessedSplits(
        X_train=pd.DataFrame(train_array, columns=selected_feature_names, index=pd.Index(ids_train, name="subject_id")),
        X_validation=pd.DataFrame(validation_array, columns=selected_feature_names, index=pd.Index(ids_validation, name="subject_id")),
        X_test=pd.DataFrame(test_array, columns=selected_feature_names, index=pd.Index(ids_test, name="subject_id")),
        subject_ids_train=ids_train,
        subject_ids_validation=ids_validation,
        subject_ids_test=ids_test,
        selected_feature_names=selected_feature_names,
        pipeline=pipeline,
        fitted_subject_ids=ids_train,
    )
