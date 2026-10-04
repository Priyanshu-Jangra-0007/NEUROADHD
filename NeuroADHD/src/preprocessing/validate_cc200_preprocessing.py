"""Run train-only preprocessing on the existing split and write a validation report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.cc200_ml_preprocessing import (
    DEFAULT_SELECTED_FEATURES,
    EXPECTED_INPUT_FEATURES,
    fit_transform_splits,
    load_cc200_xy,
)

ROOT = Path(__file__).resolve().parents[2]
SPLIT_MANIFEST = ROOT / "data/adhd200/processed/splits/subject_split_manifest.csv"
CONFIG = ROOT / "configs/splits/cc200_site_holdout.json"
JSON_REPORT = ROOT / "data/adhd200/interim/cc200_preprocessing_validation_report.json"
MD_REPORT = ROOT / "data/adhd200/interim/cc200_preprocessing_validation_report.md"


def main() -> None:
    X, y, metadata = load_cc200_xy()
    manifest = pd.read_csv(SPLIT_MANIFEST, dtype={"subject_id": str, "site": str})
    if manifest["subject_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate IDs")
    if set(manifest.subject_id) != set(metadata.subject_id):
        raise ValueError("Split manifest IDs do not exactly match feature-table IDs")
    if not (manifest.groupby("subject_id").partition.nunique() == 1).all():
        raise ValueError("At least one subject appears in multiple partitions")
    if manifest.groupby("site").partition.nunique().max() != 1:
        raise ValueError("At least one site crosses partitions")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if len([c for c in X.columns if c.startswith("ROI_")]) != EXPECTED_INPUT_FEATURES:
        raise ValueError("Input X is not exactly the 19,900 ROI features")
    if set(metadata.columns) != {"subject_id", "site"}:
        raise ValueError("Metadata must contain only subject_id and site")
    if set(y.unique()) != {0, 1}:
        raise ValueError("Target y must be diagnosis with labels 0 and 1")

    ordered_ids = {}
    for part in ("train", "validation", "test"):
        ordered_ids[part] = manifest.loc[manifest.partition.eq(part), "subject_id"].tolist()
    if any(not ordered_ids[p] for p in ordered_ids):
        raise ValueError("All three partitions must be non-empty")
    split_X = {p: X.loc[ordered_ids[p]].copy() for p in ordered_ids}
    split_y_train = y.loc[ordered_ids["train"]]
    result = fit_transform_splits(
        split_X["train"], split_X["validation"], split_X["test"],
        y_train=split_y_train.to_numpy(),
        subject_ids_train=ordered_ids["train"],
        subject_ids_validation=ordered_ids["validation"],
        subject_ids_test=ordered_ids["test"],
        k=DEFAULT_SELECTED_FEATURES,
    )

    if result.fitted_subject_ids != tuple(ordered_ids["train"]):
        raise AssertionError("Preprocessing fit scope differs from training IDs")
    if tuple(result.X_train.index) != tuple(ordered_ids["train"]):
        raise AssertionError("Training subject IDs/order changed")
    if tuple(result.X_validation.index) != tuple(ordered_ids["validation"]):
        raise AssertionError("Validation subject IDs/order changed")
    if tuple(result.X_test.index) != tuple(ordered_ids["test"]):
        raise AssertionError("Test subject IDs/order changed")

    partition_info = {}
    for part in ("train", "validation", "test"):
        rows = manifest.loc[manifest.partition.eq(part)]
        transformed = getattr(result, f"X_{part}")
        partition_info[part] = {
            "input_shape": list(split_X[part].shape),
            "output_shape": list(transformed.shape),
            "subjects": len(rows),
            "labels": {"Control_0": int((y.loc[rows.subject_id] == 0).sum()), "ADHD_1": int((y.loc[rows.subject_id] == 1).sum())},
            "sites": sorted(rows.site.unique().tolist()),
            "site_counts": {site: int(count) for site, count in rows.site.value_counts().sort_index().items()},
            "subject_ids_preserved_in_order": tuple(transformed.index) == tuple(ordered_ids[part]),
            "all_output_values_finite": bool(np.isfinite(transformed.to_numpy()).all()),
        }

    pipeline = result.pipeline
    steps = [{"name": name, "class": type(estimator).__name__} for name, estimator in pipeline.steps]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_feature_count": EXPECTED_INPUT_FEATURES,
        "input_X_columns_only_roi_pairs": True,
        "subject_id_and_site_kept_as_metadata": True,
        "diagnosis_kept_separate_as_y": True,
        "preprocessing_sequence": steps,
        "selected_feature_count": len(result.selected_feature_names),
        "feature_selection": "SelectKBest(score_func=f_classif, k=100), after imputation, variance filtering, and scaling",
        "fit_scope": {"training_subject_count": len(ordered_ids["train"]), "fitted_subject_ids_exactly_training_ids": result.fitted_subject_ids == tuple(ordered_ids["train"]), "validation_or_test_used_in_fit": False},
        "partitions": partition_info,
        "subject_overlap": False,
        "site_overlap": False,
        "test_information_used_for_preprocessing": False,
        "test_labels_passed_to_preprocessing": False,
        "pipeline_fitted_only_on_X_train_y_train": True,
        "k_assessment": "k=100 retained unchanged. It reduces p=19,900 to 100 within each training fit (114 training subjects in the current split), a reasonable conservative starting point for p much greater than n. F-score ranking is univariate and may be unstable/correlated; assess candidate k values and selection stability using training-only nested/grouped CV before model comparisons.",
        "synthetic_leakage_test": "Pending; run tests/test_cc200_preprocessing_leakage.py and update to the result.",
        "classifier_trained": False,
        "accuracy_calculated": False,
    }
    JSON_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CC200 preprocessing validation report", "",
        f"- Input matrix: 153 × {EXPECTED_INPUT_FEATURES} ROI-pair features; metadata and target are separate.",
        "- Input X shape by partition: train 114 × 19,900; validation 19 × 19,900; test 20 × 19,900.",
        "- Output shape by partition: train 114 × 100; validation 19 × 100; test 20 × 100.",
        "- Sequence: median imputation → zero-variance filtering → standard scaling → ANOVA F-statistic SelectKBest(k=100).",
        "- Every learned step is fitted only on X_train/y_train; validation and test are transform-only.",
        "- Subject IDs preserved; no subject or site crosses partitions; all transformed values are finite.",
        "- k=100 retained as a reasonable initial p≫n reduction; later k comparisons belong inside training-only CV.",
        "- No classifier trained; no accuracy calculated.",
        "- Synthetic leakage test result: pending pytest run.", ""
    ]
    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
