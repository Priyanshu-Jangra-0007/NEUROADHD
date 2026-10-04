"""Extract subject-level Fisher-z CC200 functional-connectivity features.

Reads the downloaded ROI time courses and the validated interim subject-label
mapping. Raw inputs are read-only; outputs are written under data/adhd200/features/cc200.
"""
from __future__ import annotations

import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CC200 = PROJECT_ROOT / "data/adhd200/raw/cc200"
MAPPING = PROJECT_ROOT / "data/adhd200/interim/cc200_subject_label_mapping.csv"
OUTPUT = PROJECT_ROOT / "data/adhd200/features/cc200"
N_ROIS = 200
FISHER_EPSILON = 1e-7
EXPECTED_SUBJECTS = 154


def load_roi_timeseries(path: Path) -> tuple[np.ndarray, int]:
    """Return a (timepoints, ROIs) float64 matrix and original timepoint count."""
    frame = pd.read_csv(path)
    if frame.shape[0] != N_ROIS:
        raise ValueError(f"expected 200 ROI rows, found {frame.shape[0]}")
    if frame.shape[1] < 3:
        raise ValueError(f"expected ROI ID plus multiple timepoints, found {frame.shape[1]} columns")

    roi_ids = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy()
    if not np.array_equal(roi_ids, np.arange(1, N_ROIS + 1)):
        raise ValueError("ROI row identifiers are not exactly 1..200 in order")

    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if values.shape[0] != N_ROIS:
        raise ValueError(f"unexpected ROI-by-time shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("time series contains missing or non-finite values")

    # The source is ROI x timepoint; corrcoef expects variables in columns here.
    time_by_roi = values.T
    if time_by_roi.shape[1] != N_ROIS:
        raise ValueError(f"transposed shape is {time_by_roi.shape}, expected timepoints x 200 ROIs")
    return time_by_roi, time_by_roi.shape[0]


def extract() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
    eligible = mapping.loc[mapping["status"].eq("matched_labeled")].copy()
    eligible = eligible.loc[eligible["binary_label"].isin(["ADHD", "Control"])].copy()
    eligible["diagnosis"] = eligible["binary_label"].map({"Control": 0, "ADHD": 1}).astype(int)

    if len(eligible) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"validated mapping has {len(eligible)} eligible subjects; expected {EXPECTED_SUBJECTS}")
    if eligible["cc200_subject_id"].duplicated().any():
        raise RuntimeError("validated mapping contains duplicate CC200 subject IDs")

    pairs = list(itertools.combinations(range(1, N_ROIS + 1), 2))
    feature_names = [f"ROI_{i:03d}__ROI_{j:03d}" for i, j in pairs]
    tri_i, tri_j = np.triu_indices(N_ROIS, k=1)
    if len(feature_names) != 19_900 or len(tri_i) != 19_900:
        raise RuntimeError("upper-triangle feature count is not 19,900")

    rows: list[dict] = []
    qc_rows: list[dict] = []
    skipped: list[dict] = []
    total_clipped = 0
    subjects_with_clips = 0
    total_constant_roi_signals = 0

    for item in eligible.itertuples(index=False):
        sid = item.cc200_subject_id
        # Downloaded source folders retain the C-PAC session suffix, whereas
        # the validated mapping stores the scan ID without that suffix.
        path = RAW_CC200 / f"{sid}_session_1" / "roi_CC200.csv"
        if not path.is_file():
            skipped.append({"subject_id": sid, "reason": "CC200 input file missing", "input": str(path)})
            continue
        constant_count = 0
        n_timepoints = ""
        try:
            time_by_roi, n_timepoints = load_roi_timeseries(path)
            std = np.std(time_by_roi, axis=0, ddof=0)
            constant_count = int(np.count_nonzero(std == 0.0))
            total_constant_roi_signals += constant_count
            if constant_count:
                raise ValueError(f"{constant_count} ROI time series are constant; Pearson correlations undefined")

            corr = np.corrcoef(time_by_roi, rowvar=False)
            if corr.shape != (N_ROIS, N_ROIS):
                raise ValueError(f"correlation matrix has unexpected shape {corr.shape}")
            if not np.isfinite(corr).all():
                raise ValueError("Pearson correlation matrix contains non-finite values")
            # Remove floating-point asymmetry before the Fisher transform.
            corr = (corr + corr.T) / 2.0
            np.fill_diagonal(corr, 1.0)
            edge_corr = corr[tri_i, tri_j]
            clip_mask = np.abs(edge_corr) >= (1.0 - FISHER_EPSILON)
            clipped_count = int(np.count_nonzero(clip_mask))
            z = np.arctanh(np.clip(edge_corr, -1.0 + FISHER_EPSILON, 1.0 - FISHER_EPSILON))
            if not np.isfinite(z).all():
                raise ValueError("Fisher-z vector contains non-finite values after boundary clipping")

            row = {"subject_id": sid, "site": item.site}
            row.update(zip(feature_names, z.tolist()))
            row["diagnosis"] = int(item.diagnosis)
            rows.append(row)
            total_clipped += clipped_count
            subjects_with_clips += int(clipped_count > 0)
            qc_rows.append({
                "subject_id": sid,
                "site": item.site,
                "timepoints": n_timepoints,
                "roi_count": N_ROIS,
                "constant_roi_signals": constant_count,
                "fisher_z_clipped_edges": clipped_count,
                "features_finite": bool(np.isfinite(z).all()),
                "status": "extracted",
            })
        except Exception as exc:
            skipped.append({"subject_id": sid, "reason": str(exc), "input": str(path)})
            qc_rows.append({"subject_id": sid, "site": item.site, "timepoints": n_timepoints, "roi_count": N_ROIS,
                            "constant_roi_signals": constant_count, "fisher_z_clipped_edges": "",
                            "features_finite": False, "status": "skipped: " + str(exc)})

    # Stable, documented order: subject_id, site, lexicographic upper-triangle edge order, diagnosis.
    table_columns = ["subject_id", "site", *feature_names, "diagnosis"]
    table = pd.DataFrame(rows, columns=table_columns)
    table_path = OUTPUT / "cc200_connectivity_features.csv"
    table.to_csv(table_path, index=False, float_format="%.10g")

    metadata_path = OUTPUT / "cc200_feature_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_index", "feature_name", "roi_i", "roi_j", "ordering"])
        writer.writeheader()
        for idx, ((i, j), name) in enumerate(zip(pairs, feature_names), start=1):
            writer.writerow({"feature_index": idx, "feature_name": name, "roi_i": i, "roi_j": j,
                             "ordering": "upper triangle, row-major, diagonal excluded"})

    qc_path = OUTPUT / "cc200_subject_extraction_qc.csv"
    pd.DataFrame(qc_rows, columns=["subject_id", "site", "timepoints", "roi_count", "constant_roi_signals",
                                   "fisher_z_clipped_edges", "features_finite", "status"]).to_csv(qc_path, index=False)
    skipped_path = OUTPUT / "cc200_skipped_subjects.csv"
    pd.DataFrame(skipped, columns=["subject_id", "reason", "input"]).to_csv(skipped_path, index=False)

    count_by_label = table["diagnosis"].value_counts().to_dict() if not table.empty else {}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(RAW_CC200.relative_to(PROJECT_ROOT)),
        "label_mapping": str(MAPPING.relative_to(PROJECT_ROOT)),
        "output_table": str(table_path.relative_to(PROJECT_ROOT)),
        "requested_validated_subjects": int(len(eligible)),
        "subjects_extracted": int(len(table)),
        "subjects_skipped": int(len(skipped)),
        "skipped_subjects": skipped,
        "features_per_subject": int(len(feature_names)),
        "table_shape": [int(table.shape[0]), int(table.shape[1])],
        "table_columns": {"metadata": ["subject_id", "site"], "connectivity_features": int(len(feature_names)), "label": "diagnosis"},
        "label_counts": {"Control_0": int(count_by_label.get(0, 0)), "ADHD_1": int(count_by_label.get(1, 0))},
        "input_orientation": "200 ROI rows x variable timepoints",
        "correlation_orientation": "timepoints x 200 ROIs",
        "correlation": "Pearson; upper triangle only; diagonal excluded",
        "fisher_z": {"transform": "arctanh(r)", "epsilon": FISHER_EPSILON,
                      "clipped_edge_values_total": int(total_clipped), "subjects_with_clipped_edges": int(subjects_with_clips)},
        "constant_roi_signals_total": int(total_constant_roi_signals),
        "scaling_performed": False,
        "feature_selection_performed": False,
        "train_test_split_performed": False,
        "model_training_performed": False,
    }
    summary_path = OUTPUT / "cc200_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = extract()
    print(json.dumps(result, indent=2))
