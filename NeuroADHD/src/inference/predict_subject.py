"""Single-subject ADHD pattern prediction from resting-state CC200 ROI time series.

This script executes inference on a single subject's roi_CC200.csv file using the
pre-trained and frozen NeuroADHD preprocessing pipeline and classification model.

IMPORTANT: Output is an exploratory research prediction only and DOES NOT constitute
a medical or clinical psychiatric diagnosis.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

# Dynamic path resolution to maintain portability across machines
SCRIPT_DIR = Path(__file__).resolve().parent
NEUROADHD_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = NEUROADHD_ROOT.parent
BENCHMARK_ROOT = WORKSPACE_ROOT / "NeuroADHD_ModelBenchmark"

# Add NeuroADHD to sys.path for internal imports
if str(NEUROADHD_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUROADHD_ROOT))

from src.features.extract_cc200_connectivity import (
    FISHER_EPSILON,
    N_ROIS,
    load_roi_timeseries,
)

DEFAULT_PIPELINE = BENCHMARK_ROOT / "results/final_locked/final_preprocessing_pipeline.joblib"
DEFAULT_MODEL = BENCHMARK_ROOT / "results/final_locked/final_model.joblib"

CLINICAL_DISCLAIMER = (
    "======================================================================\n"
    "RESEARCH PREDICTION NOTICE - NOT A CLINICAL DIAGNOSIS\n"
    "======================================================================\n"
    "This software generates an exploratory resting-state fMRI connectivity\n"
    "prediction for neuroscience research purposes only. It is NOT an FDA-cleared\n"
    "or CE-marked medical diagnostic device. Attention-Deficit/Hyperactivity\n"
    "Disorder (ADHD) is a complex neurodevelopmental condition diagnosed solely\n"
    "by qualified healthcare professionals using comprehensive clinical evaluations\n"
    "and standardized diagnostic criteria (DSM-5 / ICD-11).\n"
    "======================================================================"
)


def extract_connectivity_vector(time_by_roi: np.ndarray) -> np.ndarray:
    """Compute upper-triangle Fisher-z connectivity vector from (timepoints x 200 ROIs)."""
    # Check for constant/zero-variance signals
    std = np.std(time_by_roi, axis=0, ddof=0)
    constant_count = int(np.count_nonzero(std == 0.0))
    if constant_count > 0:
        raise ValueError(
            f"Input scan contains {constant_count} constant (zero-variance) ROI time series. "
            "Pearson correlation cannot be computed."
        )

    # Pearson correlation matrix (200 x 200)
    corr = np.corrcoef(time_by_roi, rowvar=False)
    if corr.shape != (N_ROIS, N_ROIS):
        raise ValueError(f"Correlation matrix has unexpected shape {corr.shape}, expected ({N_ROIS}, {N_ROIS})")
    if not np.isfinite(corr).all():
        raise ValueError("Pearson correlation matrix contains non-finite values (NaN/Inf).")

    # Symmetrize to eliminate floating-point asymmetry and set diagonal to 1.0
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    # Extract upper-triangle edges (k=1)
    tri_i, tri_j = np.triu_indices(N_ROIS, k=1)
    edge_corr = corr[tri_i, tri_j]

    # Clip to avoid arctanh singularity at +/- 1.0
    clipped = np.clip(edge_corr, -1.0 + FISHER_EPSILON, 1.0 - FISHER_EPSILON)
    z = np.arctanh(clipped)

    if not np.isfinite(z).all():
        raise ValueError("Fisher-z vector contains non-finite values after transform.")

    return z.astype(np.float64)


def predict_single_subject(
    roi_file: str | Path,
    pipeline_path: str | Path = DEFAULT_PIPELINE,
    model_path: str | Path = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Execute complete end-to-end inference for a single subject roi_CC200.csv file."""
    roi_path = Path(roi_file).resolve()
    if not roi_path.is_file():
        raise FileNotFoundError(f"Subject ROI time series file not found: {roi_path}")

    pipe_path = Path(pipeline_path).resolve()
    if not pipe_path.is_file():
        raise FileNotFoundError(f"Preprocessing pipeline artifact not found: {pipe_path}")

    mod_path = Path(model_path).resolve()
    if not mod_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {mod_path}")

    # 1. Load and validate raw CC200 ROI time series
    time_by_roi, n_timepoints = load_roi_timeseries(roi_path)

    # 2. Extract 19,900 Fisher-z connectivity features
    z_vector = extract_connectivity_vector(time_by_roi)
    if z_vector.shape != (19_900,):
        raise RuntimeError(f"Expected 19,900 connectivity features, got {z_vector.shape}")

    # 3. Load pre-fitted preprocessing pipeline and model checkpoint
    preprocessing_pipeline = joblib.load(pipe_path)
    model = joblib.load(mod_path)

    # 4. Transform features: (1, 19900) -> (1, 100)
    # Pass as 2D numpy array without DataFrame feature names to match imputer fit
    X_raw = z_vector.reshape(1, -1)
    X_selected = preprocessing_pipeline.transform(X_raw)

    if X_selected.shape != (1, 100):
        raise RuntimeError(f"Expected 100 selected features after pipeline, got {X_selected.shape}")

    # 5. Predict probabilities and class
    probabilities = model.predict_proba(X_selected)[0]
    prob_control = float(probabilities[0])
    prob_adhd = float(probabilities[1])

    predicted_class = int(prob_adhd >= 0.5)
    class_label = "ADHD Pattern" if predicted_class == 1 else "Typical Control Pattern"

    confidence_margin = abs(prob_adhd - 0.5) * 2.0  # 0.0 = completely uncertain, 1.0 = highly certain

    return {
        "subject_file": str(roi_path),
        "subject_filename": roi_path.name,
        "parent_session": roi_path.parent.name,
        "input_validation": {
            "status": "VALIDATED (PASS)",
            "roi_count": N_ROIS,
            "timepoints": int(n_timepoints),
            "features_extracted": 19_900,
            "features_selected": 100,
        },
        "model_architecture": type(model).__name__,
        "prediction": {
            "predicted_class": predicted_class,
            "class_label": class_label,
            "adhd_probability": round(prob_adhd, 4),
            "control_probability": round(prob_control, 4),
            "decision_threshold": 0.5,
            "confidence_margin": round(confidence_margin, 4),
        },
        "disclaimer": CLINICAL_DISCLAIMER,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NeuroADHD: Single-Subject fMRI Inference Tool (CC200 Atlas)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python predict_subject.py --input data/adhd200/raw/cc200/1017176_session_1/roi_CC200.csv",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to subject roi_CC200.csv file (shape: 200 ROI rows x T timepoints).",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default=str(DEFAULT_PIPELINE),
        help="Path to pre-fitted preprocessing_pipeline.joblib (default: results/final_locked).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained model.joblib checkpoint (default: results/final_locked).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable text report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save prediction results as a JSON file.",
    )

    args = parser.parse_args()

    try:
        result = predict_single_subject(
            roi_file=args.input,
            pipeline_path=args.pipeline,
            model_path=args.model,
        )

        if args.output_json:
            out_p = Path(args.output_json).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(result, indent=2), encoding="utf-8")

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            p = result["prediction"]
            val = result["input_validation"]
            print("\n" + result["disclaimer"])
            print("\n----------------------------------------------------------------------")
            print("NEUROADHD SINGLE-SUBJECT INFERENCE REPORT")
            print("----------------------------------------------------------------------")
            print(f"Subject File         : {result['subject_filename']}")
            print(f"Directory Session    : {result['parent_session']}")
            print(f"Full Path            : {result['subject_file']}")
            print(f"Input Validation     : {val['status']}")
            print(f"ROI Regions Assessed : {val['roi_count']} Craddock ROIs")
            print(f"Timepoints Processed : {val['timepoints']} BOLD frames")
            print(f"Raw Edges Calculated : {val['features_extracted']:,} pairwise functional connections")
            print(f"Features Fed to Model: {val['features_selected']} ANOVA-selected connections")
            print(f"Model Architecture   : {result['model_architecture']}")
            print("----------------------------------------------------------------------")
            print(f"PREDICTED CLASS      : {p['class_label'].upper()} (Class {p['predicted_class']})")
            print(f"ADHD Risk Probability: {p['adhd_probability'] * 100:.2f}%")
            print(f"Control Probability  : {p['control_probability'] * 100:.2f}%")
            print(f"Classification Score : {p['adhd_probability']:.4f} (Threshold = {p['decision_threshold']})")
            print("----------------------------------------------------------------------")
            if args.output_json:
                print(f"Prediction results successfully exported to: {args.output_json}\n")

    except Exception as exc:
        print(f"\n[ERROR] Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
