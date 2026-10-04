"""Individual model training and hyperparameter tuning script.

Performs leakage-safe preprocessing fitting strictly on training data, evaluates hyperparameter grid
candidates on the validation set, and records all metrics, metadata, and model artifacts into run folders.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent / "NeuroADHD"

# Import evaluation metrics & tracker
sys.path.insert(0, str(BENCHMARK_ROOT / "scripts"))
from evaluate_model import compute_classification_metrics
from run_tracker import log_timeline_event

# Model constructor imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def load_model_class(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression
    elif model_name in ("linear_svm", "rbf_svm"):
        return SVC
    elif model_name == "random_forest":
        return RandomForestClassifier
    elif model_name == "extra_trees":
        return ExtraTreesClassifier
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def make_preprocessing_pipeline(k: int = 100) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k))
    ])


def train_single_model(model_name: str, run_id: str) -> dict:
    start_time = time.time()
    log_timeline_event(run_id, f"model_training_{model_name}", "started", f"Started training {model_name}")
    
    # Paths
    config_path = BENCHMARK_ROOT / f"configs/models/{model_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config for {model_name} not found at {config_path}")
    
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    
    feature_table_path = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"
    manifest_path = PROJECT_ROOT / "data/adhd200/processed/splits/subject_split_manifest.csv"
    
    # Load dataset & metadata
    table = pd.read_csv(feature_table_path, dtype={"subject_id": str, "site": str})
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str, "site": str})
    
    feature_cols = [c for c in table.columns if c.startswith("ROI_")]
    if len(feature_cols) != 19_900:
        raise ValueError(f"Expected 19,900 ROI features, found {len(feature_cols)}")
        
    X_full = table.loc[:, feature_cols].astype("float64")
    X_full.index = table["subject_id"].astype(str)
    y_full = table["diagnosis"].astype("int64")
    y_full.index = table["subject_id"].astype(str)
    
    train_ids = manifest.loc[manifest["partition"] == "train", "subject_id"].tolist()
    val_ids = manifest.loc[manifest["partition"] == "validation", "subject_id"].tolist()
    test_ids = manifest.loc[manifest["partition"] == "test", "subject_id"].tolist()
    
    X_train = X_full.loc[train_ids]
    y_train = y_full.loc[train_ids].to_numpy()
    
    X_val = X_full.loc[val_ids]
    y_val = y_full.loc[val_ids].to_numpy()
    
    X_test = X_full.loc[test_ids]
    y_test = y_full.loc[test_ids].to_numpy()
    
    # FIT PREPROCESSING STRICTLY ON TRAIN ONLY
    preproc = make_preprocessing_pipeline(k=100)
    preproc.fit(X_train, y_train)
    
    X_train_prep = preproc.transform(X_train)
    X_val_prep = preproc.transform(X_val)
    X_test_prep = preproc.transform(X_test)
    
    # Selected feature names
    var_sup = preproc.named_steps["variance"].get_support()
    sel_sup = preproc.named_steps["select"].get_support()
    selected_features = list(np.asarray(feature_cols)[var_sup][sel_sup])
    
    # Hyperparameter search using VALIDATION ROC-AUC
    param_grid_raw = model_config["param_grid"]
    if isinstance(param_grid_raw, dict):
        grid_list = list(ParameterGrid(param_grid_raw))
    elif isinstance(param_grid_raw, list):
        grid_list = []
        for g in param_grid_raw:
            grid_list.extend(list(ParameterGrid(g)))
    else:
        raise ValueError("Invalid param_grid structure")
        
    ModelClass = load_model_class(model_name)
    
    best_score = -1.0
    best_params = None
    best_model = None
    tuning_history = []
    
    for idx, params in enumerate(grid_list):
        try:
            clf = ModelClass(**params)
            clf.fit(X_train_prep, y_train)
            
            # Predict on Val
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val_prep)[:, 1]
            elif hasattr(clf, "decision_function"):
                df_scores = clf.decision_function(X_val_prep)
                probs = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-12)
            else:
                probs = clf.predict(X_val_prep).astype(float)
                
            val_m = compute_classification_metrics(y_val, clf.predict(X_val_prep), probs)
            score = val_m["roc_auc"]
            
            param_clean = {k: str(v) if isinstance(v, type) else v for k, v in params.items()}
            tuning_history.append({
                "trial": idx + 1,
                "params": param_clean,
                "validation_roc_auc": round(score, 4),
                "validation_accuracy": round(val_m["accuracy"], 4)
            })
            
            if score > best_score or best_params is None:
                best_score = score
                best_params = params
                best_model = clf
        except Exception as exc:
            continue

    if best_model is None:
        log_timeline_event(run_id, f"model_training_{model_name}", "failed", f"Failed to fit model {model_name}")
        raise RuntimeError(f"Failed to fit any valid model for {model_name}")

    # Evaluate train & validation on best model
    t0_pred = time.time()
    train_preds = best_model.predict(X_train_prep)
    train_probs = best_model.predict_proba(X_train_prep)[:, 1] if hasattr(best_model, "predict_proba") else None
    pred_time = time.time() - t0_pred
    
    val_preds = best_model.predict(X_val_prep)
    val_probs = best_model.predict_proba(X_val_prep)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    train_metrics = compute_classification_metrics(y_train, train_preds, train_probs)
    val_metrics = compute_classification_metrics(y_val, val_preds, val_probs)
    
    total_time = time.time() - start_time
    
    # Save Run Artifacts under runs/<run_id>/
    run_folder = BENCHMARK_ROOT / "runs" / run_id
    model_subfolder = run_folder / "models" / model_name
    metrics_subfolder = run_folder / "metrics"
    preds_subfolder = run_folder / "predictions"
    cm_subfolder = run_folder / "confusion_matrices"
    
    model_subfolder.mkdir(parents=True, exist_ok=True)
    metrics_subfolder.mkdir(parents=True, exist_ok=True)
    preds_subfolder.mkdir(parents=True, exist_ok=True)
    cm_subfolder.mkdir(parents=True, exist_ok=True)
    
    # Legacy models dir fallback for compatibility
    legacy_models_dir = BENCHMARK_ROOT / "models" / run_id / model_name
    legacy_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Dump joblib objects
    joblib.dump(best_model, model_subfolder / "model.joblib")
    joblib.dump(preproc, model_subfolder / "preprocessing_pipeline.joblib")
    joblib.dump(best_model, legacy_models_dir / "model.joblib")
    joblib.dump(preproc, legacy_models_dir / "preprocessing_pipeline.joblib")
    
    result_data = {
        "model_name": model_name,
        "display_name": model_config.get("display_name", model_name),
        "model_type": model_config.get("model_type", model_name),
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": model_config.get("random_state", 42),
        "best_hyperparameters": {k: str(v) if isinstance(v, type) else v for k, v in best_params.items()},
        "hyperparameter_search": {
            "search_method": "GridSearch",
            "selection_metric": "validation_roc_auc",
            "trials_count": len(grid_list),
            "tuning_history": tuning_history
        },
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "training_data_count": len(train_ids),
        "validation_data_count": len(val_ids),
        "test_data_count": len(test_ids),
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "timing_seconds": {
            "total_execution_time": round(total_time, 3),
            "prediction_time": round(pred_time, 4)
        },
        "status": "completed"
    }
    
    # Save JSON files
    (metrics_subfolder / f"{model_name}.json").write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")
    (model_subfolder / "metrics.json").write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")
    
    # Save Confusion Matrix JSON
    (cm_subfolder / f"{model_name}_cm.json").write_text(json.dumps(val_metrics.get("confusion_matrix", {}), indent=2) + "\n", encoding="utf-8")
    
    # Save predictions CSV
    val_pred_df = pd.DataFrame({
        "subject_id": val_ids,
        "true_diagnosis": y_val,
        "predicted_diagnosis": val_preds,
        "probability_adhd": val_probs if val_probs is not None else np.nan
    })
    val_pred_df.to_csv(preds_subfolder / f"{model_name}_validation_predictions.csv", index=False)
    
    log_timeline_event(run_id, f"model_training_{model_name}", "completed", f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
    
    print(f"[{model_config.get('display_name', model_name)}]")
    print(f"  Best Params: {best_params}")
    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
    
    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model configuration name")
    parser.add_argument("--run-id", required=True, help="Benchmark run ID")
    args = parser.parse_args()
    
    train_single_model(args.model, args.run_id)


if __name__ == "__main__":
    main()
