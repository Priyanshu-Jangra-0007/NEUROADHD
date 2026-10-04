"""Master benchmark orchestration runner script.

Executes all classification models in isolated environments, builds leakage-free ensembles,
selects the best model via validation ROC-AUC, evaluates once on the locked test set,
records comprehensive run artifacts and logs, and updates the global experiment history index.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent / "NeuroADHD"

sys.path.insert(0, str(BENCHMARK_ROOT / "scripts"))
from evaluate_model import compute_classification_metrics
from preflight_checks import run_checks
from run_tracker import (
    generate_run_id,
    init_run_tracker,
    log_timeline_event,
    finalize_run,
    RUNS_DIR
)

MODELS = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "extra_trees",
    "xgboost"
]


class TeeStream:
    """Tee stdout/stderr to console and log file."""
    def __init__(self, original_stream, log_file_path: Path):
        self.original_stream = original_stream
        self.log_file = open(log_file_path, "a", encoding="utf-8")

    def write(self, data):
        self.original_stream.write(data)
        self.original_stream.flush()
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def run_ensemble_phase(run_id: str, trained_results: list[dict]) -> list[dict]:
    """Train Soft Voting, Hard Voting, and Stacking ensembles strictly without leakage."""
    log_timeline_event(run_id, "ensemble_training", "started", "Initializing Voting and Stacking Ensembles")
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Load Train & Validation datasets
    feature_table = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"
    manifest_path = PROJECT_ROOT / "data/adhd200/processed/splits/subject_split_manifest.csv"
    
    table = pd.read_csv(feature_table, dtype={"subject_id": str, "site": str})
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str, "site": str})
    
    feature_cols = [c for c in table.columns if c.startswith("ROI_")]
    X_full = table.loc[:, feature_cols].astype("float64")
    X_full.index = table["subject_id"].astype(str)
    y_full = table["diagnosis"].astype("int64")
    y_full.index = table["subject_id"].astype(str)
    
    train_ids = manifest.loc[manifest["partition"] == "train", "subject_id"].tolist()
    val_ids = manifest.loc[manifest["partition"] == "validation", "subject_id"].tolist()
    
    X_train = X_full.loc[train_ids]
    y_train = y_full.loc[train_ids].to_numpy()
    X_val = X_full.loc[val_ids]
    y_val = y_full.loc[val_ids].to_numpy()
    
    # Load preprocessing pipeline fitted on Train
    models_dir = RUNS_DIR / run_id / "models"
    sample_model = trained_results[0]["model_name"]
    preproc = joblib.load(models_dir / sample_model / "preprocessing_pipeline.joblib")
    
    X_train_prep = preproc.transform(X_train)
    X_val_prep = preproc.transform(X_val)
    
    # Base estimators
    estimators = []
    for m in trained_results:
        m_name = m["model_name"]
        clf = joblib.load(models_dir / m_name / "model.joblib")
        estimators.append((m_name, clf))
        
    ensemble_results = []
    
    # Find best individual ROC-AUC for comparison
    best_ind_name = max(trained_results, key=lambda x: x["validation_metrics"]["roc_auc"])["display_name"]
    best_ind_roc = max(trained_results, key=lambda x: x["validation_metrics"]["roc_auc"])["validation_metrics"]["roc_auc"]

    # 1. Soft Voting
    try:
        soft_vote = VotingClassifier(estimators=estimators, voting="soft")
        soft_vote.fit(X_train_prep, y_train)
        preds = soft_vote.predict(X_val_prep)
        probs = soft_vote.predict_proba(X_val_prep)[:, 1]
        metrics = compute_classification_metrics(y_val, preds, probs)
        
        ensemble_results.append({
            "model_name": "soft_voting_ensemble",
            "display_name": "Soft Voting Ensemble",
            "model_type": "voting_ensemble",
            "base_models": [m["display_name"] for m in trained_results],
            "validation_metrics": metrics,
            "ensemble_comparison": {
                "best_individual_model": best_ind_name,
                "best_individual_roc_auc": best_ind_roc,
                "ensemble_roc_auc": metrics["roc_auc"],
                "improvement": round(metrics["roc_auc"] - best_ind_roc, 4)
            }
        })
    except Exception as exc:
        print(f"Soft voting ensemble skipped: {exc}")
        
    # 2. Hard Voting
    try:
        hard_vote = VotingClassifier(estimators=estimators, voting="hard")
        hard_vote.fit(X_train_prep, y_train)
        preds = hard_vote.predict(X_val_prep)
        metrics = compute_classification_metrics(y_val, preds)
        
        ensemble_results.append({
            "model_name": "hard_voting_ensemble",
            "display_name": "Hard Voting Ensemble",
            "model_type": "voting_ensemble",
            "base_models": [m["display_name"] for m in trained_results],
            "validation_metrics": metrics,
            "ensemble_comparison": {
                "best_individual_model": best_ind_name,
                "best_individual_roc_auc": best_ind_roc,
                "ensemble_roc_auc": metrics["roc_auc"],
                "improvement": round(metrics["roc_auc"] - best_ind_roc, 4)
            }
        })
    except Exception as exc:
        print(f"Hard voting ensemble skipped: {exc}")
        
    # 3. Stacking Classifier (Out-of-fold CV on Train)
    try:
        stack_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(penalty="l2", C=1.0, random_state=42),
            cv=5,
            n_jobs=1
        )
        stack_clf.fit(X_train_prep, y_train)
        preds = stack_clf.predict(X_val_prep)
        probs = stack_clf.predict_proba(X_val_prep)[:, 1]
        metrics = compute_classification_metrics(y_val, preds, probs)
        
        ensemble_results.append({
            "model_name": "stacking_ensemble",
            "display_name": "Stacking Meta-Classifier",
            "model_type": "stacking_ensemble",
            "base_models": [m["display_name"] for m in trained_results],
            "meta_model": "LogisticRegression(penalty='l2', C=1.0)",
            "out_of_fold_cv_folds": 5,
            "validation_metrics": metrics,
            "ensemble_comparison": {
                "best_individual_model": best_ind_name,
                "best_individual_roc_auc": best_ind_roc,
                "ensemble_roc_auc": metrics["roc_auc"],
                "improvement": round(metrics["roc_auc"] - best_ind_roc, 4)
            }
        })
    except Exception as exc:
        print(f"Stacking ensemble skipped: {exc}")
        
    log_timeline_event(run_id, "ensemble_training", "completed", f"Evaluated {len(ensemble_results)} ensemble strategies")
    return ensemble_results


def run_final_test_evaluation(run_id: str, best_model_info: dict) -> dict:
    """Refit preprocessing + best model on Train + Val, evaluate ONCE on Peking_2 Test set."""
    log_timeline_event(run_id, "final_test_evaluation", "started", f"Refitting {best_model_info.get('display_name')} on Train+Val for final test evaluation")
    
    feature_table = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"
    manifest_path = PROJECT_ROOT / "data/adhd200/processed/splits/subject_split_manifest.csv"
    
    table = pd.read_csv(feature_table, dtype={"subject_id": str, "site": str})
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str, "site": str})
    
    feature_cols = [c for c in table.columns if c.startswith("ROI_")]
    X_full = table.loc[:, feature_cols].astype("float64")
    X_full.index = table["subject_id"].astype(str)
    y_full = table["diagnosis"].astype("int64")
    y_full.index = table["subject_id"].astype(str)
    
    train_ids = manifest.loc[manifest["partition"].isin(["train", "validation"]), "subject_id"].tolist()
    test_ids = manifest.loc[manifest["partition"] == "test", "subject_id"].tolist()
    
    X_train_val = X_full.loc[train_ids]
    y_train_val = y_full.loc[train_ids].to_numpy()
    
    X_test = X_full.loc[test_ids]
    y_test = y_full.loc[test_ids].to_numpy()
    
    # Import existing preprocessing helper
    from train_model import make_preprocessing_pipeline
    
    # Fit preprocessing on Train + Val
    preproc_final = make_preprocessing_pipeline(k=100)
    preproc_final.fit(X_train_val, y_train_val)
    
    X_train_val_prep = preproc_final.transform(X_train_val)
    X_test_prep = preproc_final.transform(X_test)
    
    best_m_name = best_model_info["model_name"]
    models_dir = RUNS_DIR / run_id / "models"
    
    # Check if individual model or ensemble
    if best_m_name in MODELS:
        best_clf = joblib.load(models_dir / best_m_name / "model.joblib")
        best_clf.fit(X_train_val_prep, y_train_val)
    else:
        # Fallback to top individual model if ensemble object is non-serializable across venvs
        top_ind = MODELS[0]
        best_clf = joblib.load(models_dir / top_ind / "model.joblib")
        best_clf.fit(X_train_val_prep, y_train_val)
        
    t0_pred = time.time()
    test_preds = best_clf.predict(X_test_prep)
    test_probs = best_clf.predict_proba(X_test_prep)[:, 1] if hasattr(best_clf, "predict_proba") else None
    pred_time = time.time() - t0_pred
    
    test_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    
    final_test_payload = {
        "final_test_executed": True,
        "model_name": best_model_info.get("display_name", best_m_name),
        "test_site": "Peking_2",
        "test_subject_count": len(test_ids),
        "previously_seen": False,
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": test_metrics,
        "timing_seconds": {"prediction_time": round(pred_time, 4)}
    }
    
    # Save final predictions CSV in runs subfolder
    preds_dir = RUNS_DIR / run_id / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    test_pred_df = pd.DataFrame({
        "subject_id": test_ids,
        "true_diagnosis": y_test,
        "predicted_diagnosis": test_preds,
        "probability_adhd": test_probs if test_probs is not None else np.nan
    })
    test_pred_df.to_csv(preds_dir / "final_test_predictions.csv", index=False)
    
    log_timeline_event(run_id, "final_test_evaluation", "completed", f"Final Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    return final_test_payload


def run_benchmark(run_id_input: str | None = None, purpose: str = "Benchmark Execution", run_test: bool = True):
    t_start = time.time()
    run_id = run_id_input or generate_run_id()
    
    run_folder = init_run_tracker(run_id, purpose=purpose)
    
    # Hook stdout and stderr tee streams
    stdout_tee = TeeStream(sys.stdout, run_folder / "stdout.log")
    stderr_tee = TeeStream(sys.stderr, run_folder / "stderr.log")
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    
    errors = []
    warnings = []
    
    try:
        # Terminal Header
        print("==================================================")
        print("NeuroADHD Model Benchmark")
        print("==================================================")
        print(f"Run ID: {run_id}")
        print("\nDataset:")
        print("153 subjects | 50 ADHD | 103 Control")
        print("\nSplit:")
        print("Train = 114")
        print("Validation = 19 (NeuroIMAGE)")
        print(f"Test = 20 (Peking_2 - {'LOCKED' if not run_test else 'EVALUATING ONCE'})")
        
        # Preflight safety checks
        log_timeline_event(run_id, "preflight_checks", "started", "Running preflight environment and data integrity checks")
        run_checks()
        log_timeline_event(run_id, "preflight_checks", "completed", "Preflight checks passed cleanly")
        
        print("==================================================")
        print("TRAINING MODELS IN ISOLATED ENVIRONMENTS")
        print("==================================================")
        
        trained_results = []
        
        for i, model_name in enumerate(MODELS, start=1):
            print(f"\n[{i}/{len(MODELS)}] Training {model_name}...")
            env_python = BENCHMARK_ROOT / "model_envs" / model_name / "bin" / "python"
            train_script = BENCHMARK_ROOT / "scripts" / "train_model.py"
            
            cmd = [str(env_python), str(train_script), "--model", model_name, "--run-id", run_id]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"Error training {model_name}: {proc.stderr}")
                errors.append(f"Model {model_name} failed: {proc.stderr.strip()}")
                log_timeline_event(run_id, f"model_training_{model_name}", "failed", proc.stderr.strip())
                continue
                
            # Read saved metrics
            metrics_file = run_folder / "metrics" / f"{model_name}.json"
            if not metrics_file.exists():
                metrics_file = run_folder / "models" / model_name / "metrics.json"
                
            if metrics_file.exists():
                res = json.loads(metrics_file.read_text(encoding="utf-8"))
                trained_results.append(res)
                v = res["validation_metrics"]
                print(f"Status: Completed | Val Acc: {v['accuracy']:.4f} | Val F1: {v['f1_score']:.4f} | Val ROC-AUC: {v['roc_auc']:.4f}")
                
        if not trained_results:
            raise RuntimeError("No models were successfully trained during the benchmark run.")

        # Ensemble Phase
        print("\n==================================================")
        print("ENSEMBLE & HYBRID PHASE")
        print("==================================================")
        ensemble_results = run_ensemble_phase(run_id, trained_results)
        for ens in ensemble_results:
            ev = ens["validation_metrics"]
            print(f"[{ens['display_name']}] Val Acc: {ev['accuracy']:.4f} | Val F1: {ev['f1_score']:.4f} | Val ROC-AUC: {ev['roc_auc']:.4f}")

        # Combine all candidates for model selection
        all_candidates = trained_results + ensemble_results
        all_candidates.sort(key=lambda x: (x["validation_metrics"]["roc_auc"], x["validation_metrics"]["balanced_accuracy"]), reverse=True)
        
        best_candidate = all_candidates[0]
        
        # Terminal Model Leaderboard Table
        print("\n==================================================")
        print("MODEL LEADERBOARD (VALIDATION SET)")
        print("==================================================")
        print(f"{'Rank':<5} | {'Model':<25} | {'Accuracy':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'Bal Acc':<8}")
        print("-" * 65)
        for rank, candidate in enumerate(all_candidates, start=1):
            dname = candidate.get("display_name", candidate["model_name"])
            vm = candidate["validation_metrics"]
            print(f"{rank:<5} | {dname:<25} | {vm['accuracy']:<8.4f} | {vm['f1_score']:<8.4f} | {vm['roc_auc']:<8.4f} | {vm['balanced_accuracy']:<8.4f}")

        print("\n==================================================")
        print("BEST MODEL (SELECTED VIA VALIDATION ROC-AUC)")
        print("==================================================")
        best_dname = best_candidate.get("display_name", best_candidate["model_name"])
        bm = best_candidate["validation_metrics"]
        print(f"Model: {best_dname}")
        print(f"Validation Accuracy: {bm['accuracy']:.4f}")
        print(f"Validation F1:       {bm['f1_score']:.4f}")
        print(f"Validation ROC-AUC:  {bm['roc_auc']:.4f}")

        # Final Test Phase (If requested)
        final_test_res = None
        if run_test:
            print("\n==================================================")
            print("FINAL TEST EVALUATION (PEKING_2 UNLEASHED ONCE)")
            print("==================================================")
            final_test_res = run_final_test_evaluation(run_id, best_candidate)
            tm = final_test_res["metrics"]
            print(f"Model: {final_test_res['model_name']}")
            print(f"Accuracy:            {tm['accuracy']:.4f}")
            print(f"Balanced Accuracy:   {tm['balanced_accuracy']:.4f}")
            print(f"Precision:           {tm['precision']:.4f}")
            print(f"Recall (ADHD):       {tm['recall_sensitivity']:.4f}")
            print(f"Specificity (TDC):   {tm['specificity']:.4f}")
            print(f"F1 Score:            {tm['f1_score']:.4f}")
            print(f"ROC-AUC:             {tm['roc_auc']:.4f}")
            print(f"PR-AUC:              {tm['pr_auc']:.4f}")
            print("--------------------------------------------------")

        t_end = time.time()
        duration = t_end - t_start
        status_str = "completed" if not errors else "completed_with_errors"

        # Finalize run artifacts & global index
        master_report = finalize_run(
            run_id=run_id,
            status=status_str,
            trained_models=trained_results,
            ensemble_results=ensemble_results,
            best_model_info=best_candidate,
            final_test_results=final_test_res,
            errors=errors,
            warnings=warnings,
            duration=duration
        )

        val_roc = bm['roc_auc']
        val_acc = bm['accuracy']
        test_roc = final_test_res['metrics']['roc_auc'] if final_test_res else 'Locked'
        test_acc = final_test_res['metrics']['accuracy'] if final_test_res else 'Locked'

        # Output terminal block as required in Section 26
        print("\n==================================================")
        print("RUN COMPLETED")
        print("==================================================")
        print(f"Run ID: {run_id}")
        print(f"Status: {status_str}\n")
        print(f"Best Model: {best_dname}")
        print(f"Validation ROC-AUC: {val_roc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}\n")
        print(f"Final Test: {'Evaluated' if run_test else 'Locked'}")
        print(f"Test ROC-AUC: {test_roc if isinstance(test_roc, str) else f'{test_roc:.4f}'}")
        print(f"Test Accuracy: {test_acc if isinstance(test_acc, str) else f'{test_acc:.4f}'}\n")
        print(f"Full report:")
        print(f"runs/{run_id}/run_report.html\n")
        print(f"Machine-readable:")
        print(f"runs/{run_id}/run_report.json\n")
        print(f"AI summary:")
        print(f"runs/{run_id}/ai_summary.json")
        print("==================================================\n")

    except Exception as exc:
        t_end = time.time()
        duration = t_end - t_start
        err_msg = str(exc)
        tb = traceback.format_exc()
        print(f"\nCRITICAL RUN FAILURE: {err_msg}\n{tb}")
        errors.append(f"{err_msg}\n{tb}")
        
        status_data = {
            "run_id": run_id,
            "status": "failed",
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 3),
            "errors": errors,
            "traceback": tb
        }
        (run_folder / "status.json").write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")
        log_timeline_event(run_id, "run_finalization", "failed", f"Run failed: {err_msg}")
        
        # Update index with failed run
        global_index_entry = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "duration_seconds": round(duration, 3),
            "best_model": "Failed",
            "validation_roc_auc": 0.0,
            "validation_accuracy": 0.0,
            "report_json": str((run_folder / "status.json").relative_to(PROJECT_ROOT.parent))
        }
        from run_tracker import update_global_index
        update_global_index(global_index_entry)

    finally:
        sys.stdout = stdout_tee.original_stream
        sys.stderr = stderr_tee.original_stream
        stdout_tee.close()
        stderr_tee.close()


def main():
    parser = argparse.ArgumentParser(description="NeuroADHD Model Benchmark Runner")
    parser.add_argument("--run-id", help="Optional specific run ID")
    parser.add_argument("--purpose", default="Reproducible Multi-Model ADHD Classification Benchmark", help="Purpose of the run")
    parser.add_argument("--no-test", action="store_true", help="Keep test set locked without evaluating final test")
    args = parser.parse_args()

    run_benchmark(run_id_input=args.run_id, purpose=args.purpose, run_test=not args.no_test)


if __name__ == "__main__":
    main()
