"""Backend service wrapper interfacing safely with NeuroADHD_ModelBenchmark experiment history."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DASHBOARD_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = DASHBOARD_ROOT.parent
BENCHMARK_ROOT = WORKSPACE_ROOT / "NeuroADHD_ModelBenchmark"
PROJECT_ROOT = WORKSPACE_ROOT / "NeuroADHD"
RUNS_DIR = BENCHMARK_ROOT / "runs"
INDEX_FILE = RUNS_DIR / "index.json"

# Active execution process state
CURRENT_JOB: Dict[str, Any] = {
    "status": "idle",
    "job_type": None,
    "target": None,
    "start_time": None,
    "log_output": [],
    "error": None
}


def get_dataset_info() -> Dict[str, Any]:
    return {
        "dataset_name": "ADHD-200 Consortium (CC200 Parcellation)",
        "modality": "rs-fMRI Functional Connectivity (Fisher-z arctanh)",
        "total_subjects": 153,
        "adhd_subjects": 50,
        "control_subjects": 103,
        "original_features": 19900,
        "selected_features": 100,
        "target_column": "diagnosis",
        "label_mapping": {"Control": 0, "ADHD": 1},
        "splits": {
            "train": {"subjects": 114, "adhd": 30, "control": 84, "sites": ["KKI", "NYU", "OHSU", "Peking_1", "Peking_3", "Pittsburgh"]},
            "validation": {"subjects": 19, "adhd": 10, "control": 9, "sites": ["NeuroIMAGE"], "status": "Active Validation Holdout"},
            "test": {"subjects": 20, "adhd": 10, "control": 10, "sites": ["Peking_2"], "status": "STRICTLY LOCKED"}
        }
    }


def get_all_runs_index() -> List[Dict[str, Any]]:
    """Return array of run metadata entries from global index or disk scan."""
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    if not RUNS_DIR.exists():
        return []
        
    runs_index = []
    for run_dir in sorted(RUNS_DIR.glob("run_*"), reverse=True):
        if not run_dir.is_dir():
            continue
        report_file = run_dir / "run_report.json"
        status_file = run_dir / "status.json"
        
        if report_file.exists():
            try:
                rep = json.loads(report_file.read_text(encoding="utf-8"))
                sel = rep.get("selection", {})
                ft = rep.get("final_test", {})
                runs_index.append({
                    "run_id": rep.get("run_id", run_dir.name),
                    "created_at": rep.get("created_at"),
                    "status": rep.get("status", "completed"),
                    "duration_seconds": rep.get("duration_seconds", 0),
                    "best_model": sel.get("selected_model", "Unknown"),
                    "validation_roc_auc": sel.get("selected_validation_roc_auc", 0.0),
                    "validation_accuracy": sel.get("selected_validation_accuracy", 0.0),
                    "final_test_roc_auc": ft.get("metrics", {}).get("roc_auc") if ft.get("final_test_executed") else None,
                    "final_test_accuracy": ft.get("metrics", {}).get("accuracy") if ft.get("final_test_executed") else None,
                    "report_json": f"runs/{run_dir.name}/run_report.json",
                    "report_html": f"runs/{run_dir.name}/run_report.html"
                })
            except Exception:
                continue
        elif status_file.exists():
            try:
                st = json.loads(status_file.read_text(encoding="utf-8"))
                runs_index.append({
                    "run_id": st.get("run_id", run_dir.name),
                    "created_at": st.get("start_time_utc"),
                    "status": st.get("status", "failed"),
                    "duration_seconds": st.get("duration_seconds", 0),
                    "best_model": "Failed Run",
                    "validation_roc_auc": 0.0,
                    "validation_accuracy": 0.0,
                    "final_test_roc_auc": None,
                    "final_test_accuracy": None,
                    "report_json": f"runs/{run_dir.name}/status.json",
                    "report_html": ""
                })
            except Exception:
                continue
                
    return runs_index


def get_latest_run_id() -> Optional[str]:
    runs = get_all_runs_index()
    return runs[0]["run_id"] if runs else None


def load_run_report(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run {run_id} does not exist.")
        
    report_file = run_dir / "run_report.json"
    if report_file.exists():
        return json.loads(report_file.read_text(encoding="utf-8"))
        
    status_file = run_dir / "status.json"
    if status_file.exists():
        st = json.loads(status_file.read_text(encoding="utf-8"))
        return {
            "run_id": run_id,
            "status": st.get("status", "failed"),
            "purpose": st.get("purpose", ""),
            "duration_seconds": st.get("duration_seconds", 0),
            "errors": st.get("errors", []),
            "warnings": st.get("warnings", []),
            "models": {},
            "ensembles": {},
            "selection": {"selected_model": "Failed"},
            "final_test": {"final_test_executed": False}
        }
        
    raise FileNotFoundError(f"No report files found for run {run_id}")


def load_latest_results() -> Dict[str, Any]:
    run_id = get_latest_run_id()
    if not run_id:
        return {"run_id": None, "validation_models": [], "ensemble_results": [], "final_test_result": {}, "best_model": None}
    
    try:
        report = load_run_report(run_id)
        val_models = list(report.get("models", {}).values())
        ens_models = list(report.get("ensembles", {}).values())
        return {
            "run_id": run_id,
            "validation_models": val_models,
            "ensemble_results": ens_models,
            "final_test_result": report.get("final_test", {}),
            "best_validation_model_name": report.get("selection", {}).get("selected_model"),
            "run_metadata": {
                "run_id": run_id,
                "timestamp": report.get("created_at"),
                "status": report.get("status")
            }
        }
    except Exception:
        return {"run_id": run_id, "validation_models": [], "ensemble_results": [], "final_test_result": {}, "best_model": None}


def get_all_models_summary() -> List[Dict[str, Any]]:
    latest = load_latest_results()
    val_models = latest.get("validation_models", [])
    ens_models = latest.get("ensemble_results", [])
    
    models_dict = {}
    base_names = [
        ("logistic_regression", "Logistic Regression (L1 & L2)"),
        ("linear_svm", "Linear SVM"),
        ("rbf_svm", "RBF SVM"),
        ("random_forest", "Random Forest"),
        ("extra_trees", "Extra Trees"),
        ("xgboost", "XGBoost Classifier")
    ]
    for key, disp in base_names:
        models_dict[key] = {
            "model_name": key,
            "display_name": disp,
            "type": "individual",
            "environment": f"model_envs/{key}",
            "status": "not_run",
            "validation_metrics": None,
            "timing_seconds": None,
            "best_hyperparameters": None
        }
        
    for m in val_models:
        k = m["model_name"]
        models_dict[k] = {
            "model_name": k,
            "display_name": m.get("display_name", k),
            "type": "individual",
            "environment": f"model_envs/{k}",
            "status": "completed",
            "validation_metrics": m.get("validation_metrics"),
            "timing_seconds": m.get("timing_seconds"),
            "best_hyperparameters": m.get("best_hyperparameters")
        }

    for ens in ens_models:
        k = ens["model_name"]
        models_dict[k] = {
            "model_name": k,
            "display_name": ens.get("display_name", k),
            "type": "ensemble",
            "environment": "model_envs/ensemble",
            "status": "completed",
            "validation_metrics": ens.get("validation_metrics"),
            "base_models": ens.get("base_models", [])
        }
        
    return list(models_dict.values())


def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    comparison = {
        "runs": [],
        "metrics_diff": {},
        "selected_models": {}
    }
    
    for rid in run_ids:
        try:
            rep = load_run_report(rid)
            comparison["runs"].append(rep)
        except Exception:
            continue
            
    return comparison


def get_model_performance_history() -> Dict[str, List[Dict[str, Any]]]:
    """Extract performance history for each model across all runs."""
    runs_index = get_all_runs_index()
    history: Dict[str, List[Dict[str, Any]]] = {}
    
    for entry in reversed(runs_index):
        rid = entry["run_id"]
        created = entry.get("created_at", rid)
        try:
            rep = load_run_report(rid)
            models = rep.get("models", {})
            ensembles = rep.get("ensembles", {})
            all_m = {**models, **ensembles}
            
            for m_name, m_data in all_m.items():
                if m_name not in history:
                    history[m_name] = []
                vm = m_data.get("validation_metrics", {})
                history[m_name].append({
                    "run_id": rid,
                    "date": created,
                    "display_name": m_data.get("display_name", m_name),
                    "roc_auc": vm.get("roc_auc", 0.0),
                    "accuracy": vm.get("accuracy", 0.0),
                    "f1_score": vm.get("f1_score", 0.0),
                    "balanced_accuracy": vm.get("balanced_accuracy", 0.0)
                })
        except Exception:
            continue
            
    return history


def run_benchmark_subprocess(cmd: List[str], job_type: str, target: str):
    global CURRENT_JOB
    CURRENT_JOB = {
        "status": "running",
        "job_type": job_type,
        "target": target,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "log_output": [],
        "error": None
    }
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(WORKSPACE_ROOT)
        )
        
        for line in proc.stdout:
            CURRENT_JOB["log_output"].append(line.rstrip())
            
        proc.wait()
        if proc.returncode == 0:
            CURRENT_JOB["status"] = "completed"
        else:
            CURRENT_JOB["status"] = "failed"
            CURRENT_JOB["error"] = f"Subprocess exited with code {proc.returncode}"
    except Exception as exc:
        CURRENT_JOB["status"] = "failed"
        CURRENT_JOB["error"] = str(exc)


def run_single_model_async(model_name: str) -> bool:
    if CURRENT_JOB["status"] == "running":
        return False
        
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    env_python = BENCHMARK_ROOT / "model_envs" / model_name / "bin" / "python"
    train_script = BENCHMARK_ROOT / "scripts" / "train_model.py"
    
    cmd = [str(env_python), str(train_script), "--model", model_name, "--run-id", run_id]
    
    import threading
    thread = threading.Thread(target=run_benchmark_subprocess, args=(cmd, "single_model", model_name))
    thread.start()
    return True


def run_all_models_async() -> bool:
    if CURRENT_JOB["status"] == "running":
        return False
        
    main_python = WORKSPACE_ROOT / "NeuroADHD/.venv/bin/python"
    master_script = BENCHMARK_ROOT / "scripts" / "run_all_models.py"
    
    cmd = [str(main_python), str(master_script)]
    
    import threading
    thread = threading.Thread(target=run_benchmark_subprocess, args=(cmd, "run_all", "all_models"))
    thread.start()
    return True
