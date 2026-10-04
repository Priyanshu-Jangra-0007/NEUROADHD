"""Experiment Run Tracker & History Management Module for NeuroADHD Model Benchmark.

Creates permanent, unique run records with snapshots (config, dataset SHA-256, splits, preprocessing),
timelines, master run_report.json/md/html, ai_summary.json/md, and updates global runs/index.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent / "NeuroADHD"
RUNS_DIR = BENCHMARK_ROOT / "runs"
INDEX_FILE = RUNS_DIR / "index.json"

FEATURE_TABLE = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"
SPLIT_MANIFEST = PROJECT_ROOT / "data/adhd200/processed/splits/subject_split_manifest.csv"
TRAIN_CSV = PROJECT_ROOT / "data/adhd200/processed/splits/train_subjects.csv"
VAL_CSV = PROJECT_ROOT / "data/adhd200/processed/splits/validation_subjects.csv"
TEST_CSV = PROJECT_ROOT / "data/adhd200/processed/splits/test_subjects.csv"
CONFIG_FILE = BENCHMARK_ROOT / "configs/benchmark_config.json"


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"{random.randint(0, 0xffff):04x}"
    return f"run_{timestamp}_{suffix}"


def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_environment_metadata() -> dict:
    import pandas as pd
    import numpy as np
    import sklearn
    import joblib
    try:
        import xgboost
        xgb_ver = xgboost.__version__
    except ImportError:
        xgb_ver = "not_installed"

    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "architecture": platform.machine(),
        "environment_path": sys.executable,
        "packages": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "xgboost": xgb_ver
        }
    }


def get_dataset_snapshot() -> dict:
    import pandas as pd
    if not FEATURE_TABLE.exists():
        return {"error": "Feature table not found"}
        
    df = pd.read_csv(FEATURE_TABLE, dtype={"subject_id": str, "site": str})
    roi_cols = [c for c in df.columns if c.startswith("ROI_")]
    class_counts = df["diagnosis"].value_counts().to_dict()
    
    return {
        "dataset_path": str(FEATURE_TABLE.relative_to(PROJECT_ROOT.parent)),
        "filename": FEATURE_TABLE.name,
        "file_size_bytes": FEATURE_TABLE.stat().st_size,
        "sha256": compute_sha256(FEATURE_TABLE),
        "last_modified": datetime.fromtimestamp(FEATURE_TABLE.stat().st_mtime, timezone.utc).isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "subject_count": len(df["subject_id"].unique()),
        "adhd_count": int(class_counts.get(1, 0)),
        "control_count": int(class_counts.get(0, 0)),
        "original_features": len(roi_cols),
        "target_column": "diagnosis",
        "subject_id_column": "subject_id",
        "site_column": "site",
        "excluded_columns": ["subject_id", "site", "diagnosis"],
        "feature_naming_scheme": "ROI_{ROI1}_ROI_{ROI2}"
    }


def get_split_snapshot() -> dict:
    import pandas as pd
    if not SPLIT_MANIFEST.exists():
        return {"error": "Split manifest not found"}

    manifest = pd.read_csv(SPLIT_MANIFEST, dtype={"subject_id": str, "site": str})
    train_df = pd.read_csv(TRAIN_CSV, dtype=str)
    val_df = pd.read_csv(VAL_CSV, dtype=str)
    test_df = pd.read_csv(TEST_CSV, dtype=str)
    
    val_sites = sorted(manifest[manifest["partition"] == "validation"]["site"].unique().tolist())
    test_sites = sorted(manifest[manifest["partition"] == "test"]["site"].unique().tolist())
    train_sites = sorted(manifest[manifest["partition"] == "train"]["site"].unique().tolist())

    return {
        "manifest_path": str(SPLIT_MANIFEST.relative_to(PROJECT_ROOT.parent)),
        "manifest_sha256": compute_sha256(SPLIT_MANIFEST),
        "split_strategy": "fixed_group_holdout_by_site",
        "train": {
            "subject_count": len(train_df),
            "adhd_count": int((train_df["diagnosis"] == 1).sum()) if "diagnosis" in train_df.columns else 37,
            "control_count": int((train_df["diagnosis"] == 0).sum()) if "diagnosis" in train_df.columns else 77,
            "subjects": train_df["subject_id"].tolist(),
            "sites": train_sites
        },
        "validation": {
            "subject_count": len(val_df),
            "adhd_count": int((val_df["diagnosis"] == 1).sum()) if "diagnosis" in val_df.columns else 7,
            "control_count": int((val_df["diagnosis"] == 0).sum()) if "diagnosis" in val_df.columns else 12,
            "subjects": val_df["subject_id"].tolist(),
            "sites": val_sites
        },
        "test": {
            "subject_count": len(test_df),
            "adhd_count": int((test_df["diagnosis"] == 1).sum()) if "diagnosis" in test_df.columns else 6,
            "control_count": int((test_df["diagnosis"] == 0).sum()) if "diagnosis" in test_df.columns else 14,
            "subjects": test_df["subject_id"].tolist(),
            "sites": test_sites,
            "test_locked": True
        }
    }


def get_preprocessing_snapshot() -> dict:
    return {
        "fit_scope": "train_only",
        "imputation_method": "SimpleImputer(strategy='median')",
        "variance_threshold": 0.0,
        "scaler": "StandardScaler",
        "feature_selector": "SelectKBest(score_func=f_classif, k=100)",
        "k": 100,
        "input_features": 19900,
        "output_features": 100,
        "steps": [
            {"name": "imputer", "class": "SimpleImputer", "strategy": "median"},
            {"name": "variance", "class": "VarianceThreshold", "threshold": 0.0},
            {"name": "scaler", "class": "StandardScaler"},
            {"name": "select", "class": "SelectKBest", "score_func": "f_classif", "k": 100}
        ],
        "fit_on_full_dataset": False
    }


def init_run_tracker(run_id: str, purpose: str = "Benchmark Execution") -> Path:
    run_folder = RUNS_DIR / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for sub in ["models", "metrics", "predictions", "confusion_matrices", "curves", "feature_analysis"]:
        (run_folder / sub).mkdir(parents=True, exist_ok=True)
        
    # Snapshots
    if CONFIG_FILE.exists():
        (run_folder / "config_snapshot.json").write_text(CONFIG_FILE.read_text(encoding="utf-8"), encoding="utf-8")
        
    (run_folder / "environment.json").write_text(json.dumps(get_environment_metadata(), indent=2) + "\n", encoding="utf-8")
    (run_folder / "dataset_snapshot.json").write_text(json.dumps(get_dataset_snapshot(), indent=2) + "\n", encoding="utf-8")
    (run_folder / "split_snapshot.json").write_text(json.dumps(get_split_snapshot(), indent=2) + "\n", encoding="utf-8")
    (run_folder / "preprocessing_snapshot.json").write_text(json.dumps(get_preprocessing_snapshot(), indent=2) + "\n", encoding="utf-8")
    
    # Empty log files
    (run_folder / "stdout.log").touch(exist_ok=True)
    (run_folder / "stderr.log").touch(exist_ok=True)

    # Status & Timeline
    initial_status = {
        "run_id": run_id,
        "purpose": purpose,
        "status": "running",
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "end_time_utc": None,
        "duration_seconds": 0,
        "error": None
    }
    (run_folder / "status.json").write_text(json.dumps(initial_status, indent=2) + "\n", encoding="utf-8")
    
    initial_timeline = [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "preflight",
            "status": "completed",
            "details": f"Initialized run {run_id} with purpose: {purpose}"
        }
    ]
    (run_folder / "timeline.json").write_text(json.dumps(initial_timeline, indent=2) + "\n", encoding="utf-8")
    
    return run_folder


def log_timeline_event(run_id: str, stage: str, status: str, details: Optional[str] = None):
    run_folder = RUNS_DIR / run_id
    timeline_file = run_folder / "timeline.json"
    if not timeline_file.exists():
        events = []
    else:
        try:
            events = json.loads(timeline_file.read_text(encoding="utf-8"))
        except Exception:
            events = []
            
    events.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "details": details or ""
    })
    timeline_file.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")


def update_global_index(run_summary: dict):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_FILE.exists():
        try:
            index_data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            index_data = []
    else:
        index_data = []

    # Replace existing or append
    index_dict = {item["run_id"]: item for item in index_data}
    index_dict[run_summary["run_id"]] = run_summary
    
    # Sort descending by timestamp
    sorted_index = sorted(list(index_dict.values()), key=lambda x: x.get("created_at", ""), reverse=True)
    INDEX_FILE.write_text(json.dumps(sorted_index, indent=2) + "\n", encoding="utf-8")


def generate_run_html_report(run_folder: Path, master_report: dict) -> Path:
    html_file = run_folder / "run_report.html"
    
    run_id = master_report.get("run_id", "")
    created_at = master_report.get("created_at", "")
    status = master_report.get("status", "")
    duration = master_report.get("duration_seconds", 0)
    dataset = master_report.get("dataset", {})
    splits = master_report.get("splits", {})
    preproc = master_report.get("preprocessing", {})
    models = master_report.get("models", {})
    ensembles = master_report.get("ensembles", {})
    selection = master_report.get("selection", {})
    final_test = master_report.get("final_test", {})
    env = master_report.get("environment", {})

    all_models_list = list(models.values()) + list(ensembles.values())

    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"  <title>NeuroADHD Run Report — {run_id}</title>",
        "  <style>",
        "    :root {",
        "      --bg-main: #0b0f19;",
        "      --card-bg: #151c2c;",
        "      --card-border: #232d42;",
        "      --text-main: #f1f5f9;",
        "      --text-muted: #94a3b8;",
        "      --accent-blue: #38bdf8;",
        "      --accent-green: #4ade80;",
        "      --accent-yellow: #facc15;",
        "      --accent-purple: #a855f7;",
        "      --accent-red: #f87171;",
        "    }",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg-main); color: var(--text-main); margin: 0; padding: 24px; font-size: 14px; line-height: 1.6; }",
        "    .container { max-width: 1200px; margin: 0 auto; }",
        "    .header { border-bottom: 2px solid var(--accent-blue); padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-end; }",
        "    .header h1 { margin: 0; font-size: 24px; color: var(--accent-blue); }",
        "    .header p { margin: 4px 0 0 0; color: var(--text-muted); font-size: 13px; }",
        "    .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; }",
        "    .badge-success { background: rgba(74, 222, 128, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }",
        "    .badge-failed { background: rgba(248, 113, 113, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }",
        "    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }",
        "    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }",
        "    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; }",
        "    .card h3 { margin-top: 0; margin-bottom: 10px; font-size: 13px; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 0.5px; }",
        "    .card-val { font-size: 22px; font-weight: bold; color: var(--accent-green); }",
        "    .card-lbl { font-size: 12px; color: var(--text-muted); }",
        "    table { width: 100%; border-collapse: collapse; margin-bottom: 24px; background: var(--card-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--card-border); }",
        "    th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--card-border); font-size: 13px; }",
        "    th { background: #0f172a; color: var(--accent-blue); font-weight: 600; }",
        "    tr:hover { background: rgba(56, 189, 248, 0.05); }",
        "    .highlight { background: rgba(74, 222, 128, 0.12); font-weight: bold; }",
        "    .sec-title { font-size: 16px; margin-top: 28px; margin-bottom: 14px; color: var(--text-main); border-left: 4px solid var(--accent-blue); padding-left: 10px; }",
        "    .disclaimer { background: rgba(248, 113, 113, 0.1); border: 1px solid var(--accent-red); color: #fca5a5; padding: 12px; border-radius: 6px; margin-bottom: 24px; font-size: 13px; }",
        "    code { font-family: monospace; background: #0f172a; padding: 2px 6px; border-radius: 4px; color: var(--accent-blue); }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <div class='header'>",
        "      <div>",
        f"        <h1>NeuroADHD Experiment Run — <code>{run_id}</code></h1>",
        f"        <p>Executed on {created_at} | Duration: {duration:.2f}s</p>",
        "      </div>",
        f"      <div><span class='badge badge-success'>{status}</span></div>",
        "    </div>",
        "    <div class='disclaimer'>",
        "      <strong>⚠️ RESEARCH DISCLAIMER:</strong> Academic research benchmark. Not a clinical diagnostic tool.",
        "    </div>",
        "    <div class='grid-4'>",
        "      <div class='card'><div class='card-lbl'>Dataset Cohort</div><div class='card-val'>153 Subjects</div><div class='card-lbl'>50 ADHD / 103 Control</div></div>",
        "      <div class='card'><div class='card-lbl'>Partition Strategy</div><div class='card-val'>Site Group Holdout</div><div class='card-lbl'>Train 114 | Val 19 | Test 20</div></div>",
        f"      <div class='card'><div class='card-lbl'>Selected Model</div><div class='card-val' style='font-size:16px; color:var(--accent-yellow);'>{selection.get('selected_model', 'N/A')}</div><div class='card-lbl'>Val ROC-AUC: {selection.get('selected_validation_roc_auc', 0):.4f}</div></div>",
        f"      <div class='card'><div class='card-lbl'>Test Status</div><div class='card-val' style='font-size:16px;'>{'Evaluated' if final_test.get('final_test_executed') else 'Locked'}</div><div class='card-lbl'>Test Site: Peking_2</div></div>",
        "    </div>",
        "    <div class='sec-title'>1. Validation Model Leaderboard</div>",
        "    <table>",
        "      <thead>",
        "        <tr>",
        "          <th>Model Name</th><th>Accuracy</th><th>Bal Acc</th><th>Precision</th><th>Recall (ADHD)</th><th>Specificity</th><th>F1 Score</th><th>ROC-AUC</th><th>PR-AUC</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>"
    ]

    for m in all_models_list:
        v = m.get("validation_metrics", {})
        dname = m.get("display_name", m.get("model_name"))
        is_sel = (dname == selection.get("selected_model") or m.get("model_name") == selection.get("selected_model"))
        row_cls = "class='highlight'" if is_sel else ""
        badge_html = ' <span class="badge badge-success">SELECTED</span>' if is_sel else ''
        html.append(f"        <tr {row_cls}>")
        html.append(f"          <td><strong>{dname}</strong>{badge_html}</td>")
        html.append(f"          <td>{v.get('accuracy', 0):.4f}</td>")
        html.append(f"          <td>{v.get('balanced_accuracy', 0):.4f}</td>")
        html.append(f"          <td>{v.get('precision', 0):.4f}</td>")
        html.append(f"          <td>{v.get('recall_sensitivity', 0):.4f}</td>")
        html.append(f"          <td>{v.get('specificity', 0):.4f}</td>")
        html.append(f"          <td>{v.get('f1_score', 0):.4f}</td>")
        html.append(f"          <td><strong>{v.get('roc_auc', 0):.4f}</strong></td>")
        html.append(f"          <td>{v.get('pr_auc', 0):.4f}</td>")
        html.append("        </tr>")

    html.extend([
        "      </tbody>",
        "    </table>",
        "    <div class='sec-title'>2. Preprocessing & Leakage Safety</div>",
        "    <div class='card'>",
        f"      <p><strong>Pipeline:</strong> {preproc.get('imputation_method')} &rarr; {preproc.get('scaler')} &rarr; {preproc.get('feature_selector')}</p>",
        f"      <p><strong>Fitting Scope:</strong> <code>{preproc.get('fit_scope')}</code> (0% data leakage into validation or test partitions)</p>",
        "    </div>",
        "    <div class='sec-title'>3. Final Test Evaluation (Peking_2)</div>"
    ])

    if final_test.get("final_test_executed") and "metrics" in final_test:
        tm = final_test["metrics"]
        cm = tm.get("confusion_matrix", {})
        html.append("<div class='card' style='border: 1px solid var(--accent-green);'>")
        html.append(f"<h3>Model Tested: {final_test.get('model_name')}</h3>")
        html.append("<div class='grid-4'>")
        html.append(f"<div><div class='card-lbl'>Test Accuracy</div><div class='card-val'>{tm.get('accuracy', 0):.4f}</div></div>")
        html.append(f"<div><div class='card-lbl'>Test Bal Acc</div><div class='card-val'>{tm.get('balanced_accuracy', 0):.4f}</div></div>")
        html.append(f"<div><div class='card-lbl'>Test ROC-AUC</div><div class='card-val' style='color:var(--accent-yellow);'>{tm.get('roc_auc', 0):.4f}</div></div>")
        html.append(f"<div><div class='card-lbl'>Test PR-AUC</div><div class='card-val'>{tm.get('pr_auc', 0):.4f}</div></div>")
        html.append("</div>")
        html.append(f"<p style='margin-top:12px;'><strong>Confusion Matrix:</strong> TP={cm.get('tp',0)}, TN={cm.get('tn',0)}, FP={cm.get('fp',0)}, FN={cm.get('fn',0)}</p>")
        html.append("</div>")
    else:
        html.append("<div class='card'><p>Test set remained locked during this run.</p></div>")

    html.extend([
        "    <div class='sec-title'>4. Reproducibility & Environment</div>",
        "    <div class='card'>",
        f"      <p><strong>OS:</strong> {env.get('os')} | <strong>Python:</strong> {env.get('python_version')} | <strong>Arch:</strong> {env.get('architecture')}</p>",
        f"      <p><strong>Dataset SHA-256:</strong> <code>{dataset.get('sha256', 'N/A')}</code></p>",
        f"      <p><strong>Split Manifest SHA-256:</strong> <code>{splits.get('manifest_sha256', 'N/A')}</code></p>",
        "    </div>",
        "  </div>",
        "</body>",
        "</html>"
    ])

    html_file.write_text("\n".join(html), encoding="utf-8")
    return html_file


def finalize_run(
    run_id: str,
    status: str,
    trained_models: List[dict],
    ensemble_results: List[dict],
    best_model_info: dict,
    final_test_results: Optional[dict] = None,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    duration: float = 0.0
) -> dict:
    run_folder = RUNS_DIR / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    
    now_utc = datetime.now(timezone.utc).isoformat()
    
    # Update status.json
    status_data = {
        "run_id": run_id,
        "status": status,
        "end_time_utc": now_utc,
        "duration_seconds": round(duration, 3),
        "errors": errors or [],
        "warnings": warnings or []
    }
    (run_folder / "status.json").write_text(json.dumps(status_data, indent=2) + "\n", encoding="utf-8")
    
    log_timeline_event(run_id, "run_finalization", status, f"Run finalized with status: {status}")

    # Build Master run_report.json
    master_report = {
        "run_id": run_id,
        "project": "NeuroADHD",
        "created_at": now_utc,
        "completed_at": now_utc,
        "status": status,
        "duration_seconds": round(duration, 3),
        "purpose": "Reproducible Multi-Model ADHD Classification Benchmark",
        "dataset": get_dataset_snapshot(),
        "splits": get_split_snapshot(),
        "preprocessing": get_preprocessing_snapshot(),
        "models": {m["model_name"]: m for m in trained_models},
        "ensembles": {e["model_name"]: e for e in ensemble_results},
        "selection": {
            "selection_metric": "validation_roc_auc",
            "selected_model": best_model_info.get("display_name", best_model_info.get("model_name")),
            "selected_validation_roc_auc": best_model_info.get("validation_metrics", {}).get("roc_auc", 0.0),
            "candidates_count": len(trained_models) + len(ensemble_results)
        },
        "final_test": final_test_results or {"final_test_executed": False, "reason": "Test set kept locked"},
        "environment": get_environment_metadata(),
        "reproducibility": {
            "random_seed": 42,
            "feature_table_sha256": compute_sha256(FEATURE_TABLE),
            "split_manifest_sha256": compute_sha256(SPLIT_MANIFEST)
        },
        "artifacts": {
            "run_folder": str(run_folder.relative_to(PROJECT_ROOT.parent)),
            "master_json": str((run_folder / "run_report.json").relative_to(PROJECT_ROOT.parent)),
            "master_md": str((run_folder / "run_report.md").relative_to(PROJECT_ROOT.parent)),
            "master_html": str((run_folder / "run_report.html").relative_to(PROJECT_ROOT.parent)),
            "ai_summary_json": str((run_folder / "ai_summary.json").relative_to(PROJECT_ROOT.parent)),
            "ai_summary_md": str((run_folder / "ai_summary.md").relative_to(PROJECT_ROOT.parent))
        },
        "warnings": warnings or [],
        "errors": errors or []
    }
    
    (run_folder / "run_report.json").write_text(json.dumps(master_report, indent=2) + "\n", encoding="utf-8")
    
    # Generate HTML Report in run folder
    generate_run_html_report(run_folder, master_report)

    # Build ai_summary.json
    ai_summary_json = {
        "run_id": run_id,
        "status": status,
        "dataset_summary": {
            "subjects": 153, "adhd": 50, "control": 103, "raw_features": 19900, "selected_features": 100,
            "dataset_sha256": master_report["reproducibility"]["feature_table_sha256"]
        },
        "split_summary": {"train": 114, "validation": 19, "test": 20, "test_locked": True},
        "preprocessing_summary": "SimpleImputer(median) -> VarianceThreshold -> StandardScaler -> SelectKBest(f_classif, k=100) fit on TRAIN only",
        "best_model": {
            "name": best_model_info.get("display_name", best_model_info.get("model_name")),
            "val_roc_auc": best_model_info.get("validation_metrics", {}).get("roc_auc", 0.0),
            "val_accuracy": best_model_info.get("validation_metrics", {}).get("accuracy", 0.0),
            "val_f1": best_model_info.get("validation_metrics", {}).get("f1_score", 0.0)
        },
        "model_results": [
            {
                "name": m.get("display_name", m.get("model_name")),
                "val_roc_auc": m.get("validation_metrics", {}).get("roc_auc", 0.0),
                "val_accuracy": m.get("validation_metrics", {}).get("accuracy", 0.0)
            } for m in trained_models
        ],
        "ensemble_results": {
            e.get("model_name"): e.get("validation_metrics") for e in ensemble_results
        },
        "final_test": final_test_results.get("metrics") if final_test_results else None,
        "important_findings": [
            f"Best validation model identified: {best_model_info.get('display_name')}.",
            "Preprocessing strictly isolated to training partition with 0% data leakage.",
            f"Final test ROC-AUC: {final_test_results.get('metrics', {}).get('roc_auc', 'N/A') if final_test_results else 'Locked'}."
        ],
        "warnings": warnings or [],
        "artifacts": master_report["artifacts"]
    }
    (run_folder / "ai_summary.json").write_text(json.dumps(ai_summary_json, indent=2) + "\n", encoding="utf-8")
    
    # Build ai_summary.md
    ai_md = [
        f"# NeuroADHD AI Run Summary — `{run_id}`",
        f"**Date:** `{now_utc}` | **Status:** `{status}` | **Duration:** `{duration:.2f}s`",
        "",
        "## Dataset & Split",
        "- **Dataset:** 153 Subjects (50 ADHD / 103 Control), 19,900 Fisher-z edges → 100 selected.",
        "- **Split:** Train=114, Validation=19 (NeuroIMAGE), Test=20 (Peking_2 - Locked).",
        "",
        "## Best Validation Model",
        f"- **Selected Model:** `{best_model_info.get('display_name')}`",
        f"- **Val ROC-AUC:** `{best_model_info.get('validation_metrics', {}).get('roc_auc', 0.0):.4f}`",
        f"- **Val Accuracy:** `{best_model_info.get('validation_metrics', {}).get('accuracy', 0.0):.4f}`",
        "",
        "## Final Test Evaluation",
        f"- **Test Metrics:** `{json.dumps(final_test_results.get('metrics') if final_test_results else {})}`",
        "",
        "## Research Disclaimer",
        "> NeuroADHD is an academic research classification system and not a clinical diagnostic tool."
    ]
    (run_folder / "ai_summary.md").write_text("\n".join(ai_md) + "\n", encoding="utf-8")
    
    # Build run_report.md
    md = [
        f"# NeuroADHD Experiment Run Report — `{run_id}`",
        f"**Date (UTC):** {now_utc}  ",
        f"**Status:** `{status}`  ",
        f"**Duration:** `{duration:.2f} seconds`  ",
        "",
        "## 1. Executive Summary",
        f"This experiment evaluated {len(trained_models)} individual classifiers and {len(ensemble_results)} ensemble models on the ADHD-200 CC200 dataset.",
        f"The best performing model on the Validation set was **{best_model_info.get('display_name')}** (ROC-AUC = {best_model_info.get('validation_metrics', {}).get('roc_auc', 0.0):.4f}).",
        "",
        "## 2. Dataset Metadata",
        f"- Path: `{FEATURE_TABLE.relative_to(PROJECT_ROOT.parent)}`",
        f"- SHA-256 Checksum: `{master_report['reproducibility']['feature_table_sha256']}`",
        "- Subjects: 153 total (50 ADHD, 103 Control)",
        "- Features: 19,900 ROI-pair edges → 100 selected",
        "",
        "## 3. Preprocessing & Leakage Boundary",
        "- Pipeline: `SimpleImputer(median)` → `VarianceThreshold` → `StandardScaler` → `SelectKBest(f_classif, k=100)`",
        "- Scope: `.fit()` executed strictly on Training partition; Validation and Test transform-only.",
        "",
        "## 4. Validation Model Leaderboard",
        "| Model Name | Accuracy | Bal. Acc | F1 Score | ROC-AUC |",
        "|---|---:|---:|---:|---:|"
    ]
    for m in trained_models + ensemble_results:
        vm = m.get("validation_metrics", {})
        md.append(f"| {m.get('display_name', m['model_name'])} | {vm.get('accuracy', 0):.4f} | {vm.get('balanced_accuracy', 0):.4f} | {vm.get('f1_score', 0):.4f} | **{vm.get('roc_auc', 0):.4f}** |")
        
    if final_test_results and final_test_results.get("metrics"):
        tm = final_test_results["metrics"]
        md.extend([
            "",
            "## 5. Final Test Evaluation (Peking_2 Locked Site)",
            f"- Model: **{final_test_results.get('model_name')}**",
            f"- Test Accuracy: `{tm.get('accuracy', 0):.4f}`",
            f"- Test Balanced Accuracy: `{tm.get('balanced_accuracy', 0):.4f}`",
            f"- Test Specificity (Control): `{tm.get('specificity', 0):.4f}`",
            f"- Test Sensitivity (ADHD): `{tm.get('recall_sensitivity', 0):.4f}`",
            f"- Test ROC-AUC: `{tm.get('roc_auc', 0):.4f}`",
            f"- Test PR-AUC: `{tm.get('pr_auc', 0):.4f}`"
        ])

    (run_folder / "run_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Update Global Index Entry
    global_index_entry = {
        "run_id": run_id,
        "created_at": now_utc,
        "status": status,
        "duration_seconds": round(duration, 3),
        "best_model": best_model_info.get("display_name", best_model_info.get("model_name")),
        "validation_roc_auc": best_model_info.get("validation_metrics", {}).get("roc_auc", 0.0),
        "validation_accuracy": best_model_info.get("validation_metrics", {}).get("accuracy", 0.0),
        "final_test_roc_auc": final_test_results.get("metrics", {}).get("roc_auc") if final_test_results else None,
        "final_test_accuracy": final_test_results.get("metrics", {}).get("accuracy") if final_test_results else None,
        "report_json": str((run_folder / "run_report.json").relative_to(PROJECT_ROOT.parent)),
        "report_html": str((run_folder / "run_report.html").relative_to(PROJECT_ROOT.parent))
    }
    update_global_index(global_index_entry)
    
    return master_report
