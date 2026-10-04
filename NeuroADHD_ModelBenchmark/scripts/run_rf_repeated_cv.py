#!/usr/bin/env python3
"""
Repeated Cross-Validation Experiment for Final Balanced Random Forest (3 Additional Repetitions + Combined 8-Rep Analysis)
- Dataset: 133 Development Subjects (93 Control, 40 ADHD)
- Peking_2 ($N=20$) is STRICTLY LOCKED & UNTOUCHED.
- Model: Balanced Random Forest (K=100, n_estimators=50, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", class_weight="balanced")
- Evaluates seeds 47, 48, 49 (new) and combines with 42, 43, 44, 45, 46 (total 8 repetitions).
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent / "NeuroADHD"
SPLITS_DIR = PROJECT_ROOT / "data/adhd200/processed/splits"
FEATURES_PATH = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"
MANIFEST_PATH = SPLITS_DIR / "subject_split_manifest.csv"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import precision_recall_curve, auc
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(recall_curve, precision_curve))
    except Exception:
        pr_auc = float(np.mean(y_true))
        
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        }
    }


def make_pipeline(k: int = 100) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k))
    ])


def instantiate_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=50, max_depth=3, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", class_weight="balanced", random_state=seed
    )


def compute_stats(records, metric_keys):
    stats = {}
    for k in metric_keys:
        vals = [r[k] for r in records]
        stats[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        }
    return stats


def run_experiment():
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{run_timestamp}_3more_rf_repeated"
    run_dir = BENCHMARK_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    log_lines = []
    def log(msg: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_lines.append(line)

    log(f"=== Starting Balanced Random Forest Repeated-CV Robustness Experiment: {run_id} ===")

    # 1. Load Data & Filter to 133 Development Subjects
    manifest = pd.read_csv(MANIFEST_PATH)
    features = pd.read_csv(FEATURES_PATH)
    data = pd.merge(manifest, features, on=["subject_id", "site", "diagnosis"])
    feature_cols = [c for c in features.columns if c.startswith("ROI_")]

    dev_df = data[data["partition"].isin(["train", "validation"])].copy().reset_index(drop=True)
    test_df = data[data["partition"] == "test"].copy().reset_index(drop=True)

    log(f"Development subjects for Repeated CV: {len(dev_df)} (Control: {(dev_df['diagnosis']==0).sum()}, ADHD: {(dev_df['diagnosis']==1).sum()}) across {dev_df['site'].nunique()} sites")
    log(f"Held-out Peking_2 Test set: {len(test_df)} subjects — STRICTLY LOCKED & UNTOUCHED!")

    X_dev = dev_df[feature_cols].values
    y_dev = dev_df["diagnosis"].values.astype(int)

    # 2. All 8 Seeds
    all_seeds = [42, 43, 44, 45, 46, 47, 48, 49]
    existing_seeds = [42, 43, 44, 45, 46]
    new_seeds = [47, 48, 49]

    all_repetition_records = []

    log("\n--- Executing 5-Fold Stratified CV across all 8 Repetitions ---")

    for rep_idx, seed in enumerate(all_seeds, 1):
        tag = "NEW" if seed in new_seeds else "EXISTING"
        log(f"\n[Repetition {rep_idx}/8 ({tag})] — Seed {seed}: Initializing stratified 5-fold CV...")
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        oof_rf_probs = np.zeros(len(y_dev))

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_dev, y_dev), 1):
            X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
            X_te, y_te = X_dev[test_idx], y_dev[test_idx]

            # Fit leakage-safe preprocessing pipeline (K=100) ONLY on training fold
            pipe = make_pipeline(k=100)
            X_tr_trans = pipe.fit_transform(X_tr, y_tr)
            X_te_trans = pipe.transform(X_te)

            # Fit Random Forest on training fold
            rf = instantiate_rf(seed=seed)
            rf.fit(X_tr_trans, y_tr)

            rf_fold_probs = rf.predict_proba(X_te_trans)[:, 1]
            oof_rf_probs[test_idx] = rf_fold_probs

        oof_rf_preds = (oof_rf_probs >= 0.5).astype(int)
        m = calculate_metrics(y_dev, oof_rf_preds, oof_rf_probs)
        
        rep_entry = {
            "repetition": rep_idx,
            "seed": seed,
            "is_new": bool(seed in new_seeds),
            **m
        }
        all_repetition_records.append(rep_entry)

        log(f"  RF (Seed {seed}): ROC-AUC={m['roc_auc']:.4f} | BalAcc={m['balanced_accuracy']:.4f} | Sens={m['recall_sensitivity']:.4f} | Spec={m['specificity']:.4f} | F1={m['f1_score']:.4f} | Acc={m['accuracy']:.4f}")

    # 3. Calculate Summary Statistics
    metric_keys = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "recall_sensitivity", "specificity", "f1_score"]

    existing_recs = [r for r in all_repetition_records if r["seed"] in existing_seeds]
    new_recs = [r for r in all_repetition_records if r["seed"] in new_seeds]

    stats_existing = compute_stats(existing_recs, metric_keys)
    stats_new = compute_stats(new_recs, metric_keys)
    stats_all = compute_stats(all_repetition_records, metric_keys)

    summary_stats = {
        "new_repetitions_seeds_47_48_49": stats_new,
        "existing_repetitions_seeds_42_46": stats_existing,
        "all_8_repetitions": stats_all
    }

    # 4. Compare with Hybrids & Historical Single Model Reference
    hybrid_benchmarks = {
        "Soft Voting (0.6 RF / 0.4 XGB)": {
            "roc_auc": 0.5926, "roc_auc_std": 0.0229,
            "balanced_accuracy": 0.5798, "balanced_accuracy_std": 0.0226,
            "recall_sensitivity": 0.4500, "recall_sensitivity_std": 0.0354,
            "f1_score": 0.4234, "f1_score_std": 0.0299,
            "accuracy": 0.6316, "accuracy_std": 0.0196
        },
        "Hard Voting (RF + XGBoost)": {
            "roc_auc": 0.5926, "roc_auc_std": 0.0229,
            "balanced_accuracy": 0.5798, "balanced_accuracy_std": 0.0226,
            "recall_sensitivity": 0.4500, "recall_sensitivity_std": 0.0354,
            "f1_score": 0.4234, "f1_score_std": 0.0299,
            "accuracy": 0.6316, "accuracy_std": 0.0196
        },
        "Stacking (Balanced LogReg)": {
            "roc_auc": 0.5555, "roc_auc_std": 0.0280,
            "balanced_accuracy": 0.5428, "balanced_accuracy_std": 0.0191,
            "recall_sensitivity": 0.5050, "recall_sensitivity_std": 0.0485,
            "f1_score": 0.4065, "f1_score_std": 0.0263,
            "accuracy": 0.5579, "accuracy_std": 0.0129
        }
    }

    historical_single_ref = {
        "model": "Balanced Random Forest (NeuroIMAGE Single Val)",
        "roc_auc": 0.7000,
        "balanced_accuracy": 0.6833,
        "recall_sensitivity": 0.7000,
        "f1_score": 0.7000,
        "accuracy": 0.6842
    }

    comparison_data = []
    # Standalone RF 8-rep row
    comparison_data.append({
        "model": "Balanced Random Forest (8-Rep CV Mean)",
        "protocol": "5-Fold Repeated CV (8 Seeds)",
        "roc_auc": f"{stats_all['roc_auc']['mean']:.4f} ± {stats_all['roc_auc']['std']:.4f}",
        "balanced_accuracy": f"{stats_all['balanced_accuracy']['mean']:.4f} ± {stats_all['balanced_accuracy']['std']:.4f}",
        "sensitivity": f"{stats_all['recall_sensitivity']['mean']:.4f} ± {stats_all['recall_sensitivity']['std']:.4f}",
        "specificity": f"{stats_all['specificity']['mean']:.4f} ± {stats_all['specificity']['std']:.4f}",
        "f1_score": f"{stats_all['f1_score']['mean']:.4f} ± {stats_all['f1_score']['std']:.4f}",
        "accuracy": f"{stats_all['accuracy']['mean']:.4f} ± {stats_all['accuracy']['std']:.4f}"
    })
    # Standalone RF 3 new reps row
    comparison_data.append({
        "model": "Balanced Random Forest (New 3 Reps Mean: 47,48,49)",
        "protocol": "5-Fold Repeated CV (3 Seeds)",
        "roc_auc": f"{stats_new['roc_auc']['mean']:.4f} ± {stats_new['roc_auc']['std']:.4f}",
        "balanced_accuracy": f"{stats_new['balanced_accuracy']['mean']:.4f} ± {stats_new['balanced_accuracy']['std']:.4f}",
        "sensitivity": f"{stats_new['recall_sensitivity']['mean']:.4f} ± {stats_new['recall_sensitivity']['std']:.4f}",
        "specificity": f"{stats_new['specificity']['mean']:.4f} ± {stats_new['specificity']['std']:.4f}",
        "f1_score": f"{stats_new['f1_score']['mean']:.4f} ± {stats_new['f1_score']['std']:.4f}",
        "accuracy": f"{stats_new['accuracy']['mean']:.4f} ± {stats_new['accuracy']['std']:.4f}"
    })
    # Hybrids
    for h_name, h_m in hybrid_benchmarks.items():
        comparison_data.append({
            "model": h_name,
            "protocol": "5-Fold Repeated CV (5 Seeds)",
            "roc_auc": f"{h_m['roc_auc']:.4f} ± {h_m['roc_auc_std']:.4f}",
            "balanced_accuracy": f"{h_m['balanced_accuracy']:.4f} ± {h_m['balanced_accuracy_std']:.4f}",
            "sensitivity": f"{h_m['recall_sensitivity']:.4f} ± {h_m['recall_sensitivity_std']:.4f}",
            "specificity": "—",
            "f1_score": f"{h_m['f1_score']:.4f} ± {h_m['f1_score_std']:.4f}",
            "accuracy": f"{h_m['accuracy']:.4f} ± {h_m['accuracy_std']:.4f}"
        })
    # Historical
    comparison_data.append({
        "model": "Balanced Random Forest (Historical Ref)",
        "protocol": "Single Partition (NeuroIMAGE Val)",
        "roc_auc": "0.7000",
        "balanced_accuracy": "0.6833",
        "sensitivity": "0.7000",
        "specificity": "0.6667",
        "f1_score": "0.7000",
        "accuracy": "0.6842"
    })

    # 5. Generate Distribution Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0b0f19")
    for ax in axes.ravel():
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1f2937")

    metrics_for_plot = [
        ("roc_auc", "ROC-AUC Distribution across 8 Repetitions", axes[0, 0]),
        ("balanced_accuracy", "Balanced Accuracy Distribution across 8 Repetitions", axes[0, 1]),
        ("recall_sensitivity", "Sensitivity (ADHD Recall) Distribution", axes[1, 0]),
        ("f1_score", "F1 Score Distribution", axes[1, 1])
    ]

    for m_key, title, ax in metrics_for_plot:
        data_to_plot = [r[m_key] for r in all_repetition_records]
        bplot = ax.boxplot([data_to_plot], patch_artist=True, medianprops=dict(color="#facc15", linewidth=2.5))
        ax.set_xticks([1])
        ax.set_xticklabels(["Balanced RF (8 Reps)"], color="#e2e8f0", fontsize=10, fontweight="bold")
        for patch in bplot["boxes"]:
            patch.set_facecolor("#38bdf8")
            patch.set_alpha(0.7)
            patch.set_edgecolor("#38bdf8")
        # Scatter the individual seed points
        x_pts = np.random.normal(1, 0.04, size=len(data_to_plot))
        colors = ["#4ade80" if r["is_new"] else "#94a3b8" for r in all_repetition_records]
        ax.scatter(x_pts, data_to_plot, c=colors, s=50, zorder=3, edgecolors="#0f172a", linewidth=1)
        ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.3, color="#334155")

    plt.tight_layout()
    plots_path = run_dir / "plots" / "rf_repeated_distributions.png"
    plt.savefig(plots_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    log(f"Plots saved to: {plots_path}")

    # 6. Save JSON & CSV Artifacts
    # CSV Table of RF repetitions
    csv_rows = []
    for r in all_repetition_records:
        csv_rows.append({
            "repetition": r["repetition"],
            "seed": r["seed"],
            "group": "New (47-49)" if r["is_new"] else "Existing (42-46)",
            "accuracy": r["accuracy"],
            "balanced_accuracy": r["balanced_accuracy"],
            "precision": r["precision"],
            "recall_sensitivity": r["recall_sensitivity"],
            "specificity": r["specificity"],
            "f1_score": r["f1_score"],
            "roc_auc": r["roc_auc"],
            "pr_auc": r["pr_auc"],
            "tp": r["confusion_matrix"]["tp"],
            "tn": r["confusion_matrix"]["tn"],
            "fp": r["confusion_matrix"]["fp"],
            "fn": r["confusion_matrix"]["fn"]
        })
    df_reps = pd.DataFrame(csv_rows)
    csv_path = run_dir / "rf_repeated_results.csv"
    df_reps.to_csv(csv_path, index=False)
    log(f"Repetition CSV saved to: {csv_path}")

    # JSON of repetitions
    with open(run_dir / "rf_repeated_results.json", "w") as f:
        json.dump(all_repetition_records, f, indent=2)

    # Summary Statistics JSON
    with open(run_dir / "rf_summary_statistics.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Comparison CSV & JSON
    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(run_dir / "rf_vs_hybrid_comparison.csv", index=False)
    with open(run_dir / "rf_vs_hybrid_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # 7. Generate Markdown & HTML Reports
    md_content = f"""# Balanced Random Forest Repeated Cross-Validation Robustness Report

**Run ID:** `{run_id}`  
**Experiment Type:** 3 Additional Repetitions (Seeds: 47, 48, 49) + Full 8-Repetition Analysis (Seeds: 42–49)  
**Dataset:** 133 Development Subjects (93 Control, 40 ADHD across 7 sites)  
**Peking_2 Test Status:** **STRICTLY LOCKED & UNTOUCHED (N=20)**  

---

## 1. Executive Summary

### A. New 3 Repetitions (Seeds: 47, 48, 49)
- **Mean ROC-AUC:** `{stats_new['roc_auc']['mean']:.4f} ± {stats_new['roc_auc']['std']:.4f}` (Min: `{stats_new['roc_auc']['min']:.4f}`, Max: `{stats_new['roc_auc']['max']:.4f}`)
- **Mean Balanced Accuracy:** `{stats_new['balanced_accuracy']['mean']:.4f} ± {stats_new['balanced_accuracy']['std']:.4f}` (Min: `{stats_new['balanced_accuracy']['min']:.4f}`, Max: `{stats_new['balanced_accuracy']['max']:.4f}`)
- **Mean Sensitivity (ADHD Recall):** `{stats_new['recall_sensitivity']['mean']:.4f} ± {stats_new['recall_sensitivity']['std']:.4f}` (Min: `{stats_new['recall_sensitivity']['min']:.4f}`, Max: `{stats_new['recall_sensitivity']['max']:.4f}`)
- **Mean F1 Score:** `{stats_new['f1_score']['mean']:.4f} ± {stats_new['f1_score']['std']:.4f}` (Min: `{stats_new['f1_score']['min']:.4f}`, Max: `{stats_new['f1_score']['max']:.4f}`)
- **Mean Accuracy:** `{stats_new['accuracy']['mean']:.4f} ± {stats_new['accuracy']['std']:.4f}`

### B. Full 8 Repetitions (Seeds: 42, 43, 44, 45, 46, 47, 48, 49)
- **Mean ROC-AUC ± SD:** `{stats_all['roc_auc']['mean']:.4f} ± {stats_all['roc_auc']['std']:.4f}` (Min: `{stats_all['roc_auc']['min']:.4f}`, Max: `{stats_all['roc_auc']['max']:.4f}`)
- **Mean PR-AUC ± SD:** `{stats_all['pr_auc']['mean']:.4f} ± {stats_all['pr_auc']['std']:.4f}`
- **Mean Accuracy ± SD:** `{stats_all['accuracy']['mean']:.4f} ± {stats_all['accuracy']['std']:.4f}`
- **Mean Balanced Accuracy ± SD:** `{stats_all['balanced_accuracy']['mean']:.4f} ± {stats_all['balanced_accuracy']['std']:.4f}` (Min: `{stats_all['balanced_accuracy']['min']:.4f}`, Max: `{stats_all['balanced_accuracy']['max']:.4f}`)
- **Mean Sensitivity ± SD:** `{stats_all['recall_sensitivity']['mean']:.4f} ± {stats_all['recall_sensitivity']['std']:.4f}`
- **Mean Specificity ± SD:** `{stats_all['specificity']['mean']:.4f} ± {stats_all['specificity']['std']:.4f}`
- **Mean F1 ± SD:** `{stats_all['f1_score']['mean']:.4f} ± {stats_all['f1_score']['std']:.4f}`

---

## 2. Comparison under Identical Repeated-CV Protocol

| Model / Architecture | Protocol | ROC-AUC (Mean ± SD) | Balanced Accuracy (Mean ± SD) | Sensitivity (Mean ± SD) | F1 Score (Mean ± SD) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Balanced Random Forest** | **8-Rep Repeated CV** | **{stats_all['roc_auc']['mean']:.4f} ± {stats_all['roc_auc']['std']:.4f}** | **{stats_all['balanced_accuracy']['mean']:.4f} ± {stats_all['balanced_accuracy']['std']:.4f}** | **{stats_all['recall_sensitivity']['mean']:.4f} ± {stats_all['recall_sensitivity']['std']:.4f}** | **{stats_all['f1_score']['mean']:.4f} ± {stats_all['f1_score']['std']:.4f}** |
| **Balanced Random Forest** | **New 3-Rep CV (47-49)** | **{stats_new['roc_auc']['mean']:.4f} ± {stats_new['roc_auc']['std']:.4f}** | **{stats_new['balanced_accuracy']['mean']:.4f} ± {stats_new['balanced_accuracy']['std']:.4f}** | **{stats_new['recall_sensitivity']['mean']:.4f} ± {stats_new['recall_sensitivity']['std']:.4f}** | **{stats_new['f1_score']['mean']:.4f} ± {stats_new['f1_score']['std']:.4f}** |
| Soft Voting (RF + XGB) | 5-Rep Repeated CV | 0.5926 ± 0.0229 | 0.5798 ± 0.0226 | 0.4500 ± 0.0354 | 0.4234 ± 0.0299 |
| Hard Voting (RF + XGB) | 5-Rep Repeated CV | 0.5926 ± 0.0229 | 0.5798 ± 0.0226 | 0.4500 ± 0.0354 | 0.4234 ± 0.0299 |
| Stacking (Balanced LogReg) | 5-Rep Repeated CV | 0.5555 ± 0.0280 | 0.5428 ± 0.0191 | 0.5050 ± 0.0485 | 0.4065 ± 0.0263 |

---

## 3. Critical Protocol Distinction

- **Historical Single Validation (NeuroIMAGE split, N=19):** ROC-AUC = 0.7000, BalAcc = 0.6833, Sensitivity = 0.7000, F1 = 0.7000.
- **Repeated 5-Fold Cross-Validation (Full 133 subjects, 8 independent splits):** ROC-AUC = {stats_all['roc_auc']['mean']:.4f} ± {stats_all['roc_auc']['std']:.4f}, BalAcc = {stats_all['balanced_accuracy']['mean']:.4f} ± {stats_all['balanced_accuracy']['std']:.4f}.
- The single partition score reflects a favorable holdout site, whereas the 8-repetition CV provides an unbiased generalization estimate across all sites.
"""
    with open(run_dir / "run_report.md", "w") as f:
        f.write(md_content)
    with open(run_dir / "ai_summary.md", "w") as f:
        f.write(md_content)

    ai_summary_json = {
        "run_id": run_id,
        "new_repetitions_stats": stats_new,
        "all_8_repetitions_stats": stats_all,
        "hybrid_comparison": comparison_data
    }
    with open(run_dir / "ai_summary.json", "w") as f:
        json.dump(ai_summary_json, f, indent=2)

    # HTML Report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NeuroADHD — Balanced RF Repeated-CV Robustness Report</title>
  <style>
    :root {{ --bg: #0b0f19; --card-bg: #111827; --card-border: #1f2937; --text: #f9fafb; --text-muted: #9ca3af; --cyan: #38bdf8; --green: #4ade80; --amber: #facc15; --rose: #fb7185; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 32px 20px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; background: #064e3b; color: var(--green); }}
    .badge-locked {{ background: #451a03; color: #fbbf24; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; margin: 16px 0; overflow: hidden; }}
    th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--card-border); font-size: 0.9rem; }}
    th {{ background: #1a2234; color: var(--cyan); }}
  </style>
</head>
<body>
  <div class="container">
    <span class="badge">RF 8-Repetition CV Complete</span>
    <span class="badge badge-locked">Peking_2 Locked</span>
    <h1 style="margin: 12px 0 6px 0;">Balanced Random Forest — Repeated-CV Robustness (8 Repetitions)</h1>
    <p style="color: var(--text-muted); margin-bottom: 24px;">Run ID: <code>{run_id}</code> | Seeds: 42, 43, 44, 45, 46, 47, 48, 49</p>

    <h2 style="color: var(--cyan); margin-top: 20px;">Overall 8-Repetition CV Performance</h2>
    <table>
      <thead><tr><th>Metric</th><th>Mean &plusmn; Std</th><th>Min</th><th>Max</th></tr></thead>
      <tbody>
        <tr><td><strong>ROC-AUC</strong></td><td>{stats_all['roc_auc']['mean']:.4f} &plusmn; {stats_all['roc_auc']['std']:.4f}</td><td>{stats_all['roc_auc']['min']:.4f}</td><td>{stats_all['roc_auc']['max']:.4f}</td></tr>
        <tr><td><strong>Balanced Accuracy</strong></td><td>{stats_all['balanced_accuracy']['mean']:.4f} &plusmn; {stats_all['balanced_accuracy']['std']:.4f}</td><td>{stats_all['balanced_accuracy']['min']:.4f}</td><td>{stats_all['balanced_accuracy']['max']:.4f}</td></tr>
        <tr><td><strong>Sensitivity (ADHD Recall)</strong></td><td>{stats_all['recall_sensitivity']['mean']:.4f} &plusmn; {stats_all['recall_sensitivity']['std']:.4f}</td><td>{stats_all['recall_sensitivity']['min']:.4f}</td><td>{stats_all['recall_sensitivity']['max']:.4f}</td></tr>
        <tr><td><strong>Specificity</strong></td><td>{stats_all['specificity']['mean']:.4f} &plusmn; {stats_all['specificity']['std']:.4f}</td><td>{stats_all['specificity']['min']:.4f}</td><td>{stats_all['specificity']['max']:.4f}</td></tr>
        <tr><td><strong>F1 Score</strong></td><td>{stats_all['f1_score']['mean']:.4f} &plusmn; {stats_all['f1_score']['std']:.4f}</td><td>{stats_all['f1_score']['min']:.4f}</td><td>{stats_all['f1_score']['max']:.4f}</td></tr>
      </tbody>
    </table>

    <div style="text-align: center; margin: 24px 0;">
      <img src="plots/rf_repeated_distributions.png" alt="RF Distributions" style="max-width: 100%; border-radius: 8px;">
    </div>
  </div>
</body>
</html>
"""
    with open(run_dir / "run_report.html", "w") as f:
        f.write(html_content)

    # 8. Save Logs
    with open(run_dir / "logs" / "execution.log", "w") as f:
        f.write("\n".join(log_lines))

    # 9. Update runs/index.json
    index_path = BENCHMARK_ROOT / "runs" / "index.json"
    index_data = []
    if index_path.exists():
        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)
        except Exception:
            index_data = []
    
    new_entry = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "best_model": "Balanced Random Forest (8-Rep CV)",
        "validation_roc_auc": round(float(stats_all["roc_auc"]["mean"]), 4),
        "validation_accuracy": round(float(stats_all["accuracy"]["mean"]), 4),
        "final_test_roc_auc": None,
        "final_test_accuracy": None,
        "test_status": "LOCKED",
        "report_json": f"runs/{run_id}/ai_summary.json",
        "report_html": f"runs/{run_id}/run_report.html"
    }
    index_data.insert(0, new_entry)
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    log(f"\nExperiment {run_id} complete! All artifacts saved.")
    return run_id, stats_new, stats_all


if __name__ == "__main__":
    run_experiment()
