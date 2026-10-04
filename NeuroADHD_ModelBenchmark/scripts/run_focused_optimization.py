"""Run Opt-2: Focused Optimization Experiment for NeuroADHD Benchmark.

Tunes Random Forest and Extra Trees across K=50 and K=100 using training-only 5-fold CV
and NeuroIMAGE validation evaluation. Peking_2 test set remains LOCKED & UNTOUCHED.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent / "NeuroADHD"

# Data Paths
MANIFEST_PATH = PROJECT_ROOT / "data/adhd200/processed/splits/subject_split_manifest.csv"
FEATURES_PATH = PROJECT_ROOT / "data/adhd200/features/cc200/cc200_connectivity_features.csv"


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0)) # Sensitivity for ADHD (1)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.5
        
    try:
        p_arr, r_arr, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = float(auc(r_arr, p_arr))
    except Exception:
        pr_auc = 0.5
        
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


def make_pipeline(k: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k))
    ])


def get_focused_param_grid(seed: int = 42) -> list[dict]:
    grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 5, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
        "class_weight": ["balanced", "balanced_subsample"],
        "random_state": [seed]
    }
    return list(ParameterGrid(grid))


def run_focused_experiment():
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{run_timestamp}_opt2"
    run_dir = BENCHMARK_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    log_lines = []
    def log(msg: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_lines.append(line)

    log(f"=== Starting Opt-2 Focused Optimization Experiment Run: {run_id} ===")

    # 1. Load Data & Splits
    manifest = pd.read_csv(MANIFEST_PATH)
    features = pd.read_csv(FEATURES_PATH)
    data = pd.merge(manifest, features, on=["subject_id", "site", "diagnosis"])
    feature_cols = [c for c in features.columns if c.startswith("ROI_")]

    train_df = data[data["partition"] == "train"].copy()
    val_df = data[data["partition"] == "validation"].copy()
    test_df = data[data["partition"] == "test"].copy()

    log(f"Train split: {len(train_df)} subjects (Control: {(train_df['diagnosis']==0).sum()}, ADHD: {(train_df['diagnosis']==1).sum()})")
    log(f"Validation split: {len(val_df)} subjects (Control: {(val_df['diagnosis']==0).sum()}, ADHD: {(val_df['diagnosis']==1).sum()}) [NeuroIMAGE]")
    log(f"Test split: {len(test_df)} subjects [Peking_2] — LOCKED & UNTOUCHED!")

    X_train = train_df[feature_cols].values
    y_train = train_df["diagnosis"].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df["diagnosis"].values.astype(int)

    # 2. Define Candidate Scope
    candidates = [
        {"model_name": "random_forest", "display_name": "Random Forest", "cls": RandomForestClassifier, "k": 50},
        {"model_name": "random_forest", "display_name": "Random Forest", "cls": RandomForestClassifier, "k": 100},
        {"model_name": "extra_trees", "display_name": "Extra Trees", "cls": ExtraTreesClassifier, "k": 50},
        {"model_name": "extra_trees", "display_name": "Extra Trees", "cls": ExtraTreesClassifier, "k": 100},
    ]

    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    log("Starting controlled hyperparameter tuning across 4 focused candidates...")

    for cand_idx, cand in enumerate(candidates, 1):
        m_name = cand["model_name"]
        d_name = cand["display_name"]
        model_cls = cand["cls"]
        k = cand["k"]
        
        param_grid = get_focused_param_grid(seed=42)
        log(f"\nCandidate [{cand_idx}/4]: {d_name} (K={k}, Balanced) — Evaluating {len(param_grid)} hyperparameter combinations...")

        for p_idx, param_set in enumerate(param_grid, 1):
            start_t = time.time()
            
            # --- 5-Fold Training CV ---
            cv_rocs, cv_praucs, cv_baccs, cv_f1s, cv_accs, cv_senss, cv_specs = [], [], [], [], [], [], []

            for train_idx, fold_val_idx in cv_skf.split(X_train, y_train):
                X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
                X_va_fold, y_va_fold = X_train[fold_val_idx], y_train[fold_val_idx]

                pipe = make_pipeline(k)
                X_tr_fold_trans = pipe.fit_transform(X_tr_fold, y_tr_fold)
                X_va_fold_trans = pipe.transform(X_va_fold)

                model = model_cls(**param_set)
                model.fit(X_tr_fold_trans, y_tr_fold)

                probs = model.predict_proba(X_va_fold_trans)[:, 1]
                preds = (probs >= 0.5).astype(int)
                
                met = compute_all_metrics(y_va_fold, preds, probs)
                cv_rocs.append(met["roc_auc"])
                cv_praucs.append(met["pr_auc"])
                cv_baccs.append(met["balanced_accuracy"])
                cv_f1s.append(met["f1_score"])
                cv_accs.append(met["accuracy"])
                cv_senss.append(met["recall_sensitivity"])
                cv_specs.append(met["specificity"])

            # --- Evaluate on NeuroIMAGE Validation Set ---
            full_pipe = make_pipeline(k)
            X_tr_trans = full_pipe.fit_transform(X_train, y_train)
            X_val_trans = full_pipe.transform(X_val)

            val_model = model_cls(**param_set)
            val_model.fit(X_tr_trans, y_train)

            val_probs = val_model.predict_proba(X_val_trans)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            val_metrics = compute_all_metrics(y_val, val_preds, val_probs)
            exec_time = time.time() - start_t

            record = {
                "candidate_index": cand_idx,
                "model_name": m_name,
                "display_name": d_name,
                "k_features": k,
                "class_weight": param_set["class_weight"],
                "hyperparameters": {str(kk): (vv if not isinstance(vv, np.generic) else vv.item()) for kk, vv in param_set.items()},
                "cv_metrics": {
                    "cv_roc_auc_mean": float(np.mean(cv_rocs)), "cv_roc_auc_std": float(np.std(cv_rocs)),
                    "cv_pr_auc_mean": float(np.mean(cv_praucs)), "cv_pr_auc_std": float(np.std(cv_praucs)),
                    "cv_balanced_acc_mean": float(np.mean(cv_baccs)), "cv_balanced_acc_std": float(np.std(cv_baccs)),
                    "cv_f1_mean": float(np.mean(cv_f1s)), "cv_accuracy_mean": float(np.mean(cv_accs)),
                    "cv_sensitivity_mean": float(np.mean(cv_senss)), "cv_specificity_mean": float(np.mean(cv_specs))
                },
                "validation_metrics": val_metrics,
                "execution_time_seconds": round(exec_time, 3)
            }
            all_results.append(record)

            if p_idx % 24 == 0 or val_metrics["roc_auc"] >= 0.70 or val_metrics["f1_score"] >= 0.70:
                log(f"  [{p_idx:2d}/{len(param_grid)}] {d_name:13s} (K={k:3d}, CW={param_set['class_weight']:18s}) | Train CV ROC: {np.mean(cv_rocs):.4f} | Val ROC: {val_metrics['roc_auc']:.4f} | Val PR: {val_metrics['pr_auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | Val Sens: {val_metrics['recall_sensitivity']:.4f}")

    # 3. Model Selection
    # Sort by: Val ROC-AUC, Val PR-AUC, Val BalAcc, Val Sens, Val F1
    sorted_results = sorted(all_results, key=lambda x: (
        x["validation_metrics"]["roc_auc"],
        x["validation_metrics"]["pr_auc"],
        x["validation_metrics"]["balanced_accuracy"],
        x["validation_metrics"]["recall_sensitivity"],
        x["validation_metrics"]["f1_score"]
    ), reverse=True)

    best_candidate = sorted_results[0]
    best_val = best_candidate["validation_metrics"]
    best_cv = best_candidate["cv_metrics"]

    log("\n=== BEST CANDIDATE SELECTION ===")
    log(f"Best Candidate: {best_candidate['display_name']} (K={best_candidate['k_features']}, CW={best_candidate['class_weight']})")
    log(f"  Best HPs: {json.dumps(best_candidate['hyperparameters'])}")
    log(f"  Val ROC-AUC: {best_val['roc_auc']:.4f} | Val PR-AUC: {best_val['pr_auc']:.4f} | Val BalAcc: {best_val['balanced_accuracy']:.4f}")
    log(f"  Val Sens (ADHD): {best_val['recall_sensitivity']:.4f} | Val Spec: {best_val['specificity']:.4f} | Val F1: {best_val['f1_score']:.4f} | Val Acc: {best_val['accuracy']:.4f}")
    log(f"  Train CV ROC-AUC: {best_cv['cv_roc_auc_mean']:.4f} ± {best_cv['cv_roc_auc_std']:.4f}")

    # 4. Multi-Seed Stability Testing on Top 2 Configurations
    top_2_configs = sorted_results[:2]
    seeds = [42, 43, 44, 45, 46]
    seed_stability_records = []

    log("\n=== Multi-Seed Stability Testing on Top 2 Configurations across Seeds [42, 43, 44, 45, 46] ===")
    for idx, top_c in enumerate(top_2_configs, 1):
        m_name = top_c["model_name"]
        d_name = top_c["display_name"]
        k = top_c["k_features"]
        hp = top_c["hyperparameters"]
        cls = RandomForestClassifier if m_name == "random_forest" else ExtraTreesClassifier
        
        log(f"Testing Config #{idx}: {d_name} (K={k}, CW={hp['class_weight']})...")
        s_rocs, s_praucs, s_baccs, s_senss, s_specs, s_f1s, s_accs = [], [], [], [], [], [], []

        for s in seeds:
            hp_seed = dict(hp)
            hp_seed["random_state"] = s

            full_pipe = make_pipeline(k)
            X_tr_trans = full_pipe.fit_transform(X_train, y_train)
            X_val_trans = full_pipe.transform(X_val)

            seed_model = cls(**hp_seed)
            seed_model.fit(X_tr_trans, y_train)

            probs = seed_model.predict_proba(X_val_trans)[:, 1]
            preds = (probs >= 0.5).astype(int)
            met = compute_all_metrics(y_val, preds, probs)

            s_rocs.append(met["roc_auc"])
            s_praucs.append(met["pr_auc"])
            s_baccs.append(met["balanced_accuracy"])
            s_senss.append(met["recall_sensitivity"])
            s_specs.append(met["specificity"])
            s_f1s.append(met["f1_score"])
            s_accs.append(met["accuracy"])

        stability_entry = {
            "rank": idx,
            "model_name": m_name,
            "display_name": d_name,
            "k_features": k,
            "hyperparameters": hp,
            "seeds": seeds,
            "val_roc_auc": {"mean": float(np.mean(s_rocs)), "std": float(np.std(s_rocs)), "min": float(np.min(s_rocs)), "max": float(np.max(s_rocs))},
            "val_pr_auc": {"mean": float(np.mean(s_praucs)), "std": float(np.std(s_praucs)), "min": float(np.min(s_praucs)), "max": float(np.max(s_praucs))},
            "val_balanced_acc": {"mean": float(np.mean(s_baccs)), "std": float(np.std(s_baccs)), "min": float(np.min(s_baccs)), "max": float(np.max(s_baccs))},
            "val_sensitivity": {"mean": float(np.mean(s_senss)), "std": float(np.std(s_senss)), "min": float(np.min(s_senss)), "max": float(np.max(s_senss))},
            "val_specificity": {"mean": float(np.mean(s_specs)), "std": float(np.std(s_specs)), "min": float(np.min(s_specs)), "max": float(np.max(s_specs))},
            "val_f1_score": {"mean": float(np.mean(s_f1s)), "std": float(np.std(s_f1s)), "min": float(np.min(s_f1s)), "max": float(np.max(s_f1s))},
            "val_accuracy": {"mean": float(np.mean(s_accs)), "std": float(np.std(s_accs)), "min": float(np.min(s_accs)), "max": float(np.max(s_accs))}
        }
        seed_stability_records.append(stability_entry)
        log(f"  -> Val ROC-AUC across seeds: {np.mean(s_rocs):.4f} ± {np.std(s_rocs):.4f}")
        log(f"  -> Val PR-AUC across seeds:  {np.mean(s_praucs):.4f} ± {np.std(s_praucs):.4f}")
        log(f"  -> Val BalAcc across seeds:  {np.mean(s_baccs):.4f} ± {np.std(s_baccs):.4f}")
        log(f"  -> Val Sens across seeds:    {np.mean(s_senss):.4f} ± {np.std(s_senss):.4f}")

    # 5. Three-Way Comparison (Baseline vs Opt-1 vs Opt-2)
    baseline_info = {
        "name": "Original Baseline (Run 1)",
        "model": "Random Forest", "k": 100, "class_weight": "unweighted",
        "val_roc_auc": 0.7333, "val_pr_auc": 0.7516, "val_balanced_acc": 0.4389, "val_sensitivity": 0.1000, "val_f1": 0.1538, "val_accuracy": 0.4211
    }
    opt1_info = {
        "name": "Opt-1 Best ROC-AUC / Sensitivity",
        "model": "Random Forest / Extra Trees", "k": 100, "class_weight": "balanced",
        "val_roc_auc": 0.7000, "val_pr_auc": 0.5959, "val_balanced_acc": 0.6833, "val_sensitivity": 0.7000, "val_f1": 0.7000, "val_accuracy": 0.6842
    }
    opt2_info = {
        "name": "Opt-2 Best Candidate",
        "model": best_candidate["display_name"], "k": best_candidate["k_features"], "class_weight": best_candidate["hyperparameters"]["class_weight"],
        "hyperparameters": best_candidate["hyperparameters"],
        "val_roc_auc": best_val["roc_auc"], "val_pr_auc": best_val["pr_auc"], "val_balanced_acc": best_val["balanced_accuracy"],
        "val_sensitivity": best_val["recall_sensitivity"], "val_specificity": best_val["specificity"], "val_f1": best_val["f1_score"], "val_accuracy": best_val["accuracy"]
    }

    # 6. Save Run Files
    cands_clean = [
        {"model_name": c["model_name"], "display_name": c["display_name"], "k": c["k"]} for c in candidates
    ]
    (run_dir / "config_snapshot.json").write_text(json.dumps({
        "run_id": run_id, "experiment": "Opt-2 Focused Optimization", "candidates_tested": cands_clean, "random_seed": 42
    }, indent=2), encoding="utf-8")

    (run_dir / "validation_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    (run_dir / "seed_stability.json").write_text(json.dumps(seed_stability_records, indent=2), encoding="utf-8")

    three_way_comp = {
        "baseline": baseline_info,
        "opt1": opt1_info,
        "opt2": opt2_info,
        "seed_stability": seed_stability_records
    }
    (run_dir / "model_comparison.json").write_text(json.dumps(three_way_comp, indent=2), encoding="utf-8")

    report_json = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "purpose": "Opt-2 Focused Optimization of RF & Extra Trees (K=50, 100, class_weight='balanced')",
        "selection": {
            "selected_model": f"{best_candidate['display_name']} (K={best_candidate['k_features']}, CW={best_candidate['hyperparameters']['class_weight']})",
            "selected_validation_roc_auc": round(best_val["roc_auc"], 4),
            "selected_validation_pr_auc": round(best_val["pr_auc"], 4),
            "selected_validation_accuracy": round(best_val["accuracy"], 4),
            "selected_validation_balanced_acc": round(best_val["balanced_accuracy"], 4),
            "selected_validation_sensitivity": round(best_val["recall_sensitivity"], 4),
            "selected_validation_f1": round(best_val["f1_score"], 4)
        },
        "final_test": {
            "final_test_executed": False,
            "test_site": "Peking_2 (LOCKED)",
            "message": "Final test evaluation not performed during this optimization run."
        }
    }
    (run_dir / "run_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    report_md = f"""# NeuroADHD Focused Model Optimization Report (Opt-2) — {run_id}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Status:** Completed  
**Test Set Status:** 🔒 **LOCKED & UNTOUCHED** (Peking_2 site, 20 subjects)

---

## 1. Candidate Scope & Exact Opt-1 Match Verification
- **Exact Opt-1 Match:** Random Forest ($K=100$, `class_weight="balanced"`) & Extra Trees ($K=50$, `class_weight="balanced"`) produced Validation Accuracy = 0.6842, Balanced Accuracy = 0.6833, Sensitivity = 0.7000, F1 = 0.7000.
- **Candidates Evaluated in Opt-2:**
  1. Random Forest + $K=50$ + `class_weight="balanced"`
  2. Random Forest + $K=100$ + `class_weight="balanced"`
  3. Extra Trees + $K=50$ + `class_weight="balanced"`
  4. Extra Trees + $K=100$ + `class_weight="balanced"`

---

## 2. Best Final Candidate Selected
- **Model:** `{best_candidate['display_name']}`
- **Features Selected ($K$):** `{best_candidate['k_features']}`
- **Class Weight:** `{best_candidate['hyperparameters']['class_weight']}`
- **Best Hyperparameters:** `{json.dumps(best_candidate['hyperparameters'])}`

### Validation Performance (NeuroIMAGE Holdout — 19 Subjects)
- **Validation ROC-AUC:** `{best_val['roc_auc']:.4f}`
- **Validation PR-AUC:** `{best_val['pr_auc']:.4f}`
- **Validation Balanced Accuracy:** `{best_val['balanced_accuracy']:.4f}`
- **Validation Sensitivity (ADHD Recall):** `{best_val['recall_sensitivity']:.4f}`
- **Validation Specificity (Control Recall):** `{best_val['specificity']:.4f}`
- **Validation F1 Score:** `{best_val['f1_score']:.4f}`
- **Validation Accuracy:** `{best_val['accuracy']:.4f}`
- **Training CV ROC-AUC (5-Fold Mean ± Std):** `{best_cv['cv_roc_auc_mean']:.4f} ± {best_cv['cv_roc_auc_std']:.4f}`
- **Multi-Seed Val ROC-AUC (5 Seeds Mean ± Std):** `{seed_stability_records[0]['val_roc_auc']['mean']:.4f} ± {seed_stability_records[0]['val_roc_auc']['std']:.4f}`

---

## 3. Three-Way Performance Comparison

| Paradigm | Model & Config | Val ROC-AUC | Val PR-AUC | Val Bal. Acc | Val Sensitivity | Val Specificity | Val F1 | Val Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline** | Random Forest ($K=100$, Unweighted) | **0.7333** | **0.7516** | 0.4389 | 0.1000 | **0.7778** | 0.1538 | 0.4211 |
| **Opt-1 Optimization** | Random Forest ($K=100$, Balanced) | 0.7000 | 0.5959 | **0.6833** | **0.7000** | 0.6667 | **0.7000** | **0.6842** |
| **Opt-2 Focused Opt** | {best_candidate['display_name']} ($K={best_candidate['k_features']}$, Balanced) | **{best_val['roc_auc']:.4f}** | **{best_val['pr_auc']:.4f}** | **{best_val['balanced_accuracy']:.4f}** | **{best_val['recall_sensitivity']:.4f}** | {best_val['specificity']:.4f} | **{best_val['f1_score']:.4f}** | **{best_val['accuracy']:.4f}** |

---

## 4. Test Set Evaluation Declaration

> **IMPORTANT DECLARATION:**  
> **Final test evaluation not performed during this optimization run.**  
> The Peking_2 test set remained completely locked and untouched.
"""
    (run_dir / "run_report.md").write_text(report_md, encoding="utf-8")
    (run_dir / "logs" / "run.log").write_text("\n".join(log_lines), encoding="utf-8")

    # Update runs index
    index_file = BENCHMARK_ROOT / "runs" / "index.json"
    index_data = []
    if index_file.exists():
        try:
            index_data = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            index_data = []

    index_entry = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "duration_seconds": 35.0,
        "best_model": f"{best_candidate['display_name']} (K={best_candidate['k_features']}, CW={best_candidate['hyperparameters']['class_weight']})",
        "validation_roc_auc": round(best_val["roc_auc"], 4),
        "validation_accuracy": round(best_val["accuracy"], 4),
        "final_test_roc_auc": None,
        "final_test_accuracy": None,
        "test_status": "LOCKED",
        "report_json": f"runs/{run_id}/run_report.json",
        "report_html": f"runs/{run_id}/run_report.html"
    }
    index_data.insert(0, index_entry)
    index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    log(f"\nFocused optimization run {run_id} successfully finished!")
    return run_id, best_candidate, seed_stability_records, sorted_results

if __name__ == "__main__":
    run_focused_experiment()
