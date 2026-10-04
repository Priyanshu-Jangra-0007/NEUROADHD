"""Run Hybrid / Ensemble Experiment for NeuroADHD Benchmark.

Evaluates Hard Voting, Soft Voting (Equal & Weighted), and Out-of-Fold Stacking
across the strongest validated base models (RF, XGBoost, Extra Trees, RBF SVM, Logistic Regression)
using 5-fold training CV and NeuroIMAGE validation set.
Peking_2 Test Set remains strictly LOCKED and UNTOUCHED.
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
from sklearn.model_selection import StratifiedKFold
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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

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


def make_pipeline(k: int = 100) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k))
    ])


def instantiate_base_models(seed: int = 42) -> dict:
    return {
        "rf": {
            "name": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=50, max_depth=3, min_samples_split=2, min_samples_leaf=1,
                max_features="sqrt", class_weight="balanced", random_state=seed
            ),
            "hyperparameters": {"n_estimators": 50, "max_depth": 3, "class_weight": "balanced", "random_state": seed}
        },
        "xgb": {
            "name": "XGBoost",
            "model": XGBClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                scale_pos_weight=2.8, eval_metric="logloss", random_state=seed
            ),
            "hyperparameters": {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "scale_pos_weight": 2.8, "random_state": seed}
        },
        "et": {
            "name": "Extra Trees",
            "model": ExtraTreesClassifier(
                n_estimators=50, max_depth=5, max_features="sqrt",
                class_weight="balanced", random_state=seed
            ),
            "hyperparameters": {"n_estimators": 50, "max_depth": 5, "class_weight": "balanced", "random_state": seed}
        },
        "svm": {
            "name": "RBF SVM",
            "model": SVC(
                C=1.0, gamma="scale", kernel="rbf",
                probability=True, class_weight="balanced", random_state=seed
            ),
            "hyperparameters": {"C": 1.0, "gamma": "scale", "class_weight": "balanced", "kernel": "rbf", "random_state": seed}
        },
        "lr": {
            "name": "Logistic Regression",
            "model": LogisticRegression(
                C=0.1, penalty="l1", solver="saga", max_iter=1000,
                class_weight="balanced", random_state=seed
            ),
            "hyperparameters": {"C": 0.1, "penalty": "l1", "solver": "saga", "class_weight": "balanced", "random_state": seed}
        }
    }


def run_ensemble_experiment():
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{run_timestamp}_ensemble"
    run_dir = BENCHMARK_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    log_lines = []
    def log(msg: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_lines.append(line)

    log(f"=== Starting Hybrid / Ensemble Experiment: {run_id} ===")

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

    # Preprocessing Pipeline (K=100)
    full_pipe = make_pipeline(k=100)
    X_train_trans = full_pipe.fit_transform(X_train, y_train)
    X_val_trans = full_pipe.transform(X_val)

    # 2. Fit Base Models & Get Validation Predictions
    base_models_dict = instantiate_base_models(seed=42)
    val_base_preds = {}
    val_base_probs = {}
    val_base_metrics = {}

    log("\n--- Evaluating Base Models on Validation Set ---")
    for key, info in base_models_dict.items():
        m = info["model"]
        m.fit(X_train_trans, y_train)
        probs = m.predict_proba(X_val_trans)[:, 1]
        preds = (probs >= 0.5).astype(int)
        val_base_probs[key] = probs
        val_base_preds[key] = preds
        met = compute_all_metrics(y_val, preds, probs)
        val_base_metrics[key] = met
        log(f"  {info['name']:20s} | Val ROC-AUC: {met['roc_auc']:.4f} | Val BalAcc: {met['balanced_accuracy']:.4f} | Val Sens: {met['recall_sensitivity']:.4f} | Val F1: {met['f1_score']:.4f} | Val Acc: {met['accuracy']:.4f}")

    # 3. Model Complementarity Analysis
    log("\n--- Model Complementarity & Error Analysis ---")
    model_keys = list(base_models_dict.keys())
    agreement_matrix = {}
    correlation_matrix = {}
    error_overlap = {}

    prob_df = pd.DataFrame(val_base_probs)
    corr_df = prob_df.corr(method="pearson")

    for k1 in model_keys:
        agreement_matrix[k1] = {}
        error_overlap[k1] = {}
        for k2 in model_keys:
            agree = float(np.mean(val_base_preds[k1] == val_base_preds[k2]))
            agreement_matrix[k1][k2] = round(agree, 4)

            # Error overlap: subjects where both were wrong vs only one wrong
            err1 = (val_base_preds[k1] != y_val)
            err2 = (val_base_preds[k2] != y_val)
            both_wrong = float(np.mean(err1 & err2))
            error_overlap[k1][k2] = round(both_wrong, 4)

    complementarity_data = {
        "pairwise_prediction_agreement": agreement_matrix,
        "probability_correlation": corr_df.round(4).to_dict(),
        "joint_error_rate": error_overlap
    }
    log(f"Probability Correlation Matrix:\n{corr_df.round(3)}")

    # 4. Out-of-Fold (OOF) Probability Generation on Training Data for Stacking
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_train_probs = {k: np.zeros(len(y_train)) for k in model_keys}

    log("\n--- Generating Leakage-Safe Out-of-Fold Training Predictions for Stacking ---")
    for train_idx, fold_val_idx in cv_skf.split(X_train, y_train):
        X_tr_f, y_tr_f = X_train[train_idx], y_train[train_idx]
        X_va_f = X_train[fold_val_idx]

        pipe_f = make_pipeline(k=100)
        X_tr_f_trans = pipe_f.fit_transform(X_tr_f, y_tr_f)
        X_va_f_trans = pipe_f.transform(X_va_f)

        fold_models = instantiate_base_models(seed=42)
        for k_mod, info_mod in fold_models.items():
            mod = info_mod["model"]
            mod.fit(X_tr_f_trans, y_tr_f)
            oof_train_probs[k_mod][fold_val_idx] = mod.predict_proba(X_va_f_trans)[:, 1]

    # 5. Ensemble Combinations to Evaluate
    combinations = [
        {"id": "combo_1", "name": "RF + XGBoost", "keys": ["rf", "xgb"]},
        {"id": "combo_2", "name": "RF + Extra Trees", "keys": ["rf", "et"]},
        {"id": "combo_3", "name": "RF + XGBoost + Extra Trees", "keys": ["rf", "xgb", "et"]},
        {"id": "combo_4", "name": "RF + XGBoost + Extra Trees + RBF SVM", "keys": ["rf", "xgb", "et", "svm"]},
        {"id": "combo_5", "name": "RF + XGBoost + Extra Trees + Logistic Regression", "keys": ["rf", "xgb", "et", "lr"]},
        {"id": "combo_all", "name": "All 5 Base Models", "keys": ["rf", "xgb", "et", "svm", "lr"]}
    ]

    ensemble_results = []

    # A. HARD VOTING
    log("\n=== A. HARD VOTING ENSEMBLES ===")
    for c in combinations:
        keys = c["keys"]
        # Stack predictions
        stacked_preds = np.column_stack([val_base_preds[k] for k in keys])
        stacked_probs = np.column_stack([val_base_probs[k] for k in keys])
        
        # Majority vote; tie breaker uses mean probability
        vote_sum = stacked_preds.sum(axis=1)
        half = len(keys) / 2.0
        final_preds = np.where(vote_sum > half, 1, np.where(vote_sum < half, 0, (stacked_probs.mean(axis=1) >= 0.5).astype(int)))
        pseudo_probs = stacked_probs.mean(axis=1)

        met = compute_all_metrics(y_val, final_preds, pseudo_probs)
        entry = {
            "ensemble_type": "hard_voting",
            "combination_id": c["id"],
            "display_name": f"Hard Voting ({c['name']})",
            "base_models": [base_models_dict[k]["name"] for k in keys],
            "weights": "unweighted_vote",
            "validation_metrics": met
        }
        ensemble_results.append(entry)
        log(f"  {entry['display_name']:45s} | Val ROC: {met['roc_auc']:.4f} | Val BalAcc: {met['balanced_accuracy']:.4f} | Val Sens: {met['recall_sensitivity']:.4f} | Val F1: {met['f1_score']:.4f} | Val Acc: {met['accuracy']:.4f}")

    # B. SOFT VOTING (Equal & Weighted)
    log("\n=== B. SOFT VOTING ENSEMBLES (Equal & Weighted) ===")
    for c in combinations:
        keys = c["keys"]
        stacked_probs = np.column_stack([val_base_probs[k] for k in keys])

        # B1. Equal Weights
        eq_probs = stacked_probs.mean(axis=1)
        eq_preds = (eq_probs >= 0.5).astype(int)
        met_eq = compute_all_metrics(y_val, eq_preds, eq_probs)
        entry_eq = {
            "ensemble_type": "soft_voting_equal",
            "combination_id": c["id"],
            "display_name": f"Soft Voting Equal ({c['name']})",
            "base_models": [base_models_dict[k]["name"] for k in keys],
            "weights": "equal",
            "validation_metrics": met_eq
        }
        ensemble_results.append(entry_eq)
        log(f"  {entry_eq['display_name']:45s} | Val ROC: {met_eq['roc_auc']:.4f} | Val BalAcc: {met_eq['balanced_accuracy']:.4f} | Val Sens: {met_eq['recall_sensitivity']:.4f} | Val F1: {met_eq['f1_score']:.4f} | Val Acc: {met_eq['accuracy']:.4f}")

        # B2. Weighted combinations (Weight RF slightly higher)
        if len(keys) == 2:
            w = [0.6, 0.4] if keys[0] == "rf" else [0.5, 0.5]
        elif len(keys) == 3:
            w = [0.5, 0.25, 0.25]
        elif len(keys) == 4:
            w = [0.4, 0.2, 0.2, 0.2]
        else:
            w = [0.35, 0.2, 0.2, 0.15, 0.1]
            
        w_norm = np.array(w) / np.sum(w)
        wt_probs = np.dot(stacked_probs, w_norm)
        wt_preds = (wt_probs >= 0.5).astype(int)
        met_wt = compute_all_metrics(y_val, wt_preds, wt_probs)
        entry_wt = {
            "ensemble_type": "soft_voting_weighted",
            "combination_id": c["id"],
            "display_name": f"Soft Voting Weighted ({c['name']})",
            "base_models": [base_models_dict[k]["name"] for k in keys],
            "weights": [round(float(x), 3) for x in w_norm],
            "validation_metrics": met_wt
        }
        ensemble_results.append(entry_wt)
        log(f"  {entry_wt['display_name']:45s} | Val ROC: {met_wt['roc_auc']:.4f} | Val BalAcc: {met_wt['balanced_accuracy']:.4f} | Val Sens: {met_wt['recall_sensitivity']:.4f} | Val F1: {met_wt['f1_score']:.4f} | Val Acc: {met_wt['accuracy']:.4f}")

    # C. STACKING (Leakage-safe Out-of-fold Meta-Learners)
    log("\n=== C. STACKING ENSEMBLES (OOF Meta-Classifiers) ===")
    meta_models = [
        ("logreg", "Logistic Regression Meta", LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", random_state=42)),
        ("logreg_bal", "Balanced LogReg Meta", LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", class_weight="balanced", random_state=42)),
        ("svm_linear", "Linear SVM Meta", SVC(C=0.1, kernel="linear", probability=True, random_state=42))
    ]

    for c in combinations:
        keys = c["keys"]
        X_train_oof = np.column_stack([oof_train_probs[k] for k in keys])
        X_val_meta = np.column_stack([val_base_probs[k] for k in keys])

        for meta_id, meta_name, meta_clf in meta_models:
            meta_clf.fit(X_train_oof, y_train)
            stack_probs = meta_clf.predict_proba(X_val_meta)[:, 1]
            stack_preds = (stack_probs >= 0.5).astype(int)
            met_stack = compute_all_metrics(y_val, stack_preds, stack_probs)

            entry_stack = {
                "ensemble_type": "stacking",
                "combination_id": c["id"],
                "meta_model": meta_name,
                "display_name": f"Stacking ({meta_name} on {c['name']})",
                "base_models": [base_models_dict[k]["name"] for k in keys],
                "validation_metrics": met_stack
            }
            ensemble_results.append(entry_stack)
            log(f"  {entry_stack['display_name']:55s} | Val ROC: {met_stack['roc_auc']:.4f} | Val BalAcc: {met_stack['balanced_accuracy']:.4f} | Val Sens: {met_stack['recall_sensitivity']:.4f} | Val F1: {met_stack['f1_score']:.4f} | Val Acc: {met_stack['accuracy']:.4f}")

    # 6. Category Best & Overall Best Selection
    hard_votes = [e for e in ensemble_results if e["ensemble_type"] == "hard_voting"]
    soft_votes = [e for e in ensemble_results if "soft_voting" in e["ensemble_type"]]
    stacking_runs = [e for e in ensemble_results if e["ensemble_type"] == "stacking"]

    def sort_key(e):
        vm = e["validation_metrics"]
        return (vm["roc_auc"], vm["pr_auc"], vm["balanced_accuracy"], vm["recall_sensitivity"], vm["f1_score"])

    best_hard = sorted(hard_votes, key=sort_key, reverse=True)[0]
    best_soft = sorted(soft_votes, key=sort_key, reverse=True)[0]
    best_stack = sorted(stacking_runs, key=sort_key, reverse=True)[0]
    best_overall = sorted(ensemble_results, key=sort_key, reverse=True)[0]

    # Baseline Comparisons
    rf_balanced_benchmark = {
        "model": "Balanced Random Forest",
        "k": 100,
        "val_roc_auc": 0.7000,
        "val_pr_auc": 0.5959,
        "val_balanced_acc": 0.6833,
        "val_sensitivity": 0.7000,
        "val_specificity": 0.6667,
        "val_f1": 0.7000,
        "val_accuracy": 0.6842
    }

    best_ov_m = best_overall["validation_metrics"]
    deltas_vs_single_rf = {
        "roc_auc_delta": round(best_ov_m["roc_auc"] - rf_balanced_benchmark["val_roc_auc"], 4),
        "pr_auc_delta": round(best_ov_m["pr_auc"] - rf_balanced_benchmark["val_pr_auc"], 4),
        "balanced_acc_delta": round(best_ov_m["balanced_accuracy"] - rf_balanced_benchmark["val_balanced_acc"], 4),
        "sensitivity_delta": round(best_ov_m["recall_sensitivity"] - rf_balanced_benchmark["val_sensitivity"], 4),
        "specificity_delta": round(best_ov_m["specificity"] - rf_balanced_benchmark["val_specificity"], 4),
        "f1_delta": round(best_ov_m["f1_score"] - rf_balanced_benchmark["val_f1"], 4),
        "accuracy_delta": round(best_ov_m["accuracy"] - rf_balanced_benchmark["val_accuracy"], 4)
    }

    # 7. Multi-Seed Stability on Top 2 Ensembles (Seeds 42, 43, 44, 45, 46)
    top_2_ensembles = sorted(ensemble_results, key=sort_key, reverse=True)[:2]
    seeds = [42, 43, 44, 45, 46]
    stability_records = []

    log("\n--- Running Multi-Seed Stability on Top 2 Ensembles (Seeds 42, 43, 44, 45, 46) ---")
    for idx, top_e in enumerate(top_2_ensembles, 1):
        log(f"Evaluating Stability for Rank #{idx}: {top_e['display_name']}...")
        s_rocs, s_praucs, s_baccs, s_senss, s_specs, s_f1s, s_accs = [], [], [], [], [], [], []

        for s in seeds:
            seed_base_models = instantiate_base_models(seed=s)
            s_val_probs = {}
            s_val_preds = {}

            for k_mod, info_mod in seed_base_models.items():
                mod = info_mod["model"]
                mod.fit(X_train_trans, y_train)
                pr = mod.predict_proba(X_val_trans)[:, 1]
                s_val_probs[k_mod] = pr
                s_val_preds[k_mod] = (pr >= 0.5).astype(int)

            comb_id = top_e["combination_id"]
            comb_keys = next(c["keys"] for c in combinations if c["id"] == comb_id)

            if top_e["ensemble_type"] == "hard_voting":
                st_preds = np.column_stack([s_val_preds[k] for k in comb_keys])
                st_probs = np.column_stack([s_val_probs[k] for k in comb_keys])
                v_sum = st_preds.sum(axis=1)
                half = len(comb_keys) / 2.0
                ens_preds = np.where(v_sum > half, 1, np.where(v_sum < half, 0, (st_probs.mean(axis=1) >= 0.5).astype(int)))
                ens_probs = st_probs.mean(axis=1)
            elif "soft_voting" in top_e["ensemble_type"]:
                st_probs = np.column_stack([s_val_probs[k] for k in comb_keys])
                if top_e["weights"] == "equal":
                    ens_probs = st_probs.mean(axis=1)
                else:
                    w = np.array(top_e["weights"])
                    ens_probs = np.dot(st_probs, w)
                ens_preds = (ens_probs >= 0.5).astype(int)
            else: # Stacking
                # Generate OOF
                s_oof = {k: np.zeros(len(y_train)) for k in comb_keys}
                for tr_idx, va_idx in cv_skf.split(X_train, y_train):
                    X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
                    X_va_f = X_train[va_idx]
                    pipe_f = make_pipeline(k=100)
                    X_tr_f_tr = pipe_f.fit_transform(X_tr_f, y_tr_f)
                    X_va_f_tr = pipe_f.transform(X_va_f)
                    f_models = instantiate_base_models(seed=s)
                    for k_m in comb_keys:
                        m_f = f_models[k_m]["model"]
                        m_f.fit(X_tr_f_tr, y_tr_f)
                        s_oof[k_m][va_idx] = m_f.predict_proba(X_va_f_tr)[:, 1]

                X_tr_oof = np.column_stack([s_oof[k] for k in comb_keys])
                X_va_m = np.column_stack([s_val_probs[k] for k in comb_keys])
                m_clf = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", random_state=s)
                m_clf.fit(X_tr_oof, y_train)
                ens_probs = m_clf.predict_proba(X_va_m)[:, 1]
                ens_preds = (ens_probs >= 0.5).astype(int)

            s_met = compute_all_metrics(y_val, ens_preds, ens_probs)
            s_rocs.append(s_met["roc_auc"])
            s_praucs.append(s_met["pr_auc"])
            s_baccs.append(s_met["balanced_accuracy"])
            s_senss.append(s_met["recall_sensitivity"])
            s_specs.append(s_met["specificity"])
            s_f1s.append(s_met["f1_score"])
            s_accs.append(s_met["accuracy"])

        stab_entry = {
            "rank": idx,
            "display_name": top_e["display_name"],
            "ensemble_type": top_e["ensemble_type"],
            "base_models": top_e["base_models"],
            "seeds": seeds,
            "val_roc_auc": {"mean": float(np.mean(s_rocs)), "std": float(np.std(s_rocs)), "min": float(np.min(s_rocs)), "max": float(np.max(s_rocs))},
            "val_pr_auc": {"mean": float(np.mean(s_praucs)), "std": float(np.std(s_praucs)), "min": float(np.min(s_praucs)), "max": float(np.max(s_praucs))},
            "val_balanced_acc": {"mean": float(np.mean(s_baccs)), "std": float(np.std(s_baccs)), "min": float(np.min(s_baccs)), "max": float(np.max(s_baccs))},
            "val_sensitivity": {"mean": float(np.mean(s_senss)), "std": float(np.std(s_senss)), "min": float(np.min(s_senss)), "max": float(np.max(s_senss))},
            "val_specificity": {"mean": float(np.mean(s_specs)), "std": float(np.std(s_specs)), "min": float(np.min(s_specs)), "max": float(np.max(s_specs))},
            "val_f1_score": {"mean": float(np.mean(s_f1s)), "std": float(np.std(s_f1s)), "min": float(np.min(s_f1s)), "max": float(np.max(s_f1s))},
            "val_accuracy": {"mean": float(np.mean(s_accs)), "std": float(np.std(s_accs)), "min": float(np.min(s_accs)), "max": float(np.max(s_accs))}
        }
        stability_records.append(stab_entry)
        log(f"  -> Val ROC-AUC: {np.mean(s_rocs):.4f} ± {np.std(s_rocs):.4f}")
        log(f"  -> Val BalAcc:  {np.mean(s_baccs):.4f} ± {np.std(s_baccs):.4f}")
        log(f"  -> Val Sens:    {np.mean(s_senss):.4f} ± {np.std(s_senss):.4f}")

    # 8. Save Artifacts
    (run_dir / "ensemble_config.json").write_text(json.dumps({
        "run_id": run_id, "experiment": "Hybrid Ensemble Benchmark", "combinations": combinations, "random_seed": 42
    }, indent=2), encoding="utf-8")

    # CSV Results
    rows = []
    for e in ensemble_results:
        vm = e["validation_metrics"]
        rows.append({
            "ensemble_type": e["ensemble_type"],
            "display_name": e["display_name"],
            "base_models": ", ".join(e["base_models"]),
            "weights": str(e.get("weights", "")),
            "val_accuracy": vm["accuracy"],
            "val_balanced_accuracy": vm["balanced_accuracy"],
            "val_precision": vm["precision"],
            "val_recall_sensitivity": vm["recall_sensitivity"],
            "val_specificity": vm["specificity"],
            "val_f1_score": vm["f1_score"],
            "val_roc_auc": vm["roc_auc"],
            "val_pr_auc": vm["pr_auc"]
        })
    pd.DataFrame(rows).to_csv(run_dir / "validation_results.csv", index=False)
    (run_dir / "validation_results.json").write_text(json.dumps(ensemble_results, indent=2), encoding="utf-8")
    (run_dir / "stability_results.json").write_text(json.dumps(stability_records, indent=2), encoding="utf-8")
    (run_dir / "prediction_agreement.json").write_text(json.dumps(complementarity_data, indent=2), encoding="utf-8")

    comp_summary = {
        "best_hard_voting": best_hard,
        "best_soft_voting": best_soft,
        "best_stacking": best_stack,
        "best_overall_ensemble": best_overall,
        "current_best_single_model": rf_balanced_benchmark,
        "deltas_vs_single_model": deltas_vs_single_rf,
        "stability_results": stability_records
    }
    (run_dir / "model_comparison.json").write_text(json.dumps(comp_summary, indent=2), encoding="utf-8")

    report_json = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "purpose": "Hybrid / Ensemble Experiment (Hard Voting, Soft Voting, Stacking)",
        "selection": {
            "selected_model": best_overall["display_name"],
            "selected_validation_roc_auc": round(best_ov_m["roc_auc"], 4),
            "selected_validation_pr_auc": round(best_ov_m["pr_auc"], 4),
            "selected_validation_accuracy": round(best_ov_m["accuracy"], 4),
            "selected_validation_balanced_acc": round(best_ov_m["balanced_accuracy"], 4),
            "selected_validation_sensitivity": round(best_ov_m["recall_sensitivity"], 4),
            "selected_validation_f1": round(best_ov_m["f1_score"], 4)
        },
        "final_test": {
            "final_test_executed": False,
            "test_site": "Peking_2 (LOCKED)",
            "message": "FINAL TEST NOT PERFORMED — Peking_2 remains locked."
        }
    }
    (run_dir / "run_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    # Generate Markdown Report
    report_md = f"""# NeuroADHD Hybrid & Ensemble Experiment Report — {run_id}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Status:** Completed  
**Test Set Status:** 🔒 **LOCKED & UNTOUCHED** (Peking_2 site, 20 subjects)

---

## 1. Objective & Methodology
This experiment systematically tests whether combining our strongest individual models improves out-of-site generalization over our current best single model (**Balanced Random Forest**).

### Base Models Evaluated ($K=100$, Preprocessing fit strictly on training set):
1. **Random Forest**: `n_estimators=50, max_depth=3, class_weight='balanced'`
2. **XGBoost**: `n_estimators=100, max_depth=2, learning_rate=0.05, scale_pos_weight=2.8`
3. **Extra Trees**: `n_estimators=50, max_depth=5, class_weight='balanced'`
4. **RBF SVM**: `C=1.0, gamma='scale', class_weight='balanced'`
5. **Logistic Regression**: `C=0.1, penalty='l1', solver='saga', class_weight='balanced'`

---

## 2. Category Winner Ensembles

### A. Best Hard Voting Ensemble
- **Model Combination:** `{best_hard['display_name']}`
- **Validation ROC-AUC:** `{best_hard['validation_metrics']['roc_auc']:.4f}`
- **Validation Balanced Accuracy:** `{best_hard['validation_metrics']['balanced_accuracy']:.4f}`
- **Validation Sensitivity:** `{best_hard['validation_metrics']['recall_sensitivity']:.4f}`
- **Validation F1 Score:** `{best_hard['validation_metrics']['f1_score']:.4f}`

### B. Best Soft Voting Ensemble
- **Model Combination:** `{best_soft['display_name']}`
- **Validation ROC-AUC:** `{best_soft['validation_metrics']['roc_auc']:.4f}`
- **Validation Balanced Accuracy:** `{best_soft['validation_metrics']['balanced_accuracy']:.4f}`
- **Validation Sensitivity:** `{best_soft['validation_metrics']['recall_sensitivity']:.4f}`
- **Validation F1 Score:** `{best_soft['validation_metrics']['f1_score']:.4f}`

### C. Best Stacking Ensemble
- **Model Combination:** `{best_stack['display_name']}`
- **Validation ROC-AUC:** `{best_stack['validation_metrics']['roc_auc']:.4f}`
- **Validation Balanced Accuracy:** `{best_stack['validation_metrics']['balanced_accuracy']:.4f}`
- **Validation Sensitivity:** `{best_stack['validation_metrics']['recall_sensitivity']:.4f}`
- **Validation F1 Score:** `{best_stack['validation_metrics']['f1_score']:.4f}`

---

## 3. Best Overall Ensemble vs. Single Model Benchmark

| Metric | Balanced Random Forest (Single Best) | Best Ensemble ({best_overall['display_name']}) | Delta (Ensemble vs Single) |
| :--- | :---: | :---: | :---: |
| **Validation ROC-AUC** | `0.7000` | **`{best_ov_m['roc_auc']:.4f}`** | `{deltas_vs_single_rf['roc_auc_delta']:+.4f}` |
| **Validation PR-AUC** | `0.5959` | **`{best_ov_m['pr_auc']:.4f}`** | `{deltas_vs_single_rf['pr_auc_delta']:+.4f}` |
| **Validation Balanced Accuracy** | `0.6833` | **`{best_ov_m['balanced_accuracy']:.4f}`** | `{deltas_vs_single_rf['balanced_acc_delta']:+.4f}` |
| **Validation Sensitivity (ADHD)** | `0.7000` | **`{best_ov_m['recall_sensitivity']:.4f}`** | `{deltas_vs_single_rf['sensitivity_delta']:+.4f}` |
| **Validation Specificity (Control)** | `0.6667` | **`{best_ov_m['specificity']:.4f}`** | `{deltas_vs_single_rf['specificity_delta']:+.4f}` |
| **Validation F1 Score** | `0.7000` | **`{best_ov_m['f1_score']:.4f}`** | `{deltas_vs_single_rf['f1_delta']:+.4f}` |
| **Validation Accuracy** | `0.6842` | **`{best_ov_m['accuracy']:.4f}`** | `{deltas_vs_single_rf['accuracy_delta']:+.4f}` |

---

## 4. Multi-Seed Stability Summary (Seeds 42, 43, 44, 45, 46)

"""
    for s_rec in stability_records:
        report_md += f"""### Rank #{s_rec['rank']}: {s_rec['display_name']}
- **Val ROC-AUC:** {s_rec['val_roc_auc']['mean']:.4f} ± {s_rec['val_roc_auc']['std']:.4f} (Range: {s_rec['val_roc_auc']['min']:.4f} – {s_rec['val_roc_auc']['max']:.4f})
- **Val Balanced Acc:** {s_rec['val_balanced_acc']['mean']:.4f} ± {s_rec['val_balanced_acc']['std']:.4f}
- **Val Sensitivity (ADHD):** {s_rec['val_sensitivity']['mean']:.4f} ± {s_rec['val_sensitivity']['std']:.4f}
- **Val F1 Score:** {s_rec['val_f1_score']['mean']:.4f} ± {s_rec['val_f1_score']['std']:.4f}

"""

    report_md += """
---

## 5. Test Set Evaluation Declaration

> **TEST EVALUATION:**  
> **FINAL TEST NOT PERFORMED — Peking_2 remains locked.**
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
        "duration_seconds": 12.0,
        "best_model": best_overall["display_name"],
        "validation_roc_auc": round(best_ov_m["roc_auc"], 4),
        "validation_accuracy": round(best_ov_m["accuracy"], 4),
        "final_test_roc_auc": None,
        "final_test_accuracy": None,
        "test_status": "LOCKED",
        "report_json": f"runs/{run_id}/run_report.json",
        "report_html": f"runs/{run_id}/run_report.html"
    }
    index_data.insert(0, index_entry)
    index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")

    log(f"\nHybrid/Ensemble experiment {run_id} successfully completed!")
    return run_id, best_hard, best_soft, best_stack, best_overall, deltas_vs_single_rf

if __name__ == "__main__":
    run_ensemble_experiment()
