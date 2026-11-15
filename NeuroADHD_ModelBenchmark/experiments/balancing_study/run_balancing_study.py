"""Balancing Study Execution Script.

Evaluates 12 balancing methods across 10 classifiers under:
  Scheme A: Repeated Stratified 5-Fold CV (5 repeats: seeds 42-46)
  Scheme B: Leave-One-Institution-Out CV (6 folds)

Pipeline per fold:
  SimpleImputer(median) -> VarianceThreshold() -> StandardScaler() ->
  SelectKBest(f_classif, k=100) -> [balancing] -> classifier
"""
from __future__ import annotations

import json
import time
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    brier_score_loss
)

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN
)
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN


MODELS = [
    "Logistic Regression (L2)",
    "Linear SVM (calibrated)",
    "RBF SVM",
    "Random Forest (n=50, d=3)",
    "Random Forest (n=200, d=None)",
    "Extra Trees",
    "XGBoost",
    "Gradient Boosting",
    "KNN (k=5)",
    "LDA with shrinkage"
]

BALANCING_METHODS = [
    "None",
    "class_weight='balanced'",
    "RandomOverSampler",
    "SMOTE",
    "BorderlineSMOTE",
    "SVMSMOTE",
    "ADASYN",
    "RandomUnderSampler",
    "NearMiss",
    "SMOTETomek",
    "SMOTEENN",
    "Threshold tuning"
]

SKIPPED_COMBINATIONS = [
    ("Gradient Boosting", "class_weight='balanced'"),
    ("KNN (k=5)", "class_weight='balanced'"),
    ("LDA with shrinkage", "class_weight='balanced'")
]


def is_valid_combination(model_name: str, method_name: str) -> bool:
    return (model_name, method_name) not in SKIPPED_COMBINATIONS


def get_classifier(model_name: str, method_name: str, seed: int, y_train: np.ndarray | None = None):
    cw = "balanced" if method_name == "class_weight='balanced'" else None
    
    if model_name == "Logistic Regression (L2)":
        return LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=seed, solver="lbfgs", class_weight=cw)
    elif model_name == "Linear SVM (calibrated)":
        base_svc = LinearSVC(dual=False, C=1.0, random_state=seed, class_weight=cw)
        return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
    elif model_name == "RBF SVM":
        return SVC(kernel="rbf", probability=True, C=1.0, random_state=seed, class_weight=cw)
    elif model_name == "Random Forest (n=50, d=3)":
        return RandomForestClassifier(n_estimators=50, max_depth=3, random_state=seed, class_weight=cw, n_jobs=1)
    elif model_name == "Random Forest (n=200, d=None)":
        return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, class_weight=cw, n_jobs=1)
    elif model_name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=100, random_state=seed, class_weight=cw, n_jobs=1)
    elif model_name == "XGBoost":
        scale_pos = 1.0
        if method_name == "class_weight='balanced'" and y_train is not None:
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale_pos = float(n_neg / n_pos) if n_pos > 0 else 1.0
        return xgb.XGBClassifier(eval_metric="logloss", random_state=seed, scale_pos_weight=scale_pos, n_jobs=1)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=seed)
    elif model_name == "KNN (k=5)":
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == "LDA with shrinkage":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_sampler(method_name: str, seed: int):
    if method_name in ("None", "class_weight='balanced'", "Threshold tuning"):
        return None
    elif method_name == "RandomOverSampler":
        return RandomOverSampler(random_state=seed)
    elif method_name == "SMOTE":
        return SMOTE(k_neighbors=3, random_state=seed)
    elif method_name == "BorderlineSMOTE":
        return BorderlineSMOTE(k_neighbors=3, random_state=seed)
    elif method_name == "SVMSMOTE":
        return SVMSMOTE(k_neighbors=3, random_state=seed)
    elif method_name == "ADASYN":
        return ADASYN(n_neighbors=3, random_state=seed)
    elif method_name == "RandomUnderSampler":
        return RandomUnderSampler(random_state=seed)
    elif method_name == "NearMiss":
        return NearMiss(version=1, n_neighbors=3)
    elif method_name == "SMOTETomek":
        return SMOTETomek(smote=SMOTE(k_neighbors=3, random_state=seed), random_state=seed)
    elif method_name == "SMOTEENN":
        return SMOTEENN(smote=SMOTE(k_neighbors=3, random_state=seed), random_state=seed)
    else:
        raise ValueError(f"Unknown balancing method: {method_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        p_arr, r_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_arr, p_arr))
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "mcc": mcc,
        "brier": brier,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    accs = []
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        accs.append(accuracy_score(y_b, y_pred[idx]))
        aucs.append(roc_auc_score(y_b, y_prob[idx]))
    acc_ci = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
    auc_ci = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    return acc_ci, auc_ci


def tune_threshold_inner_cv(X_tr: np.ndarray, y_tr: np.ndarray, model_name: str, seed: int) -> float:
    """Find threshold maximizing balanced accuracy via inner 3-fold CV on training fold."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    inner_oof_probs = np.zeros(len(y_tr))
    
    for in_tr, in_val in skf.split(X_tr, y_tr):
        clf = get_classifier(model_name, "None", seed, y_tr[in_tr])
        clf.fit(X_tr[in_tr], y_tr[in_tr])
        inner_oof_probs[in_val] = clf.predict_proba(X_tr[in_val])[:, 1]
        
    thresholds = np.linspace(0.10, 0.90, 81)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        score = balanced_accuracy_score(y_tr, (inner_oof_probs >= t).astype(int))
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)


def main():
    t_start_total = time.time()
    out_dir = Path("NeuroADHD_ModelBenchmark/experiments/balancing_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("================================================================================")
    print("NEUROADHD CLASS-BALANCING BENCHMARK STUDY")
    print("================================================================================")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models ({len(MODELS)}): {', '.join(MODELS)}")
    print(f"Balancing Methods ({len(BALANCING_METHODS)}): {', '.join(BALANCING_METHODS)}")
    print(f"Skipped invalid combinations: {SKIPPED_COMBINATIONS}")
    
    # 1. Load Data
    data_path = Path("NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv")
    df = pd.read_csv(data_path)
    df["institution"] = df["site"].replace({"Peking_1": "Peking", "Peking_2": "Peking", "Peking_3": "Peking"})
    
    feature_cols = [c for c in df.columns if c.startswith("ROI_")]
    X = df[feature_cols].to_numpy()
    y = df["diagnosis"].to_numpy()
    institutions = df["institution"].to_numpy()
    unique_institutions = np.unique(institutions)
    
    n_total = len(df)
    n_adhd = int(np.sum(y == 1))
    n_ctrl = int(np.sum(y == 0))
    print(f"\nCohort: {n_total} subjects ({n_ctrl} Control, {n_adhd} ADHD | {len(feature_cols)} features)")
    print(f"Institutions ({len(unique_institutions)}): {', '.join(unique_institutions)}")
    
    # ============================================================================
    # SCHEME A: REPEATED STRATIFIED 5-FOLD CV (5 REPEATS: SEEDS 42-46)
    # ============================================================================
    print("\n--------------------------------------------------------------------------------")
    print("RUNNING SCHEME A: REPEATED STRATIFIED 5-FOLD CV (5 REPEATS)")
    print("--------------------------------------------------------------------------------")
    seeds = [42, 43, 44, 45, 46]
    
    # Store predictions: repeat_idx -> (model, method) -> (y_pred, y_prob)
    # Also record reference predictions
    scheme_a_repeat_metrics = {m: {b: [] for b in BALANCING_METHODS if is_valid_combination(m, b)} for m in MODELS}
    scheme_a_ref_majority = []
    scheme_a_ref_site = []
    
    # Store pooled predictions of the first repeat (seed 42) for ROC curves and confusion matrices
    scheme_a_seed42_preds = {}

    for r_idx, seed in enumerate(seeds):
        t_rep_start = time.time()
        print(f"\n[Scheme A] Repeat {r_idx + 1}/5 (Seed={seed})...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        # Out-of-fold prediction storage for this repeat
        oof_preds = {m: {b: np.zeros(n_total) for b in BALANCING_METHODS if is_valid_combination(m, b)} for m in MODELS}
        oof_probs = {m: {b: np.zeros(n_total) for b in BALANCING_METHODS if is_valid_combination(m, b)} for m in MODELS}
        oof_site_probs = np.zeros(n_total)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # 1. Feature Preprocessing & Selection (Strictly fit on training fold)
            imputer = SimpleImputer(strategy="median")
            X_tr_imp = imputer.fit_transform(X_tr)
            X_val_imp = imputer.transform(X_val)
            
            vt = VarianceThreshold()
            X_tr_vt = vt.fit_transform(X_tr_imp)
            X_val_vt = vt.transform(X_val_imp)
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_vt)
            X_val_sc = scaler.transform(X_val_vt)
            
            kbest = SelectKBest(f_classif, k=100)
            X_tr_kbest = kbest.fit_transform(X_tr_sc, y_tr)
            X_val_kbest = kbest.transform(X_val_sc)
            
            # Site-only baseline: fit Logistic Regression on one-hot site dummies
            site_dummies_tr = pd.get_dummies(df.iloc[train_idx]["institution"]).reindex(columns=unique_institutions, fill_value=0).astype(float)
            site_dummies_val = pd.get_dummies(df.iloc[val_idx]["institution"]).reindex(columns=unique_institutions, fill_value=0).astype(float)
            site_clf = LogisticRegression(C=1.0, random_state=seed)
            site_clf.fit(site_dummies_tr, y_tr)
            oof_site_probs[val_idx] = site_clf.predict_proba(site_dummies_val)[:, 1]
            
            # Precompute resampled datasets for each balancing method
            resampled_datasets = {}
            for method in BALANCING_METHODS:
                if method in ("None", "class_weight='balanced'", "Threshold tuning"):
                    resampled_datasets[method] = (X_tr_kbest, y_tr)
                else:
                    sampler = get_sampler(method, seed)
                    try:
                        X_res, y_res = sampler.fit_resample(X_tr_kbest, y_tr)
                        if len(np.unique(y_res)) < 2:
                            X_res, y_res = X_tr_kbest, y_tr
                    except Exception as e:
                        X_res, y_res = X_tr_kbest, y_tr
                    resampled_datasets[method] = (X_res, y_res)
            
            # Loop over all models and methods
            for model_name in MODELS:
                for method_name in BALANCING_METHODS:
                    if not is_valid_combination(model_name, method_name):
                        continue
                    
                    if method_name == "Threshold tuning":
                        X_tr_m, y_tr_m = resampled_datasets["None"]
                        t_star = tune_threshold_inner_cv(X_tr_m, y_tr_m, model_name, seed)
                        clf = get_classifier(model_name, "None", seed, y_tr_m)
                        clf.fit(X_tr_m, y_tr_m)
                        prob_val = clf.predict_proba(X_val_kbest)[:, 1]
                        pred_val = (prob_val >= t_star).astype(int)
                    else:
                        X_tr_m, y_tr_m = resampled_datasets[method_name]
                        clf = get_classifier(model_name, method_name, seed, y_tr_m)
                        clf.fit(X_tr_m, y_tr_m)
                        prob_val = clf.predict_proba(X_val_kbest)[:, 1]
                        pred_val = clf.predict(X_val_kbest)
                        
                    oof_preds[model_name][method_name][val_idx] = pred_val
                    oof_probs[model_name][method_name][val_idx] = prob_val
                    
        # Compute metrics for this repeat
        for model_name in MODELS:
            for method_name in BALANCING_METHODS:
                if not is_valid_combination(model_name, method_name):
                    continue
                m_dict = compute_metrics(y, oof_preds[model_name][method_name], oof_probs[model_name][method_name])
                scheme_a_repeat_metrics[model_name][method_name].append(m_dict)
                if r_idx == 0:
                    scheme_a_seed42_preds[(model_name, method_name)] = (
                        oof_preds[model_name][method_name].copy(),
                        oof_probs[model_name][method_name].copy()
                    )
                    
        # Reference models for this repeat
        # Majority: always 0
        maj_preds = np.zeros(n_total, dtype=int)
        maj_probs = np.zeros(n_total)
        scheme_a_ref_majority.append(compute_metrics(y, maj_preds, maj_probs))
        
        # Site-only
        site_preds = (oof_site_probs >= 0.5).astype(int)
        scheme_a_ref_site.append(compute_metrics(y, site_preds, oof_site_probs))
        
        print(f"  Finished Repeat {r_idx + 1}/5 in {time.time() - t_rep_start:.1f}s")
        
    # Aggregate Scheme A across 5 repeats
    scheme_a_rows = []
    for model_name in MODELS:
        for method_name in BALANCING_METHODS:
            if not is_valid_combination(model_name, method_name):
                continue
            rep_list = scheme_a_repeat_metrics[model_name][method_name]
            row = {
                "scheme": "Scheme A (Stratified 5-Fold x 5 Repeats)",
                "model": model_name,
                "balancing_method": method_name
            }
            for metric in ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "precision", "recall_sensitivity", "specificity", "f1", "mcc", "brier"]:
                vals = [r[metric] for r in rep_list]
                row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{metric}_sd"] = round(float(np.std(vals, ddof=1)), 4)
                
            # Average confusion matrix
            row["tp_mean"] = round(float(np.mean([r["tp"] for r in rep_list])), 1)
            row["tn_mean"] = round(float(np.mean([r["tn"] for r in rep_list])), 1)
            row["fp_mean"] = round(float(np.mean([r["fp"] for r in rep_list])), 1)
            row["fn_mean"] = round(float(np.mean([r["fn"] for r in rep_list])), 1)
            scheme_a_rows.append(row)
            
    # Add Reference rows for Scheme A
    for name, ref_list in [("Majority Class (Always Control)", scheme_a_ref_majority), ("Site-Only Baseline (Grouped CV)", scheme_a_ref_site)]:
        row = {
            "scheme": "Scheme A (Stratified 5-Fold x 5 Repeats)",
            "model": name,
            "balancing_method": "N/A (Reference)"
        }
        for metric in ["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "precision", "recall_sensitivity", "specificity", "f1", "mcc", "brier"]:
            vals = [r[metric] for r in ref_list]
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{metric}_sd"] = round(float(np.std(vals, ddof=1)), 4)
        row["tp_mean"] = round(float(np.mean([r["tp"] for r in ref_list])), 1)
        row["tn_mean"] = round(float(np.mean([r["tn"] for r in ref_list])), 1)
        row["fp_mean"] = round(float(np.mean([r["fp"] for r in ref_list])), 1)
        row["fn_mean"] = round(float(np.mean([r["fn"] for r in ref_list])), 1)
        scheme_a_rows.append(row)
        
    df_scheme_a = pd.DataFrame(scheme_a_rows)
    df_scheme_a.to_csv(out_dir / "results_stratified_cv.csv", index=False)
    print(f"\nSaved Scheme A results: {out_dir / 'results_stratified_cv.csv'} ({len(df_scheme_a)} rows)")

    # ============================================================================
    # SCHEME B: LEAVE-ONE-INSTITUTION-OUT CV (6 FOLDS)
    # ============================================================================
    print("\n--------------------------------------------------------------------------------")
    print("RUNNING SCHEME B: LEAVE-ONE-INSTITUTION-OUT CV (6 FOLDS)")
    print("--------------------------------------------------------------------------------")
    t_loio_start = time.time()
    
    oof_loio_preds = {m: {b: np.zeros(n_total) for b in BALANCING_METHODS if is_valid_combination(m, b)} for m in MODELS}
    oof_loio_probs = {m: {b: np.zeros(n_total) for b in BALANCING_METHODS if is_valid_combination(m, b)} for m in MODELS}
    oof_loio_site_probs = np.zeros(n_total)
    
    # Track per-site metrics for KKI, NeuroIMAGE, Peking, NYU, OHSU, Pittsburgh
    site_fold_metrics = {m: {b: {} for b in BALANCING_METHODS if is_valid_combination(m, b)} for m in MODELS}
    
    seed_loio = 42

    for inst_idx, inst in enumerate(unique_institutions):
        t_fold_start = time.time()
        val_idx = np.where(institutions == inst)[0]
        train_idx = np.where(institutions != inst)[0]
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Site-only LOIO baseline: training empirical base-rate
        p_train = float(np.mean(y_tr))
        oof_loio_site_probs[val_idx] = p_train
        
        # 1. Feature Preprocessing & Selection on training institutions
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_val_imp = imputer.transform(X_val)
        
        vt = VarianceThreshold()
        X_tr_vt = vt.fit_transform(X_tr_imp)
        X_val_vt = vt.transform(X_val_imp)
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_vt)
        X_val_sc = scaler.transform(X_val_vt)
        
        kbest = SelectKBest(f_classif, k=100)
        X_tr_kbest = kbest.fit_transform(X_tr_sc, y_tr)
        X_val_kbest = kbest.transform(X_val_sc)
        
        # Precompute resampled datasets for this fold
        resampled_datasets = {}
        for method in BALANCING_METHODS:
            if method in ("None", "class_weight='balanced'", "Threshold tuning"):
                resampled_datasets[method] = (X_tr_kbest, y_tr)
            else:
                sampler = get_sampler(method, seed_loio)
                try:
                    X_res, y_res = sampler.fit_resample(X_tr_kbest, y_tr)
                    if len(np.unique(y_res)) < 2:
                        X_res, y_res = X_tr_kbest, y_tr
                except Exception:
                    X_res, y_res = X_tr_kbest, y_tr
                resampled_datasets[method] = (X_res, y_res)
                
        # Loop over models and methods
        for model_name in MODELS:
            for method_name in BALANCING_METHODS:
                if not is_valid_combination(model_name, method_name):
                    continue
                
                if method_name == "Threshold tuning":
                    X_tr_m, y_tr_m = resampled_datasets["None"]
                    t_star = tune_threshold_inner_cv(X_tr_m, y_tr_m, model_name, seed_loio)
                    clf = get_classifier(model_name, "None", seed_loio, y_tr_m)
                    clf.fit(X_tr_m, y_tr_m)
                    prob_val = clf.predict_proba(X_val_kbest)[:, 1]
                    pred_val = (prob_val >= t_star).astype(int)
                else:
                    X_tr_m, y_tr_m = resampled_datasets[method_name]
                    clf = get_classifier(model_name, method_name, seed_loio, y_tr_m)
                    clf.fit(X_tr_m, y_tr_m)
                    prob_val = clf.predict_proba(X_val_kbest)[:, 1]
                    pred_val = clf.predict(X_val_kbest)
                    
                oof_loio_preds[model_name][method_name][val_idx] = pred_val
                oof_loio_probs[model_name][method_name][val_idx] = prob_val
                
                # Record fold-level metrics for this institution
                if len(np.unique(y_val)) > 1:
                    fold_auc = float(roc_auc_score(y_val, prob_val))
                    fold_bal_acc = float(balanced_accuracy_score(y_val, pred_val))
                else:
                    fold_auc = np.nan
                    fold_bal_acc = float(balanced_accuracy_score(y_val, pred_val))
                site_fold_metrics[model_name][method_name][inst] = {
                    "n": len(val_idx),
                    "n_adhd": int(np.sum(y_val == 1)),
                    "auc": fold_auc,
                    "bal_acc": fold_bal_acc,
                    "mean_prob": float(np.mean(prob_val)),
                    "spec": float(np.mean(pred_val == 0)) if np.all(y_val == 0) else None
                }
                
        print(f"  Finished Fold {inst_idx + 1}/6 ({inst}, N={len(val_idx)}) in {time.time() - t_fold_start:.1f}s")
        
    # Aggregate Scheme B Pooled Metrics
    scheme_b_rows = []
    for model_name in MODELS:
        for method_name in BALANCING_METHODS:
            if not is_valid_combination(model_name, method_name):
                continue
            preds = oof_loio_preds[model_name][method_name]
            probs = oof_loio_probs[model_name][method_name]
            m_dict = compute_metrics(y, preds, probs)
            row = {
                "scheme": "Scheme B (Leave-One-Institution-Out CV)",
                "model": model_name,
                "balancing_method": method_name
            }
            for k, v in m_dict.items():
                if isinstance(v, float):
                    row[k] = round(v, 4)
                else:
                    row[k] = v
            # Add specific site AUCs
            sm = site_fold_metrics[model_name][method_name]
            row["kki_auc"] = round(sm["KKI"]["auc"], 4) if not np.isnan(sm["KKI"]["auc"]) else np.nan
            row["neuroimage_auc"] = round(sm["NeuroIMAGE"]["auc"], 4) if not np.isnan(sm["NeuroIMAGE"]["auc"]) else np.nan
            row["peking_auc"] = round(sm["Peking"]["auc"], 4) if not np.isnan(sm["Peking"]["auc"]) else np.nan
            row["nyu_auc_exploratory"] = round(sm["NYU"]["auc"], 4) if not np.isnan(sm["NYU"]["auc"]) else np.nan
            row["ohsu_spec"] = round(sm["OHSU"]["spec"], 4)
            row["ohsu_mean_prob"] = round(sm["OHSU"]["mean_prob"], 4)
            row["pitt_spec"] = round(sm["Pittsburgh"]["spec"], 4)
            row["pitt_mean_prob"] = round(sm["Pittsburgh"]["mean_prob"], 4)
            scheme_b_rows.append(row)
            
    # Reference rows for Scheme B
    # Majority
    maj_preds = np.zeros(n_total, dtype=int)
    maj_probs = np.zeros(n_total)
    row_maj = {
        "scheme": "Scheme B (Leave-One-Institution-Out CV)",
        "model": "Majority Class (Always Control)",
        "balancing_method": "N/A (Reference)"
    }
    for k, v in compute_metrics(y, maj_preds, maj_probs).items():
        row_maj[k] = round(v, 4) if isinstance(v, float) else v
    row_maj["kki_auc"] = 0.5; row_maj["neuroimage_auc"] = 0.5; row_maj["peking_auc"] = 0.5
    row_maj["nyu_auc_exploratory"] = 0.5; row_maj["ohsu_spec"] = 1.0; row_maj["ohsu_mean_prob"] = 0.0
    row_maj["pitt_spec"] = 1.0; row_maj["pitt_mean_prob"] = 0.0
    scheme_b_rows.append(row_maj)
    
    # Site-only LOIO
    site_preds = (oof_loio_site_probs >= 0.5).astype(int)
    row_site = {
        "scheme": "Scheme B (Leave-One-Institution-Out CV)",
        "model": "Site-Only Baseline (LOIO Training Base-Rate)",
        "balancing_method": "N/A (Reference)"
    }
    for k, v in compute_metrics(y, site_preds, oof_loio_site_probs).items():
        row_site[k] = round(v, 4) if isinstance(v, float) else v
    row_site["kki_auc"] = np.nan; row_site["neuroimage_auc"] = np.nan; row_site["peking_auc"] = np.nan
    row_site["nyu_auc_exploratory"] = np.nan; row_site["ohsu_spec"] = 1.0; row_site["ohsu_mean_prob"] = round(float(np.mean(oof_loio_site_probs[institutions=="OHSU"])), 4)
    row_site["pitt_spec"] = 1.0; row_site["pitt_mean_prob"] = round(float(np.mean(oof_loio_site_probs[institutions=="Pittsburgh"])), 4)
    scheme_b_rows.append(row_site)
    
    df_scheme_b = pd.DataFrame(scheme_b_rows)
    df_scheme_b.to_csv(out_dir / "results_loio_cv.csv", index=False)
    print(f"\nSaved Scheme B results: {out_dir / 'results_loio_cv.csv'} ({len(df_scheme_b)} rows)")

    # ============================================================================
    # BOOTSTRAP 95% CI FOR TOP 10 CONFIGURATIONS
    # ============================================================================
    print("\n--------------------------------------------------------------------------------")
    print("COMPUTING BOOTSTRAP 95% CIs (1000 RESAMPLES) FOR TOP 10 CONFIGURATIONS")
    print("--------------------------------------------------------------------------------")
    
    # Filter out reference rows for ranking
    clf_rows_a = [r for r in scheme_a_rows if not r["model"].startswith("Majority") and not r["model"].startswith("Site-Only")]
    clf_rows_b = [r for r in scheme_b_rows if not r["model"].startswith("Majority") and not r["model"].startswith("Site-Only")]
    
    top10_a_acc = sorted(clf_rows_a, key=lambda x: x["accuracy_mean"], reverse=True)[:10]
    top10_a_auc = sorted(clf_rows_a, key=lambda x: x["roc_auc_mean"], reverse=True)[:10]
    
    top10_b_acc = sorted(clf_rows_b, key=lambda x: x["accuracy"], reverse=True)[:10]
    top10_b_auc = sorted(clf_rows_b, key=lambda x: x["roc_auc"], reverse=True)[:10]
    
    # Compute Bootstrap CIs
    ci_results = {"scheme_a": {}, "scheme_b": {}}
    
    # For Scheme A, compute on the pooled predictions of seed 42
    top10_a_unique = { (r["model"], r["balancing_method"]) for r in top10_a_acc + top10_a_auc }
    for m, b in top10_a_unique:
        preds, probs = scheme_a_seed42_preds[(m, b)]
        acc_ci, auc_ci = compute_bootstrap_ci(y, preds, probs, n_bootstraps=1000, random_state=42)
        ci_results["scheme_a"][f"{m} + {b}"] = {
            "accuracy_ci_95": acc_ci,
            "roc_auc_ci_95": auc_ci
        }
        
    # For Scheme B, compute on the pooled LOIO predictions
    top10_b_unique = { (r["model"], r["balancing_method"]) for r in top10_b_acc + top10_b_auc }
    for m, b in top10_b_unique:
        preds = oof_loio_preds[m][b]
        probs = oof_loio_probs[m][b]
        acc_ci, auc_ci = compute_bootstrap_ci(y, preds, probs, n_bootstraps=1000, random_state=42)
        ci_results["scheme_b"][f"{m} + {b}"] = {
            "accuracy_ci_95": acc_ci,
            "roc_auc_ci_95": auc_ci
        }
        
    (out_dir / "bootstrap_ci_top10.json").write_text(json.dumps(ci_results, indent=2), encoding="utf-8")
    print(f"Saved bootstrap CIs: {out_dir / 'bootstrap_ci_top10.json'}")
    
    # Save seed 42 predictions and LOIO predictions for plotting
    np.savez_compressed(
        out_dir / "predictions_cache.npz",
        y_true=y,
        **{f"scheme_a_prob__{m}__{b}": scheme_a_seed42_preds[(m, b)][1] for (m, b) in scheme_a_seed42_preds},
        **{f"scheme_a_pred__{m}__{b}": scheme_a_seed42_preds[(m, b)][0] for (m, b) in scheme_a_seed42_preds},
        **{f"scheme_b_prob__{m}__{b}": oof_loio_probs[m][b] for m in MODELS for b in BALANCING_METHODS if is_valid_combination(m, b)},
        **{f"scheme_b_pred__{m}__{b}": oof_loio_preds[m][b] for m in MODELS for b in BALANCING_METHODS if is_valid_combination(m, b)}
    )
    print(f"Saved predictions cache: {out_dir / 'predictions_cache.npz'}")
    print(f"\nTotal execution time: {time.time() - t_start_total:.1f}s")


if __name__ == "__main__":
    main()
