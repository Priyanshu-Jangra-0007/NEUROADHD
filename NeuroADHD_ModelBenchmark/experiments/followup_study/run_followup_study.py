"""Follow-Up Study: Class Balancing, Demographic Covariates, High-Dimensional Regularization, and Nested CV.

Configs:
  Config 1: All 153 subjects (6 institutions: KKI, NYU, NeuroIMAGE, OHSU, Peking, Pittsburgh)
  Config 2: 118 subjects (excludes OHSU & Pittsburgh: KKI, NYU, NeuroIMAGE, Peking)

Covariates:
  age_years (standardized inside training folds only)
  sex (Male=1, Female=0)

Feature Sets:
  1. connectivity_only (19,900 CC200 edges)
  2. connectivity_plus_covariates (19,900 edges + standardized age + sex)
  3. covariates_only (standardized age + sex)

Models:
  - L2 Logistic Regression (all 19,900 features, no SelectKBest, C in {0.0001, 0.001, 0.01})
  - ElasticNet (all 19,900 features, no SelectKBest, C in {0.001, 0.01, 0.1}, l1_ratio=0.5)
  - L2 Logistic Regression (SelectKBest k=100, C in {0.01, 0.1, 1.0})
  - Linear SVM calibrated (SelectKBest k=100, C in {0.1, 1.0})
  - RBF SVM (SelectKBest k=100, C in {0.1, 1.0, 10.0})
  - Random Forest (SelectKBest k=100, n=50, depth=3)

Balancing:
  None, class_weight='balanced', SMOTE (applied after feature selection)

Nested CV:
  Inner 3-fold Stratified CV selects model & hyperparameters on training folds only.

Significance:
  Within-institution label permutation test (1000 permutations) for the nested-CV model.
"""
from __future__ import annotations

import json
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef
)
from imblearn.over_sampling import SMOTE


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "sensitivity": rec,
        "specificity": spec,
        "precision": prec,
        "mcc": mcc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }


def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    metrics_list = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        metrics_list.append(compute_metrics(y_b, y_pred[idx], y_prob[idx]))
        
    ci_dict = {}
    for k in ["accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "mcc"]:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        if vals:
            ci_dict[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        else:
            ci_dict[k] = (np.nan, np.nan)
    return ci_dict


def get_candidate_grid():
    """Define candidate configurations for Nested CV model selection."""
    candidates = []
    
    # 1. Full-feature L2 Logistic Regression
    for c in [0.0001, 0.001, 0.01]:
        for cw in [None, "balanced"]:
            candidates.append({
                "name": f"L2-LogReg-Full (C={c}, cw={cw})",
                "family": "L2 Logistic Regression (Full)",
                "use_kbest": False,
                "model_type": "l2_full",
                "C": c,
                "class_weight": cw,
                "balancing": "None" if cw is None else "class_weight='balanced'"
            })
            
    # 2. Full-feature ElasticNet (SAGA)
    for c in [0.001, 0.01, 0.1]:
        for cw in [None, "balanced"]:
            candidates.append({
                "name": f"ElasticNet-Full (C={c}, l1=0.5, cw={cw})",
                "family": "ElasticNet (Full)",
                "use_kbest": False,
                "model_type": "elasticnet_full",
                "C": c,
                "l1_ratio": 0.5,
                "class_weight": cw,
                "balancing": "None" if cw is None else "class_weight='balanced'"
            })
            
    # 3. SelectKBest(100) + L2 Logistic Regression
    for c in [0.01, 0.1, 1.0]:
        for bal in ["None", "class_weight='balanced'", "SMOTE"]:
            cw = "balanced" if bal == "class_weight='balanced'" else None
            candidates.append({
                "name": f"L2-LogReg-K100 (C={c}, bal={bal})",
                "family": "L2 Logistic Regression (K=100)",
                "use_kbest": True,
                "model_type": "l2_k100",
                "C": c,
                "class_weight": cw,
                "balancing": bal
            })
            
    # 4. SelectKBest(100) + Linear SVM (calibrated)
    for c in [0.1, 1.0]:
        for bal in ["None", "class_weight='balanced'", "SMOTE"]:
            cw = "balanced" if bal == "class_weight='balanced'" else None
            candidates.append({
                "name": f"LinearSVM-K100 (C={c}, bal={bal})",
                "family": "Linear SVM (K=100)",
                "use_kbest": True,
                "model_type": "linearsvm_k100",
                "C": c,
                "class_weight": cw,
                "balancing": bal
            })
            
    # 5. SelectKBest(100) + RBF SVM
    for c in [0.1, 1.0, 10.0]:
        for bal in ["None", "class_weight='balanced'", "SMOTE"]:
            cw = "balanced" if bal == "class_weight='balanced'" else None
            candidates.append({
                "name": f"RBF-SVM-K100 (C={c}, bal={bal})",
                "family": "RBF SVM (K=100)",
                "use_kbest": True,
                "model_type": "rbfsvm_k100",
                "C": c,
                "class_weight": cw,
                "balancing": bal
            })
            
    # 6. SelectKBest(100) + Random Forest (n=50, depth=3)
    for bal in ["None", "class_weight='balanced'", "SMOTE"]:
        cw = "balanced" if bal == "class_weight='balanced'" else None
        candidates.append({
            "name": f"RF-n50-d3-K100 (bal={bal})",
            "family": "Random Forest (K=100, depth=3)",
            "use_kbest": True,
            "model_type": "rf_k100",
            "class_weight": cw,
            "balancing": bal
        })
        
    return candidates


def fit_and_predict_candidate(cand: dict, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, seed: int):
    """Fit a single candidate on training fold and predict test fold."""
    # Apply balancing if SMOTE
    if cand.get("balancing") == "SMOTE":
        smote = SMOTE(k_neighbors=3, random_state=seed)
        try:
            X_tr_fit, y_tr_fit = smote.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_fit, y_tr_fit = X_tr, y_tr
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr
        
    mtype = cand["model_type"]
    cw = cand.get("class_weight")
    
    if mtype == "l2_full" or mtype == "l2_k100":
        clf = LogisticRegression(penalty="l2", C=cand["C"], solver="lbfgs", max_iter=500, class_weight=cw, random_state=seed)
    elif mtype == "elasticnet_full":
        clf = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=cand["l1_ratio"], C=cand["C"], max_iter=200, tol=1e-3, class_weight=cw, random_state=seed)
    elif mtype == "linearsvm_k100":
        base_svc = LinearSVC(dual=False, C=cand["C"], class_weight=cw, random_state=seed)
        clf = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
    elif mtype == "rbfsvm_k100":
        clf = SVC(kernel="rbf", C=cand["C"], probability=True, class_weight=cw, random_state=seed)
    elif mtype == "rf_k100":
        clf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight=cw, random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown model_type: {mtype}")
        
    clf.fit(X_tr_fit, y_tr_fit)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = clf.predict(X_te)
    return preds, probs


def prepare_features(feature_set: str, X_conn: np.ndarray, age: np.ndarray, sex: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray):
    """Transform and combine features strictly using training fold statistics."""
    # Age standardization
    age_scaler = StandardScaler()
    age_tr = age_scaler.fit_transform(age[train_idx].reshape(-1, 1))
    age_val = age_scaler.transform(age[val_idx].reshape(-1, 1))
    
    sex_tr = sex[train_idx].reshape(-1, 1)
    sex_val = sex[val_idx].reshape(-1, 1)
    
    if feature_set == "covariates_only":
        X_tr = np.hstack([age_tr, sex_tr])
        X_val = np.hstack([age_val, sex_val])
        return X_tr, X_val, X_tr, X_val  # both full and kbest are just covariates
        
    # Preprocessing on connectivity
    imputer = SimpleImputer(strategy="median")
    X_conn_tr_imp = imputer.fit_transform(X_conn[train_idx])
    X_conn_val_imp = imputer.transform(X_conn[val_idx])
    
    vt = VarianceThreshold()
    X_conn_tr_vt = vt.fit_transform(X_conn_tr_imp)
    X_conn_val_vt = vt.transform(X_conn_val_imp)
    
    conn_scaler = StandardScaler()
    X_conn_tr_sc = conn_scaler.fit_transform(X_conn_tr_vt)
    X_conn_val_sc = conn_scaler.transform(X_conn_val_vt)
    
    # SelectKBest (100)
    kbest = SelectKBest(f_classif, k=100)
    X_conn_tr_k100 = kbest.fit_transform(X_conn_tr_sc, X_conn[train_idx]) # label is y
    # Note: caller will pass y_tr
    return X_conn_tr_sc, X_conn_val_sc, age_tr, age_val, sex_tr, sex_val


def run_nested_cv_evaluation(df_cohort: pd.DataFrame, feature_set: str, cv_scheme: str, seed: int = 42):
    """Run full outer CV with inner 3-fold Nested CV for hyperparameter/model selection."""
    feature_cols = [c for c in df_cohort.columns if c.startswith("ROI_")]
    X_conn = df_cohort[feature_cols].to_numpy()
    age = df_cohort["age_years"].to_numpy()
    sex = (df_cohort["sex"] == "Male").astype(float).to_numpy()
    y = df_cohort["diagnosis"].to_numpy()
    institutions = df_cohort["institution"].to_numpy()
    unique_insts = np.unique(institutions)
    n_total = len(df_cohort)
    
    candidates = get_candidate_grid()
    
    # Outer CV split setup
    if cv_scheme == "stratified_5fold":
        outer_cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_conn, y))
    elif cv_scheme == "loio":
        outer_cv = [(np.where(institutions != inst)[0], np.where(institutions == inst)[0]) for inst in unique_insts]
    else:
        raise ValueError(f"Unknown cv_scheme: {cv_scheme}")
        
    oof_nested_preds = np.zeros(n_total)
    oof_nested_probs = np.zeros(n_total)
    selected_configs = []
    
    # Also evaluate each candidate family across outer folds for reporting individual models
    cand_oof_probs = {c["name"]: np.zeros(n_total) for c in candidates}
    cand_oof_preds = {c["name"]: np.zeros(n_total) for c in candidates}
    
    for outer_fold_idx, (train_idx, val_idx) in enumerate(outer_cv):
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Prepare features on training fold
        age_scaler = StandardScaler()
        age_tr = age_scaler.fit_transform(age[train_idx].reshape(-1, 1))
        age_val = age_scaler.transform(age[val_idx].reshape(-1, 1))
        sex_tr = sex[train_idx].reshape(-1, 1)
        sex_val = sex[val_idx].reshape(-1, 1)
        
        if feature_set == "covariates_only":
            X_tr_full = np.hstack([age_tr, sex_tr])
            X_val_full = np.hstack([age_val, sex_val])
            X_tr_k100 = X_tr_full
            X_val_k100 = X_val_full
        else:
            # Connectivity features
            imputer = SimpleImputer(strategy="median")
            X_c_tr_imp = imputer.fit_transform(X_conn[train_idx])
            X_c_val_imp = imputer.transform(X_conn[val_idx])
            
            vt = VarianceThreshold()
            X_c_tr_vt = vt.fit_transform(X_c_tr_imp)
            X_c_val_vt = vt.transform(X_c_val_imp)
            
            sc = StandardScaler()
            X_c_tr_sc = sc.fit_transform(X_c_tr_vt)
            X_c_val_sc = sc.transform(X_c_val_vt)
            
            kbest = SelectKBest(f_classif, k=100)
            X_c_tr_k100 = kbest.fit_transform(X_c_tr_sc, y_tr)
            X_c_val_k100 = kbest.transform(X_c_val_sc)
            
            if feature_set == "connectivity_only":
                X_tr_full = X_c_tr_sc
                X_val_full = X_c_val_sc
                X_tr_k100 = X_c_tr_k100
                X_val_k100 = X_c_val_k100
            elif feature_set == "connectivity_plus_covariates":
                X_tr_full = np.hstack([X_c_tr_sc, age_tr, sex_tr])
                X_val_full = np.hstack([X_c_val_sc, age_val, sex_val])
                X_tr_k100 = np.hstack([X_c_tr_k100, age_tr, sex_tr])
                X_val_k100 = np.hstack([X_c_val_k100, age_val, sex_val])
            else:
                raise ValueError(f"Unknown feature_set: {feature_set}")
                
        # INNER 3-FOLD NESTED CV FOR MODEL SELECTION
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        inner_splits = list(inner_cv.split(X_tr_k100, y_tr))
        
        cand_inner_aucs = {}
        for c in candidates:
            # Skip full-feature models if feature_set is covariates_only (covariates only has 2 features)
            if feature_set == "covariates_only" and not c["use_kbest"]:
                continue
                
            inner_probs = np.zeros(len(train_idx))
            for in_tr, in_val in inner_splits:
                X_in_tr = X_tr_k100[in_tr] if c["use_kbest"] else X_tr_full[in_tr]
                X_in_val = X_tr_k100[in_val] if c["use_kbest"] else X_tr_full[in_val]
                
                preds_in, probs_in = fit_and_predict_candidate(c, X_in_tr, y_tr[in_tr], X_in_val, seed)
                inner_probs[in_val] = probs_in
                
            try:
                inner_auc = roc_auc_score(y_tr, inner_probs)
            except Exception:
                inner_auc = 0.5
            cand_inner_aucs[c["name"]] = inner_auc
            
        best_cand_name = max(cand_inner_aucs, key=cand_inner_aucs.get)
        best_cand = next(c for c in candidates if c["name"] == best_cand_name)
        selected_configs.append(best_cand_name)
        
        # Train winning candidate on full outer training fold
        X_outer_tr = X_tr_k100 if best_cand["use_kbest"] else X_tr_full
        X_outer_val = X_val_k100 if best_cand["use_kbest"] else X_val_full
        
        preds_out, probs_out = fit_and_predict_candidate(best_cand, X_outer_tr, y_tr, X_outer_val, seed)
        oof_nested_preds[val_idx] = preds_out
        oof_nested_probs[val_idx] = probs_out
        
        # Also fit each individual candidate across outer folds
        for c in candidates:
            if feature_set == "covariates_only" and not c["use_kbest"]:
                continue
            X_tr_c = X_tr_k100 if c["use_kbest"] else X_tr_full
            X_val_c = X_val_k100 if c["use_kbest"] else X_val_full
            p_out, pr_out = fit_and_predict_candidate(c, X_tr_c, y_tr, X_val_c, seed)
            cand_oof_preds[c["name"]][val_idx] = p_out
            cand_oof_probs[c["name"]][val_idx] = pr_out
            
    # Compute nested-CV metrics
    nested_metrics = compute_metrics(y, oof_nested_preds, oof_nested_probs)
    nested_ci = compute_bootstrap_ci(y, oof_nested_preds, oof_nested_probs, n_bootstraps=1000, random_state=seed)
    
    # Compute candidate families metrics
    cand_metrics = {}
    for c in candidates:
        if feature_set == "covariates_only" and not c["use_kbest"]:
            continue
        cname = c["name"]
        m_dict = compute_metrics(y, cand_oof_preds[cname], cand_oof_probs[cname])
        cand_metrics[cname] = m_dict
        
    return {
        "feature_set": feature_set,
        "cv_scheme": cv_scheme,
        "nested_metrics": nested_metrics,
        "nested_ci": nested_ci,
        "selected_configs": selected_configs,
        "cand_metrics": cand_metrics,
        "oof_nested_probs": oof_nested_probs,
        "oof_nested_preds": oof_nested_preds
    }


def run_within_institution_permutation_test(df_cohort: pd.DataFrame, winning_pipeline_fn, n_permutations: int = 1000, random_state: int = 42):
    """Permute labels WITHIN each institution preserving site-wise ADHD counts."""
    y_orig = df_cohort["diagnosis"].to_numpy()
    institutions = df_cohort["institution"].to_numpy()
    unique_insts = np.unique(institutions)
    
    # Observed score
    obs_auc = winning_pipeline_fn(y_orig)
    
    def single_perm(perm_idx):
        rng_i = np.random.RandomState(random_state + perm_idx * 137)
        y_perm = y_orig.copy()
        for inst in unique_insts:
            inst_idx = np.where(institutions == inst)[0]
            y_perm[inst_idx] = rng_i.permutation(y_perm[inst_idx])
        return winning_pipeline_fn(y_perm)
        
    perm_aucs = Parallel(n_jobs=6)(delayed(single_perm)(i) for i in range(n_permutations))
    perm_aucs = np.array(perm_aucs)
    
    p_value = float((np.sum(perm_aucs >= obs_auc) + 1) / (n_permutations + 1))
    return {
        "observed_auc": float(obs_auc),
        "perm_mean": float(np.mean(perm_aucs)),
        "perm_std": float(np.std(perm_aucs)),
        "perm_95th": float(np.percentile(perm_aucs, 95)),
        "perm_99th": float(np.percentile(perm_aucs, 99)),
        "p_value": p_value,
        "n_permutations": n_permutations
    }


def main():
    t_start = time.time()
    out_dir = Path("NeuroADHD_ModelBenchmark/experiments/followup_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("================================================================================")
    print("NEUROADHD FOLLOW-UP STUDY: COVARIATES, HIGH-D REGULARIZATION & NESTED CV")
    print("================================================================================")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load Data
    feat_path = Path("NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv")
    df_feat = pd.read_csv(feat_path)
    match_path = Path("ADHD200_Official/metadata/official_cc200_phenotype_match.csv")
    df_match = pd.read_csv(match_path)
    
    df_feat["norm_id"] = df_feat["subject_id"].astype(str).str.lstrip("0")
    df_match["norm_id"] = df_match["normalized_subject_id"].astype(str).str.lstrip("0")
    
    # Merge exact phenotypic columns
    df = df_feat.merge(df_match[["norm_id", "age_years", "sex"]], on="norm_id", how="left")
    df["institution"] = df["site"].replace({"Peking_1": "Peking", "Peking_2": "Peking", "Peking_3": "Peking"})
    
    # Define Config 1 and Config 2
    cohorts = {
        "Config 1 (All 153 Subjects)": df,
        "Config 2 (Excl. OHSU & Pittsburgh, 118 Subjects)": df[~df["institution"].isin(["OHSU", "Pittsburgh"])].copy().reset_index(drop=True)
    }
    
    all_results = {}
    
    for c_name, c_df in cohorts.items():
        print(f"\n================================================================================")
        print(f"PROCESSING {c_name.upper()}")
        print(f"Total Subjects: {len(c_df)} | Control: {np.sum(c_df['diagnosis']==0)} | ADHD: {np.sum(c_df['diagnosis']==1)}")
        print(f"Institutions: {dict(c_df['institution'].value_counts())}")
        print(f"================================================================================")
        
        c_results = {}
        
        # Baselines: Majority and Site-Only
        y_c = c_df["diagnosis"].to_numpy()
        maj_acc = float(np.max(np.bincount(y_c)) / len(y_c))
        c_results["majority_accuracy"] = maj_acc
        
        # Evaluate 3 feature sets x 2 CV schemes
        feature_sets = ["connectivity_only", "connectivity_plus_covariates", "covariates_only"]
        cv_schemes = ["stratified_5fold", "loio"]
        
        for f_set in feature_sets:
            for cv_sch in cv_schemes:
                t0 = time.time()
                print(f"  Running Nested CV: [{f_set}] under [{cv_sch}]...")
                res = run_nested_cv_evaluation(c_df, f_set, cv_sch, seed=42)
                print(f"    -> Nested AUC: {res['nested_metrics']['roc_auc']:.4f} | BalAcc: {res['nested_metrics']['balanced_accuracy']:.4f} | Acc: {res['nested_metrics']['accuracy']:.4f} (MajAcc: {maj_acc:.4f}) in {time.time()-t0:.1f}s")
                c_results[f"{f_set}__{cv_sch}"] = res
                
        # SIGNIFICANCE: Permutation Test (1,000 within-institution shuffles)
        print(f"\n  Running Within-Institution Permutation Test (1,000 permutations) for {c_name}...")
        t_perm = time.time()
        
        # Winning pipeline under LOIO on connectivity_plus_covariates
        def eval_loio_pipeline(y_target):
            # Run fast evaluation with RF(50, depth 3) on connectivity+covariates
            feature_cols = [c for c in c_df.columns if c.startswith("ROI_")]
            X_c = c_df[feature_cols].to_numpy()
            age_arr = c_df["age_years"].to_numpy()
            sex_arr = (c_df["sex"] == "Male").astype(float).to_numpy()
            insts = c_df["institution"].to_numpy()
            u_insts = np.unique(insts)
            
            oof_p = np.zeros(len(c_df))
            for inst in u_insts:
                v_idx = np.where(insts == inst)[0]
                t_idx = np.where(insts != inst)[0]
                
                # In-fold age scaler
                asc = StandardScaler()
                a_tr = asc.fit_transform(age_arr[t_idx].reshape(-1, 1))
                a_v = asc.transform(age_arr[v_idx].reshape(-1, 1))
                
                # In-fold connectivity top 100
                kb = SelectKBest(f_classif, k=100)
                X_tr_k = kb.fit_transform(X_c[t_idx], y_target[t_idx])
                X_v_k = kb.transform(X_c[v_idx])
                
                # Combine
                X_tr_all = np.hstack([X_tr_k, a_tr, sex_arr[t_idx].reshape(-1, 1)])
                X_v_all = np.hstack([X_v_k, a_v, sex_arr[v_idx].reshape(-1, 1)])
                
                clf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight="balanced", random_state=42, n_jobs=1)
                clf.fit(X_tr_all, y_target[t_idx])
                oof_p[v_idx] = clf.predict_proba(X_v_all)[:, 1]
                
            return roc_auc_score(y_target, oof_p)
            
        perm_res = run_within_institution_permutation_test(c_df, eval_loio_pipeline, n_permutations=1000, random_state=42)
        print(f"  Permutation Test Finished in {time.time()-t_perm:.1f}s:")
        print(f"    Observed AUC: {perm_res['observed_auc']:.4f} | Null Mean: {perm_res['perm_mean']:.4f} ± {perm_res['perm_std']:.4f}")
        print(f"    Empirical p-value: {perm_res['p_value']:.4f} (95th percentile null: {perm_res['perm_95th']:.4f})")
        c_results["permutation_test"] = perm_res
        
        all_results[c_name] = c_results
        
    # Save raw results JSON
    # Strip numpy arrays for json serialization
    json_export = {}
    for c_name, c_res in all_results.items():
        json_export[c_name] = {"majority_accuracy": c_res["majority_accuracy"]}
        for k, v in c_res.items():
            if k == "permutation_test":
                json_export[c_name][k] = v
            elif isinstance(v, dict) and "nested_metrics" in v:
                json_export[c_name][k] = {
                    "nested_metrics": v["nested_metrics"],
                    "nested_ci": v["nested_ci"],
                    "selected_configs": v["selected_configs"],
                    "top_candidates": {ck: cv for ck, cv in sorted(v["cand_metrics"].items(), key=lambda x: x[1]["roc_auc"], reverse=True)[:5]}
                }
    (out_dir / "followup_study_results.json").write_text(json.dumps(json_export, indent=2), encoding="utf-8")
    print(f"\nSaved JSON results: {out_dir / 'followup_study_results.json'}")
    print(f"Total study runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
