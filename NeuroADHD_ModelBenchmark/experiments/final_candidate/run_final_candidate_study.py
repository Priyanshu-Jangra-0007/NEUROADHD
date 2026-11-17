"""Comprehensive Final Candidate Benchmark on Config 2 (118 subjects).

Optimized and Resumable Implementation:
- Feature space: Pearson (19,900), Partial corr (19,900), Tangent-space embeddings (19,900)
- Feature selection: SelectKBest(f_classif) with k in {50, 100, 200}
- Covariates: With Age/Sex (standardized in-fold age + binary sex) vs without
- Base classifiers (Top 30% configuration space, 7 models):
  * L2-LogReg C=0.01 balanced
  * L2-LogReg C=0.1 balanced
  * LinearSVC C=0.01 balanced (calibrated)
  * LinearSVC C=0.1 balanced (calibrated)
  * RBFSVM C=1.0 balanced (probability=True)
  * Random Forest (50 trees, max_depth=3, balanced)
  * Extra Trees (50 trees, max_depth=3, balanced)
  Total inner configurations per fold = 3 x 3 x 2 x 7 = 126 configurations.

Evaluation & Resumability:
- Checkpoints to study_checkpoint.json after every outer fold and every 10 permutations.
- LOIO CV (4 institutions: KKI, NYU, NeuroIMAGE, Peking).
- Repeated Stratified 5-Fold CV (5 repeats, seeds 42..46).
- 200 within-institution label permutations (re-running full inner selection).
- Uses all CPU cores (n_jobs = os.cpu_count() or 8).
- Measures and prints timing per fold and estimated total before continuing.
- Generates candidate_study_results.json and fits final_candidate_pipeline.joblib.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
from nilearn.connectome.connectivity_matrices import _geometric_mean, _map_eigenvalues, sym_matrix_to_vec


# ----------------------------------------------------------------------
# Metrics & Bootstrapping Helpers
# ----------------------------------------------------------------------

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


def compute_paired_bootstrap(y_true: np.ndarray, prob_final: np.ndarray, prob_covars: np.ndarray, n_bootstraps: int = 2000, random_state: int = 42):
    """Paired subject-level bootstrap test: H0: AUC(final) <= AUC(covars)."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc1 = roc_auc_score(yb, prob_final[idx])
        auc2 = roc_auc_score(yb, prob_covars[idx])
        diffs.append(auc1 - auc2)
        
    diffs = np.array(diffs)
    delta_mean = float(np.mean(diffs))
    delta_ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    p_val = float(np.mean(diffs <= 0.0))
    return {
        "delta_mean": delta_mean,
        "delta_ci_95": delta_ci,
        "p_value_one_sided": p_val
    }


# ----------------------------------------------------------------------
# Configuration Space Definition (Top 30% of Space)
# ----------------------------------------------------------------------

def get_base_classifier_specs():
    """Top performing classifier configurations based on systematic follow-up studies."""
    return [
        {"clf_family": "LogReg", "C": 0.01, "class_weight": "balanced", "name": "LogReg(C=0.01, cw=balanced)"},
        {"clf_family": "LogReg", "C": 0.1, "class_weight": "balanced", "name": "LogReg(C=0.1, cw=balanced)"},
        {"clf_family": "LinearSVM", "C": 0.01, "class_weight": "balanced", "name": "LinearSVM(C=0.01, cw=balanced)"},
        {"clf_family": "LinearSVM", "C": 0.1, "class_weight": "balanced", "name": "LinearSVM(C=0.1, cw=balanced)"},
        {"clf_family": "RBFSVM", "C": 1.0, "class_weight": "balanced", "name": "RBFSVM(C=1.0, cw=balanced)"},
        {"clf_family": "RF", "n_estimators": 50, "max_depth": 3, "class_weight": "balanced", "name": "RF(n=50, d=3, cw=balanced)"},
        {"clf_family": "ExtraTrees", "n_estimators": 50, "max_depth": 3, "class_weight": "balanced", "name": "ExtraTrees(n=50, d=3, cw=balanced)"},
    ]


def instantiate_classifier(spec: dict, seed: int):
    cw = spec.get("class_weight")
    fam = spec["clf_family"]
    if fam == "LogReg":
        return LogisticRegression(C=spec["C"], solver="lbfgs", max_iter=500, class_weight=cw, random_state=seed)
    elif fam == "LinearSVM":
        base_svc = LinearSVC(dual=False, C=spec["C"], class_weight=cw, random_state=seed)
        return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=3)
    elif fam == "RBFSVM":
        return SVC(kernel="rbf", C=spec["C"], probability=True, class_weight=cw, random_state=seed)
    elif fam == "RF":
        return RandomForestClassifier(n_estimators=spec["n_estimators"], max_depth=spec["max_depth"], class_weight=cw, random_state=seed, n_jobs=1)
    elif fam == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=spec["n_estimators"], max_depth=spec["max_depth"], class_weight=cw, random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown family: {fam}")


# ----------------------------------------------------------------------
# Resumable Checkpointing Helpers
# ----------------------------------------------------------------------

def save_checkpoint(data: dict, checkpoint_path: Path):
    """Atomically write checkpoint to disk."""
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load existing checkpoint if available."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[CHECKPOINT] Loaded existing checkpoint from {checkpoint_path} (updated {data.get('updated_at')})")
            return data
        except Exception as e:
            print(f"[CHECKPOINT WARN] Error reading {checkpoint_path}: {e}")
            return None
    return None


# ----------------------------------------------------------------------
# Candidate Benchmark Engine with Fully Precomputed Fold Manifolds
# ----------------------------------------------------------------------

class CandidateStudyEngine:
    def __init__(self, data_cache_dir: Path):
        self.data_cache_dir = data_cache_dir
        self.meta = pd.read_csv(data_cache_dir / "metadata.csv")
        self.y = self.meta["diagnosis"].to_numpy(dtype=int)
        self.institutions = self.meta["institution"].to_numpy()
        self.sites = self.meta["site"].to_numpy()
        self.age = self.meta["age_years"].to_numpy(dtype=float)
        self.sex = (self.meta["sex"] == "Male").astype(float).to_numpy()
        
        self.X_pearson = np.load(data_cache_dir / "X_pearson.npy")
        self.X_partial = np.load(data_cache_dir / "X_partial.npy")
        self.covs = np.load(data_cache_dir / "covs.npy")
        self.covs = np.array([0.5 * (c + c.T) for c in self.covs])
        
        self.base_specs = get_base_classifier_specs()
        
        # Precompute in-fold manifold features for LOIO outer and inner folds
        self.loio_cached_folds = self._build_loio_feature_cache()

    def _extract_scaled_conn(self, conn_type: str, tr_idx: np.ndarray, val_idx: np.ndarray):
        """Extract and scale connectivity features strictly on training fold."""
        if conn_type == "pearson":
            X_tr_raw = self.X_pearson[tr_idx]
            X_val_raw = self.X_pearson[val_idx]
        elif conn_type == "partial":
            X_tr_raw = self.X_partial[tr_idx]
            X_val_raw = self.X_partial[val_idx]
        elif conn_type == "tangent":
            covs_tr = self.covs[tr_idx]
            covs_val = self.covs[val_idx]
            mean_cov = _geometric_mean(covs_tr, max_iter=25, tol=1e-5)
            whitening = _map_eigenvalues(lambda x: 1.0 / np.sqrt(x), mean_cov)
            mats_tr = [_map_eigenvalues(np.log, whitening.dot(c).dot(whitening)) for c in covs_tr]
            mats_val = [_map_eigenvalues(np.log, whitening.dot(c).dot(whitening)) for c in covs_val]
            X_tr_raw = sym_matrix_to_vec(np.array(mats_tr), discard_diagonal=True)
            X_val_raw = sym_matrix_to_vec(np.array(mats_val), discard_diagonal=True)
        else:
            raise ValueError(f"Unknown conn_type: {conn_type}")
            
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr_raw)
        X_val_imp = imp.transform(X_val_raw)
        
        vt = VarianceThreshold()
        X_tr_vt = vt.fit_transform(X_tr_imp)
        X_val_vt = vt.transform(X_val_imp)
        
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_vt)
        X_val_sc = sc.transform(X_val_vt)
        return X_tr_sc, X_val_sc

    def _build_loio_feature_cache(self):
        """Precompute outer and inner split matrices for all 4 LOIO folds, with disk caching."""
        cache_file = self.data_cache_dir / "loio_cached_folds.joblib"
        if cache_file.exists():
            print(f"Loading precomputed LOIO manifolds from {cache_file}...")
            return load(cache_file)

        print("Precomputing all outer and inner split manifolds for LOIO...")
        t0 = time.time()
        unique_insts = sorted(list(set(self.institutions)))
        cached_folds = []
        
        for inst in unique_insts:
            outer_te = np.where(self.institutions == inst)[0]
            outer_tr = np.where(self.institutions != inst)[0]
            
            # Outer features
            outer_conn = {}
            for ct in ["pearson", "partial", "tangent"]:
                outer_conn[ct] = self._extract_scaled_conn(ct, outer_tr, outer_te)
                
            asc = StandardScaler()
            age_tr = asc.fit_transform(self.age[outer_tr].reshape(-1, 1))
            age_te = asc.transform(self.age[outer_te].reshape(-1, 1))
            outer_cov_tr = np.hstack([age_tr, self.sex[outer_tr].reshape(-1, 1)])
            outer_cov_te = np.hstack([age_te, self.sex[outer_te].reshape(-1, 1)])
            
            # Inner 3-fold splits on outer_tr (stratified by training labels)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            inner_splits_idx = list(skf.split(np.zeros(len(outer_tr)), self.y[outer_tr]))
            
            inner_splits_data = []
            for in_tr_local, in_val_local in inner_splits_idx:
                in_tr_global = outer_tr[in_tr_local]
                in_val_global = outer_tr[in_val_local]
                
                # Precompute inner connectivity for all 3 types
                in_conn = {}
                for ct in ["pearson", "partial", "tangent"]:
                    in_conn[ct] = self._extract_scaled_conn(ct, in_tr_global, in_val_global)
                    
                asc_in = StandardScaler()
                age_in_tr = asc_in.fit_transform(self.age[in_tr_global].reshape(-1, 1))
                age_in_val = asc_in.transform(self.age[in_val_global].reshape(-1, 1))
                cov_in_tr = np.hstack([age_in_tr, self.sex[in_tr_global].reshape(-1, 1)])
                cov_in_val = np.hstack([age_in_val, self.sex[in_val_global].reshape(-1, 1)])
                
                inner_splits_data.append({
                    "in_tr_local": in_tr_local,
                    "in_val_local": in_val_local,
                    "in_tr_global": in_tr_global,
                    "in_val_global": in_val_global,
                    "in_conn": in_conn,
                    "cov_in_tr": cov_in_tr,
                    "cov_in_val": cov_in_val
                })
                
            cached_folds.append({
                "inst": inst,
                "outer_tr": outer_tr,
                "outer_te": outer_te,
                "outer_conn": outer_conn,
                "outer_cov_tr": outer_cov_tr,
                "outer_cov_te": outer_cov_te,
                "inner_splits": inner_splits_data
            })
            
        print(f"Precomputed LOIO manifolds in {time.time()-t0:.2f}s. Caching to {cache_file}...")
        dump(cached_folds, cache_file)
        return cached_folds

    def select_and_evaluate_loio_fold(self, fold_data: dict, y_current: np.ndarray, seed: int = 42):
        """Run inner CV selection on fold_data using y_current and predict held-out outer fold."""
        outer_tr = fold_data["outer_tr"]
        outer_te = fold_data["outer_te"]
        y_tr_full = y_current[outer_tr]
        
        conn_types = ["pearson", "partial", "tangent"]
        k_values = [50, 100, 200]
        covar_options = [True, False]
        
        cand_inner_probs = {}
        
        for conn_type in conn_types:
            for sdata in fold_data["inner_splits"]:
                y_in_tr = y_current[sdata["in_tr_global"]]
                y_in_val = y_current[sdata["in_val_global"]]
                in_val_local = sdata["in_val_local"]
                
                X_c_tr, X_c_val = sdata["in_conn"][conn_type]
                cov_tr = sdata["cov_in_tr"]
                cov_val = sdata["cov_in_val"]
                
                for k in k_values:
                    kb = SelectKBest(f_classif, k=k)
                    X_k_tr = kb.fit_transform(X_c_tr, y_in_tr)
                    X_k_val = kb.transform(X_c_val)
                    
                    for use_cov in covar_options:
                        if use_cov:
                            X_tr_f = np.hstack([X_k_tr, cov_tr])
                            X_val_f = np.hstack([X_k_val, cov_val])
                        else:
                            X_tr_f = X_k_tr
                            X_val_f = X_k_val
                            
                        for spec in self.base_specs:
                            cfg_key = f"{conn_type}__k{k}__cov{use_cov}__{spec['name']}"
                            if cfg_key not in cand_inner_probs:
                                cand_inner_probs[cfg_key] = np.zeros(len(outer_tr))
                                
                            clf = instantiate_classifier(spec, seed=seed)
                            clf.fit(X_tr_f, y_in_tr)
                            pr = clf.predict_proba(X_val_f)[:, 1]
                            cand_inner_probs[cfg_key][in_val_local] = pr
                            
        cand_aucs = {}
        for k_str, probs in cand_inner_probs.items():
            try:
                auc = float(roc_auc_score(y_tr_full, probs))
            except Exception:
                auc = 0.5
            cand_aucs[k_str] = auc
            
        sorted_cands = sorted(cand_aucs.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_cands[:3]
        
        # Soft-voting ensemble of top 3
        ens_probs = np.mean([cand_inner_probs[c[0]] for c in top3], axis=0)
        ens_auc = float(roc_auc_score(y_tr_full, ens_probs))
        
        if ens_auc > top3[0][1]:
            winner = {"type": "ensemble", "top3_keys": [c[0] for c in top3], "auc": ens_auc, "name": "Ensemble_Top3"}
        else:
            winner = {"type": "single", "key": top3[0][0], "auc": top3[0][1], "name": top3[0][0]}
            
        def fit_and_predict_key(cfg_key):
            parts = cfg_key.split("__")
            conn_type = parts[0]
            k = int(parts[1][1:])
            use_cov = parts[2].replace("cov", "") == "True"
            spec_name = parts[3]
            spec = next(s for s in self.base_specs if s["name"] == spec_name)
            
            X_tr_sc, X_te_sc = fold_data["outer_conn"][conn_type]
            kb = SelectKBest(f_classif, k=k)
            X_tr_k = kb.fit_transform(X_tr_sc, y_tr_full)
            X_te_k = kb.transform(X_te_sc)
            
            if use_cov:
                X_tr_f = np.hstack([X_tr_k, fold_data["outer_cov_tr"]])
                X_te_f = np.hstack([X_te_k, fold_data["outer_cov_te"]])
            else:
                X_tr_f = X_tr_k
                X_te_f = X_te_k
                
            clf = instantiate_classifier(spec, seed=seed)
            clf.fit(X_tr_f, y_tr_full)
            return clf.predict_proba(X_te_f)[:, 1]
            
        if winner["type"] == "single":
            prob_out = fit_and_predict_key(winner["key"])
        else:
            probs = [fit_and_predict_key(k_str) for k_str in winner["top3_keys"]]
            prob_out = np.mean(probs, axis=0)
            
        return prob_out, winner, len(cand_inner_probs) + 1


# ----------------------------------------------------------------------
# Resumable Evaluation Orchestration
# ----------------------------------------------------------------------

def run_evaluation_suite(engine: CandidateStudyEngine, checkpoint: dict, checkpoint_path: Path, seed: int = 42) -> dict:
    n_total = len(engine.y)
    unique_insts = sorted(list(set(engine.institutions)))
    
    # 1. Age/Sex Baseline
    if "covariates_only_loio" not in checkpoint:
        print("\n--- Running Age/Sex Only Baseline ---")
        covars_raw = np.hstack([engine.age.reshape(-1, 1), engine.sex.reshape(-1, 1)])
        oof_cov_loio = np.zeros(n_total)
        for inst in unique_insts:
            te = np.where(engine.institutions == inst)[0]
            tr = np.where(engine.institutions != inst)[0]
            asc = StandardScaler()
            X_tr = asc.fit_transform(covars_raw[tr])
            X_te = asc.transform(covars_raw[te])
            clf = LogisticRegression(class_weight="balanced", random_state=seed)
            clf.fit(X_tr, engine.y[tr])
            oof_cov_loio[te] = clf.predict_proba(X_te)[:, 1]
            
        cov_loio_metrics = compute_metrics(engine.y, (oof_cov_loio >= 0.5).astype(int), oof_cov_loio)
        cov_loio_ci = compute_bootstrap_ci(engine.y, (oof_cov_loio >= 0.5).astype(int), oof_cov_loio, random_state=seed)
        checkpoint["covariates_only_loio"] = {
            "metrics": cov_loio_metrics,
            "ci": cov_loio_ci,
            "probs": oof_cov_loio.tolist()
        }
        save_checkpoint(checkpoint, checkpoint_path)
        print(f"Age/Sex LOIO AUC: {cov_loio_metrics['roc_auc']:.4f} [{cov_loio_ci['roc_auc'][0]:.3f}, {cov_loio_ci['roc_auc'][1]:.3f}]")
    else:
        print("\n[RESUME] Age/Sex baseline already complete. Loaded from checkpoint.")

    oof_cov_loio = np.array(checkpoint["covariates_only_loio"]["probs"])

    # 2. Leave-One-Institution-Out CV (LOIO) with Nested Selection
    if "loio" not in checkpoint:
        checkpoint["loio"] = {
            "completed_folds": [],
            "fold_results": {},
            "oof_probs": [0.0] * n_total,
            "oof_preds": [0] * n_total,
            "selected_configs": [],
            "total_configs": 0
        }
        save_checkpoint(checkpoint, checkpoint_path)

    loio_state = checkpoint["loio"]
    completed_folds = set(loio_state.get("completed_folds", []))
    oof_loio_probs = np.array(loio_state["oof_probs"])
    oof_loio_preds = np.array(loio_state["oof_preds"])
    loio_selected = loio_state.get("selected_configs", [])

    print(f"\n--- Running Leave-One-Institution-Out Nested CV ({len(unique_insts)} Institutions) ---")
    first_fold_measured = False
    
    for fold_idx, fold_data in enumerate(engine.loio_cached_folds):
        inst = fold_data["inst"]
        if inst in completed_folds:
            print(f"  [RESUME] Fold for institution '{inst}' already completed.")
            continue
            
        te = fold_data["outer_te"]
        tr = fold_data["outer_tr"]
        print(f"  LOIO Fold {len(completed_folds)+1}/{len(unique_insts)}: Leaving out {inst} (N_val={len(te)}, N_tr={len(tr)})...")
        
        t0_f = time.time()
        prob, winner, n_configs = engine.select_and_evaluate_loio_fold(fold_data, engine.y, seed=seed)
        dt_f = time.time() - t0_f
        
        oof_loio_probs[te] = prob
        oof_loio_preds[te] = (prob >= 0.5).astype(int)
        loio_selected.append({"held_out": inst, "winner": winner["name"], "inner_auc": winner["auc"]})
        completed_folds.add(inst)
        
        loio_state["completed_folds"] = list(completed_folds)
        loio_state["oof_probs"] = oof_loio_probs.tolist()
        loio_state["oof_preds"] = oof_loio_preds.tolist()
        loio_state["selected_configs"] = loio_selected
        loio_state["total_configs"] = n_configs
        
        save_checkpoint(checkpoint, checkpoint_path)
        print(f"    Selected Winner: {winner['name']} (Inner AUC: {winner['auc']:.4f})")
        print(f"    Saved checkpoint for fold '{inst}' (completed in {dt_f:.2f}s).")
        
        if not first_fold_measured:
            rem = len(unique_insts) - len(completed_folds)
            est_rem = dt_f * rem
            print(f"    [TIMING] Measured time per LOIO fold: {dt_f:.2f}s. Estimated time for remaining {rem} folds: {est_rem:.1f}s ({est_rem/60:.2f} min).")
            first_fold_measured = True

    if "metrics" not in loio_state:
        loio_metrics = compute_metrics(engine.y, oof_loio_preds, oof_loio_probs)
        loio_ci = compute_bootstrap_ci(engine.y, oof_loio_preds, oof_loio_probs, random_state=seed)
        loio_state["metrics"] = loio_metrics
        loio_state["ci"] = loio_ci
        
        # Per-site breakdown
        site_breakdown = {}
        for inst in unique_insts:
            idx = np.where(engine.institutions == inst)[0]
            y_site = engine.y[idx]
            p_site = oof_loio_probs[idx]
            pr_site = oof_loio_preds[idx]
            n_site = len(y_site)
            n_adhd = int(np.sum(y_site))
            
            if len(np.unique(y_site)) > 1:
                auc_s = float(roc_auc_score(y_site, p_site))
                ci_s = compute_bootstrap_ci(y_site, pr_site, p_site, n_bootstraps=500, random_state=seed)
                auc_ci_s = ci_s["roc_auc"]
            else:
                auc_s = np.nan
                auc_ci_s = (np.nan, np.nan)
                
            m_s = compute_metrics(y_site, pr_site, p_site)
            site_breakdown[inst] = {
                "n_total": n_site,
                "n_adhd": n_adhd,
                "n_control": n_site - n_adhd,
                "metrics": m_s,
                "auc_ci": auc_ci_s
            }
        loio_state["site_breakdown"] = site_breakdown
        save_checkpoint(checkpoint, checkpoint_path)

    print(f"LOIO Results: AUC = {loio_state['metrics']['roc_auc']:.4f} [{loio_state['ci']['roc_auc'][0]:.3f}, {loio_state['ci']['roc_auc'][1]:.3f}]")

    # 3. Repeated Stratified 5-Fold CV (5 Repeats, Seeds 42..46)
    if "stratified_5fold" not in checkpoint:
        checkpoint["stratified_5fold"] = {
            "completed_repeats": [],
            "rep_metrics": [],
            "all_strat_probs": [],
            "all_strat_preds": []
        }
        save_checkpoint(checkpoint, checkpoint_path)

    strat_state = checkpoint["stratified_5fold"]
    completed_reps = set(strat_state.get("completed_repeats", []))
    rep_metrics = strat_state.get("rep_metrics", [])
    all_strat_probs = strat_state.get("all_strat_probs", [])
    all_strat_preds = strat_state.get("all_strat_preds", [])

    print("\n--- Running Repeated Stratified 5-Fold Nested CV (5 Repeats) ---")
    rep_seeds = list(range(42, 47))
    first_rep_measured = False

    for rep_idx, rep_seed in enumerate(rep_seeds):
        if rep_idx in completed_reps:
            print(f"  [RESUME] Repeat {rep_idx+1} (seed {rep_seed}) already complete.")
            continue
            
        print(f"  Starting Repeat {rep_idx+1}/5 (seed {rep_seed})...")
        t0_rep = time.time()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep_seed)
        oof_p = np.zeros(n_total)
        oof_pr = np.zeros(n_total)
        
        for fold_idx, (tr, te) in enumerate(skf.split(np.zeros(n_total), engine.y)):
            fold_conn = {ct: engine._extract_scaled_conn(ct, tr, te) for ct in ["pearson", "partial", "tangent"]}
            asc = StandardScaler()
            age_tr = asc.fit_transform(engine.age[tr].reshape(-1, 1))
            age_te = asc.transform(engine.age[te].reshape(-1, 1))
            cov_tr = np.hstack([age_tr, engine.sex[tr].reshape(-1, 1)])
            cov_te = np.hstack([age_te, engine.sex[te].reshape(-1, 1)])
            
            skf_in = StratifiedKFold(n_splits=3, shuffle=True, random_state=rep_seed)
            in_splits = list(skf_in.split(np.zeros(len(tr)), engine.y[tr]))
            in_sdata = []
            for in_tr_l, in_val_l in in_splits:
                in_tr_g = tr[in_tr_l]
                in_val_g = tr[in_val_l]
                in_conn = {ct: engine._extract_scaled_conn(ct, in_tr_g, in_val_g) for ct in ["pearson", "partial", "tangent"]}
                asc_in = StandardScaler()
                a_tr = asc_in.fit_transform(engine.age[in_tr_g].reshape(-1, 1))
                a_val = asc_in.transform(engine.age[in_val_g].reshape(-1, 1))
                c_tr = np.hstack([a_tr, engine.sex[in_tr_g].reshape(-1, 1)])
                c_val = np.hstack([a_val, engine.sex[in_val_g].reshape(-1, 1)])
                in_sdata.append({
                    "in_tr_local": in_tr_l, "in_val_local": in_val_l,
                    "in_tr_global": in_tr_g, "in_val_global": in_val_g,
                    "in_conn": in_conn, "cov_in_tr": c_tr, "cov_in_val": c_val
                })
                
            strat_fold_data = {
                "outer_tr": tr, "outer_te": te,
                "outer_conn": fold_conn, "outer_cov_tr": cov_tr, "outer_cov_te": cov_te,
                "inner_splits": in_sdata
            }
            
            prob, winner, _ = engine.select_and_evaluate_loio_fold(strat_fold_data, engine.y, seed=rep_seed)
            oof_p[te] = prob
            oof_pr[te] = (prob >= 0.5).astype(int)

        dt_rep = time.time() - t0_rep
        m = compute_metrics(engine.y, oof_pr, oof_p)
        rep_metrics.append(m)
        all_strat_probs.append(oof_p.tolist())
        all_strat_preds.append(oof_pr.tolist())
        completed_reps.add(rep_idx)
        
        strat_state["completed_repeats"] = list(completed_reps)
        strat_state["rep_metrics"] = rep_metrics
        strat_state["all_strat_probs"] = all_strat_probs
        strat_state["all_strat_preds"] = all_strat_preds
        save_checkpoint(checkpoint, checkpoint_path)
        
        print(f"  Repeat {rep_idx+1} (seed {rep_seed}) finished in {dt_rep:.2f}s: AUC={m['roc_auc']:.4f}, BalAcc={m['balanced_accuracy']:.4f}")
        if not first_rep_measured:
            rem_reps = len(rep_seeds) - len(completed_reps)
            est_rem_reps = dt_rep * rem_reps
            print(f"  [TIMING] Measured time per repeat (5 folds): {dt_rep:.2f}s. Estimated time for remaining {rem_reps} repeats: {est_rem_reps:.1f}s ({est_rem_reps/60:.2f} min).")
            first_rep_measured = True

    if "metrics" not in strat_state:
        mean_strat_probs = np.mean(np.array(all_strat_probs), axis=0)
        mean_strat_preds = (mean_strat_probs >= 0.5).astype(int)
        strat_metrics = compute_metrics(engine.y, mean_strat_preds, mean_strat_probs)
        strat_ci = compute_bootstrap_ci(engine.y, mean_strat_preds, mean_strat_probs, random_state=seed)
        strat_state["metrics"] = strat_metrics
        strat_state["ci"] = strat_ci
        strat_state["repeats_mean"] = {k: float(np.mean([m[k] for m in rep_metrics])) for k in strat_metrics}
        strat_state["repeats_std"] = {k: float(np.std([m[k] for m in rep_metrics])) for k in strat_metrics}
        save_checkpoint(checkpoint, checkpoint_path)

    print(f"Stratified CV: Mean AUC = {strat_state['repeats_mean']['roc_auc']:.4f} ± {strat_state['repeats_std']['roc_auc']:.4f}")

    # 4. Paired Bootstrap Test: Final Candidate Model vs Age/Sex Only
    if "paired_bootstrap_vs_covars" not in checkpoint:
        print("\n--- Running Paired Bootstrap Test (Final Model vs Age/Sex Only) ---")
        paired_boot = compute_paired_bootstrap(engine.y, oof_loio_probs, oof_cov_loio, n_bootstraps=2000, random_state=seed)
        checkpoint["paired_bootstrap_vs_covars"] = paired_boot
        save_checkpoint(checkpoint, checkpoint_path)
        print(f"  Delta AUC: {paired_boot['delta_mean']:+.4f} [{paired_boot['delta_ci_95'][0]:+.3f}, {paired_boot['delta_ci_95'][1]:+.3f}] | p = {paired_boot['p_value_one_sided']:.4f}")
    else:
        print("\n[RESUME] Paired bootstrap test already complete.")

    return checkpoint


# ----------------------------------------------------------------------
# Resumable Permutation Significance Tests
# ----------------------------------------------------------------------

def run_within_institution_permutation_significance(engine: CandidateStudyEngine, checkpoint: dict, checkpoint_path: Path, n_permutations: int = 200, random_state: int = 42) -> dict:
    """Permutation test re-running FULL selection inside each permutation, checkpointed every 10 permutations."""
    print(f"\n================================================================================")
    print(f"SIGNIFICANCE: Within-Institution Label Permutation Test ({n_permutations} Shuffles)")
    print(f"================================================================================")
    
    unique_insts = sorted(list(set(engine.institutions)))
    covars_raw = np.hstack([engine.age.reshape(-1, 1), engine.sex.reshape(-1, 1)])
    
    def eval_full_loio_selection(y_perm):
        oof_p = np.zeros(len(y_perm))
        for fold_data in engine.loio_cached_folds:
            te = fold_data["outer_te"]
            prob, _, _ = engine.select_and_evaluate_loio_fold(fold_data, y_perm, seed=42)
            oof_p[te] = prob
        return float(roc_auc_score(y_perm, oof_p))

    def eval_covars_loio(y_perm):
        oof_p = np.zeros(len(y_perm))
        for inst in unique_insts:
            te = np.where(engine.institutions == inst)[0]
            tr = np.where(engine.institutions != inst)[0]
            asc = StandardScaler()
            X_tr = asc.fit_transform(covars_raw[tr])
            X_te = asc.transform(covars_raw[te])
            clf = LogisticRegression(class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_perm[tr])
            oof_p[te] = clf.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_perm, oof_p))

    if "permutations" not in checkpoint:
        print("Evaluating observed score with full selection...")
        obs_auc = eval_full_loio_selection(engine.y)
        obs_cov_auc = eval_covars_loio(engine.y)
        checkpoint["permutations"] = {
            "completed_count": 0,
            "perm_results": [],
            "obs_auc": obs_auc,
            "obs_cov_auc": obs_cov_auc
        }
        save_checkpoint(checkpoint, checkpoint_path)
        print(f"Observed LOIO ROC-AUC: Final Model = {obs_auc:.4f}, Age/Sex = {obs_cov_auc:.4f}")

    perm_state = checkpoint["permutations"]
    obs_auc = perm_state["obs_auc"]
    obs_cov_auc = perm_state["obs_cov_auc"]
    completed_count = perm_state["completed_count"]
    perm_results = perm_state.get("perm_results", [])
    
    def single_perm_worker(idx):
        rng = np.random.RandomState(random_state + idx * 317)
        y_p = engine.y.copy()
        for inst in unique_insts:
            inst_idx = np.where(engine.institutions == inst)[0]
            y_p[inst_idx] = rng.permutation(y_p[inst_idx])
            
        auc_final = eval_full_loio_selection(y_p)
        auc_cov = eval_covars_loio(y_p)
        return auc_final, auc_cov

    n_workers = min(os.cpu_count() or 8, 8)
    batch_size = 10
    first_batch_measured = False

    while completed_count < n_permutations:
        batch_end = min(completed_count + batch_size, n_permutations)
        batch_indices = list(range(completed_count, batch_end))
        print(f"  Running permutations batch {completed_count+1} to {batch_end} of {n_permutations} across {n_workers} workers...")
        
        t0_batch = time.time()
        batch_out = Parallel(n_jobs=n_workers)(delayed(single_perm_worker)(i) for i in batch_indices)
        dt_batch = time.time() - t0_batch
        
        perm_results.extend(batch_out)
        completed_count = batch_end
        perm_state["completed_count"] = completed_count
        perm_state["perm_results"] = perm_results
        save_checkpoint(checkpoint, checkpoint_path)
        
        print(f"  [CHECKPOINT] Permutations {completed_count}/{n_permutations} complete (batch took {dt_batch:.2f}s).")
        if not first_batch_measured:
            rem_perms = n_permutations - completed_count
            est_rem_perms = (dt_batch / len(batch_indices)) * rem_perms
            print(f"  [TIMING] Measured time for 10 permutations: {dt_batch:.2f}s. Estimated time for remaining {rem_perms} permutations: {est_rem_perms:.1f}s ({est_rem_perms/60:.2f} min).")
            first_batch_measured = True

    final_null = np.array([r[0] for r in perm_results])
    cov_null = np.array([r[1] for r in perm_results])
    
    p_final = float((np.sum(final_null >= obs_auc) + 1) / (n_permutations + 1))
    p_cov = float((np.sum(cov_null >= obs_cov_auc) + 1) / (n_permutations + 1))
    
    summary = {
        "n_permutations": n_permutations,
        "final_model": {
            "observed_auc": obs_auc,
            "null_mean": float(np.mean(final_null)),
            "null_std": float(np.std(final_null)),
            "null_95th": float(np.percentile(final_null, 95)),
            "null_99th": float(np.percentile(final_null, 99)),
            "p_value": p_final
        },
        "covariates_only_model": {
            "observed_auc": obs_cov_auc,
            "null_mean": float(np.mean(cov_null)),
            "null_std": float(np.std(cov_null)),
            "null_95th": float(np.percentile(cov_null, 95)),
            "null_99th": float(np.percentile(cov_null, 99)),
            "p_value": p_cov
        }
    }
    
    perm_state["summary"] = summary
    save_checkpoint(checkpoint, checkpoint_path)
    
    print(f"\nPermutations Complete:")
    print(f"  Final Model LOIO AUC: {obs_auc:.4f} | Null: {summary['final_model']['null_mean']:.4f} ± {summary['final_model']['null_std']:.4f} | p = {p_final:.4f}")
    print(f"  Age/Sex Model LOIO AUC: {obs_cov_auc:.4f} | Null: {summary['covariates_only_model']['null_mean']:.4f} ± {summary['covariates_only_model']['null_std']:.4f} | p = {p_cov:.4f}")
    return summary


# ----------------------------------------------------------------------
# Refit Final Model on All 118 Subjects
# ----------------------------------------------------------------------

def refit_and_save_final_model(engine: CandidateStudyEngine, output_dir: Path, checkpoint: dict, checkpoint_path: Path, seed: int = 42):
    """Run full inner CV selection on all 118 subjects, refit winner, and save artifacts."""
    if "final_model_refit" in checkpoint:
        print("\n[RESUME] Final model refit already complete. Loaded from checkpoint.")
        return checkpoint["final_model_refit"]["winner_name"], Path(checkpoint["final_model_refit"]["pipeline_path"])

    print("\n================================================================================")
    print("FINAL MODEL: Refitting Winner on All 118 Subjects")
    print("================================================================================")
    all_idx = np.arange(len(engine.y))
    
    fold_conn = {ct: engine._extract_scaled_conn(ct, all_idx, all_idx) for ct in ["pearson", "partial", "tangent"]}
    asc = StandardScaler()
    age_all = asc.fit_transform(engine.age.reshape(-1, 1))
    cov_all = np.hstack([age_all, engine.sex.reshape(-1, 1)])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    inner_splits = []
    for in_tr_l, in_val_l in skf.split(np.zeros(len(all_idx)), engine.y):
        in_conn = {ct: engine._extract_scaled_conn(ct, in_tr_l, in_val_l) for ct in ["pearson", "partial", "tangent"]}
        asc_in = StandardScaler()
        a_tr = asc_in.fit_transform(engine.age[in_tr_l].reshape(-1, 1))
        a_val = asc_in.transform(engine.age[in_val_l].reshape(-1, 1))
        c_tr = np.hstack([a_tr, engine.sex[in_tr_l].reshape(-1, 1)])
        c_val = np.hstack([a_val, engine.sex[in_val_l].reshape(-1, 1)])
        inner_splits.append({
            "in_tr_local": in_tr_l, "in_val_local": in_val_l,
            "in_tr_global": in_tr_l, "in_val_global": in_val_l,
            "in_conn": in_conn, "cov_in_tr": c_tr, "cov_in_val": c_val
        })
        
    full_fold_data = {
        "outer_tr": all_idx, "outer_te": all_idx,
        "outer_conn": fold_conn, "outer_cov_tr": cov_all, "outer_cov_te": cov_all,
        "inner_splits": inner_splits
    }
    
    _, winner, _ = engine.select_and_evaluate_loio_fold(full_fold_data, engine.y, seed=seed)
    print(f"Selected Global Winner: {winner['name']} (Inner 3-Fold AUC: {winner['auc']:.4f})")
    
    def fit_single_pipeline(cfg_key):
        parts = cfg_key.split("__")
        conn_type = parts[0]
        k = int(parts[1][1:])
        use_cov = parts[2].replace("cov", "") == "True"
        spec_name = parts[3]
        spec = next(s for s in engine.base_specs if s["name"] == spec_name)
        
        X_all_sc, _ = full_fold_data["outer_conn"][conn_type]
        kb = SelectKBest(f_classif, k=k)
        X_all_k = kb.fit_transform(X_all_sc, engine.y)
        
        if use_cov:
            X_final = np.hstack([X_all_k, cov_all])
        else:
            X_final = X_all_k
            
        clf = instantiate_classifier(spec, seed=seed)
        clf.fit(X_final, engine.y)
        
        return {
            "cfg_key": cfg_key,
            "conn_type": conn_type,
            "k": k,
            "use_covars": use_cov,
            "kbest": kb,
            "classifier": clf,
            "classifier_spec": spec
        }
        
    if winner["type"] == "single":
        final_pipeline = fit_single_pipeline(winner["key"])
    else:
        final_pipeline = {
            "type": "ensemble",
            "models": [fit_single_pipeline(k_str) for k_str in winner["top3_keys"]]
        }
        
    pipeline_path = output_dir / "final_candidate_pipeline.joblib"
    dump(final_pipeline, pipeline_path)
    print(f"Saved final pipeline artifact: {pipeline_path}")
    
    checkpoint["final_model_refit"] = {
        "winner_name": winner["name"],
        "winner_auc": winner["auc"],
        "pipeline_path": str(pipeline_path)
    }
    save_checkpoint(checkpoint, checkpoint_path)
    return winner["name"], pipeline_path


# ----------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------

def main():
    t_start = time.time()
    out_dir = Path(__file__).resolve().parent
    cache_dir = out_dir / "cache"
    checkpoint_path = out_dir / "study_checkpoint.json"
    
    print("================================================================================")
    print("NEUROADHD: BEST HONEST CANDIDATE BENCHMARK (CONFIG 2: 118 SUBJECTS)")
    print("Resumable Execution · Caffeinate Safe · Checkpoint Every Outer Fold & 10 Perms")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("================================================================================")
    
    engine = CandidateStudyEngine(cache_dir)
    print(f"Loaded cohort of {len(engine.y)} subjects (ADHD: {int(engine.y.sum())}, Control: {len(engine.y) - int(engine.y.sum())}).")
    print(f"Majority-class accuracy = {float(np.max(np.bincount(engine.y)) / len(engine.y))*100:.2f}%")
    print(f"Active base classifier specifications: {len(engine.base_specs)} (Top 30% grid)")
    
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        checkpoint = {
            "status": "in_progress",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "n_subjects": len(engine.y),
            "cpu_cores": os.cpu_count() or 8
        }
        save_checkpoint(checkpoint, checkpoint_path)
        
    # 1. Run nested CV evaluation suite (LOIO + Stratified 5-Fold)
    checkpoint = run_evaluation_suite(engine, checkpoint, checkpoint_path, seed=42)
    
    # 2. Run within-institution permutation tests (200 permutations re-running full selection)
    perm_summary = run_within_institution_permutation_significance(engine, checkpoint, checkpoint_path, n_permutations=200, random_state=42)
    
    # 3. Refit final model and save
    winner_name, pipeline_path = refit_and_save_final_model(engine, out_dir, checkpoint, checkpoint_path, seed=42)
    
    # 4. Save consolidated final results
    final_results = {
        "metadata": {
            "cohort": "Config 2 (118 subjects)",
            "n_subjects": len(engine.y),
            "majority_accuracy": float(np.max(np.bincount(engine.y)) / len(engine.y)),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": float(time.time() - t_start)
        },
        "covariates_only_loio": checkpoint["covariates_only_loio"],
        "loio": checkpoint["loio"],
        "stratified_5fold": checkpoint["stratified_5fold"],
        "paired_bootstrap_vs_covars": checkpoint["paired_bootstrap_vs_covars"],
        "permutation_significance": perm_summary,
        "final_model_refit": checkpoint["final_model_refit"]
    }
    
    json_path = out_dir / "candidate_study_results.json"
    json_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    
    checkpoint["status"] = "completed"
    save_checkpoint(checkpoint, checkpoint_path)
    
    print("\n================================================================================")
    print("CANDIDATE STUDY COMPLETED SUCCESSFULLY!")
    print(f"Results JSON: {json_path}")
    print(f"Checkpoint JSON: {checkpoint_path}")
    print(f"Final Pipeline: {pipeline_path}")
    print(f"Total benchmark runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.2f} min)")
    print("================================================================================")


if __name__ == "__main__":
    main()
