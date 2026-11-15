"""Recompute Within-Institution Permutations with Distinct Random Seeds."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def eval_loio_pipeline_for_cohort(c_df: pd.DataFrame, y_target: np.ndarray) -> float:
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
        
        asc = StandardScaler()
        a_tr = asc.fit_transform(age_arr[t_idx].reshape(-1, 1))
        a_v = asc.transform(age_arr[v_idx].reshape(-1, 1))
        
        kb = SelectKBest(f_classif, k=100)
        X_tr_k = kb.fit_transform(X_c[t_idx], y_target[t_idx])
        X_v_k = kb.transform(X_c[v_idx])
        
        X_tr_all = np.hstack([X_tr_k, a_tr, sex_arr[t_idx].reshape(-1, 1)])
        X_v_all = np.hstack([X_v_k, a_v, sex_arr[v_idx].reshape(-1, 1)])
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight="balanced", random_state=42, n_jobs=1)
        clf.fit(X_tr_all, y_target[t_idx])
        oof_p[v_idx] = clf.predict_proba(X_v_all)[:, 1]
        
    return float(roc_auc_score(y_target, oof_p))


def run_within_institution_perm(c_df: pd.DataFrame, n_permutations: int = 1000, seed: int = 42):
    y_orig = c_df["diagnosis"].to_numpy()
    institutions = c_df["institution"].to_numpy()
    unique_insts = np.unique(institutions)
    
    obs_auc = eval_loio_pipeline_for_cohort(c_df, y_orig)
    
    def single_perm(perm_idx: int):
        rng_i = np.random.RandomState(seed + perm_idx * 137)
        y_perm = y_orig.copy()
        for inst in unique_insts:
            inst_idx = np.where(institutions == inst)[0]
            y_perm[inst_idx] = rng_i.permutation(y_perm[inst_idx])
        return eval_loio_pipeline_for_cohort(c_df, y_perm)
        
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
    out_dir = Path("NeuroADHD_ModelBenchmark/experiments/followup_study")
    results = json.loads((out_dir / "followup_study_results.json").read_text())

    feat_path = Path("NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv")
    df_feat = pd.read_csv(feat_path)
    match_path = Path("ADHD200_Official/metadata/official_cc200_phenotype_match.csv")
    df_match = pd.read_csv(match_path)

    df_feat["norm_id"] = df_feat["subject_id"].astype(str).str.lstrip("0")
    df_match["norm_id"] = df_match["normalized_subject_id"].astype(str).str.lstrip("0")

    df = df_feat.merge(df_match[["norm_id", "age_years", "sex"]], on="norm_id", how="left")
    df["institution"] = df["site"].replace({"Peking_1": "Peking", "Peking_2": "Peking", "Peking_3": "Peking"})

    cohorts = {
        "Config 1 (All 153 Subjects)": df,
        "Config 2 (Excl. OHSU & Pittsburgh, 118 Subjects)": df[~df["institution"].isin(["OHSU", "Pittsburgh"])].copy().reset_index(drop=True)
    }

    for c_name, c_df in cohorts.items():
        print(f"Recomputing permutation test for {c_name}...")
        t0 = time.time()
        perm_res = run_within_institution_perm(c_df, n_permutations=1000, seed=42)
        print(f"Done in {time.time()-t0:.1f}s:")
        print(f"  Observed AUC: {perm_res['observed_auc']:.4f}")
        print(f"  Null Mean: {perm_res['perm_mean']:.4f} ± {perm_res['perm_std']:.4f}")
        print(f"  Null 95th Percentile: {perm_res['perm_95th']:.4f}")
        print(f"  Empirical p-value: {perm_res['p_value']:.4f}")
        results[c_name]["permutation_test"] = perm_res

    (out_dir / "followup_study_results.json").write_text(json.dumps(results, indent=2))
    print("Updated followup_study_results.json")


if __name__ == "__main__":
    main()
