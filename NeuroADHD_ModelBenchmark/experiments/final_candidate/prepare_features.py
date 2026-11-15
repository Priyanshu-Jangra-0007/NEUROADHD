"""Extract and precompute connectivity features for Config 2 (118 subjects).

Saves to cache/:
1. metadata.csv: subject_id, site, institution, diagnosis, age_years, sex
2. X_pearson.npy: (118, 19900) Pearson-Fisher z correlation edges
3. X_partial.npy: (118, 19900) Partial correlation edges via Ledoit-Wolf precision
4. covs.npy: (118, 200, 200) Regularized Ledoit-Wolf covariance matrices
5. X_tangent_global.npy: (118, 19900) Tangent-space embeddings at full-cohort mean
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from nilearn.connectome.connectivity_matrices import _geometric_mean, _map_eigenvalues, sym_matrix_to_vec


def main():
    t0 = time.time()
    cache_dir = Path("NeuroADHD_ModelBenchmark/experiments/final_candidate/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load CC200 features and phenotypes
    feat_df = pd.read_csv("NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv")
    pheno_df = pd.read_csv("ADHD200_Official/metadata/official_cc200_phenotype_match.csv")
    
    site_map = {"Peking_1": "Peking", "Peking_2": "Peking", "Peking_3": "Peking"}
    feat_df["institution"] = feat_df["site"].replace(site_map)
    
    # Config 2: Exclude zero-ADHD sites (OHSU, Pittsburgh)
    c2_df = feat_df[~feat_df["institution"].isin(["OHSU", "Pittsburgh"])].copy().reset_index(drop=True)
    
    c2_df["sid_str"] = c2_df["subject_id"].astype(str)
    pheno_df["sid_str"] = pheno_df["phenotype_subject_id"].astype(str)
    
    # Merge phenotypes
    merged = pd.merge(
        c2_df[["sid_str", "site", "institution", "diagnosis"]],
        pheno_df[["sid_str", "age_years", "sex"]],
        on="sid_str",
        how="inner"
    )
    merged.rename(columns={"sid_str": "subject_id"}, inplace=True)
    if len(merged) != 118:
        raise ValueError(f"Expected 118 subjects, found {len(merged)}")
        
    print(f"Loaded {len(merged)} subjects for Config 2.")
    print("Site breakdown:")
    for inst, grp in merged.groupby("institution"):
        n_tot = len(grp)
        n_adhd = int(grp["diagnosis"].sum())
        print(f"  {inst}: Total={n_tot}, ADHD={n_adhd} ({n_adhd/n_tot*100:.1f}%), Control={n_tot - n_adhd}")
        
    merged.to_csv(cache_dir / "metadata.csv", index=False)
    
    # 2. Pearson-Fisher z
    feature_cols = [c for c in c2_df.columns if c.startswith("ROI_")]
    X_pearson = c2_df[feature_cols].to_numpy(dtype=np.float64)
    np.save(cache_dir / "X_pearson.npy", X_pearson)
    print(f"X_pearson shape: {X_pearson.shape}, all finite: {np.isfinite(X_pearson).all()}")
    
    # 3. Load raw ROI time series and compute Ledoit-Wolf Covariance & Partial Correlation
    raw_dir = Path("NeuroADHD/data/adhd200/raw/cc200")
    c2_ids = list(merged["subject_id"].astype(str))
    
    N_ROIS = 200
    tri_i, tri_j = np.triu_indices(N_ROIS, k=1)
    
    covs = []
    partial_edges = []
    available_count = 0
    
    for sid in c2_ids:
        clean_sid = sid.lstrip("0")
        matched = list(raw_dir.glob(f"{sid}_session_1")) or list(raw_dir.glob(f"{int(sid):07d}_session_1")) or list(raw_dir.glob(f"{clean_sid}_session_1"))
        if not matched:
            raise FileNotFoundError(f"Missing scan directory for subject in Config 2")
        roi_file = matched[0] / "roi_CC200.csv"
        if not roi_file.exists():
            raise FileNotFoundError(f"Missing roi_CC200.csv for subject in Config 2")
            
        available_count += 1
        frame = pd.read_csv(roi_file)
        vals = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        ts = vals.T  # (T, 200)
        
        lw = LedoitWolf().fit(ts)
        cov = lw.covariance_
        prec = lw.precision_
        covs.append(cov)
        
        d = np.sqrt(np.diag(prec))
        part_corr = - prec / np.outer(d, d)
        np.fill_diagonal(part_corr, 1.0)
        p_edge = part_corr[tri_i, tri_j]
        z_edge = np.arctanh(np.clip(p_edge, -1.0 + 1e-7, 1.0 - 1e-7))
        partial_edges.append(z_edge)
        
    print(f"Local roi_CC200.csv files available: {available_count} / {len(c2_ids)}")
    covs = np.array(covs, dtype=np.float64)
    partial_edges = np.array(partial_edges, dtype=np.float64)
    
    np.save(cache_dir / "covs.npy", covs)
    np.save(cache_dir / "X_partial.npy", partial_edges)
    print(f"covs shape: {covs.shape}, all finite: {np.isfinite(covs).all()}")
    print(f"X_partial shape: {partial_edges.shape}, all finite: {np.isfinite(partial_edges).all()}")
    
    # 4. Tangent-space projection at global Fréchet mean
    mean_cov = _geometric_mean(covs, max_iter=25, tol=1e-6)
    whitening = _map_eigenvalues(lambda x: 1.0 / np.sqrt(x), mean_cov)
    tangent_mats = [_map_eigenvalues(np.log, whitening.dot(c).dot(whitening)) for c in covs]
    X_tangent_global = sym_matrix_to_vec(np.array(tangent_mats), discard_diagonal=True)
    np.save(cache_dir / "X_tangent_global.npy", X_tangent_global)
    print(f"X_tangent_global shape: {X_tangent_global.shape}, all finite: {np.isfinite(X_tangent_global).all()}")
    
    print(f"\nCompleted feature preparation in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
