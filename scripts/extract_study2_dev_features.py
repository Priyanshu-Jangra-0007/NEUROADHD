#!/usr/bin/env python3
"""
Study 2 Development-Site Feature Extraction (Pearson r -> Fisher z).

Extracts CC200 resting-state BOLD connectivity features strictly for the 404
development-site rest_1 scans (KKI, NeuroIMAGE, OHSU, Peking 1-3) per
docs/study2/DEV_FEATURE_SET_PLAN.md and docs/study2/DEVIATIONS.md Section 3.5.

Strict Guardrails:
- Reads ONLY development-site members of ADHD200_CC200_TCs_filtfix.tar.gz.
- Strictly skips NYU, WashU, and Pittsburgh (zero NYU members read or extracted).
- Defines the valid ROI set ONCE across all 404 development scans (100% finite,
  non-zero temporal variance); shared by both primary (N=368) and no-exclusion (N=404).
- Never prints individual subject IDs.
- Saves all outputs under Git-ignored NeuroADHD/data/athena/features/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DEV_SITES_CC200 = ("KKI", "NeuroIMAGE", "OHSU", "Peking_1", "Peking_2", "Peking_3")
FISHER_CLIP_EPS = 1e-7


def parse_athena_1d_timeseries(text: str) -> Tuple[List[str], np.ndarray]:
    """
    Parse an Athena CC200 .1D time-course file (1 header row + T data rows).
    Header columns: ['File', 'Sub-brick', 'Mean_<roi_1>', ...].
    Returns (roi_names, ts_array) where ts_array has shape (T, K_raw) float64.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Expected header + >=2 time points, got {len(lines)} lines")

    header = lines[0].split()
    if len(header) < 4 or header[0] != "File" or header[1] != "Sub-brick":
        raise ValueError(f"Unexpected Athena .1D header prefix: {header[:4]}")
    roi_names = header[2:]
    k_raw = len(roi_names)

    rows: List[List[float]] = []
    for line_idx, ln in enumerate(lines[1:], start=1):
        parts = ln.split()
        if len(parts) != k_raw + 2:
            raise ValueError(
                f"Row {line_idx} column mismatch: expected {k_raw + 2}, got {len(parts)}"
            )
        rows.append([float(v) for v in parts[2:]])

    ts = np.asarray(rows, dtype=np.float64)
    return roi_names, ts


def compute_valid_roi_mask(ts_list: List[np.ndarray]) -> np.ndarray:
    """
    Define valid ROI mask ONCE across all provided development scans:
    True iff ROI k has finite values and strictly positive temporal std (> 0)
    in 100% of scans.
    """
    if not ts_list:
        raise ValueError("ts_list must be non-empty")
    k_raw = ts_list[0].shape[1]
    valid = np.ones(k_raw, dtype=bool)
    for ts in ts_list:
        if ts.shape[1] != k_raw:
            raise ValueError(f"Inconsistent ROI width: expected {k_raw}, got {ts.shape[1]}")
        finite_col = np.all(np.isfinite(ts), axis=0)
        std_col = np.std(ts, axis=0)
        valid &= finite_col & (std_col > 0.0)
    return valid


def timeseries_to_fisher_z_edges(
    ts_valid: np.ndarray,
    clip_eps: float = FISHER_CLIP_EPS,
) -> np.ndarray:
    """
    Compute K x K pairwise Pearson correlation matrix, validate symmetry/diagonal/bounds,
    clip off-diagonal values to [-1 + clip_eps, 1 - clip_eps], apply Fisher z = arctanh(r),
    and return the strict upper triangle vector of length K*(K-1)//2 (float64).
    """
    if ts_valid.ndim != 2 or ts_valid.shape[0] < 3 or ts_valid.shape[1] < 2:
        raise ValueError(f"Invalid valid-ROI time-series shape: {ts_valid.shape}")
    if not np.all(np.isfinite(ts_valid)):
        raise ValueError("Non-finite values in ts_valid")

    corr = np.corrcoef(ts_valid, rowvar=False)
    k = ts_valid.shape[1]
    if corr.shape != (k, k):
        raise ValueError(f"Unexpected correlation shape: {corr.shape}")
    if not np.all(np.isfinite(corr)):
        raise ValueError("Non-finite correlation values encountered")

    sym_err = float(np.max(np.abs(corr - corr.T)))
    if sym_err >= 1e-12:
        raise ValueError(f"Correlation matrix symmetry violation: {sym_err:.2e}")

    diag_err = float(np.max(np.abs(np.diag(corr) - 1.0)))
    if diag_err >= 1e-12:
        raise ValueError(f"Correlation matrix unit-diagonal violation: {diag_err:.2e}")

    if float(np.min(corr)) < -1.0 - 1e-12 or float(np.max(corr)) > 1.0 + 1e-12:
        raise ValueError("Correlation matrix out of [-1, 1] bounds")

    triu_i, triu_j = np.triu_indices(k, k=1)
    r_edges = np.clip(corr[triu_i, triu_j], -1.0 + clip_eps, 1.0 - clip_eps)
    z_edges = np.arctanh(r_edges).astype(np.float64)
    if not np.all(np.isfinite(z_edges)):
        raise ValueError("Non-finite Fisher z edges encountered")
    return z_edges


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_development_features(
    staging_dir: Path,
    motion_csv: Path,
    pheno_tsv: Path,
    out_dir: Path,
) -> Dict[str, object]:
    cc200_archive = staging_dir / "ADHD200_CC200_TCs_filtfix.tar.gz"
    if not cc200_archive.is_file():
        raise FileNotFoundError(f"Missing CC200 archive: {cc200_archive}")
    if not motion_csv.is_file():
        raise FileNotFoundError(f"Missing motion CSV: {motion_csv}")
    if not pheno_tsv.is_file():
        raise FileNotFoundError(f"Missing phenotype TSV: {pheno_tsv}")

    # 1. Load blinded motion metrics for the 404 development scans (ordered deterministically)
    motion_rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(motion_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["site"], row["subject_id_norm"])
            motion_rows[key] = row

    # 2. Load phenotype DX, Age, Gender strictly for joining to dev scans
    pheno_map: Dict[str, Dict[str, object]] = {}
    with open(pheno_tsv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["ScanDir ID"].strip().lstrip("0") or "0"
            dx_raw = row["DX"].strip()
            if dx_raw == "0":
                dx_bin = 0
            elif dx_raw in ("1", "2", "3"):
                dx_bin = 1
            else:
                dx_bin = -1
            age_val = float(row["Age"]) if row["Age"].strip() not in ("", "N/A", "NA") else np.nan
            sex_val = float(row["Gender"]) if row["Gender"].strip() not in ("", "N/A", "NA") else np.nan
            pheno_map[sid] = {"dx": dx_bin, "age": age_val, "gender": sex_val}

    # 3. Stream ONLY development-site filtered rest_1 CC200 .1D members (ZERO NYU read!)
    ts_by_key: Dict[Tuple[str, str], np.ndarray] = {}
    roi_names_ref: List[str] | None = None
    non_dev_read = 0

    with tarfile.open(cc200_archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            site_dir = parts[0]
            if site_dir not in DEV_SITES_CC200:
                continue  # Strictly skip NYU, WashU, Pittsburgh
            fname = parts[-1]
            if not (fname.startswith("sfnwmrda") and "rest_1" in fname and fname.endswith(".1D")):
                continue

            sub_norm = parts[1].lstrip("0") or "0"
            key = (site_dir, sub_norm)
            if key not in motion_rows:
                continue

            stream = tf.extractfile(member)
            if stream is None:
                continue
            roi_names, ts = parse_athena_1d_timeseries(
                stream.read().decode("utf-8", errors="replace")
            )
            if roi_names_ref is None:
                roi_names_ref = roi_names
            elif roi_names != roi_names_ref:
                raise ValueError("Inconsistent ROI header columns across development scans")

            # Athena .1D has 1 header row + T data rows, so T + 1 == total lines counted earlier
            expected_lines = int(motion_rows[key]["frames_cc200"])
            if ts.shape[0] + 1 != expected_lines:
                raise ValueError(
                    f"Frame count mismatch: ts rows ({ts.shape[0]}) + 1 header != {expected_lines}"
                )
            ts_by_key[key] = ts

    assert non_dev_read == 0
    if roi_names_ref is None or len(ts_by_key) != len(motion_rows):
        raise ValueError(
            f"Expected {len(motion_rows)} development scans, parsed {len(ts_by_key)}"
        )

    # Deterministic ordering: sorted by (site, integer subject_id)
    ordered_keys = sorted(ts_by_key.keys(), key=lambda kv: (kv[0], int(kv[1])))
    ts_ordered = [ts_by_key[k] for k in ordered_keys]

    # 4. Define unified valid ROI set ONCE across all 404 development scans
    valid_mask = compute_valid_roi_mask(ts_ordered)
    k_raw = len(roi_names_ref)
    k_kept = int(np.sum(valid_mask))
    k_dropped = k_raw - k_kept
    kept_roi_names = [name for name, keep in zip(roi_names_ref, valid_mask) if keep]
    dropped_roi_names = [name for name, keep in zip(roi_names_ref, valid_mask) if not keep]

    # Verify min std > 0 across all 404 scans for every kept ROI
    min_kept_std = min(float(np.min(np.std(ts[:, valid_mask], axis=0))) for ts in ts_ordered)
    assert min_kept_std > 0.0

    # 5. Compute Fisher-z upper-triangle edges for every development scan
    p_edges = k_kept * (k_kept - 1) // 2
    n_total = len(ordered_keys)
    X_all = np.empty((n_total, p_edges), dtype=np.float64)
    y_all = np.empty(n_total, dtype=np.int64)
    institutions = np.empty(n_total, dtype="<U16")
    sites = np.empty(n_total, dtype="<U16")
    primary_keep = np.empty(n_total, dtype=bool)
    motion_X = np.empty((n_total, 2), dtype=np.float64)
    age_sex_X = np.empty((n_total, 2), dtype=np.float64)
    t_lengths = np.empty(n_total, dtype=np.int64)

    for idx, key in enumerate(ordered_keys):
        site_dir, sub_norm = key
        m_row = motion_rows[key]
        p_row = pheno_map[sub_norm]
        if int(p_row["dx"]) not in (0, 1):
            raise ValueError("Missing binary DX in development cohort")

        ts_valid = ts_by_key[key][:, valid_mask]
        X_all[idx] = timeseries_to_fisher_z_edges(ts_valid)
        y_all[idx] = int(p_row["dx"])
        institutions[idx] = m_row["institution"]
        sites[idx] = site_dir
        primary_keep[idx] = int(m_row["exclude_primary_050"]) == 0
        motion_X[idx, 0] = float(m_row["mean_fd_mm"])
        motion_X[idx, 1] = float(m_row["max_disp_mm"])
        age_sex_X[idx, 0] = float(p_row["age"])
        age_sex_X[idx, 1] = float(p_row["gender"])
        t_lengths[idx] = ts_valid.shape[0]

    # Mandatory final matrix assertions
    assert X_all.shape == (404, p_edges)
    assert X_all.dtype == np.float64
    assert np.all(np.isfinite(X_all))
    assert int(np.sum(primary_keep)) == 368

    X_primary = X_all[primary_keep]
    assert X_primary.shape == (368, p_edges)
    assert np.all(np.isfinite(X_primary))

    # 6. Save under Git-ignored NeuroADHD/data/athena/features/ (NO subject IDs saved/printed)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_all_path = out_dir / "dev_cc200_fisher_z_all404.npz"
    npz_prim_path = out_dir / "dev_cc200_fisher_z_primary368.npz"
    roi_json_path = out_dir / "dev_valid_roi_set.json"

    np.savez_compressed(
        npz_all_path,
        X=X_all,
        y=y_all,
        institutions=institutions,
        sites=sites,
        primary_keep=primary_keep,
        motion_X=motion_X,
        age_sex_X=age_sex_X,
        t_lengths=t_lengths,
    )
    np.savez_compressed(
        npz_prim_path,
        X=X_primary,
        y=y_all[primary_keep],
        institutions=institutions[primary_keep],
        sites=sites[primary_keep],
        motion_X=motion_X[primary_keep],
        age_sex_X=age_sex_X[primary_keep],
        t_lengths=t_lengths[primary_keep],
    )
    roi_meta = {
        "roi_rule": "finite and std > 0 in 100% of all 404 development scans (shared by N=368 and N=404)",
        "k_raw": k_raw,
        "k_kept": k_kept,
        "k_dropped": k_dropped,
        "p_edges": p_edges,
        "min_kept_temporal_std": min_kept_std,
        "kept_roi_names": kept_roi_names,
        "dropped_roi_names": dropped_roi_names,
        "kept_roi_indices": [int(i) for i in np.where(valid_mask)[0]],
    }
    with open(roi_json_path, "w", encoding="utf-8") as f:
        json.dump(roi_meta, f, indent=2)

    hashes = {
        npz_all_path.name: sha256_file(npz_all_path),
        npz_prim_path.name: sha256_file(npz_prim_path),
        roi_json_path.name: sha256_file(roi_json_path),
    }
    sizes = {
        npz_all_path.name: npz_all_path.stat().st_size,
        npz_prim_path.name: npz_prim_path.stat().st_size,
        roi_json_path.name: roi_json_path.stat().st_size,
    }

    return {
        "n_all": n_total,
        "n_primary": int(np.sum(primary_keep)),
        "k_raw": k_raw,
        "k_kept": k_kept,
        "k_dropped": k_dropped,
        "p_edges": p_edges,
        "shape_all": list(X_all.shape),
        "shape_primary": list(X_primary.shape),
        "age_sex_missing": int(np.sum(~np.isfinite(age_sex_X))),
        "motion_missing": int(np.sum(~np.isfinite(motion_X))),
        "min_kept_std": min_kept_std,
        "dropped_roi_names": dropped_roi_names,
        "file_sizes_bytes": sizes,
        "file_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Study 2 development Fisher-z CC200 connectivity features (no NYU)."
    )
    parser.add_argument("--staging-dir", type=Path, default=Path("NeuroADHD/data/athena/staging"))
    parser.add_argument(
        "--motion-csv",
        type=Path,
        default=Path("NeuroADHD/data/athena/motion/dev_niak_fd_metrics.csv"),
    )
    parser.add_argument(
        "--pheno-tsv",
        type=Path,
        default=Path("NeuroADHD/data/athena/phenotypes/adhd200_preprocessed_phenotypics.tsv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("NeuroADHD/data/athena/features"))
    args = parser.parse_args()

    res = extract_development_features(
        args.staging_dir, args.motion_csv, args.pheno_tsv, args.out_dir
    )
    print("=== Study 2 Development Feature Extraction Complete (Counts Only, Zero NYU) ===")
    print(f"ROIs raw: {res['k_raw']} | ROIs kept (100% of 404 dev scans): {res['k_kept']} | Dropped: {res['k_dropped']}")
    print(f"Dropped ROI names: {res['dropped_roi_names']}")
    print(f"Upper-triangle Fisher-z edges (P): {res['p_edges']}")
    print(f"Shape X_all_404:     {tuple(res['shape_all'])} (0 NaN, 0 Inf)")
    print(f"Shape X_primary_368: {tuple(res['shape_primary'])} (0 NaN, 0 Inf)")
    print(f"Missing values -> motion_X: {res['motion_missing']} | age_sex_X: {res['age_sex_missing']}")
    print(f"Min temporal std across kept ROIs: {res['min_kept_std']:.6f}")
    print("\nSaved Files (Git-ignored under NeuroADHD/data/athena/features/):")
    for fname, sha in res["file_sha256"].items():
        sz = res["file_sizes_bytes"][fname]
        print(f"  {fname:36s} | {sz:10,d} bytes | self-computed SHA-256: {sha}")


if __name__ == "__main__":
    main()
