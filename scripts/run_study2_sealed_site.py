#!/usr/bin/env python3
"""
Study 2 Sealed External Validation Runner (NYU — One-Shot Scoring).

Implements docs/study2/SEALED_RUN_PLAN.md strictly per docs/study2/PREREGISTRATION.md
(SHA-256: e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec):
- Imports frozen pipeline & statistical functions from scripts/run_study2_dev_analysis.py
  and frozen motion/feature functions from scripts/compute_athena_fd.py and
  scripts/extract_study2_dev_features.py without modifying them.
- Refuses to read NYU data unless BOTH:
    1) --execute-real-sealed is explicitly passed, AND
    2) local annotated Git tag `study2-sealed-frozen` is an ancestor of HEAD.
- Refuses to overwrite an existing sealed results file.
- Refits once on pooled primary_368 -> scores once on primary-FD-filtered NYU (governs Section 8.2 zone).
- Refits once on pooled no_exclusion_404 -> scores once on all complete NYU scans (Section 5.4 sensitivity).
- Applies Section 5.1 zero-edge imputation for any missing/constant NYU ROI without re-selecting features.
- Never prints individual subject IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import fisher_exact

from scripts.compute_athena_fd import (
    DUMMY_FRAMES_DROPPED,
    HEAD_RADIUS_MM,
    center_shift_transf,
    compute_power_fd,
    extract_6dof_params,
    parse_octave_transf_text,
    read_nifti_volume_center_world,
)
from scripts.extract_study2_dev_features import (
    FISHER_CLIP_EPS,
    parse_athena_1d_timeseries,
    timeseries_to_fisher_z_edges,
)
from scripts.run_study2_dev_analysis import (
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_K_BEST,
    LOCKED_RANDOM_STATE,
    build_linear_baseline_pipeline,
    build_locked_fmri_pipeline,
    stratified_bootstrap_auc,
)

REQUIRED_FREEZE_TAG = "study2-sealed-frozen"
DEFAULT_SEALED_RESULTS_JSON = Path("NeuroADHD/data/athena/results/study2_sealed_results.json")


def classify_section_8_2_zone(auc: float, ci_lo: float) -> str:
    """
    Classify sealed-site ROC-AUC into the registered Section 8.2 outcome zones:
      - 'confirmed':    95% bootstrap CI lower bound > 0.50
      - 'inconclusive': point estimate > 0.50, and 95% CI lower bound <= 0.50
      - 'no_evidence':  point estimate <= 0.50
    """
    if ci_lo > 0.50:
        return "confirmed"
    if auc > 0.50:
        return "inconclusive"
    return "no_evidence"


def check_freeze_tag_is_ancestor(
    tag_name: str = REQUIRED_FREEZE_TAG,
    repo_root: Path | None = None,
) -> bool:
    """Return True iff `tag_name` exists in Git and is an ancestor of HEAD."""
    cwd = str(repo_root) if repo_root is not None else None
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag_name, "HEAD"],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def enforce_sealed_guards(
    execute_real_sealed: bool,
    out_json: Path,
    tag_name: str = REQUIRED_FREEZE_TAG,
    repo_root: Path | None = None,
) -> None:
    """Enforce all three hard guards prior to reading any NYU file."""
    if not execute_real_sealed:
        raise RuntimeError(
            "Refusing to read NYU data without --execute-real-sealed. "
            "Run synthetic tests in tests/test_run_study2_sealed_site.py instead."
        )
    if not check_freeze_tag_is_ancestor(tag_name=tag_name, repo_root=repo_root):
        raise RuntimeError(
            f"Refusing to read NYU data: Git tag '{tag_name}' does not exist on an ancestor of HEAD."
        )
    if out_json.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing sealed results file: {out_json}. "
            "Sealed evaluation is strictly one-shot."
        )


def sealed_timeseries_to_fisher_z_edges(
    roi_names: Sequence[str],
    ts_raw: np.ndarray,
    dev_kept_roi_names: Sequence[str],
    clip_eps: float = FISHER_CLIP_EPS,
) -> Tuple[np.ndarray, int]:
    """
    Map a sealed-site subject's (T, K_raw) time series onto the frozen development ROI set
    (`dev_kept_roi_names`, length K) and compute Fisher-z upper-triangle edges.

    Per PREREGISTRATION.md Section 5.1:
    - If every ROI in `dev_kept_roi_names` is present, finite, and has temporal std > 0,
      delegates directly to `timeseries_to_fisher_z_edges`.
    - If any ROI is missing, non-finite, or constant (std == 0) for this subject, computes
      correlations/Fisher-z across valid ROIs and sets every edge incident to any bad ROI
      to 0.0 (prior mean r=0, z=0) without altering the K-ROI feature space.
    Returns (z_edges, n_bad_rois).
    """
    name_to_col = {name: i for i, name in enumerate(roi_names)}
    k = len(dev_kept_roi_names)
    t_len = ts_raw.shape[0]
    aligned = np.empty((t_len, k), dtype=np.float64)
    good_mask = np.ones(k, dtype=bool)

    for out_idx, req_name in enumerate(dev_kept_roi_names):
        col_idx = name_to_col.get(req_name)
        if col_idx is None:
            good_mask[out_idx] = False
            aligned[:, out_idx] = 0.0
            continue
        col = ts_raw[:, col_idx]
        if not np.all(np.isfinite(col)) or float(np.std(col)) <= 0.0:
            good_mask[out_idx] = False
            aligned[:, out_idx] = 0.0
        else:
            aligned[:, out_idx] = col

    n_bad = int(np.sum(~good_mask))
    if n_bad == 0:
        return timeseries_to_fisher_z_edges(aligned, clip_eps=clip_eps), 0

    # Zero-fill edges for bad ROIs while computing exact Fisher-z among good ROIs
    z_mat = np.zeros((k, k), dtype=np.float64)
    good_idx = np.where(good_mask)[0]
    if len(good_idx) >= 2:
        ts_good = aligned[:, good_idx]
        corr_good = np.corrcoef(ts_good, rowvar=False)
        corr_good = np.clip(corr_good, -1.0 + clip_eps, 1.0 - clip_eps)
        z_good = np.arctanh(corr_good).astype(np.float64)
        np.fill_diagonal(z_good, 0.0)
        z_mat[np.ix_(good_idx, good_idx)] = z_good

    triu_i, triu_j = np.triu_indices(k, k=1)
    z_edges = z_mat[triu_i, triu_j]
    assert np.all(np.isfinite(z_edges))
    return z_edges, n_bad


def evaluate_sealed_regime(
    X_dev: np.ndarray,
    X_motion_dev: np.ndarray,
    X_agesex_dev: np.ndarray,
    y_dev: np.ndarray,
    X_sealed: np.ndarray,
    X_motion_sealed: np.ndarray,
    X_agesex_sealed: np.ndarray,
    y_sealed: np.ndarray,
    k_best: int = DEFAULT_K_BEST,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = LOCKED_RANDOM_STATE,
    site_label: str = "NYU",
) -> Tuple[Dict[str, object], np.ndarray]:
    """
    Refit the locked pipeline exactly once on pooled development data and score exactly once
    on the sealed site. Returns (aggregate_results_dict, fmri_predicted_probs).
    """
    k_eff = min(k_best, X_dev.shape[1])

    # 1. Refit frozen fMRI pipeline exactly once on pooled development set
    fmri_pipe = build_locked_fmri_pipeline(k_best=k_eff)
    fmri_pipe.fit(X_dev, y_dev)
    probs_fmri = fmri_pipe.predict_proba(X_sealed)[:, 1]

    strata_sealed = np.full(len(y_sealed), site_label, dtype="<U16")
    fmri_stats = stratified_bootstrap_auc(
        y_sealed, probs_fmri, strata_sealed, n_boot=n_boot, seed=seed
    )
    fmri_stats["outcome_zone"] = classify_section_8_2_zone(
        fmri_stats["auc"], fmri_stats["ci_lo"]
    )

    # 2. Descriptive baselines fit once on pooled development set and scored once on sealed site
    motion_pipe = build_linear_baseline_pipeline()
    motion_pipe.fit(X_motion_dev, y_dev)
    probs_motion = motion_pipe.predict_proba(X_motion_sealed)[:, 1]
    motion_stats = stratified_bootstrap_auc(
        y_sealed, probs_motion, strata_sealed, n_boot=n_boot, seed=seed
    )

    agesex_pipe = build_linear_baseline_pipeline()
    agesex_pipe.fit(X_agesex_dev, y_dev)
    probs_agesex = agesex_pipe.predict_proba(X_agesex_sealed)[:, 1]
    agesex_stats = stratified_bootstrap_auc(
        y_sealed, probs_agesex, strata_sealed, n_boot=n_boot, seed=seed
    )

    summary = {
        "n_dev_train": int(len(y_dev)),
        "n_sealed_scored": int(len(y_sealed)),
        "n_sealed_tdc": int(np.sum(y_sealed == 0)),
        "n_sealed_adhd": int(np.sum(y_sealed == 1)),
        "n_features": int(X_dev.shape[1]),
        "k_selected": int(k_eff),
        "fmri": fmri_stats,
        "descriptive_motion_baseline": motion_stats,
        "descriptive_age_sex_baseline": agesex_stats,
    }
    return summary, probs_fmri


def compute_nyu_fd_from_archive(
    nyu_intermediate_tar: Path,
    drop_frames: int = DUMMY_FRAMES_DROPPED,
) -> Dict[str, Dict[str, object]]:
    """
    Compute label-blind NYU FD metrics strictly using the frozen functions from
    scripts/compute_athena_fd.py (volume-centered NIAK transf, first 3 frames dropped).
    Returns mapping: normalized_sub_id -> motion metric dict (NO diagnosis labels).
    """
    if not nyu_intermediate_tar.is_file():
        raise FileNotFoundError(
            f"Missing NYU NIAK motion archive: {nyu_intermediate_tar}. "
            "Acquire intermediate_nyu.tar (NITRC /frs/downloadlink.php/3808) before sealed scoring."
        )

    motion_by_sub: Dict[str, Dict[str, object]] = {}
    with tarfile.open(nyu_intermediate_tar, "r") as tf:
        members = tf.getmembers()
        targets: Dict[str, tarfile.TarInfo] = {}
        for m in members:
            if m.isfile() and "motion_target" in m.name and m.name.endswith(".nii.gz"):
                hit = re.search(r"X_(\d+)_session_1_run1", m.name)
                if hit:
                    targets[hit.group(1).lstrip("0") or "0"] = m

        for m in members:
            if not m.isfile() or "motion_parameters" not in m.name or not m.name.endswith(".mat"):
                continue
            hit = re.search(r"X_(\d+)_session_1_run1", m.name)
            if not hit:
                continue
            sub_norm = hit.group(1).lstrip("0") or "0"
            if sub_norm not in targets:
                raise FileNotFoundError("Missing NYU motion_target NIfTI for session_1_run1")

            center_world = read_nifti_volume_center_world(tf.extractfile(targets[sub_norm]).read())
            transf = parse_octave_transf_text(
                tf.extractfile(m).read().decode("utf-8", errors="replace")
            )
            n_raw = transf.shape[2]
            if n_raw <= drop_frames:
                raise ValueError(f"Too few frames ({n_raw}) to drop {drop_frames}")

            transf_centered = center_shift_transf(transf[:, :, drop_frames:], center_world)
            params = extract_6dof_params(transf_centered)
            fd = compute_power_fd(params, radius_mm=HEAD_RADIUS_MM)

            mean_fd = float(np.mean(fd))
            rel_trans = params[:, 3:6] - params[0, 3:6]
            max_disp_mm = float(np.max(np.linalg.norm(rel_trans, axis=1)))
            pct_gt_050 = float(100.0 * np.mean(fd > 0.50))
            ex_primary = (mean_fd > 0.50) or (pct_gt_050 > 20.0)

            motion_by_sub[sub_norm] = {
                "frames_niak_raw": n_raw,
                "frames_retained": int(transf_centered.shape[2]),
                "mean_fd_mm": mean_fd,
                "max_disp_mm": max_disp_mm,
                "pct_frames_gt_050": pct_gt_050,
                "exclude_primary_050": int(ex_primary),
            }

    return motion_by_sub


def run_real_sealed_evaluation(
    staging_dir: Path,
    features_dir: Path,
    pheno_tsv: Path,
    out_json: Path,
    execute_real_sealed: bool,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = LOCKED_RANDOM_STATE,
    repo_root: Path | None = None,
) -> Dict[str, object]:
    """
    Execute the single one-shot sealed NYU evaluation after passing all hard guards.
    Never overwrites an existing results file and never prints individual subject IDs.
    """
    enforce_sealed_guards(
        execute_real_sealed=execute_real_sealed,
        out_json=out_json,
        tag_name=REQUIRED_FREEZE_TAG,
        repo_root=repo_root,
    )

    # 1. Load frozen 190-ROI development set
    with open(features_dir / "dev_valid_roi_set.json", encoding="utf-8") as f:
        dev_roi_meta = json.load(f)
    dev_kept_roi_names: List[str] = list(dev_roi_meta["kept_roi_names"])
    p_edges = len(dev_kept_roi_names) * (len(dev_kept_roi_names) - 1) // 2

    # 2. Compute label-blind NYU FD from intermediate_nyu.tar
    nyu_motion = compute_nyu_fd_from_archive(staging_dir / "intermediate_nyu.tar")

    # 3. Load NYU phenotype DX, Age, Gender
    nyu_pheno: Dict[str, Dict[str, object]] = {}
    with open(pheno_tsv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Site"].strip() != "5":
                continue
            sid = row["ScanDir ID"].strip().lstrip("0") or "0"
            dx_raw = row["DX"].strip()
            if dx_raw == "0":
                dx_bin = 0
            elif dx_raw in ("1", "2", "3"):
                dx_bin = 1
            else:
                continue
            age_val = float(row["Age"]) if row["Age"].strip() not in ("", "N/A", "NA") else np.nan
            sex_val = float(row["Gender"]) if row["Gender"].strip() not in ("", "N/A", "NA") else np.nan
            nyu_pheno[sid] = {"dx": dx_bin, "age": age_val, "gender": sex_val}

    # 4. Read NYU filtered rest_1 CC200 time series exclusively inside this guarded runner
    cc200_archive = staging_dir / "ADHD200_CC200_TCs_filtfix.tar.gz"
    nyu_edges_by_sub: Dict[str, np.ndarray] = {}
    scans_with_zero_roi = 0
    total_zero_roi_instances = 0

    with tarfile.open(cc200_archive, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 3 or parts[0] != "NYU":
                continue
            fname = parts[-1]
            if not (fname.startswith("sfnwmrda") and "rest_1" in fname and fname.endswith(".1D")):
                continue
            sub_norm = parts[1].lstrip("0") or "0"
            if sub_norm not in nyu_pheno or sub_norm not in nyu_motion:
                continue

            stream = tf.extractfile(m)
            if stream is None:
                continue
            roi_names, ts_raw = parse_athena_1d_timeseries(
                stream.read().decode("utf-8", errors="replace")
            )
            z_edges, n_bad = sealed_timeseries_to_fisher_z_edges(
                roi_names, ts_raw, dev_kept_roi_names
            )
            if n_bad > 0:
                scans_with_zero_roi += 1
                total_zero_roi_instances += n_bad
            nyu_edges_by_sub[sub_norm] = z_edges

    ordered_subs = sorted(nyu_edges_by_sub.keys(), key=int)
    n_nyu_total = len(ordered_subs)
    X_nyu_all = np.empty((n_nyu_total, p_edges), dtype=np.float64)
    y_nyu_all = np.empty(n_nyu_total, dtype=np.int64)
    motion_nyu_all = np.empty((n_nyu_total, 2), dtype=np.float64)
    agesex_nyu_all = np.empty((n_nyu_total, 2), dtype=np.float64)
    primary_keep_nyu = np.empty(n_nyu_total, dtype=bool)

    for idx, sid in enumerate(ordered_subs):
        X_nyu_all[idx] = nyu_edges_by_sub[sid]
        y_nyu_all[idx] = int(nyu_pheno[sid]["dx"])
        motion_nyu_all[idx, 0] = float(nyu_motion[sid]["mean_fd_mm"])
        motion_nyu_all[idx, 1] = float(nyu_motion[sid]["max_disp_mm"])
        agesex_nyu_all[idx, 0] = float(nyu_pheno[sid]["age"])
        agesex_nyu_all[idx, 1] = float(nyu_pheno[sid]["gender"])
        primary_keep_nyu[idx] = int(nyu_motion[sid]["exclude_primary_050"]) == 0

    # Aggregate NYU attrition counts (zero subject IDs)
    tdc_all = int(np.sum(y_nyu_all == 0))
    adhd_all = int(np.sum(y_nyu_all == 1))
    tdc_ret = int(np.sum((y_nyu_all == 0) & primary_keep_nyu))
    adhd_ret = int(np.sum((y_nyu_all == 1) & primary_keep_nyu))
    tdc_ex = tdc_all - tdc_ret
    adhd_ex = adhd_all - adhd_ret
    _, fisher_p = fisher_exact([[tdc_ex, tdc_ret], [adhd_ex, adhd_ret]], alternative="two-sided")

    # 5. Load frozen development sets and score each regime exactly once
    dev_prim = np.load(features_dir / "dev_cc200_fisher_z_primary368.npz", allow_pickle=False)
    dev_all = np.load(features_dir / "dev_cc200_fisher_z_all404.npz", allow_pickle=False)

    primary_res, _ = evaluate_sealed_regime(
        X_dev=dev_prim["X"],
        X_motion_dev=dev_prim["motion_X"],
        X_agesex_dev=dev_prim["age_sex_X"],
        y_dev=dev_prim["y"],
        X_sealed=X_nyu_all[primary_keep_nyu],
        X_motion_sealed=motion_nyu_all[primary_keep_nyu],
        X_agesex_sealed=agesex_nyu_all[primary_keep_nyu],
        y_sealed=y_nyu_all[primary_keep_nyu],
        k_best=DEFAULT_K_BEST,
        n_boot=n_boot,
        seed=seed,
    )

    sens_res, _ = evaluate_sealed_regime(
        X_dev=dev_all["X"],
        X_motion_dev=dev_all["motion_X"],
        X_agesex_dev=dev_all["age_sex_X"],
        y_dev=dev_all["y"],
        X_sealed=X_nyu_all,
        X_motion_sealed=motion_nyu_all,
        X_agesex_sealed=agesex_nyu_all,
        y_sealed=y_nyu_all,
        k_best=DEFAULT_K_BEST,
        n_boot=n_boot,
        seed=seed,
    )

    payload: Dict[str, object] = {
        "nyu_counts": {
            "total_complete": n_nyu_total,
            "total_tdc": tdc_all,
            "total_adhd": adhd_all,
            "primary_excluded_total": tdc_ex + adhd_ex,
            "primary_excluded_tdc": tdc_ex,
            "primary_excluded_adhd": adhd_ex,
            "primary_retained_total": tdc_ret + adhd_ret,
            "primary_retained_tdc": tdc_ret,
            "primary_retained_adhd": adhd_ret,
            "primary_exclusion_fisher_p": float(fisher_p),
            "scans_with_zero_imputed_roi": scans_with_zero_roi,
            "total_zero_imputed_roi_instances": total_zero_roi_instances,
        },
        "primary_sealed": primary_res,
        "sensitivity_no_exclusion": sens_res,
        "governing_section_8_2_zone": primary_res["fmri"]["outcome_zone"],
        "seeds_and_B": {
            "rf_random_state": LOCKED_RANDOM_STATE,
            "bootstrap_seed": seed,
            "bootstrap_B": n_boot,
            "k_selected": DEFAULT_K_BEST,
        },
    }

    # Final overwrite guard right before writing
    if out_json.exists():
        raise FileExistsError(f"Refusing to overwrite existing sealed results file: {out_json}")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Study 2 one-shot sealed NYU evaluation (requires --execute-real-sealed and study2-sealed-frozen tag)."
    )
    parser.add_argument("--staging-dir", type=Path, default=Path("NeuroADHD/data/athena/staging"))
    parser.add_argument("--features-dir", type=Path, default=Path("NeuroADHD/data/athena/features"))
    parser.add_argument(
        "--pheno-tsv",
        type=Path,
        default=Path("NeuroADHD/data/athena/phenotypes/adhd200_preprocessed_phenotypics.tsv"),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SEALED_RESULTS_JSON)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument(
        "--execute-real-sealed",
        action="store_true",
        help="Explicit safety guard required to execute on sealed NYU data.",
    )
    args = parser.parse_args()

    payload = run_real_sealed_evaluation(
        staging_dir=args.staging_dir,
        features_dir=args.features_dir,
        pheno_tsv=args.pheno_tsv,
        out_json=args.out_json,
        execute_real_sealed=args.execute_real_sealed,
        n_boot=args.n_boot,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
