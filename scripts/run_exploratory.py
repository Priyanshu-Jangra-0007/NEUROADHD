#!/usr/bin/env python3
"""
Exploratory Analysis Runner for Multi-Site Resting-State fMRI Connectivity in ADHD-200.

Implements the protocol in docs/exploratory/EXPLORATORY_PLAN_DRAFT.md:
- Strictly exploratory and hypothesis-generating.
- Strictly guarded: refuses to read real label files unless BOTH:
    1) --execute-real is passed on CLI, AND
    2) Git tag 'exploratory-frozen' is an ancestor of HEAD.
- 12 Primary LOIO configurations across 5 multi-class sites (Peking, KKI, NeuroIMAGE, OHSU, NYU) using CC200 (190 populated ROIs).
- 4 Multi-Run configurations (NYU + OHSU sub-analysis, subject-grouped folds, separate family with Bonferroni correction for 4 tests).
- Deterministic motion-matched evaluation (1:1 within-site matching on mean FD).
- Primary exploratory contrast: Motion+Age+Sex vs Motion+Age+Sex+Connectivity.
- Max-statistic within-institution permutation test repeating full configuration selection:
    1) Test 1 (AUC > 0.5): Label-permutation max-AUC test, permuting labels strictly WITHIN each institution (as in Study 2).
    2) Test 2 (Incremental contrast): Connectivity-permutation max-delta-AUC test (covariates & labels stay paired).
- NEVER prints subject IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import re
import resource
import subprocess
import sys
import tarfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
import gc
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from nilearn.connectome.connectivity_matrices import _form_symmetric, _geometric_mean, _map_eigenvalues
from scipy import linalg as sp_linalg
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.compute_athena_fd import (
    DUMMY_FRAMES_DROPPED,
    HEAD_RADIUS_MM,
    center_shift_transf,
    compute_power_fd,
    extract_6dof_params,
    parse_octave_transf_text,
    read_nifti_volume_center_world,
)

REQUIRED_FREEZE_TAG = "exploratory-frozen"
DEFAULT_RANDOM_STATE = 42
DEFAULT_BOOTSTRAP_B = 2000
DEFAULT_PERMUTATION_B = 200
FISHER_CLIP_EPS = 1e-7
PEAK_MEMORY_LIMIT_MB = 6000.0
DEFAULT_MAX_WORKERS = 2


def resolve_n_workers(n_workers: Optional[int] = None, max_cap: int = DEFAULT_MAX_WORKERS) -> int:
    """Resolve worker count capped to keep total memory comfortably below 6 GB."""
    cpu_cnt = os.cpu_count() or 1
    safe_default = max(1, min(max_cap, cpu_cnt // 2 if cpu_cnt >= 2 else 1))
    if n_workers is None or n_workers <= 0:
        return safe_default
    return max(1, min(int(n_workers), max_cap))

PRIMARY_CONFIGS: List[Tuple[str, str, str, str]] = [
    ("CFG_01", "pearson_fisher_z", "k100", "rf"),
    ("CFG_02", "pearson_fisher_z", "k100", "l2_lr"),
    ("CFG_03", "pearson_fisher_z", "all", "rf"),
    ("CFG_04", "pearson_fisher_z", "all", "l2_lr"),
    ("CFG_05", "tangent_space", "k100", "rf"),
    ("CFG_06", "tangent_space", "k100", "l2_lr"),
    ("CFG_07", "tangent_space", "all", "rf"),
    ("CFG_08", "tangent_space", "all", "l2_lr"),
    ("CFG_09", "motion_residualized", "k100", "rf"),
    ("CFG_10", "motion_residualized", "k100", "l2_lr"),
    ("CFG_11", "motion_residualized", "all", "rf"),
    ("CFG_12", "motion_residualized", "all", "l2_lr"),
]

MULTIRUN_CONFIGS: List[Tuple[str, str, str, str]] = [
    ("CFG_13", "multi_run_avg", "k100", "rf"),
    ("CFG_14", "multi_run_avg", "k100", "l2_lr"),
    ("CFG_15", "multi_run_avg", "all", "rf"),
    ("CFG_16", "multi_run_avg", "all", "l2_lr"),
]

SITE_CODE_TO_INSTITUTION = {
    "1": "Peking",
    "3": "KKI",
    "4": "NeuroIMAGE",
    "5": "NYU",
    "6": "OHSU",
}
INCLUDED_ARCHIVE_SITES = {"Peking_1", "Peking_2", "Peking_3", "KKI", "NeuroIMAGE", "OHSU", "NYU"}


def check_freeze_tag_is_ancestor(
    tag_name: str = REQUIRED_FREEZE_TAG,
    repo_root: Optional[Path] = None,
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


def enforce_exploratory_guards(
    execute_real: bool,
    tag_name: str = REQUIRED_FREEZE_TAG,
    repo_root: Optional[Path] = None,
) -> None:
    """Enforce hard guards before opening or reading any real label/phenotype file."""
    if not execute_real:
        raise RuntimeError(
            "Refusing to read real labels without --execute-real. "
            "Run synthetic tests in tests/test_run_exploratory.py instead."
        )
    if not check_freeze_tag_is_ancestor(tag_name=tag_name, repo_root=repo_root):
        raise RuntimeError(
            f"Refusing to read real labels: Git tag '{tag_name}' does not exist on an ancestor of HEAD."
        )


# =====================================================================
# Feature Extraction & Representation Transforms
# =====================================================================

def timeseries_to_fisher_z(
    ts: np.ndarray,
    clip_eps: float = FISHER_CLIP_EPS,
) -> np.ndarray:
    """
    Compute Fisher-z transformed Pearson correlation upper triangle (excluding diagonal).
    If any ROI column has zero temporal variance, its correlation edges are imputed with 0.0
    (matching Study 2 PREREGISTRATION / ATHENA_ACQUISITION_MANIFEST.md rule).
    Input: (T, K)
    Output: 1D array of shape (K*(K-1)//2,)
    """
    std_cols = np.std(ts, axis=0)
    k = ts.shape[1]
    if np.any(std_cols <= 0.0):
        valid = std_cols > 0.0
        corr = np.eye(k, dtype=np.float64)
        if np.sum(valid) >= 2:
            sub_corr = np.corrcoef(ts[:, valid], rowvar=False)
            idx_valid = np.where(valid)[0]
            corr[np.ix_(idx_valid, idx_valid)] = sub_corr
    else:
        corr = np.corrcoef(ts, rowvar=False)
    triu_i, triu_j = np.triu_indices(k, k=1)
    r_edges = np.clip(corr[triu_i, triu_j], -1.0 + clip_eps, 1.0 - clip_eps)
    z_edges = np.arctanh(r_edges).astype(np.float64)
    if not np.all(np.isfinite(z_edges)):
        raise ValueError("Non-finite Fisher z edges encountered")
    return z_edges


def precompute_subject_covariances(
    timeseries_list: Sequence[np.ndarray],
) -> np.ndarray:
    """Compute Ledoit-Wolf shrinkage covariances once per subject (exact match to ConnectivityMeasure)."""
    if not timeseries_list:
        return np.empty((0, 0, 0), dtype=np.float64)
    covs = [LedoitWolf(store_precision=False).fit(ts).covariance_ for ts in timeseries_list]
    return np.asarray(covs, dtype=np.float64)


def _fast_geometric_mean(
    matrices: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-7,
) -> np.ndarray:
    """Batched Riemannian Fréchet mean matching nilearn's _geometric_mean within <1e-12."""
    gmean = np.mean(matrices, axis=0)
    norm_old = np.inf
    step = 1.0
    for _ in range(max_iter):
        vals_gmean, vecs_gmean = sp_linalg.eigh(gmean)
        gmean_inv_sqrt = _form_symmetric(np.sqrt, 1.0 / vals_gmean, vecs_gmean)
        whitened = gmean_inv_sqrt @ matrices @ gmean_inv_sqrt
        whitened = 0.5 * (whitened + np.swapaxes(whitened, -1, -2))
        ev, evec = np.linalg.eigh(whitened)
        logs = np.einsum("nik,nk,njk->nij", evec, np.log(ev), evec, optimize=True)
        logs_mean = np.mean(logs, axis=0)
        norm = float(np.linalg.norm(logs_mean))
        vals_log, vecs_log = sp_linalg.eigh(logs_mean)
        gmean_sqrt = _form_symmetric(np.sqrt, vals_gmean, vecs_gmean)
        gmean = gmean_sqrt.dot(_form_symmetric(np.exp, vals_log * step, vecs_log)).dot(gmean_sqrt)
        if norm < norm_old:
            norm_old = norm
        elif norm > norm_old:
            step = step / 2.0
            norm = norm_old
        if tol is not None and norm / gmean.size < tol:
            break
    return gmean


def _tangent_whitening_worker(args: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    """Compute Riemannian Fréchet whitening matrix for a subset of subject covariances."""
    idx, tr_covs = args
    gmean = _fast_geometric_mean(tr_covs, max_iter=30, tol=1e-7)
    whitening = _map_eigenvalues(lambda x: 1.0 / np.sqrt(x), gmean)
    return idx, whitening


def _project_covariances_to_tangent(
    covs_all: np.ndarray,
    whitening: np.ndarray,
) -> np.ndarray:
    """Project subject covariance matrices onto a fitted Riemannian tangent space."""
    w_all = whitening @ covs_all @ whitening
    w_all = 0.5 * (w_all + np.swapaxes(w_all, -1, -2))
    ev, evec = np.linalg.eigh(w_all)
    logged = np.einsum("nik,nk,njk->nij", evec, np.log(ev), evec, optimize=True)
    return sym_matrix_to_vec(logged, discard_diagonal=False)


def precompute_tangent_cache(
    timeseries_list: Sequence[np.ndarray],
    institutions: np.ndarray,
    subsets: Sequence[frozenset],
    subject_covs: Optional[np.ndarray] = None,
    n_workers: int = 1,
) -> Dict[frozenset, np.ndarray]:
    """
    Precompute tangent space projections across multiple training-site subsets,
    reusing per-subject Ledoit-Wolf covariances and parallelizing Fréchet means cleanly.
    """
    subset_list = [s for s in subsets if s]
    if not subset_list:
        return {}

    covs_all = subject_covs if subject_covs is not None else precompute_subject_covariances(timeseries_list)
    use_workers = resolve_n_workers(n_workers)
    tasks = [
        (i, covs_all[np.isin(institutions, list(tr_sites))])
        for i, tr_sites in enumerate(subset_list)
    ]

    whitenings: Dict[int, np.ndarray] = {}
    # Spawn worker pool only when cohort matrices are substantial enough to outweigh spawn cost
    if use_workers > 1 and len(subset_list) >= 4 and covs_all.shape[0] >= 120 and covs_all.shape[1] >= 40:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(use_workers, len(tasks))) as pool:
            for idx, wh in pool.map(_tangent_whitening_worker, tasks):
                whitenings[idx] = wh
    else:
        for task in tasks:
            idx, wh = _tangent_whitening_worker(task)
            whitenings[idx] = wh

    cache: Dict[frozenset, np.ndarray] = {}
    for i, tr_sites in enumerate(subset_list):
        cache[tr_sites] = _project_covariances_to_tangent(covs_all, whitenings[i])
    return cache


def fit_transform_tangent_space(
    train_timeseries: Sequence[np.ndarray],
    test_timeseries: Sequence[np.ndarray],
    vectorize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, ConnectivityMeasure]:
    """
    Fit Riemannian tangent space projection strictly on training timeseries and project test.
    Uses Ledoit-Wolf shrinkage covariance and Fréchet geometric mean.
    Returns: (X_train_tangent, X_test_tangent, fitted_cm)
    """
    cm = ConnectivityMeasure(kind="tangent", vectorize=vectorize)
    X_train_tang = cm.fit_transform(train_timeseries)
    X_test_tang = cm.transform(test_timeseries)
    return X_train_tang, X_test_tang, cm


def _fit_ols_motion_residualization(
    X_train: np.ndarray,
    motion_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit OLS weights on training connectivity vs motion covariates; return (X_tr_resid, coeffs)."""
    M_tr_ext = np.column_stack([np.ones(X_train.shape[0], dtype=np.float64), motion_train])
    coeffs, _, _, _ = np.linalg.lstsq(M_tr_ext, X_train, rcond=None)
    X_tr_resid = X_train - M_tr_ext @ coeffs
    return X_tr_resid, coeffs


def _apply_ols_motion_residualization(
    X_test: np.ndarray,
    motion_test: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Apply previously fitted OLS motion weights to a held-out connectivity block."""
    M_te_ext = np.column_stack([np.ones(X_test.shape[0], dtype=np.float64), motion_test])
    return X_test - M_te_ext @ coeffs


def fit_transform_motion_residualization(
    X_train: np.ndarray,
    motion_train: np.ndarray,
    X_test: np.ndarray,
    motion_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit edge-level ordinary least squares (OLS) regression on training motion covariates:
        X_train = M_train @ W + B + E
    and apply fitted weights to residualize both train and test.
    Returns: (X_train_resid, X_test_resid, W, B)
    """
    X_tr_resid, coeffs = _fit_ols_motion_residualization(X_train, motion_train)
    X_te_resid = _apply_ols_motion_residualization(X_test, motion_test, coeffs)
    b = coeffs[0, :]
    w = coeffs[1:, :]
    return X_tr_resid, X_te_resid, w, b


# =====================================================================
# Model Factory & Feature Selection
# =====================================================================

def build_classifier(
    classifier_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = 1,
) -> ClassifierMixin:
    """Instantiate pre-declared classifier architecture."""
    if classifier_name == "rf":
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif classifier_name == "l2_lr":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown classifier name: {classifier_name}")


def _select_k100_indices(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Select top-k100 ANOVA F-score feature indices strictly on training labels."""
    k_val = min(100, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k_val)
    selector.fit(X_train, y_train)
    return selector.get_support(indices=True)


def apply_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    selection_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Apply feature selection strictly on training labels.
    selection_mode: 'k100' or 'all'
    Returns: (X_train_selected, X_test_selected, selected_indices)
    """
    if selection_mode == "k100":
        indices = _select_k100_indices(X_train, y_train)
        return X_train[:, indices], X_test[:, indices], indices
    elif selection_mode == "all":
        return X_train, X_test, None
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")


# =====================================================================
# Cross-Validation & Metric Evaluation
# =====================================================================

@dataclass
class ConfigurationResult:
    config_id: str
    feature_kind: str
    selection_mode: str
    classifier_name: str
    pooled_auc: float
    pooled_ci_lo: float
    pooled_ci_hi: float
    balanced_acc: float
    bacc_ci_lo: float
    bacc_ci_hi: float
    per_site_auc: Dict[str, float]
    per_site_ci: Dict[str, Tuple[float, float]]
    y_true: np.ndarray
    y_pred_probs: np.ndarray
    institutions: np.ndarray


def _fast_binary_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Fast rank-based Mann-Whitney U binary ROC AUC (1e-16 identical to sklearn with average-rank ties)."""
    ranks = rankdata(y_scores, method="average")
    pos = y_true == 1
    n_pos = int(np.sum(pos))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((np.sum(ranks[pos]) - 0.5 * n_pos * (n_pos + 1)) / (n_pos * n_neg))


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, float, float]:
    """Compute point estimate and 95% bootstrap confidence interval."""
    eval_fn = _fast_binary_roc_auc if metric_fn is roc_auc_score else metric_fn
    point = float(eval_fn(y_true, y_scores))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return point, point, point

    boots = np.empty(n_boot, dtype=np.float64)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    for b in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sample_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        boots[b] = eval_fn(y_true[sample_idx], y_scores[sample_idx])

    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return point, ci_lo, ci_hi


def paired_bootstrap_auc_diff(
    y_true: np.ndarray,
    y_scores_a: np.ndarray,
    y_scores_b: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, float, float]:
    """
    Compute paired delta AUC (A - B) with 95% bootstrap CI on identical samples.
    """
    diff_point = float(_fast_binary_roc_auc(y_true, y_scores_a) - _fast_binary_roc_auc(y_true, y_scores_b))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    diffs = np.empty(n_boot, dtype=np.float64)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    for b in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sample_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        diffs[b] = _fast_binary_roc_auc(y_true[sample_idx], y_scores_a[sample_idx]) - _fast_binary_roc_auc(
            y_true[sample_idx], y_scores_b[sample_idx]
        )

    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    return diff_point, ci_lo, ci_hi


# =====================================================================
# Deterministic Motion Matching (1:1 within-site nearest neighbor)
# =====================================================================

def match_subjects_on_motion_within_site(
    subject_ids: Sequence[str],
    site_names: Sequence[str],
    mean_fd_values: Sequence[float],
    labels: Sequence[int],
    caliper_mm: float = 0.05,
) -> List[int]:
    """
    Fully deterministic greedy 1:1 nearest-neighbor matching without replacement between
    ADHD (1) and TDC (0) within each site.

    Deterministic Rules (Zero Randomness):
    1. ADHD subjects sorted by (mean_fd_mm ascending, normalized_subject_id ascending).
    2. TDC subjects sorted by (mean_fd_mm ascending, normalized_subject_id ascending).
    3. Tie-breaking rule for multiple TDC candidates with equal distance |FD_tdc - FD_adhd|:
       a) Minimum absolute distance.
       b) Candidate with smaller mean_fd_mm.
       c) Candidate with lexicographically smaller subject_id.
    """
    matched_indices: List[int] = []
    unique_sites = sorted(set(site_names))

    for site in unique_sites:
        site_indices = [i for i, s in enumerate(site_names) if s == site]

        adhd_rows = [i for i in site_indices if labels[i] == 1]
        tdc_rows = [i for i in site_indices if labels[i] == 0]

        # Deterministic sorting
        adhd_sorted = sorted(adhd_rows, key=lambda i: (mean_fd_values[i], str(subject_ids[i])))
        avail_tdc = set(tdc_rows)

        for a_idx in adhd_sorted:
            if not avail_tdc:
                break
            a_fd = mean_fd_values[a_idx]

            # Find matching TDC candidate with deterministic tie-breaking
            best_t_idx: Optional[int] = None
            best_dist = float("inf")
            best_t_fd = float("inf")
            best_t_id = ""

            for t_idx in sorted(avail_tdc, key=lambda i: (mean_fd_values[i], str(subject_ids[i]))):
                dist = abs(mean_fd_values[t_idx] - a_fd)
                if dist <= caliper_mm:
                    t_fd = mean_fd_values[t_idx]
                    t_id = str(subject_ids[t_idx])
                    # Check if strictly closer, or tied with smaller FD, or tied with smaller ID
                    if (
                        dist < best_dist
                        or (abs(dist - best_dist) < 1e-12 and t_fd < best_t_fd)
                        or (abs(dist - best_dist) < 1e-12 and abs(t_fd - best_t_fd) < 1e-12 and t_id < best_t_id)
                    ):
                        best_dist = dist
                        best_t_idx = t_idx
                        best_t_fd = t_fd
                        best_t_id = t_id

            if best_t_idx is not None:
                matched_indices.append(a_idx)
                matched_indices.append(best_t_idx)
                avail_tdc.remove(best_t_idx)

    return sorted(matched_indices)


# =====================================================================
# LOIO Evaluation Engine
# =====================================================================

def evaluate_loio_pipeline(
    feature_kind: str,
    selection_mode: str,
    classifier_name: str,
    timeseries_list: Sequence[np.ndarray],
    pearson_z_edges: np.ndarray,
    motion_covariates: np.ndarray,
    labels: np.ndarray,
    institutions: np.ndarray,
    unique_institutions: Optional[Sequence[str]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    tangent_cache: Optional[Dict[frozenset, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute 5-fold Leave-One-Institution-Out evaluation for a single configuration.
    Returns: (all_out_of_fold_probabilities, all_out_of_fold_labels)
    """
    if unique_institutions is None:
        unique_institutions = sorted(set(institutions))

    n_samples = len(labels)
    probs_out = np.zeros(n_samples, dtype=np.float64)

    for test_site in unique_institutions:
        train_mask = institutions != test_site
        test_mask = institutions == test_site

        y_train = labels[train_mask]

        if feature_kind == "pearson_fisher_z":
            X_train = pearson_z_edges[train_mask]
            X_test = pearson_z_edges[test_mask]
        elif feature_kind == "tangent_space":
            tr_sites_set = frozenset(str(s) for s in institutions[train_mask])
            if tangent_cache is not None and tr_sites_set in tangent_cache:
                tang_all = tangent_cache[tr_sites_set]
                X_train = tang_all[train_mask]
                X_test = tang_all[test_mask]
            else:
                ts_train = [timeseries_list[i] for i in np.where(train_mask)[0]]
                ts_test = [timeseries_list[i] for i in np.where(test_mask)[0]]
                X_train, X_test, _ = fit_transform_tangent_space(ts_train, ts_test, vectorize=True)
        elif feature_kind == "motion_residualized":
            X_tr_raw = pearson_z_edges[train_mask]
            X_te_raw = pearson_z_edges[test_mask]
            M_tr = motion_covariates[train_mask]
            M_te = motion_covariates[test_mask]
            X_train, X_test, _, _ = fit_transform_motion_residualization(X_tr_raw, M_tr, X_te_raw, M_te)
        elif feature_kind == "multi_run_avg":
            X_train = pearson_z_edges[train_mask]
            X_test = pearson_z_edges[test_mask]
        else:
            raise ValueError(f"Unknown feature kind: {feature_kind}")

        X_tr_sel, X_te_sel, _ = apply_feature_selection(X_train, y_train, X_test, selection_mode)

        clf = build_classifier(classifier_name, random_state=random_state, n_jobs=1)
        clf.fit(X_tr_sel, y_train)

        test_indices = np.where(test_mask)[0]
        probs_out[test_indices] = clf.predict_proba(X_te_sel)[:, 1]

    return probs_out, labels


def evaluate_non_imaging_baselines(
    motion_covariates: np.ndarray,
    age_sex_covariates: np.ndarray,
    labels: np.ndarray,
    institutions: np.ndarray,
    unique_institutions: Optional[Sequence[str]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """Evaluate LOIO predictions for Motion, Age/Sex, and Motion+Age+Sex baselines."""
    if unique_institutions is None:
        unique_institutions = sorted(set(institutions))

    n = len(labels)
    probs_motion = np.zeros(n, dtype=np.float64)
    probs_agesex = np.zeros(n, dtype=np.float64)
    probs_combined = np.zeros(n, dtype=np.float64)

    for test_site in unique_institutions:
        train_mask = institutions != test_site
        test_mask = institutions == test_site

        s_m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(motion_covariates[train_mask])
        m_tr = s_m.transform(motion_covariates[train_mask])
        m_te = s_m.transform(motion_covariates[test_mask])

        s_a = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(age_sex_covariates[train_mask])
        a_tr = s_a.transform(age_sex_covariates[train_mask])
        a_te = s_a.transform(age_sex_covariates[test_mask])

        c_tr = np.column_stack([m_tr, a_tr])
        c_te = np.column_stack([m_te, a_te])

        y_tr = labels[train_mask]
        test_idx = np.where(test_mask)[0]

        clf_m = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", random_state=random_state)
        clf_m.fit(m_tr, y_tr)
        probs_motion[test_idx] = clf_m.predict_proba(m_te)[:, 1]

        clf_a = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", random_state=random_state)
        clf_a.fit(a_tr, y_tr)
        probs_agesex[test_idx] = clf_a.predict_proba(a_te)[:, 1]

        clf_c = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", random_state=random_state)
        clf_c.fit(c_tr, y_tr)
        probs_combined[test_idx] = clf_c.predict_proba(c_te)[:, 1]

    return {
        "motion": probs_motion,
        "age_sex": probs_agesex,
        "motion_age_sex": probs_combined,
    }


def _fit_predict_connectivity_fold(
    feature_kind: str,
    selection_mode: str,
    classifier_name: str,
    X_tr_conn: np.ndarray,
    X_te_conn: np.ndarray,
    y_tr: np.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = 1,
) -> np.ndarray:
    """Apply feature selection and fit classifier on a connectivity fold, returning test probabilities."""
    X_tr_sel, X_te_sel, _ = apply_feature_selection(X_tr_conn, y_tr, X_te_conn, selection_mode)
    clf = build_classifier(classifier_name, random_state=random_state, n_jobs=n_jobs)
    clf.fit(X_tr_sel, y_tr)
    return clf.predict_proba(X_te_sel)[:, 1]


def _build_conn_fold_features(
    feature_kind: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    timeseries_list: Sequence[np.ndarray],
    pearson_z_edges: np.ndarray,
    motion_covariates: np.ndarray,
    tangent_all: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract connectivity features for a train/test split without touching test labels."""
    if feature_kind == "pearson_fisher_z" or feature_kind == "multi_run_avg":
        return pearson_z_edges[train_mask], pearson_z_edges[test_mask]
    elif feature_kind == "tangent_space":
        if tangent_all is not None:
            return tangent_all[train_mask], tangent_all[test_mask]
        ts_train = [timeseries_list[i] for i in np.where(train_mask)[0]]
        ts_test = [timeseries_list[i] for i in np.where(test_mask)[0]]
        X_tr, X_te, _ = fit_transform_tangent_space(ts_train, ts_test, vectorize=True)
        return X_tr, X_te
    elif feature_kind == "motion_residualized":
        X_tr, X_te, _, _ = fit_transform_motion_residualization(
            pearson_z_edges[train_mask],
            motion_covariates[train_mask],
            pearson_z_edges[test_mask],
            motion_covariates[test_mask],
        )
        return X_tr, X_te
    else:
        raise ValueError(f"Unknown feature kind: {feature_kind}")


def fit_evaluate_late_fusion_outer_fold(
    test_site: str,
    unique_institutions: Sequence[str],
    feature_kind: str,
    selection_mode: str,
    classifier_name: str,
    timeseries_list: Sequence[np.ndarray],
    pearson_z_edges: np.ndarray,
    motion_covariates: np.ndarray,
    age_sex_covariates: np.ndarray,
    labels: np.ndarray,
    institutions: np.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
    tangent_cache: Optional[Dict[frozenset, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Fit late fusion for a single outer LOIO fold (`test_site` held out):
      - Inner 4-fold LOIO strictly within the 4 training institutions.
      - 2-parameter LogisticRegression meta-learner fit on `[s_cov_oof, s_conn_oof]`.
      - Outer base models fit on all 4 training institutions and applied to `test_site`.
    Returns: (outer_test_probs, fold_artifacts_dict)
    """
    n = len(labels)
    covariates = np.column_stack([motion_covariates, age_sex_covariates])
    outer_train_mask = institutions != test_site
    outer_test_mask = institutions == test_site
    inner_sites = [s for s in unique_institutions if s != test_site]

    # 1-2. Inner OOF scores strictly within the training institution(s)
    s_cov_oof = np.zeros(n, dtype=np.float64)
    s_conn_oof = np.zeros(n, dtype=np.float64)

    if len(inner_sites) >= 2:
        inner_splits = [
            (outer_train_mask & (institutions != iv), institutions == iv)
            for iv in inner_sites
        ]
    else:
        # 2-institution cohort (e.g. NYU + OHSU): 5-fold subject-grouped StratifiedKFold within training site
        tr_idx = np.where(outer_train_mask)[0]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        inner_splits = []
        for tr_rel, val_rel in skf.split(tr_idx, labels[tr_idx]):
            in_tr_m = np.zeros(n, dtype=bool)
            in_val_m = np.zeros(n, dtype=bool)
            in_tr_m[tr_idx[tr_rel]] = True
            in_val_m[tr_idx[val_rel]] = True
            inner_splits.append((in_tr_m, in_val_m))

    for inner_tr_mask, inner_val_mask in inner_splits:
        y_in_tr = labels[inner_tr_mask]

        sc_in = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(covariates[inner_tr_mask])
        clf_cov_in = LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=random_state
        )
        clf_cov_in.fit(sc_in.transform(covariates[inner_tr_mask]), y_in_tr)
        s_cov_oof[inner_val_mask] = clf_cov_in.predict_proba(sc_in.transform(covariates[inner_val_mask]))[:, 1]

        in_sites_set = frozenset(str(s) for s in institutions[inner_tr_mask])
        tang_in = tangent_cache.get(in_sites_set) if tangent_cache is not None else None
        X_in_tr, X_in_val = _build_conn_fold_features(
            feature_kind, inner_tr_mask, inner_val_mask, timeseries_list, pearson_z_edges, motion_covariates, tang_in
        )
        s_conn_oof[inner_val_mask] = _fit_predict_connectivity_fold(
            feature_kind, selection_mode, classifier_name, X_in_tr, X_in_val, y_in_tr, random_state=random_state
        )

    # 3. Fit 2-parameter LogisticRegression meta-learner on inner OOF scores
    meta_X_tr = np.column_stack([s_cov_oof[outer_train_mask], s_conn_oof[outer_train_mask]])
    meta_clf = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=random_state
    )
    meta_clf.fit(meta_X_tr, labels[outer_train_mask])

    # 4. Fit base models on all 4 training institutions and predict on outer test_site
    y_out_tr = labels[outer_train_mask]
    sc_out = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(covariates[outer_train_mask])
    clf_cov_out = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=random_state
    )
    clf_cov_out.fit(sc_out.transform(covariates[outer_train_mask]), y_out_tr)
    s_cov_test = clf_cov_out.predict_proba(sc_out.transform(covariates[outer_test_mask]))[:, 1]

    out_sites_set = frozenset(str(s) for s in institutions[outer_train_mask])
    tang_out = tangent_cache.get(out_sites_set) if tangent_cache is not None else None
    X_out_tr, X_out_te = _build_conn_fold_features(
        feature_kind, outer_train_mask, outer_test_mask, timeseries_list, pearson_z_edges, motion_covariates, tang_out
    )
    X_tr_sel, X_te_sel, sel_idx = apply_feature_selection(X_out_tr, y_out_tr, X_out_te, selection_mode)
    clf_conn_out = build_classifier(classifier_name, random_state=random_state)
    clf_conn_out.fit(X_tr_sel, y_out_tr)
    s_conn_test = clf_conn_out.predict_proba(X_te_sel)[:, 1]

    meta_X_te = np.column_stack([s_cov_test, s_conn_test])
    probs_test = meta_clf.predict_proba(meta_X_te)[:, 1]

    artifacts: Dict[str, object] = {
        "s_cov_oof_train": s_cov_oof[outer_train_mask].copy(),
        "s_conn_oof_train": s_conn_oof[outer_train_mask].copy(),
        "meta_coef": meta_clf.coef_.copy(),
        "meta_intercept": meta_clf.intercept_.copy(),
        "base_cov_coef": clf_cov_out.coef_.copy(),
        "base_cov_intercept": clf_cov_out.intercept_.copy(),
        "base_conn_sel_idx": sel_idx.copy() if sel_idx is not None else None,
        "base_conn_coef": clf_conn_out.coef_.copy() if hasattr(clf_conn_out, "coef_") else clf_conn_out.feature_importances_.copy(),
    }
    return probs_test, artifacts


def evaluate_combined_covariates_and_connectivity(
    feature_kind: str,
    selection_mode: str,
    classifier_name: str,
    timeseries_list: Sequence[np.ndarray],
    pearson_z_edges: np.ndarray,
    motion_covariates: np.ndarray,
    age_sex_covariates: np.ndarray,
    labels: np.ndarray,
    institutions: np.ndarray,
    unique_institutions: Optional[Sequence[str]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    tangent_cache: Optional[Dict[frozenset, np.ndarray]] = None,
) -> np.ndarray:
    """
    Evaluate outer LOIO predictions for Motion + Age + Sex + Connectivity via LATE FUSION.
    """
    if unique_institutions is None:
        unique_institutions = sorted(set(institutions))

    n = len(labels)
    probs_out = np.zeros(n, dtype=np.float64)

    for test_site in unique_institutions:
        outer_test_mask = institutions == test_site
        probs_test, _ = fit_evaluate_late_fusion_outer_fold(
            test_site=test_site,
            unique_institutions=unique_institutions,
            feature_kind=feature_kind,
            selection_mode=selection_mode,
            classifier_name=classifier_name,
            timeseries_list=timeseries_list,
            pearson_z_edges=pearson_z_edges,
            motion_covariates=motion_covariates,
            age_sex_covariates=age_sex_covariates,
            labels=labels,
            institutions=institutions,
            random_state=random_state,
            tangent_cache=tangent_cache,
        )
        probs_out[outer_test_mask] = probs_test

    return probs_out


# =====================================================================
# Permutation Tests (Label Permutation & Connectivity Permutation)
# =====================================================================

def save_permutation_checkpoint(
    checkpoint_path: Optional[Path],
    test_key: str,
    completed_values: Sequence[float],
) -> None:
    """
    Save ONLY the completed permutation null statistics to disk so a crash can resume
    without exposing any observed AUCs, deltas, or partial results.
    """
    if checkpoint_path is None:
        return
    data: Dict[str, List[float]] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[test_key] = [float(v) for v in completed_values]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp_path.replace(checkpoint_path)


def load_permutation_checkpoint(
    checkpoint_path: Optional[Path],
    test_key: str,
) -> List[float]:
    """Load previously completed permutation null statistics from checkpoint, if present."""
    if checkpoint_path is None or not checkpoint_path.exists():
        return []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [float(v) for v in data.get(test_key, [])]


def _SharedArraySpec_create(arr: np.ndarray, shm_list: List[SharedMemory]) -> Tuple[str, Tuple[int, ...], str]:
    """Allocate a POSIX SharedMemory segment for a contiguous float/int ndarray."""
    contig = np.ascontiguousarray(arr)
    nbytes = max(1, contig.nbytes)
    name = f"exp_{os.getpid()}_{uuid.uuid4().hex[:12]}"
    shm = SharedMemory(create=True, size=nbytes, name=name)
    shm_list.append(shm)
    if contig.nbytes > 0:
        view = np.ndarray(contig.shape, dtype=contig.dtype, buffer=shm.buf)
        view[:] = contig
    return name, contig.shape, contig.dtype.str


def _open_shared_array(spec: Tuple[str, Tuple[int, ...], str], opened: List[SharedMemory]) -> np.ndarray:
    """Attach read-only view onto an existing SharedMemory segment."""
    name, shape, dtype_str = spec
    shm = SharedMemory(create=False, name=name)
    opened.append(shm)
    return np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)


def _current_process_rss_mb() -> float:
    """Return peak RSS of the current process in megabytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru.ru_maxrss / 1024.0


_WORKER_STATE: Dict[str, object] = {}


def _eval_single_test1_perm(args: Tuple[int, np.ndarray]) -> Tuple[int, float, float]:
    """Evaluate one Test 1 within-site label permutation deterministically."""
    b, y_perm = args
    st = _WORKER_STATE
    configs: Sequence[Tuple[str, str, str]] = st["configs"]  # type: ignore[assignment]
    unique_sites: Sequence[str] = st["unique_sites"]  # type: ignore[assignment]
    institutions: np.ndarray = st["institutions"]  # type: ignore[assignment]
    pearson_z_edges: np.ndarray = st["pearson_z_edges"]  # type: ignore[assignment]
    tangent_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = st["tangent_by_fold"]  # type: ignore[assignment]
    resid_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = st["resid_by_fold"]  # type: ignore[assignment]
    seed: int = int(st["seed"])  # type: ignore[arg-type]

    # Pre-slice fold masks once
    fold_masks = {
        site: (institutions != site, institutions == site) for site in unique_sites
    }
    # Cache k100 selected indices per (feat_kind, test_site) so rf & l2_lr share f_classif
    k100_cache: Dict[Tuple[str, str], np.ndarray] = {}

    perm_aucs: List[float] = []
    for feat_kind, sel_mode, clf_name in configs:
        try:
            probs = np.zeros(len(y_perm), dtype=np.float64)
            for test_site in unique_sites:
                train_mask, test_mask = fold_masks[test_site]
                y_tr = y_perm[train_mask]

                if feat_kind == "pearson_fisher_z" or feat_kind == "multi_run_avg":
                    X_tr = pearson_z_edges[train_mask]
                    X_te = pearson_z_edges[test_mask]
                elif feat_kind == "tangent_space":
                    X_tr, X_te = tangent_by_fold[test_site]
                elif feat_kind == "motion_residualized":
                    X_tr, X_te = resid_by_fold[test_site]
                else:
                    raise ValueError(f"Unknown feature kind: {feat_kind}")

                if sel_mode == "k100":
                    ckey = (feat_kind, test_site)
                    if ckey not in k100_cache:
                        k100_cache[ckey] = _select_k100_indices(X_tr, y_tr)
                    idx = k100_cache[ckey]
                    X_tr_sel, X_te_sel = X_tr[:, idx], X_te[:, idx]
                elif sel_mode == "all":
                    X_tr_sel, X_te_sel = X_tr, X_te
                else:
                    raise ValueError(f"Unknown selection mode: {sel_mode}")

                clf = build_classifier(clf_name, random_state=seed + b, n_jobs=1)
                clf.fit(X_tr_sel, y_tr)
                probs[test_mask] = clf.predict_proba(X_te_sel)[:, 1]

            perm_aucs.append(float(roc_auc_score(y_perm, probs)))
        except Exception:
            continue

    if not perm_aucs:
        raise RuntimeError("All configurations failed during Test 1 permutation.")
    return b, float(max(perm_aucs)), _current_process_rss_mb()


def _test1_worker_entrypoint(args: Tuple[int, np.ndarray]) -> Tuple[int, float, float]:
    return _eval_single_test1_perm(args)


def run_within_site_label_permutation_test(
    configs: Sequence[Tuple[str, str, str]],
    timeseries_list: Sequence[np.ndarray],
    pearson_z_edges: np.ndarray,
    motion_covariates: np.ndarray,
    labels: np.ndarray,
    institutions: np.ndarray,
    observed_max_auc: float,
    n_permutations: int = DEFAULT_PERMUTATION_B,
    seed: int = DEFAULT_RANDOM_STATE,
    checkpoint_path: Optional[Path] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    tangent_cache: Optional[Dict[frozenset, np.ndarray]] = None,
    n_workers: int = 1,
) -> Tuple[float, np.ndarray]:
    """
    Test 1 (AUC > 0.5): Label-permutation max-AUC test.
    Permutes labels strictly WITHIN each institution (as in Study 2), repeating full
    selection across configurations. Supports null-only checkpointing for crash recovery.
    Returns: (empirical_p_value, max_null_distribution)
    """
    rng = np.random.default_rng(seed)
    unique_sites = sorted(set(institutions))
    completed = load_permutation_checkpoint(checkpoint_path, "test1_null_max_aucs")
    null_max_aucs = np.empty(n_permutations, dtype=np.float64)
    for idx, val in enumerate(completed[:n_permutations]):
        null_max_aucs[idx] = val

    # Pre-generate deterministic per-permutation label arrays so 1 worker == N workers == resumed
    perm_labels_schedule: List[np.ndarray] = []
    site_idx_map = {site: np.where(institutions == site)[0] for site in unique_sites}
    for _ in range(n_permutations):
        y_perm = labels.copy()
        for site in unique_sites:
            s_idx = site_idx_map[site]
            y_perm[s_idx] = rng.permutation(y_perm[s_idx])
        perm_labels_schedule.append(y_perm)

    pending_indices = [b for b in range(n_permutations) if b >= len(completed)]
    if not pending_indices:
        p_val = float((1.0 + np.sum(null_max_aucs >= observed_max_auc)) / (1.0 + n_permutations))
        return p_val, null_max_aucs

    # Pre-extract tangent space (reusing primary tangent_cache when provided) and residualized features once per fold
    uses_tangent = any(fk == "tangent_space" for fk, _, _ in configs)
    uses_resid = any(fk == "motion_residualized" for fk, _, _ in configs)
    tangent_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    resid_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for test_site in unique_sites:
        train_mask = institutions != test_site
        test_mask = institutions == test_site
        tr_sites = frozenset(str(s) for s in institutions[train_mask])
        if uses_tangent:
            if tangent_cache is not None and tr_sites in tangent_cache:
                tang_all = tangent_cache[tr_sites]
                tangent_by_fold[test_site] = (tang_all[train_mask], tang_all[test_mask])
            else:
                ts_train = [timeseries_list[i] for i in np.where(train_mask)[0]]
                ts_test = [timeseries_list[i] for i in np.where(test_mask)[0]]
                X_tr_tang, X_te_tang, _ = fit_transform_tangent_space(ts_train, ts_test, vectorize=True)
                tangent_by_fold[test_site] = (X_tr_tang, X_te_tang)
        if uses_resid:
            X_tr_res, X_te_res, _, _ = fit_transform_motion_residualization(
                pearson_z_edges[train_mask],
                motion_covariates[train_mask],
                pearson_z_edges[test_mask],
                motion_covariates[test_mask],
            )
            resid_by_fold[test_site] = (X_tr_res, X_te_res)

    use_workers = resolve_n_workers(n_workers)
    tasks = [(b, perm_labels_schedule[b]) for b in pending_indices]

    if use_workers > 1 and len(tasks) >= 2:
        created_shm: List[SharedMemory] = []
        try:
            # Share sliced tangent_by_fold and resid_by_fold + pearson_z_edges without duplicating across workers
            pearson_spec = _SharedArraySpec_create(pearson_z_edges, created_shm)
            tang_fold_specs = {
                site: (_SharedArraySpec_create(xtr, created_shm), _SharedArraySpec_create(xte, created_shm))
                for site, (xtr, xte) in tangent_by_fold.items()
            }
            resid_specs = {
                site: (_SharedArraySpec_create(xtr, created_shm), _SharedArraySpec_create(xte, created_shm))
                for site, (xtr, xte) in resid_by_fold.items()
            }
            payload = {
                "configs": list(configs),
                "unique_sites": list(unique_sites),
                "institutions": institutions,
                "pearson_spec": pearson_spec,
                "tangent_by_fold": {},
                "resid_specs": resid_specs,
                "tang_fold_specs": tang_fold_specs,
                "seed": seed,
            }
            ctx = multiprocessing.get_context("spawn")
            # Initialize worker state cleanly (open both pearson, resid_specs and tang_fold_specs)
            with ctx.Pool(
                processes=min(use_workers, len(tasks)),
                initializer=_init_test1_worker,
                initargs=(payload,),
            ) as pool:
                for b, stat_val, _ in pool.imap(_test1_worker_entrypoint, tasks):
                    if progress_fn is not None:
                        progress_fn(f"Test 1 permutation {b + 1}/{n_permutations}")
                    null_max_aucs[b] = stat_val
                    save_permutation_checkpoint(checkpoint_path, "test1_null_max_aucs", null_max_aucs[: b + 1])
        finally:
            for shm in created_shm:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
    else:
        _WORKER_STATE.clear()
        _WORKER_STATE.update({
            "configs": list(configs),
            "unique_sites": list(unique_sites),
            "institutions": institutions,
            "pearson_z_edges": pearson_z_edges,
            "tangent_by_fold": tangent_by_fold,
            "resid_by_fold": resid_by_fold,
            "seed": seed,
        })
        try:
            for b, y_perm in tasks:
                if progress_fn is not None:
                    progress_fn(f"Test 1 permutation {b + 1}/{n_permutations}")
                _, stat_val, _ = _eval_single_test1_perm((b, y_perm))
                null_max_aucs[b] = stat_val
                save_permutation_checkpoint(checkpoint_path, "test1_null_max_aucs", null_max_aucs[: b + 1])
        finally:
            _WORKER_STATE.clear()

    p_val = float((1.0 + np.sum(null_max_aucs >= observed_max_auc)) / (1.0 + n_permutations))
    return p_val, null_max_aucs


def _init_test1_worker(payload: Dict[str, object]) -> None:
    """Initializer for Test 1 workers opening shared fold arrays."""
    opened: List[SharedMemory] = []
    state = dict(payload)
    state["_opened_shm"] = opened
    state["pearson_z_edges"] = _open_shared_array(payload["pearson_spec"], opened)  # type: ignore[arg-type]
    state["tangent_by_fold"] = {
        site: (_open_shared_array(tr_sp, opened), _open_shared_array(te_sp, opened))
        for site, (tr_sp, te_sp) in payload["tang_fold_specs"].items()  # type: ignore[union-attr]
    }
    state["resid_by_fold"] = {
        site: (_open_shared_array(tr_sp, opened), _open_shared_array(te_sp, opened))
        for site, (tr_sp, te_sp) in payload["resid_specs"].items()  # type: ignore[union-attr]
    }
    global _WORKER_STATE
    _WORKER_STATE = state


def _init_test2_worker(payload: Dict[str, object]) -> None:
    """Initializer for Test 2 workers opening shared read-only arrays."""
    opened: List[SharedMemory] = []
    state = dict(payload)
    state["_opened_shm"] = opened
    state["pearson_z_edges"] = _open_shared_array(payload["pearson_spec"], opened)  # type: ignore[arg-type]
    tang_specs = payload.get("tangent_specs")
    if tang_specs is not None:
        state["tangent_all_by_subset"] = {
            k: _open_shared_array(spec, opened) for k, spec in tang_specs.items()  # type: ignore[union-attr]
        }
    global _WORKER_STATE
    _WORKER_STATE = state


def _eval_single_test2_perm(args: Tuple[int, np.ndarray]) -> Tuple[int, float, float]:
    """
    Evaluate one Test 2 permutation (connectivity rows permuted within site; labels & covariates intact).
    Eliminates redundant fits across identical training splits while strictly recomputing
    OLS motion residualization inside each permutation.
    """
    b, conn_perm_indices = args
    st = _WORKER_STATE
    configs: Sequence[Tuple[str, str, str]] = st["configs"]  # type: ignore[assignment]
    unique_sites: Sequence[str] = st["unique_sites"]  # type: ignore[assignment]
    institutions: np.ndarray = st["institutions"]  # type: ignore[assignment]
    labels: np.ndarray = st["labels"]  # type: ignore[assignment]
    motion_covariates: np.ndarray = st["motion_covariates"]  # type: ignore[assignment]
    pearson_z_edges: np.ndarray = st["pearson_z_edges"]  # type: ignore[assignment]
    timeseries_list: Optional[Sequence[np.ndarray]] = st.get("timeseries_list")  # type: ignore[assignment]
    tangent_all_by_subset: Dict[frozenset, np.ndarray] = st["tangent_all_by_subset"]  # type: ignore[assignment]
    use_cached_tangent: bool = bool(st["use_cached_tangent"])
    cov_oof_by_outer: Dict[str, np.ndarray] = st["cov_oof_by_outer"]  # type: ignore[assignment]
    cov_test_by_outer: Dict[str, np.ndarray] = st["cov_test_by_outer"]  # type: ignore[assignment]
    baseline_auc: float = float(st["baseline_auc"])  # type: ignore[arg-type]
    seed: int = int(st["seed"])  # type: ignore[arg-type]

    n = len(labels)
    pearson_perm = pearson_z_edges[conn_perm_indices]

    # Build outer/inner split definitions once
    outer_defs: List[Tuple[str, np.ndarray, np.ndarray, frozenset, List[Tuple[np.ndarray, np.ndarray, Tuple[frozenset, str]]]]] = []
    unique_tr_masks: Dict[Tuple[frozenset, str], np.ndarray] = {}
    for test_site in unique_sites:
        outer_train_mask = institutions != test_site
        outer_test_mask = institutions == test_site
        outer_tr_sites = frozenset(s for s in unique_sites if s != test_site)
        inner_sites = [s for s in unique_sites if s != test_site]

        if len(inner_sites) >= 2:
            in_splits: List[Tuple[np.ndarray, np.ndarray, Tuple[frozenset, str]]] = []
            for iv in inner_sites:
                m_tr = outer_train_mask & (institutions != iv)
                m_val = institutions == iv
                inner_tr_sites = frozenset(s for s in outer_tr_sites if s != iv)
                sp_key = (inner_tr_sites, "loio")
                unique_tr_masks[sp_key] = m_tr
                in_splits.append((m_tr, m_val, sp_key))
            out_key = (outer_tr_sites, "loio")
        else:
            tr_idx = np.where(outer_train_mask)[0]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            in_splits = []
            for f_i, (tr_rel, val_rel) in enumerate(skf.split(tr_idx, labels[tr_idx])):
                m_tr = np.zeros(n, dtype=bool)
                m_val = np.zeros(n, dtype=bool)
                m_tr[tr_idx[tr_rel]] = True
                m_val[tr_idx[val_rel]] = True
                sp_key = (outer_tr_sites, f"cv{f_i}")
                unique_tr_masks[sp_key] = m_tr
                in_splits.append((m_tr, m_val, sp_key))
            out_key = (outer_tr_sites, "outer")

        unique_tr_masks[out_key] = outer_train_mask
        outer_defs.append((test_site, outer_train_mask, outer_test_mask, out_key[0], out_key[1], in_splits))  # type: ignore[misc]

    # When use_cached_tangent=False (verification test mode), recompute tangent projections per perm
    local_tangent_cache: Dict[Tuple[frozenset, str], np.ndarray] = {}
    if not use_cached_tangent and any(fk == "tangent_space" for fk, _, _ in configs):
        if timeseries_list is None:
            raise ValueError("timeseries_list is required when use_cached_tangent=False")
        perm_ts = [timeseries_list[int(i)] for i in conn_perm_indices]
        perm_covs = precompute_subject_covariances(perm_ts)
        for sp_key, tr_m in unique_tr_masks.items():
            wh = _tangent_whitening_worker((0, perm_covs[tr_m]))[1]
            local_tangent_cache[sp_key] = _project_covariances_to_tangent(perm_covs, wh)

    # Per-permutation caches (NEVER shared across permutations!).
    # Store only compact (3, P) OLS coefficients (~0.4 MB/split instead of ~75 MB X_tr_res/split)
    # to keep parallel worker private RSS low while keeping motion residualization strictly per-permutation.
    resid_coeff_cache: Dict[Tuple[frozenset, str], np.ndarray] = {}
    k100_cache: Dict[Tuple[str, Tuple[frozenset, str]], np.ndarray] = {}
    model_cache: Dict[Tuple[str, str, str, Tuple[frozenset, str]], Tuple[ClassifierMixin, Optional[np.ndarray]]] = {}

    def _get_tr_matrix(fk: str, sp_key: Tuple[frozenset, str], tr_mask: np.ndarray) -> np.ndarray:
        if fk == "pearson_fisher_z" or fk == "multi_run_avg":
            return pearson_perm[tr_mask]
        elif fk == "tangent_space":
            if use_cached_tangent:
                return tangent_all_by_subset[sp_key[0]][conn_perm_indices[tr_mask]]
            return local_tangent_cache[sp_key][tr_mask]
        elif fk == "motion_residualized":
            if sp_key not in resid_coeff_cache:
                # Strictly recompute OLS motion residualization on permuted connectivity inside perm b
                x_tr_res, coeffs = _fit_ols_motion_residualization(
                    pearson_perm[tr_mask], motion_covariates[tr_mask]
                )
                resid_coeff_cache[sp_key] = coeffs
                return x_tr_res
            return _apply_ols_motion_residualization(
                pearson_perm[tr_mask], motion_covariates[tr_mask], resid_coeff_cache[sp_key]
            )
        else:
            raise ValueError(f"Unknown feature kind: {fk}")

    def _get_te_matrix(fk: str, sp_key: Tuple[frozenset, str], tr_mask: np.ndarray, te_mask: np.ndarray, te_tag: str) -> np.ndarray:
        if fk == "pearson_fisher_z" or fk == "multi_run_avg":
            return pearson_perm[te_mask]
        elif fk == "tangent_space":
            if use_cached_tangent:
                return tangent_all_by_subset[sp_key[0]][conn_perm_indices[te_mask]]
            return local_tangent_cache[sp_key][te_mask]
        elif fk == "motion_residualized":
            if sp_key not in resid_coeff_cache:
                _, coeffs = _fit_ols_motion_residualization(
                    pearson_perm[tr_mask], motion_covariates[tr_mask]
                )
                resid_coeff_cache[sp_key] = coeffs
            return _apply_ols_motion_residualization(
                pearson_perm[te_mask], motion_covariates[te_mask], resid_coeff_cache[sp_key]
            )
        else:
            raise ValueError(f"Unknown feature kind: {fk}")

    def _fit_cached_clf(
        fk: str, sel_mode: str, clf_name: str, sp_key: Tuple[frozenset, str], tr_mask: np.ndarray
    ) -> Tuple[ClassifierMixin, Optional[np.ndarray]]:
        mkey = (fk, sel_mode, clf_name, sp_key)
        if mkey not in model_cache:
            X_tr = _get_tr_matrix(fk, sp_key, tr_mask)
            y_tr = labels[tr_mask]
            if sel_mode == "k100":
                kkey = (fk, sp_key)
                if kkey not in k100_cache:
                    k100_cache[kkey] = _select_k100_indices(X_tr, y_tr)
                sel_idx: Optional[np.ndarray] = k100_cache[kkey]
                X_tr_sel = X_tr[:, sel_idx]
            elif sel_mode == "all":
                sel_idx = None
                X_tr_sel = X_tr
            else:
                raise ValueError(f"Unknown selection mode: {sel_mode}")
            clf = build_classifier(clf_name, random_state=seed + b, n_jobs=1)
            clf.fit(X_tr_sel, y_tr)
            model_cache[mkey] = (clf, sel_idx)
        return model_cache[mkey]

    deltas: List[float] = []
    for feat_kind, sel_mode, clf_name in configs:
        try:
            fused_probs = np.zeros(n, dtype=np.float64)
            for test_site, outer_train_mask, outer_test_mask, out_sites, out_tag, in_splits in outer_defs:
                s_conn_oof = np.zeros(n, dtype=np.float64)
                for inner_tr_mask, inner_val_mask, sp_key in in_splits:
                    clf_in, sel_idx = _fit_cached_clf(feat_kind, sel_mode, clf_name, sp_key, inner_tr_mask)
                    val_tag = f"val_{sp_key[1]}_{np.flatnonzero(inner_val_mask)[0]}_{np.flatnonzero(inner_val_mask)[-1]}"
                    X_in_val = _get_te_matrix(feat_kind, sp_key, inner_tr_mask, inner_val_mask, val_tag)
                    X_val_sel = X_in_val[:, sel_idx] if sel_idx is not None else X_in_val
                    s_conn_oof[inner_val_mask] = clf_in.predict_proba(X_val_sel)[:, 1]

                s_cov_oof = cov_oof_by_outer[test_site]
                meta_X_tr = np.column_stack([s_cov_oof[outer_train_mask], s_conn_oof[outer_train_mask]])
                meta_clf = LogisticRegression(
                    C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed + b
                )
                meta_clf.fit(meta_X_tr, labels[outer_train_mask])

                out_key = (out_sites, out_tag)
                clf_out, out_sel_idx = _fit_cached_clf(feat_kind, sel_mode, clf_name, out_key, outer_train_mask)
                X_out_te = _get_te_matrix(feat_kind, out_key, outer_train_mask, outer_test_mask, f"out_{test_site}")
                X_out_sel = X_out_te[:, out_sel_idx] if out_sel_idx is not None else X_out_te
                s_conn_test = clf_out.predict_proba(X_out_sel)[:, 1]
                meta_X_te = np.column_stack([cov_test_by_outer[test_site], s_conn_test])
                fused_probs[outer_test_mask] = meta_clf.predict_proba(meta_X_te)[:, 1]

            comb_auc = roc_auc_score(labels, fused_probs)
            deltas.append(float(comb_auc - baseline_auc))
        except Exception:
            continue

    if not deltas:
        raise RuntimeError("All configurations failed during Test 2 permutation.")
    return b, float(max(deltas)), _current_process_rss_mb()


def _test2_worker_entrypoint(args: Tuple[int, np.ndarray]) -> Tuple[int, float, float]:
    return _eval_single_test2_perm(args)


def run_incremental_contrast_permutation_test(
    configs: Sequence[Tuple[str, str, str]],
    timeseries_list: Sequence[np.ndarray],
    pearson_z_edges: np.ndarray,
    motion_covariates: np.ndarray,
    age_sex_covariates: np.ndarray,
    labels: np.ndarray,
    institutions: np.ndarray,
    baseline_probs: np.ndarray,
    observed_max_delta_auc: float,
    n_permutations: int = DEFAULT_PERMUTATION_B,
    seed: int = DEFAULT_RANDOM_STATE,
    checkpoint_path: Optional[Path] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    tangent_cache: Optional[Dict[frozenset, np.ndarray]] = None,
    use_cached_tangent: bool = True,
    n_workers: int = 1,
) -> Tuple[float, np.ndarray]:
    """
    Test 2: Max-statistic permutation test for the Incremental Contrast (Late Fusion):
        delta_AUC = AUC(fused[motion+age+sex, connectivity]) - AUC(motion+age+sex)

    Null Specification:
    - Diagnostic labels (y) and non-imaging covariates (motion + age + sex) stay PAIRED and intact.
    - Only the connectivity feature rows (X) are permuted among subjects WITHIN each institution.
    - Outer & inner covariate-only scores (s_cov_oof, s_cov_test) depend only on intact covariates/labels
      and are precomputed once before the permutation loop.
    - When `use_cached_tangent=True`, per-fold tangent reference means and projected vectors are
      reused (either from `tangent_cache` or computed once), while motion residualization is always
      recomputed per permutation.
    - Supports deterministic parallel execution across worker processes and null-only checkpointing.
    """
    rng = np.random.default_rng(seed)
    unique_sites = sorted(set(institutions))
    n = len(labels)
    baseline_auc = float(roc_auc_score(labels, baseline_probs))
    completed = load_permutation_checkpoint(checkpoint_path, "test2_null_max_deltas")
    null_max_deltas = np.empty(n_permutations, dtype=np.float64)
    for idx, val in enumerate(completed[:n_permutations]):
        null_max_deltas[idx] = val

    # Pre-generate deterministic permutation schedule so 1 worker == N workers == resumed
    perm_schedule: List[np.ndarray] = []
    site_idx_map = {site: np.where(institutions == site)[0] for site in unique_sites}
    for _ in range(n_permutations):
        conn_perm_indices = np.arange(n)
        for site in unique_sites:
            s_idx = site_idx_map[site]
            conn_perm_indices[s_idx] = rng.permutation(s_idx)
        perm_schedule.append(conn_perm_indices)

    pending_indices = [b for b in range(n_permutations) if b >= len(completed)]
    if not pending_indices:
        p_val = float((1.0 + np.sum(null_max_deltas >= observed_max_delta_auc)) / (1.0 + n_permutations))
        return p_val, null_max_deltas

    # Precompute outer & inner covariate-only OOF scores once (unpermuted covariates & labels)
    covariates = np.column_stack([motion_covariates, age_sex_covariates])
    cov_oof_by_outer: Dict[str, np.ndarray] = {}
    cov_test_by_outer: Dict[str, np.ndarray] = {}
    for test_site in unique_sites:
        outer_train_mask = institutions != test_site
        outer_test_mask = institutions == test_site
        inner_sites = [s for s in unique_sites if s != test_site]

        s_cov_oof = np.zeros(n, dtype=np.float64)
        if len(inner_sites) >= 2:
            in_splits_cov = [
                (outer_train_mask & (institutions != iv), institutions == iv)
                for iv in inner_sites
            ]
        else:
            tr_idx = np.where(outer_train_mask)[0]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            in_splits_cov = []
            for tr_rel, val_rel in skf.split(tr_idx, labels[tr_idx]):
                m_tr = np.zeros(n, dtype=bool)
                m_val = np.zeros(n, dtype=bool)
                m_tr[tr_idx[tr_rel]] = True
                m_val[tr_idx[val_rel]] = True
                in_splits_cov.append((m_tr, m_val))

        for inner_tr_mask, inner_val_mask in in_splits_cov:
            sc_in = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(covariates[inner_tr_mask])
            clf_in = LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed
            )
            clf_in.fit(sc_in.transform(covariates[inner_tr_mask]), labels[inner_tr_mask])
            s_cov_oof[inner_val_mask] = clf_in.predict_proba(sc_in.transform(covariates[inner_val_mask]))[:, 1]

        sc_out = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(covariates[outer_train_mask])
        clf_out = LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed
        )
        clf_out.fit(sc_out.transform(covariates[outer_train_mask]), labels[outer_train_mask])
        cov_oof_by_outer[test_site] = s_cov_oof
        cov_test_by_outer[test_site] = clf_out.predict_proba(sc_out.transform(covariates[outer_test_mask]))[:, 1]

    uses_tangent = any(fk == "tangent_space" for fk, _, _ in configs)
    tangent_all_by_subset: Dict[frozenset, np.ndarray] = {}
    if uses_tangent and use_cached_tangent:
        subsets_needed: Set[frozenset] = set()
        for test_site in unique_sites:
            outer_sites = frozenset(s for s in unique_sites if s != test_site)
            if outer_sites:
                subsets_needed.add(outer_sites)
            for inner_val_site in outer_sites:
                inner_sites = frozenset(s for s in outer_sites if s != inner_val_site)
                if inner_sites:
                    subsets_needed.add(inner_sites)

        if tangent_cache is not None and subsets_needed.issubset(tangent_cache.keys()):
            tangent_all_by_subset = {k: tangent_cache[k] for k in subsets_needed}
        else:
            tangent_all_by_subset = precompute_tangent_cache(
                timeseries_list=timeseries_list,
                institutions=institutions,
                subsets=sorted(subsets_needed, key=lambda s: tuple(sorted(s))),
                n_workers=n_workers,
            )

    use_workers = resolve_n_workers(n_workers)
    tasks = [(b, perm_schedule[b]) for b in pending_indices]

    if use_workers > 1 and len(tasks) >= 2:
        created_shm: List[SharedMemory] = []
        try:
            pearson_spec = _SharedArraySpec_create(pearson_z_edges, created_shm)
            tangent_specs = (
                {k: _SharedArraySpec_create(v, created_shm) for k, v in tangent_all_by_subset.items()}
                if use_cached_tangent
                else None
            )
            payload: Dict[str, object] = {
                "configs": list(configs),
                "unique_sites": list(unique_sites),
                "institutions": institutions,
                "labels": labels,
                "motion_covariates": motion_covariates,
                "pearson_spec": pearson_spec,
                "tangent_specs": tangent_specs,
                "tangent_all_by_subset": {},
                "use_cached_tangent": use_cached_tangent,
                "timeseries_list": timeseries_list if not use_cached_tangent else None,
                "cov_oof_by_outer": cov_oof_by_outer,
                "cov_test_by_outer": cov_test_by_outer,
                "baseline_auc": baseline_auc,
                "seed": seed,
            }
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes=min(use_workers, len(tasks)),
                initializer=_init_test2_worker,
                initargs=(payload,),
            ) as pool:
                for b, stat_val, _ in pool.imap(_test2_worker_entrypoint, tasks):
                    if progress_fn is not None:
                        progress_fn(f"Test 2 permutation {b + 1}/{n_permutations}")
                    null_max_deltas[b] = stat_val
                    save_permutation_checkpoint(checkpoint_path, "test2_null_max_deltas", null_max_deltas[: b + 1])
        finally:
            for shm in created_shm:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
    else:
        _WORKER_STATE.clear()
        _WORKER_STATE.update({
            "configs": list(configs),
            "unique_sites": list(unique_sites),
            "institutions": institutions,
            "labels": labels,
            "motion_covariates": motion_covariates,
            "pearson_z_edges": pearson_z_edges,
            "tangent_all_by_subset": tangent_all_by_subset,
            "use_cached_tangent": use_cached_tangent,
            "timeseries_list": timeseries_list if not use_cached_tangent else None,
            "cov_oof_by_outer": cov_oof_by_outer,
            "cov_test_by_outer": cov_test_by_outer,
            "baseline_auc": baseline_auc,
            "seed": seed,
        })
        try:
            for b, conn_perm in tasks:
                if progress_fn is not None:
                    progress_fn(f"Test 2 permutation {b + 1}/{n_permutations}")
                _, stat_val, _ = _eval_single_test2_perm((b, conn_perm))
                null_max_deltas[b] = stat_val
                save_permutation_checkpoint(checkpoint_path, "test2_null_max_deltas", null_max_deltas[: b + 1])
        finally:
            _WORKER_STATE.clear()

    p_val = float((1.0 + np.sum(null_max_deltas >= observed_max_delta_auc)) / (1.0 + n_permutations))
    return p_val, null_max_deltas


# =====================================================================
# "Interesting" Criteria Conjunction, Preflight, and Write-Once Output
# =====================================================================

def evaluate_interesting_criteria(
    y_true: np.ndarray,
    institutions: np.ndarray,
    probs_combined: np.ndarray,
    probs_motion_baseline: np.ndarray,
    probs_agesex_baseline: np.ndarray,
    probs_combined_baseline: np.ndarray,
    incremental_perm_p: float,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, object]:
    """
    Evaluate the strict 3-part conjunction required for an "interesting" finding:
      (a) Beats motion-only baseline AND age/sex baseline with 95% paired-bootstrap CI
          lower bounds strictly > 0.
      (b) Pooled delta-AUC of the fused model over the covariate-only model is > 0 with
          95% paired-bootstrap CI lower bound > 0 AND max-statistic permutation p < 0.05.
      (c) Per-institution delta-AUC > 0 in at least 4 of 5 institutions.
    Anything short of all three is "not interesting".
    """
    diff_motion, ci_lo_motion, ci_hi_motion = paired_bootstrap_auc_diff(
        y_true, probs_combined, probs_motion_baseline, n_boot=n_boot, seed=seed
    )
    diff_agesex, ci_lo_agesex, ci_hi_agesex = paired_bootstrap_auc_diff(
        y_true, probs_combined, probs_agesex_baseline, n_boot=n_boot, seed=seed
    )
    diff_cov, ci_lo_cov, ci_hi_cov = paired_bootstrap_auc_diff(
        y_true, probs_combined, probs_combined_baseline, n_boot=n_boot, seed=seed
    )
    cond_a = (ci_lo_motion > 0.0) and (ci_lo_agesex > 0.0)
    cond_b = (diff_cov > 0.0) and (ci_lo_cov > 0.0) and (incremental_perm_p < 0.05)

    per_site_deltas: Dict[str, float] = {}
    unique_sites = sorted(set(institutions))
    for site in unique_sites:
        mask = institutions == site
        auc_comb = float(roc_auc_score(y_true[mask], probs_combined[mask]))
        auc_base = float(roc_auc_score(y_true[mask], probs_combined_baseline[mask]))
        per_site_deltas[str(site)] = auc_comb - auc_base

    n_sites_positive = sum(1 for d in per_site_deltas.values() if d > 0.0)
    cond_c = n_sites_positive >= 4

    is_interesting = cond_a and cond_b and cond_c
    return {
        "verdict": "interesting" if is_interesting else "not interesting",
        "is_interesting": is_interesting,
        "cond_a_beats_baselines_ci_gt_0": cond_a,
        "diff_vs_motion": (diff_motion, ci_lo_motion, ci_hi_motion),
        "diff_vs_agesex": (diff_agesex, ci_lo_agesex, ci_hi_agesex),
        "cond_b_fused_ci_gt_0_and_perm_p_lt_005": cond_b,
        "diff_vs_covariate_baseline": (diff_cov, ci_lo_cov, ci_hi_cov),
        "incremental_perm_p": incremental_perm_p,
        "cond_c_sites_positive_ge_4": cond_c,
        "n_sites_positive": n_sites_positive,
        "per_site_deltas": per_site_deltas,
    }


def run_preflight(data_root: Path = Path("NeuroADHD/data/athena")) -> Dict[str, object]:
    """
    Verify that every required input exists and has expected shapes, columns, and counts
    WITHOUT reading any diagnostic label column (DX, Secondary Dx, ADHD Measure, etc.).
    Fails clearly if any required input is missing or malformed.
    """
    required_files = {
        "features_primary368": data_root / "features" / "dev_cc200_fisher_z_primary368.npz",
        "features_all404": data_root / "features" / "dev_cc200_fisher_z_all404.npz",
        "valid_roi_set": data_root / "features" / "dev_valid_roi_set.json",
        "phenotypes_tsv": data_root / "phenotypes" / "adhd200_preprocessed_phenotypics.tsv",
        "dev_fd_csv": data_root / "motion" / "dev_niak_fd_metrics.csv",
        "nyu_motion_csv": data_root / "staging" / "NYU_motion.csv",
        "nyu_intermediate_tar": data_root / "staging" / "intermediate_nyu.tar",
        "cc200_tcs_tar": data_root / "staging" / "ADHD200_CC200_TCs_filtfix.tar.gz",
    }
    for name, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Preflight failed: missing required input '{name}' at {path}")

    # 1. Check NPZ feature shapes WITHOUT reading 'y'
    summary: Dict[str, object] = {}
    for key, expected_n in [("features_primary368", 368), ("features_all404", 404)]:
        with np.load(required_files[key], allow_pickle=False) as d:
            for req_arr in ["X", "motion_X", "age_sex_X", "t_lengths", "institutions"]:
                if req_arr not in d.files:
                    raise ValueError(f"Preflight failed: '{req_arr}' missing in {required_files[key]}")
            x_shape = d["X"].shape
            m_shape = d["motion_X"].shape
            a_shape = d["age_sex_X"].shape
            t_shape = d["t_lengths"].shape
            if x_shape != (expected_n, 17955):
                raise ValueError(f"Preflight failed: {key} X shape {x_shape} != ({expected_n}, 17955)")
            if m_shape != (expected_n, 2) or a_shape != (expected_n, 2) or t_shape != (expected_n,):
                raise ValueError(f"Preflight failed: {key} covariate/length shape mismatch")
            summary[key] = {"X_shape": x_shape, "motion_shape": m_shape, "age_sex_shape": a_shape}

    # 2. Check ROI set JSON (190 populated CC200 ROIs)
    with open(required_files["valid_roi_set"], "r", encoding="utf-8") as f:
        roi_meta = json.load(f)
    if roi_meta.get("k_kept") != 190 or roi_meta.get("p_edges") != 17955:
        raise ValueError(
            f"Preflight failed: unexpected ROI set metadata k_kept={roi_meta.get('k_kept')}, p_edges={roi_meta.get('p_edges')}"
        )
    summary["valid_roi_keys"] = len(roi_meta)
    summary["valid_roi_k_kept"] = roi_meta["k_kept"]

    # 3. Check phenotype TSV non-label columns ONLY (never read DX or ADHD columns)
    with open(required_files["phenotypes_tsv"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        for req_col in ["ScanDir ID", "Site", "Gender", "Age"]:
            if req_col not in cols:
                raise ValueError(f"Preflight failed: missing column '{req_col}' in phenotypes TSV")
        n_pheno_rows = 0
        n_multiclass_rows = 0
        n_missing_age_sex = 0
        for row in reader:
            n_pheno_rows += 1
            if str(row.get("Site", "")).strip() in SITE_CODE_TO_INSTITUTION:
                n_multiclass_rows += 1
                try:
                    a_val = float(str(row.get("Age", "")).strip())
                    s_val = float(str(row.get("Gender", "")).strip())
                    if not (math.isfinite(a_val) and math.isfinite(s_val)):
                        n_missing_age_sex += 1
                except ValueError:
                    n_missing_age_sex += 1
    if n_pheno_rows < 900:
        raise ValueError(f"Preflight failed: unexpected phenotype row count {n_pheno_rows}")
    summary["phenotypes_rows"] = n_pheno_rows
    summary["phenotypes_multiclass_rows"] = n_multiclass_rows
    summary["phenotypes_missing_age_sex"] = n_missing_age_sex

    # 4. Check motion & matching inputs (dev_niak_fd_metrics.csv and NYU_motion.csv)
    with open(required_files["dev_fd_csv"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fd_cols = reader.fieldnames or []
        for req_col in ["subject_id_norm", "institution", "mean_fd_mm", "max_disp_mm", "frames_cc200", "exclude_primary_050"]:
            if req_col not in fd_cols:
                raise ValueError(f"Preflight failed: missing column '{req_col}' in dev_niak_fd_metrics.csv")
        n_fd_rows = 0
        n_dev_retained = 0
        for row in reader:
            n_fd_rows += 1
            if int(row["exclude_primary_050"]) == 0:
                n_dev_retained += 1
    if n_fd_rows != 404 or n_dev_retained != 368:
        raise ValueError(
            f"Preflight failed: dev FD counts mismatch (total={n_fd_rows} != 404 or retained={n_dev_retained} != 368)"
        )
    summary["dev_fd_rows"] = n_fd_rows
    summary["dev_fd_retained"] = n_dev_retained

    with open(required_files["nyu_motion_csv"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        nyu_cols = reader.fieldnames or []
        if "File" not in nyu_cols or "Max Motion (mm)" not in nyu_cols:
            raise ValueError("Preflight failed: missing expected columns in NYU_motion.csv")
        n_nyu_motion = 0
        n_nyu_rest1 = 0
        n_nyu_rest2 = 0
        for row in reader:
            n_nyu_motion += 1
            fname = row.get("File", "")
            if "rest_1" in fname:
                n_nyu_rest1 += 1
            elif "rest_2" in fname:
                n_nyu_rest2 += 1
    if n_nyu_rest1 < 216:
        raise ValueError(f"Preflight failed: expected >= 216 NYU rest_1 motion rows, found {n_nyu_rest1}")
    summary["nyu_motion_rows"] = n_nyu_motion
    summary["nyu_motion_rest1"] = n_nyu_rest1
    summary["nyu_motion_rest2"] = n_nyu_rest2

    # 5. Check archives are readable tar files (no full extraction needed)
    with tarfile.open(required_files["nyu_intermediate_tar"], "r:") as tf:
        members = tf.getmembers()
        n_nyu_tar = len(members)
        n_nyu_files = sum(1 for m in members if m.isfile())
        n_nyu_dirs = sum(1 for m in members if m.isdir())
    if n_nyu_files == 0:
        raise ValueError("Preflight failed: intermediate_nyu.tar has no regular files")
    summary["nyu_tar_entries"] = n_nyu_tar
    summary["nyu_tar_files"] = n_nyu_files
    summary["nyu_tar_dirs"] = n_nyu_dirs
    summary["cc200_tcs_tar_bytes"] = required_files["cc200_tcs_tar"].stat().st_size

    # 6. Check label-free cache staleness if cache exists
    cache_path = data_root / "features" / "exploratory_label_free_cache.npz"
    if cache_path.is_file():
        cache_mtime = cache_path.stat().st_mtime
        for src_key in ["cc200_tcs_tar", "nyu_intermediate_tar", "dev_fd_csv", "phenotypes_tsv"]:
            if required_files[src_key].stat().st_mtime > cache_mtime:
                raise ValueError(
                    f"Preflight failed: label-free cache {cache_path} is STALE relative to {src_key}. "
                    "Rebuild cache with --label-free-check."
                )
        with np.load(cache_path, allow_pickle=True) as cd:
            if "meta_json" not in cd.files or cd["pearson_z_edges"].shape != (620, 17955):
                raise ValueError(
                    f"Preflight failed: label-free cache {cache_path} has stale schema/shape."
                )
        summary["label_free_cache_status"] = "fresh"
    else:
        summary["label_free_cache_status"] = "not_built_yet"

    return summary


def run_label_free_check(
    data_root: Path = Path("NeuroADHD/data/athena"),
    cache_path: Path = Path("NeuroADHD/data/athena/features/exploratory_label_free_cache.npz"),
) -> Dict[str, object]:
    """
    Execute the REAL label-free loader on the real archive (no labels, no models, no AUCs)
    and verify counts, ROI set, Fisher-z reproduction vs dev_cc200_fisher_z_all404.npz,
    FD retained counts vs Study 2, NaN/Inf counts, multi-run availability, memory, and git-ignore.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    inputs = load_label_free_inputs(data_root=data_root, cache_path=cache_path, force_rebuild=True)
    build_sec = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_total = int(len(inputs.subject_ids))
    n_nyu = int(np.sum(inputs.institutions == "NYU"))
    n_dev = int(n_total - n_nyu)

    # Scan lengths per site (min / median / max)
    scan_lengths_per_site: Dict[str, Dict[str, float]] = {}
    for site in sorted(set(inputs.institutions)):
        lens = inputs.t_lengths[inputs.institutions == site]
        scan_lengths_per_site[str(site)] = {
            "n": int(len(lens)),
            "min": int(np.min(lens)),
            "median": float(np.median(lens)),
            "max": int(np.max(lens)),
        }

    # ROI set check vs dev_valid_roi_set.json (190 ROIs)
    roi_json_path = data_root / "features" / "dev_valid_roi_set.json"
    with open(roi_json_path, "r", encoding="utf-8") as f:
        roi_meta = json.load(f)
    expected_roi_names = roi_meta["kept_roi_names"]
    roi_set_matches = (len(inputs.roi_names) == 190) and (inputs.roi_names == expected_roi_names)

    # Reproduce rest_1 Fisher-z edges for the 404 dev scans vs dev_cc200_fisher_z_all404.npz (NO 'y' read!)
    dev_mask = inputs.institutions != "NYU"
    dev_z = inputs.pearson_z_edges[dev_mask]
    dev_mot = inputs.motion_covariates[dev_mask]
    dev_as = inputs.age_sex_covariates[dev_mask]
    dev_t = inputs.t_lengths[dev_mask]
    dev_inst = inputs.institutions[dev_mask]

    with np.load(data_root / "features" / "dev_cc200_fisher_z_all404.npz", allow_pickle=False) as d404:
        ref_z = d404["X"]
        ref_mot = d404["motion_X"]
        ref_as = d404["age_sex_X"]
        ref_t = d404["t_lengths"]
        ref_inst = d404["institutions"]

    # Match rows 1:1 via (institution, motion_X, age_sex_X, t_lengths)
    used_ref: Set[int] = set()
    max_abs_diff = 0.0
    matched_rows = 0
    for i in range(len(dev_z)):
        best_j = None
        for j in range(len(ref_z)):
            if j in used_ref:
                continue
            if (
                str(dev_inst[i]) == str(ref_inst[j])
                and int(dev_t[i]) == int(ref_t[j])
                and np.allclose(dev_mot[i], ref_mot[j], atol=1e-5)
                and np.allclose(dev_as[i], ref_as[j], atol=1e-4)
            ):
                best_j = j
                break
        if best_j is not None:
            used_ref.add(best_j)
            matched_rows += 1
            diff = float(np.max(np.abs(dev_z[i] - ref_z[best_j])))
            if diff > max_abs_diff:
                max_abs_diff = diff

    # Label-free FD retained counts vs Study 2 recorded values (dev=368, NYU=196)
    dev_retained = int(np.sum((inputs.institutions != "NYU") & (inputs.exclude_primary_050 == 0)))
    nyu_retained = int(np.sum((inputs.institutions == "NYU") & (inputs.exclude_primary_050 == 0)))
    study2_dev_diff = dev_retained - 368
    study2_nyu_diff = nyu_retained - 196

    # NaN / Inf counts in every array (imaging/motion must be 0; primary_564 age/sex must be 0)
    primary_mask = inputs.exclude_primary_050 == 0
    nan_inf_counts = {
        "pearson_z_edges": int(np.sum(~np.isfinite(inputs.pearson_z_edges))),
        "motion_covariates": int(np.sum(~np.isfinite(inputs.motion_covariates))),
        "age_sex_covariates_all620": int(np.sum(~np.isfinite(inputs.age_sex_covariates))),
        "age_sex_covariates_primary564": int(np.sum(~np.isfinite(inputs.age_sex_covariates[primary_mask]))),
        "multirun_pearson_z_avg": int(np.sum(~np.isfinite(inputs.multirun_pearson_z_avg))),
        "timeseries_list": int(sum(int(np.sum(~np.isfinite(ts))) for ts in inputs.timeseries_list)),
    }

    # Multi-run availability per site and per subject
    mr_per_site_subjects = {
        "NYU_subjects": int(np.sum(inputs.multirun_institutions == "NYU")),
        "OHSU_subjects": int(np.sum(inputs.multirun_institutions == "OHSU")),
        "NYU_ge2_runs": int(np.sum((inputs.multirun_institutions == "NYU") & (inputs.multirun_run_counts >= 2))),
        "OHSU_ge2_runs": int(np.sum((inputs.multirun_institutions == "OHSU") & (inputs.multirun_run_counts >= 2))),
        "OHSU_3_runs": int(np.sum((inputs.multirun_institutions == "OHSU") & (inputs.multirun_run_counts >= 3))),
    }

    # Git-ignore proof
    gi_proc = subprocess.run(
        ["git", "check-ignore", "-v", str(cache_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    git_ignore_proof = gi_proc.stdout.strip() if gi_proc.returncode == 0 else "NOT IGNORED"

    checks_pass = (
        n_total == 620
        and n_nyu == 216
        and n_dev == 404
        and roi_set_matches
        and matched_rows == 404
        and max_abs_diff < 1e-6
        and dev_retained == 368
        and nyu_retained == 196
        and nan_inf_counts["pearson_z_edges"] == 0
        and nan_inf_counts["motion_covariates"] == 0
        and nan_inf_counts["age_sex_covariates_primary564"] == 0
        and nan_inf_counts["multirun_pearson_z_avg"] == 0
        and nan_inf_counts["timeseries_list"] == 0
        and gi_proc.returncode == 0
    )

    return {
        "status": "PASS" if checks_pass else "FAIL",
        "scans_loaded": {"total": n_total, "nyu": n_nyu, "dev": n_dev},
        "phenotype_coverage": {
            "age_sex_present": n_total,
            "missing_counts": inputs.pheno_missing_age_sex_counts,
        },
        "scan_lengths_per_site": scan_lengths_per_site,
        "roi_set_matches_190": roi_set_matches,
        "zero_variance_roi_counts_per_site": inputs.zero_var_roi_counts_per_site,
        "dev_404_fisher_z_reproduction": {
            "matched_rows": matched_rows,
            "max_abs_diff": max_abs_diff,
            "row_order_note": "matched 1:1 via (institution, motion_X, age_sex_X, t_lengths); Study 2 npz ordered by site CSV, cache ordered by sorted ID",
        },
        "fd_primary_retained": {
            "dev_retained": dev_retained,
            "study2_dev_recorded": 368,
            "dev_diff": study2_dev_diff,
            "nyu_retained": nyu_retained,
            "study2_nyu_recorded": 196,
            "nyu_diff": study2_nyu_diff,
        },
        "nan_inf_counts": nan_inf_counts,
        "multirun_availability": {
            "runs_per_site": inputs.multirun_availability,
            "subjects_summary": mr_per_site_subjects,
        },
        "build_time_sec": round(build_sec, 3),
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
        "cache_path": str(cache_path),
        "git_ignore_proof": git_ignore_proof,
    }


def write_results_once(output_path: Path, payload: Dict[str, object]) -> None:
    """
    Write final results JSON strictly once. Refuses to overwrite if output_path already exists.
    """
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results file: {output_path}. "
            "Results are write-once to prevent re-running after viewing partial/prior output."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@dataclass
class LabelFreeInputs:
    subject_ids: np.ndarray
    institutions: np.ndarray
    timeseries_list: List[np.ndarray]
    pearson_z_edges: np.ndarray
    motion_covariates: np.ndarray
    age_sex_covariates: np.ndarray
    exclude_primary_050: np.ndarray
    t_lengths: np.ndarray
    multirun_subject_ids: np.ndarray
    multirun_institutions: np.ndarray
    multirun_pearson_z_avg: np.ndarray
    multirun_motion_covariates: np.ndarray
    multirun_age_sex_covariates: np.ndarray
    multirun_run_counts: np.ndarray
    roi_names: List[str] = field(default_factory=list)
    zero_var_roi_counts_per_site: Dict[str, int] = field(default_factory=dict)
    pheno_missing_age_sex_counts: Dict[str, int] = field(default_factory=dict)
    multirun_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class CohortData:
    name: str
    subject_ids: np.ndarray
    institutions: np.ndarray
    timeseries_list: List[np.ndarray]
    pearson_z_edges: np.ndarray
    motion_covariates: np.ndarray
    age_sex_covariates: np.ndarray
    labels: np.ndarray


def parse_cc200_1d_text(text: str, return_roi_names: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """Parse a CC200 .1D timecourse file into a (T, 190) float64 array (and optionally ROI names)."""
    rows: List[List[float]] = []
    roi_names: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("File\t") or s.startswith("File "):
            header = s.split()
            roi_names = header[2:]
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        rows.append([float(x) for x in parts[2:]])
    if not rows:
        raise ValueError("Empty CC200 .1D timecourse stream")
    arr = np.array(rows, dtype=np.float64)
    if return_roi_names:
        return arr, roi_names
    return arr


def _compute_nyu_rest1_motion(nyu_tar_path: Path) -> Dict[str, Tuple[float, float, int]]:
    """
    Compute Power et al. (2012) FD and max_disp_mm for NYU rest_1 scans from intermediate_nyu.tar
    using the exact Study 2 conventions in scripts/compute_athena_fd.py (first 3 frames dropped).
    Zero diagnosis labels loaded.
    """
    nyu_motion: Dict[str, Tuple[float, float, int]] = {}
    with tarfile.open(nyu_tar_path, "r") as tf:
        members = tf.getmembers()
        target_members: Dict[str, tarfile.TarInfo] = {}
        for member in members:
            if member.isfile() and "motion_target" in member.name and member.name.endswith(".nii.gz"):
                m = re.search(r"X_(\d+)_session_1_run1", member.name)
                if m:
                    target_members[m.group(1).lstrip("0") or "0"] = member

        for member in members:
            if not member.isfile():
                continue
            if "motion_parameters" not in member.name or not member.name.endswith(".mat"):
                continue
            m = re.search(r"X_(\d+)_session_(\d+)_run(\d+)", member.name)
            if not m:
                continue
            raw_id, sess_num, run_num = m.group(1), m.group(2), m.group(3)
            if sess_num != "1" or run_num != "1":
                continue
            sub_norm = raw_id.lstrip("0") or "0"
            if sub_norm not in target_members:
                continue
            t_stream = tf.extractfile(target_members[sub_norm])
            m_stream = tf.extractfile(member)
            if t_stream is None or m_stream is None:
                continue
            center_world = read_nifti_volume_center_world(t_stream.read())
            transf = parse_octave_transf_text(m_stream.read().decode("utf-8", errors="replace"))
            if transf.shape[2] <= DUMMY_FRAMES_DROPPED:
                continue
            transf_centered = center_shift_transf(transf[:, :, DUMMY_FRAMES_DROPPED:], center_world)
            params = extract_6dof_params(transf_centered)
            fd = compute_power_fd(params, radius_mm=HEAD_RADIUS_MM)
            mean_fd = float(np.mean(fd))
            rel_trans = params[:, 3:6] - params[0, 3:6]
            max_disp_mm = float(np.max(np.linalg.norm(rel_trans, axis=1)))
            pct_gt_050 = float(100.0 * np.mean(fd > 0.50))
            ex_050 = int((mean_fd > 0.50) or (pct_gt_050 > 20.0))
            nyu_motion[sub_norm] = (mean_fd, max_disp_mm, ex_050)
    return nyu_motion


def load_label_free_inputs(
    data_root: Path = Path("NeuroADHD/data/athena"),
    cache_path: Optional[Path] = Path("NeuroADHD/data/athena/features/exploratory_label_free_cache.npz"),
    force_rebuild: bool = False,
) -> LabelFreeInputs:
    """
    Load and compute all label-free inputs (CC200 Fisher-z edges, timeseries for tangent space,
    motion metrics, age, sex, and NYU+OHSU multi-run averaged Fisher-z features).
    Reads ZERO diagnosis/label columns and caches label-free arrays under the git-ignored athena folder.
    """
    if cache_path is not None and cache_path.is_file() and not force_rebuild:
        with np.load(cache_path, allow_pickle=True) as d:
            meta_json = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
            return LabelFreeInputs(
                subject_ids=d["subject_ids"],
                institutions=d["institutions"],
                timeseries_list=[np.asarray(ts, dtype=np.float64) for ts in d["timeseries_obj"]],
                pearson_z_edges=np.asarray(d["pearson_z_edges"], dtype=np.float64),
                motion_covariates=np.asarray(d["motion_covariates"], dtype=np.float64),
                age_sex_covariates=np.asarray(d["age_sex_covariates"], dtype=np.float64),
                exclude_primary_050=np.asarray(d["exclude_primary_050"], dtype=np.int64),
                t_lengths=np.asarray(d["t_lengths"], dtype=np.int64),
                multirun_subject_ids=d["multirun_subject_ids"],
                multirun_institutions=d["multirun_institutions"],
                multirun_pearson_z_avg=np.asarray(d["multirun_pearson_z_avg"], dtype=np.float64),
                multirun_motion_covariates=np.asarray(d["multirun_motion_covariates"], dtype=np.float64),
                multirun_age_sex_covariates=np.asarray(d["multirun_age_sex_covariates"], dtype=np.float64),
                multirun_run_counts=np.asarray(d["multirun_run_counts"], dtype=np.int64),
                roi_names=meta_json.get("roi_names", []),
                zero_var_roi_counts_per_site=meta_json.get("zero_var_roi_counts_per_site", {}),
                pheno_missing_age_sex_counts=meta_json.get("pheno_missing_age_sex_counts", {}),
                multirun_availability=meta_json.get("multirun_availability", {}),
            )

    phenotypes_tsv = data_root / "phenotypes" / "adhd200_preprocessed_phenotypics.tsv"
    dev_fd_csv = data_root / "motion" / "dev_niak_fd_metrics.csv"
    nyu_tar = data_root / "staging" / "intermediate_nyu.tar"
    cc200_archive = data_root / "staging" / "ADHD200_CC200_TCs_filtfix.tar.gz"

    # 1. Non-label phenotype columns ONLY (never touch DX or ADHD columns)
    pheno_cov: Dict[str, Tuple[str, float, float]] = {}
    pheno_missing: Dict[str, int] = {"missing_age": 0, "missing_sex": 0, "total_multiclass_rows": 0}
    with open(phenotypes_tsv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            site_code = str(row.get("Site", "")).strip()
            if site_code not in SITE_CODE_TO_INSTITUTION:
                continue
            pheno_missing["total_multiclass_rows"] += 1
            sid = str(row.get("ScanDir ID", "")).strip().lstrip("0") or "0"
            age_str = str(row.get("Age", "")).strip()
            sex_str = str(row.get("Gender", "")).strip()
            try:
                age_val = float(age_str)
                if not math.isfinite(age_val):
                    raise ValueError
            except ValueError:
                pheno_missing["missing_age"] += 1
                age_val = float("nan")
            try:
                sex_val = float(sex_str)
                if not math.isfinite(sex_val):
                    raise ValueError
            except ValueError:
                pheno_missing["missing_sex"] += 1
                sex_val = float("nan")
            pheno_cov[sid] = (SITE_CODE_TO_INSTITUTION[site_code], age_val, sex_val)

    # 2. Motion metrics: saved dev metrics + NYU from intermediate_nyu.tar
    motion_map: Dict[str, Tuple[float, float, int]] = {}
    with open(dev_fd_csv, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            sid = str(row["subject_id_norm"]).strip().lstrip("0") or "0"
            motion_map[sid] = (
                float(row["mean_fd_mm"]),
                float(row["max_disp_mm"]),
                int(row["exclude_primary_050"]),
            )
    nyu_motion = _compute_nyu_rest1_motion(nyu_tar)
    motion_map.update(nyu_motion)

    # 3. Stream CC200 archive in memory (rest_1 ONLY for primary/sensitivity; rest_2/3 tracked for CFG_13-16)
    rest1_data: Dict[str, Tuple[str, np.ndarray, np.ndarray]] = {}
    multirun_runs: Dict[str, List[np.ndarray]] = defaultdict(list)
    multirun_avail: Dict[str, Dict[str, int]] = {
        "NYU": {"rest_1": 0, "rest_2": 0, "rest_3": 0},
        "OHSU": {"rest_1": 0, "rest_2": 0, "rest_3": 0},
    }
    zero_var_per_site: Dict[str, int] = defaultdict(int)
    observed_roi_names: List[str] = []

    with tarfile.open(cc200_archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            site_dir = parts[0]
            if site_dir not in INCLUDED_ARCHIVE_SITES:
                continue
            fname = parts[-1]
            if not (fname.startswith("sfnwmrda") and fname.endswith(".1D")):
                continue
            m_run = re.search(r"rest_(\d+)", fname)
            if not m_run:
                continue
            run_num = int(m_run.group(1))
            inst = "Peking" if site_dir.startswith("Peking") else site_dir
            sid = parts[1].lstrip("0") or "0"
            if sid not in pheno_cov or sid not in motion_map:
                continue

            if run_num == 1 or inst in {"NYU", "OHSU"}:
                stream = tf.extractfile(member)
                if stream is None:
                    continue
                ts, r_names = parse_cc200_1d_text(stream.read().decode("utf-8", errors="replace"), return_roi_names=True)
                if not observed_roi_names and r_names:
                    observed_roi_names = r_names
                n_zero_var = int(np.sum(np.std(ts, axis=0) <= 0.0))
                if run_num == 1:
                    zero_var_per_site[inst] += n_zero_var
                    z_edges = timeseries_to_fisher_z(ts)
                    rest1_data[sid] = (inst, ts, z_edges)
                else:
                    z_edges = timeseries_to_fisher_z(ts)
                if inst in {"NYU", "OHSU"} and run_num in {1, 2, 3}:
                    multirun_runs[sid].append(z_edges)
                    multirun_avail[inst][f"rest_{run_num}"] += 1

    sorted_sids = sorted(rest1_data.keys())
    institutions_arr = np.array([rest1_data[s][0] for s in sorted_sids], dtype="<U16")
    ts_list = [rest1_data[s][1] for s in sorted_sids]
    z_arr = np.vstack([rest1_data[s][2] for s in sorted_sids])
    motion_arr = np.array([[motion_map[s][0], motion_map[s][1]] for s in sorted_sids], dtype=np.float64)
    excl_arr = np.array([motion_map[s][2] for s in sorted_sids], dtype=np.int64)
    age_sex_arr = np.array([[pheno_cov[s][1], pheno_cov[s][2]] for s in sorted_sids], dtype=np.float64)
    t_len_arr = np.array([ts.shape[0] for ts in ts_list], dtype=np.int64)

    # Multi-run NYU + OHSU (one row per subject)
    mr_sids = sorted(s for s in multirun_runs.keys() if s in rest1_data)
    mr_inst = np.array([rest1_data[s][0] for s in mr_sids], dtype="<U16")
    mr_z_avg = np.vstack([np.mean(multirun_runs[s], axis=0) for s in mr_sids])
    mr_mot = np.array([[motion_map[s][0], motion_map[s][1]] for s in mr_sids], dtype=np.float64)
    mr_as = np.array([[pheno_cov[s][1], pheno_cov[s][2]] for s in mr_sids], dtype=np.float64)
    mr_counts = np.array([len(multirun_runs[s]) for s in mr_sids], dtype=np.int64)

    for inst_name in sorted(set(institutions_arr)):
        zero_var_per_site.setdefault(str(inst_name), 0)

    inputs = LabelFreeInputs(
        subject_ids=np.array(sorted_sids, dtype="<U16"),
        institutions=institutions_arr,
        timeseries_list=ts_list,
        pearson_z_edges=z_arr,
        motion_covariates=motion_arr,
        age_sex_covariates=age_sex_arr,
        exclude_primary_050=excl_arr,
        t_lengths=t_len_arr,
        multirun_subject_ids=np.array(mr_sids, dtype="<U16"),
        multirun_institutions=mr_inst,
        multirun_pearson_z_avg=mr_z_avg,
        multirun_motion_covariates=mr_mot,
        multirun_age_sex_covariates=mr_as,
        multirun_run_counts=mr_counts,
        roi_names=observed_roi_names,
        zero_var_roi_counts_per_site=dict(zero_var_per_site),
        pheno_missing_age_sex_counts=pheno_missing,
        multirun_availability=multirun_avail,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ts_obj = np.empty(len(ts_list), dtype=object)
        for i, ts in enumerate(ts_list):
            ts_obj[i] = ts.astype(np.float32)
        meta_payload = json.dumps(
            {
                "roi_names": inputs.roi_names,
                "zero_var_roi_counts_per_site": inputs.zero_var_roi_counts_per_site,
                "pheno_missing_age_sex_counts": inputs.pheno_missing_age_sex_counts,
                "multirun_availability": inputs.multirun_availability,
            }
        )
        np.savez(
            cache_path,
            subject_ids=inputs.subject_ids,
            institutions=inputs.institutions,
            timeseries_obj=ts_obj,
            pearson_z_edges=inputs.pearson_z_edges,
            motion_covariates=inputs.motion_covariates,
            age_sex_covariates=inputs.age_sex_covariates,
            exclude_primary_050=inputs.exclude_primary_050,
            t_lengths=inputs.t_lengths,
            multirun_subject_ids=inputs.multirun_subject_ids,
            multirun_institutions=inputs.multirun_institutions,
            multirun_pearson_z_avg=inputs.multirun_pearson_z_avg,
            multirun_motion_covariates=inputs.multirun_motion_covariates,
            multirun_age_sex_covariates=inputs.multirun_age_sex_covariates,
            multirun_run_counts=inputs.multirun_run_counts,
            meta_json=np.array(meta_payload),
        )
    return inputs


def load_labels_last(
    phenotypes_tsv: Path,
    subject_ids: Sequence[str],
) -> np.ndarray:
    """
    Read diagnostic labels (DX) ONLY after all label-free inputs have been built.
    Maps DX='0' -> 0 (TDC), DX in {'1', '2', '3'} -> 1 (ADHD).
    """
    dx_map: Dict[str, int] = {}
    with open(phenotypes_tsv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = str(row.get("ScanDir ID", "")).strip().lstrip("0") or "0"
            dx_raw = str(row.get("DX", "")).strip()
            if dx_raw == "0":
                dx_map[sid] = 0
            elif dx_raw in {"1", "2", "3"}:
                dx_map[sid] = 1
    out = []
    for sid in subject_ids:
        s_norm = str(sid).strip().lstrip("0") or "0"
        if s_norm not in dx_map:
            raise ValueError("Missing binary DX label for a subject in cohort")
        out.append(dx_map[s_norm])
    return np.array(out, dtype=np.int64)


def _subset_cohort(name: str, inputs: LabelFreeInputs, labels: np.ndarray, indices: Sequence[int]) -> CohortData:
    idx_arr = np.asarray(indices, dtype=np.int64)
    return CohortData(
        name=name,
        subject_ids=inputs.subject_ids[idx_arr],
        institutions=inputs.institutions[idx_arr],
        timeseries_list=[inputs.timeseries_list[int(i)] for i in idx_arr],
        pearson_z_edges=inputs.pearson_z_edges[idx_arr],
        motion_covariates=inputs.motion_covariates[idx_arr],
        age_sex_covariates=inputs.age_sex_covariates[idx_arr],
        labels=labels[idx_arr],
    )


def build_cohorts(
    inputs: LabelFreeInputs,
    labels_rest1: np.ndarray,
    labels_multirun: np.ndarray,
) -> Dict[str, CohortData]:
    """
    Construct the primary N=564 cohort, N=620 all-scans cohort, 215-pair and 226-pair
    deterministic motion-matched cohorts, and the CFG_13-16 NYU+OHSU multi-run cohort.
    """
    all_idx = np.arange(len(labels_rest1))
    clean_idx = np.where(inputs.exclude_primary_050 == 0)[0]

    matched_clean_rel = match_subjects_on_motion_within_site(
        subject_ids=inputs.subject_ids[clean_idx],
        site_names=inputs.institutions[clean_idx],
        mean_fd_values=inputs.motion_covariates[clean_idx, 0],
        labels=labels_rest1[clean_idx],
        caliper_mm=0.05,
    )
    matched_clean_idx = clean_idx[matched_clean_rel]

    matched_all_idx = match_subjects_on_motion_within_site(
        subject_ids=inputs.subject_ids,
        site_names=inputs.institutions,
        mean_fd_values=inputs.motion_covariates[:, 0],
        labels=labels_rest1,
        caliper_mm=0.05,
    )

    multirun_cohort = CohortData(
        name="multirun_nyu_ohsu",
        subject_ids=inputs.multirun_subject_ids,
        institutions=inputs.multirun_institutions,
        timeseries_list=[],
        pearson_z_edges=inputs.multirun_pearson_z_avg,
        motion_covariates=inputs.multirun_motion_covariates,
        age_sex_covariates=inputs.multirun_age_sex_covariates,
        labels=labels_multirun,
    )

    return {
        "primary_564": _subset_cohort("primary_564", inputs, labels_rest1, clean_idx),
        "all_620": _subset_cohort("all_620", inputs, labels_rest1, all_idx),
        "matched_215": _subset_cohort("matched_215", inputs, labels_rest1, matched_clean_idx),
        "matched_226": _subset_cohort("matched_226", inputs, labels_rest1, matched_all_idx),
        "multirun_nyu_ohsu": multirun_cohort,
    }


def _evaluate_cohort_grid(
    cohort: CohortData,
    configs: Sequence[Tuple[str, str, str, str]],
    n_boot: int,
    seed: int,
    failing_config_ids: Optional[Set[str]] = None,
    progress_fn: Optional[Callable[[str], None]] = None,
    progress_prefix: str = "Cohort",
    subject_covs: Optional[np.ndarray] = None,
    n_workers: int = 1,
) -> Dict[str, object]:
    """Evaluate baselines and all candidate configurations (late fusion + connectivity-only) on a cohort."""
    phase_timings: Dict[str, float] = {}
    t_base0 = time.perf_counter()
    baselines_probs = evaluate_non_imaging_baselines(
        motion_covariates=cohort.motion_covariates,
        age_sex_covariates=cohort.age_sex_covariates,
        labels=cohort.labels,
        institutions=cohort.institutions,
        random_state=seed,
    )
    baselines_summary: Dict[str, object] = {}
    t_boot_acc = 0.0
    for b_name, b_probs in baselines_probs.items():
        tb0 = time.perf_counter()
        auc_pt, auc_lo, auc_hi = stratified_bootstrap_ci(
            cohort.labels, b_probs, roc_auc_score, n_boot=n_boot, seed=seed
        )
        t_boot_acc += time.perf_counter() - tb0
        baselines_summary[b_name] = {"auc": auc_pt, "ci_lo": auc_lo, "ci_hi": auc_hi}
    phase_timings["baselines_sec"] = round(time.perf_counter() - t_base0, 3)

    base_cov_probs = baselines_probs["motion_age_sex"]
    unique_sites = sorted(set(cohort.institutions))

    # Precompute 15-split tangent space cache once per cohort if any config uses tangent_space
    t_tang0 = time.perf_counter()
    tangent_cache: Optional[Dict[frozenset, np.ndarray]] = None
    if any(fk == "tangent_space" for _, fk, _, _ in configs) and len(unique_sites) >= 2:
        subsets: Set[frozenset] = set()
        for ts_site in unique_sites:
            outer_s = frozenset(s for s in unique_sites if s != ts_site)
            subsets.add(outer_s)
            for iv_site in outer_s:
                subsets.add(frozenset(s for s in outer_s if s != iv_site))
        tangent_cache = precompute_tangent_cache(
            timeseries_list=cohort.timeseries_list,
            institutions=cohort.institutions,
            subsets=sorted(subsets, key=lambda s: tuple(sorted(s))),
            subject_covs=subject_covs,
            n_workers=n_workers,
        )
    phase_timings["tangent_precompute_sec"] = round(time.perf_counter() - t_tang0, 3)

    config_results: Dict[str, Dict[str, object]] = {}
    fused_probs_by_cfg: Dict[str, np.ndarray] = {}
    conn_probs_by_cfg: Dict[str, np.ndarray] = {}
    failed_ids: List[str] = []

    for idx_cfg, (cfg_id, fk, sel, clf_name) in enumerate(configs, start=1):
        if progress_fn is not None:
            progress_fn(f"{progress_prefix}: config {idx_cfg}/{len(configs)}")
        try:
            if failing_config_ids and cfg_id in failing_config_ids:
                raise RuntimeError(f"Injected configuration failure for {cfg_id}")

            conn_probs, _ = evaluate_loio_pipeline(
                feature_kind=fk,
                selection_mode=sel,
                classifier_name=clf_name,
                timeseries_list=cohort.timeseries_list,
                pearson_z_edges=cohort.pearson_z_edges,
                motion_covariates=cohort.motion_covariates,
                labels=cohort.labels,
                institutions=cohort.institutions,
                random_state=seed,
                tangent_cache=tangent_cache,
            )
            fused_probs = evaluate_combined_covariates_and_connectivity(
                feature_kind=fk,
                selection_mode=sel,
                classifier_name=clf_name,
                timeseries_list=cohort.timeseries_list,
                pearson_z_edges=cohort.pearson_z_edges,
                motion_covariates=cohort.motion_covariates,
                age_sex_covariates=cohort.age_sex_covariates,
                labels=cohort.labels,
                institutions=cohort.institutions,
                random_state=seed,
                tangent_cache=tangent_cache,
            )

            tb0 = time.perf_counter()
            c_auc, c_lo, c_hi = stratified_bootstrap_ci(cohort.labels, conn_probs, roc_auc_score, n_boot=n_boot, seed=seed)
            f_auc, f_lo, f_hi = stratified_bootstrap_ci(cohort.labels, fused_probs, roc_auc_score, n_boot=n_boot, seed=seed)
            d_cov, d_cov_lo, d_cov_hi = paired_bootstrap_auc_diff(
                cohort.labels, fused_probs, base_cov_probs, n_boot=n_boot, seed=seed
            )
            d_mot, d_mot_lo, d_mot_hi = paired_bootstrap_auc_diff(
                cohort.labels, fused_probs, baselines_probs["motion"], n_boot=n_boot, seed=seed
            )
            d_as, d_as_lo, d_as_hi = paired_bootstrap_auc_diff(
                cohort.labels, fused_probs, baselines_probs["age_sex"], n_boot=n_boot, seed=seed
            )

            per_site: Dict[str, Dict[str, object]] = {}
            for site in unique_sites:
                m = cohort.institutions == site
                y_s = cohort.labels[m]
                sc_auc, sc_lo, sc_hi = stratified_bootstrap_ci(y_s, conn_probs[m], roc_auc_score, n_boot=n_boot, seed=seed)
                sf_auc, sf_lo, sf_hi = stratified_bootstrap_ci(y_s, fused_probs[m], roc_auc_score, n_boot=n_boot, seed=seed)
                sb_auc, sb_lo, sb_hi = stratified_bootstrap_ci(y_s, base_cov_probs[m], roc_auc_score, n_boot=n_boot, seed=seed)
                sd_pt, sd_lo, sd_hi = paired_bootstrap_auc_diff(y_s, fused_probs[m], base_cov_probs[m], n_boot=n_boot, seed=seed)
                per_site[str(site)] = {
                    "conn_only_auc": sc_auc,
                    "conn_only_ci": [sc_lo, sc_hi],
                    "fused_auc": sf_auc,
                    "fused_auc_ci": [sf_lo, sf_hi],
                    "cov_auc": sb_auc,
                    "cov_auc_ci": [sb_lo, sb_hi],
                    "delta_auc": sd_pt,
                    "delta_auc_ci": [sd_lo, sd_hi],
                }
            t_boot_acc += time.perf_counter() - tb0

            fused_probs_by_cfg[cfg_id] = fused_probs
            conn_probs_by_cfg[cfg_id] = conn_probs
            config_results[cfg_id] = {
                "status": "ok",
                "feature_kind": fk,
                "selection_mode": sel,
                "classifier": clf_name,
                "conn_only_auc": c_auc,
                "conn_only_ci": [c_lo, c_hi],
                "fused_auc": f_auc,
                "fused_ci": [f_lo, f_hi],
                "delta_auc_vs_cov": d_cov,
                "delta_auc_vs_cov_ci": [d_cov_lo, d_cov_hi],
                "delta_auc_vs_motion": d_mot,
                "delta_auc_vs_motion_ci": [d_mot_lo, d_mot_hi],
                "delta_auc_vs_age_sex": d_as,
                "delta_auc_vs_age_sex_ci": [d_as_lo, d_as_hi],
                "per_institution": per_site,
            }
        except Exception as exc:
            failed_ids.append(cfg_id)
            config_results[cfg_id] = {
                "status": "failed",
                "feature_kind": fk,
                "selection_mode": sel,
                "classifier": clf_name,
                "error": str(exc),
            }

    valid_ids = [cid for cid, res in config_results.items() if res["status"] == "ok"]
    if not valid_ids:
        raise RuntimeError(f"All configurations failed on cohort {cohort.name}")

    # Chosen configuration = largest pooled delta-AUC (fused minus covariate-only) among non-failed configs
    chosen_id = max(valid_ids, key=lambda cid: float(config_results[cid]["delta_auc_vs_cov"]))
    max_conn_auc = max(float(config_results[cid]["conn_only_auc"]) for cid in valid_ids)
    max_delta_auc = float(config_results[chosen_id]["delta_auc_vs_cov"])

    phase_timings["bootstrap_cis_sec"] = round(t_boot_acc, 3)
    phase_timings["total_grid_sec"] = round(time.perf_counter() - t_base0, 3)

    return {
        "n_subjects": int(len(cohort.labels)),
        "baselines": baselines_summary,
        "configs": config_results,
        "failed_configs": failed_ids,
        "chosen_config_id": chosen_id,
        "observed_max_conn_auc": max_conn_auc,
        "observed_max_delta_auc": max_delta_auc,
        "_internal": {
            "baselines_probs": baselines_probs,
            "fused_probs_by_cfg": fused_probs_by_cfg,
            "valid_configs": [(cid, fk, sel, clf) for cid, fk, sel, clf in configs if cid in valid_ids],
            "tangent_cache": tangent_cache,
            "phase_timings": phase_timings,
        },
    }


def run_full_analysis(
    cohorts: Dict[str, CohortData],
    n_permutations: int = DEFAULT_PERMUTATION_B,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_RANDOM_STATE,
    checkpoint_path: Optional[Path] = None,
    failing_config_ids: Optional[Set[str]] = None,
    progress_fn: Optional[Callable[[str], None]] = print,
    n_workers: Optional[int] = None,
) -> Dict[str, object]:
    """
    Execute the complete pre-specified exploratory analysis:
      1. Primary cohort (N=564): baselines, 12 configs, late fusion, Test 1 & Test 2 permutations,
         per-institution CIs, chosen configuration, 3-part conjunction, and outcome wording.
      2. Sensitivity analyses (N=620, 215-pair matched, 226-pair matched).
      3. Separate CFG_13-16 family (NYU + OHSU, 2-fold subject-grouped, Bonferroni 4 tests).
    Prints ONLY progress counters via progress_fn; never prints AUCs, p-values, or subject IDs.
    """
    use_workers = resolve_n_workers(n_workers)
    analysis_timings: Dict[str, object] = {"n_workers": use_workers}

    # Precompute Ledoit-Wolf shrinkage covariances ONCE across the superset of single-run cohorts
    superset_key = "all_620" if "all_620" in cohorts and cohorts["all_620"].timeseries_list else "primary_564"
    covs_by_sid: Dict[str, np.ndarray] = {}
    if cohorts[superset_key].timeseries_list:
        t_cov0 = time.perf_counter()
        sup_covs = precompute_subject_covariances(cohorts[superset_key].timeseries_list)
        for idx_s, sid in enumerate(cohorts[superset_key].subject_ids):
            covs_by_sid[str(sid)] = sup_covs[idx_s]
        analysis_timings["cov_precompute_sec"] = round(time.perf_counter() - t_cov0, 3)

    def _cohort_covs(cohort_obj: CohortData) -> Optional[np.ndarray]:
        if not cohort_obj.timeseries_list or not covs_by_sid:
            return None
        if all(str(sid) in covs_by_sid for sid in cohort_obj.subject_ids):
            return np.asarray([covs_by_sid[str(sid)] for sid in cohort_obj.subject_ids], dtype=np.float64)
        return None

    primary_cohort = cohorts["primary_564"]
    prim_eval = _evaluate_cohort_grid(
        primary_cohort,
        PRIMARY_CONFIGS,
        n_boot=n_boot,
        seed=seed,
        failing_config_ids=failing_config_ids,
        progress_fn=progress_fn,
        progress_prefix="Primary N=564",
        subject_covs=_cohort_covs(primary_cohort),
        n_workers=use_workers,
    )
    analysis_timings["primary_observed"] = prim_eval["_internal"]["phase_timings"]
    valid_tuples = [(fk, sel, clf) for _, fk, sel, clf in prim_eval["_internal"]["valid_configs"]]
    base_probs = prim_eval["_internal"]["baselines_probs"]
    primary_tangent_cache = prim_eval["_internal"]["tangent_cache"]
    chosen_id = str(prim_eval["chosen_config_id"])
    chosen_fused_probs = prim_eval["_internal"]["fused_probs_by_cfg"][chosen_id]

    t_t1 = time.perf_counter()
    p_test1, _ = run_within_site_label_permutation_test(
        configs=valid_tuples,
        timeseries_list=primary_cohort.timeseries_list,
        pearson_z_edges=primary_cohort.pearson_z_edges,
        motion_covariates=primary_cohort.motion_covariates,
        labels=primary_cohort.labels,
        institutions=primary_cohort.institutions,
        observed_max_auc=float(prim_eval["observed_max_conn_auc"]),
        n_permutations=n_permutations,
        seed=seed,
        checkpoint_path=checkpoint_path,
        progress_fn=progress_fn,
        tangent_cache=primary_tangent_cache,
        n_workers=use_workers,
    )
    analysis_timings["test1_sec"] = round(time.perf_counter() - t_t1, 3)

    t_t2 = time.perf_counter()
    p_test2, _ = run_incremental_contrast_permutation_test(
        configs=valid_tuples,
        timeseries_list=primary_cohort.timeseries_list,
        pearson_z_edges=primary_cohort.pearson_z_edges,
        motion_covariates=primary_cohort.motion_covariates,
        age_sex_covariates=primary_cohort.age_sex_covariates,
        labels=primary_cohort.labels,
        institutions=primary_cohort.institutions,
        baseline_probs=base_probs["motion_age_sex"],
        observed_max_delta_auc=float(prim_eval["observed_max_delta_auc"]),
        n_permutations=n_permutations,
        seed=seed,
        checkpoint_path=checkpoint_path,
        progress_fn=progress_fn,
        tangent_cache=primary_tangent_cache,
        use_cached_tangent=True,
        n_workers=use_workers,
    )
    analysis_timings["test2_sec"] = round(time.perf_counter() - t_t2, 3)

    t_cj = time.perf_counter()
    conjunction = evaluate_interesting_criteria(
        y_true=primary_cohort.labels,
        institutions=primary_cohort.institutions,
        probs_combined=chosen_fused_probs,
        probs_motion_baseline=base_probs["motion"],
        probs_agesex_baseline=base_probs["age_sex"],
        probs_combined_baseline=base_probs["motion_age_sex"],
        incremental_perm_p=p_test2,
        n_boot=n_boot,
        seed=seed,
    )
    analysis_timings["conjunction_boot_sec"] = round(time.perf_counter() - t_cj, 3)
    outcome_wording = (
        "a hypothesis for a new cohort; NYU is not sealed, so this is not confirmation"
        if conjunction["is_interesting"]
        else "no evidence of incremental connectivity signal in this design"
    )

    # Free primary tangent cache before running sensitivity grids to bound peak memory
    del primary_tangent_cache
    del prim_eval["_internal"]
    gc.collect()
    prim_eval["permutation_tests"] = {
        "n_permutations": int(n_permutations),
        "test1_label_perm_p": p_test1,
        "test2_incremental_perm_p": p_test2,
    }
    prim_eval["interesting_conjunction"] = conjunction
    prim_eval["outcome_wording"] = outcome_wording

    # Sensitivity analyses (not used for the interesting rule)
    sensitivity: Dict[str, object] = {}
    for sens_key in ["all_620", "matched_215", "matched_226"]:
        if sens_key in cohorts:
            s_cohort = cohorts[sens_key]
            s_eval = _evaluate_cohort_grid(
                s_cohort,
                PRIMARY_CONFIGS,
                n_boot=n_boot,
                seed=seed,
                failing_config_ids=failing_config_ids,
                progress_fn=progress_fn,
                progress_prefix=f"Sensitivity {sens_key}",
                subject_covs=_cohort_covs(s_cohort),
                n_workers=use_workers,
            )
            analysis_timings[f"sensitivity_{sens_key}"] = s_eval["_internal"]["phase_timings"]
            del s_eval["_internal"]
            gc.collect()
            s_eval["used_for_interesting_rule"] = False
            sensitivity[sens_key] = s_eval

    # Separate CFG_13-16 family (NYU + OHSU, 2-fold subject-grouped, Bonferroni across 4 tests)
    mr_cohort = cohorts["multirun_nyu_ohsu"]
    mr_eval = _evaluate_cohort_grid(
        mr_cohort,
        MULTIRUN_CONFIGS,
        n_boot=n_boot,
        seed=seed,
        failing_config_ids=failing_config_ids,
        progress_fn=progress_fn,
        progress_prefix="MultiRun CFG_13-16",
        n_workers=use_workers,
    )
    analysis_timings["cfg13_16_observed"] = mr_eval["_internal"]["phase_timings"]
    mr_base_probs = mr_eval["_internal"]["baselines_probs"]
    mr_bonf_alpha = 0.05 / 4.0
    mr_perm_p: Dict[str, float] = {}
    t_mr_perm = time.perf_counter()
    for cid, fk, sel, clf in mr_eval["_internal"]["valid_configs"]:
        p_cfg, _ = run_incremental_contrast_permutation_test(
            configs=[(fk, sel, clf)],
            timeseries_list=mr_cohort.timeseries_list,
            pearson_z_edges=mr_cohort.pearson_z_edges,
            motion_covariates=mr_cohort.motion_covariates,
            age_sex_covariates=mr_cohort.age_sex_covariates,
            labels=mr_cohort.labels,
            institutions=mr_cohort.institutions,
            baseline_probs=mr_base_probs["motion_age_sex"],
            observed_max_delta_auc=float(mr_eval["configs"][cid]["delta_auc_vs_cov"]),
            n_permutations=n_permutations,
            seed=seed,
            progress_fn=progress_fn,
            n_workers=use_workers,
        )
        mr_perm_p[cid] = p_cfg
    analysis_timings["cfg13_16_perm_sec"] = round(time.perf_counter() - t_mr_perm, 3)
    del mr_eval["_internal"]
    mr_eval["bonferroni_alpha"] = mr_bonf_alpha
    mr_eval["per_config_perm_p"] = mr_perm_p
    mr_eval["bonferroni_significant_configs"] = [cid for cid, pv in mr_perm_p.items() if pv < mr_bonf_alpha]

    return {
        "protocol_version": "exploratory_12_plus_4_late_fusion",
        "primary_analysis": prim_eval,
        "sensitivity_analyses": sensitivity,
        "multirun_family_cfg13_16": mr_eval,
        "_phase_timings": analysis_timings,
    }


def run_exploratory_synthetic_e2e(
    cohorts: Dict[str, CohortData],
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    n_permutations: int = 4,
    n_boot: int = 100,
    seed: int = DEFAULT_RANDOM_STATE,
    failing_config_ids: Optional[Set[str]] = None,
    progress_fn: Optional[Callable[[str], None]] = print,
    n_workers: int = 1,
) -> Dict[str, object]:
    """
    Test-only entry point for synthetic end-to-end verification.
    Strictly refuses any path under NeuroADHD/data or the real CC200 archive.
    """
    for p in [output_path, checkpoint_path]:
        if p is not None:
            p_str = str(p.resolve()) if p.is_absolute() else str(p)
            if "NeuroADHD/data" in p_str or "ADHD200_CC200_TCs_filtfix" in p_str:
                raise ValueError(f"Synthetic E2E entry point refuses real data path: {p}")

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results file: {output_path}")

    results = run_full_analysis(
        cohorts=cohorts,
        n_permutations=n_permutations,
        n_boot=n_boot,
        seed=seed,
        checkpoint_path=checkpoint_path,
        failing_config_ids=failing_config_ids,
        progress_fn=progress_fn,
        n_workers=n_workers,
    )
    results.pop("_phase_timings", None)
    write_results_once(output_path, results)
    return results


def run_rehearsal(
    data_root: Path = Path("NeuroADHD/data/athena"),
    cache_path: Optional[Path] = Path("NeuroADHD/data/athena/features/exploratory_label_free_cache.npz"),
    rehearsal_results_path: Path = Path("NeuroADHD/data/athena/results/rehearsal_results.json"),
    rehearsal_checkpoint_path: Path = Path("NeuroADHD/data/athena/results/rehearsal_perm_checkpoint.json"),
    n_permutations: int = 2,
    n_boot: int = 50,
    seed: int = DEFAULT_RANDOM_STATE,
    n_workers: Optional[int] = None,
) -> Dict[str, object]:
    """
    Full-pipeline crash/runtime rehearsal on REAL label-free arrays with SYNTHETIC labels
    (stratified by institution at a fixed 45% positive rate).
    - NEVER reads any real label column (DX, etc.).
    - NEVER reads or writes exploratory_results.json or exploratory_perm_checkpoint.json.
    - Writes ONLY to rehearsal_* paths under git-ignored NeuroADHD/data/athena/results/.
    - Prints/returns NO AUCs, delta-AUCs, or p-values.
    """
    import warnings

    for p in [rehearsal_results_path, rehearsal_checkpoint_path]:
        if p.name in {"exploratory_results.json", "exploratory_perm_checkpoint.json"}:
            raise ValueError(f"Rehearsal refuses to touch real output/checkpoint path: {p}")

    # Remove prior rehearsal artifacts if re-running rehearsal
    if rehearsal_results_path.exists():
        rehearsal_results_path.unlink()
    if rehearsal_checkpoint_path.exists():
        rehearsal_checkpoint_path.unlink()

    timings: Dict[str, float] = {}
    stages: Dict[str, str] = {}

    t0 = time.perf_counter()
    inputs = load_label_free_inputs(data_root=data_root, cache_path=cache_path, force_rebuild=False)
    timings["load_label_free_cache_sec"] = round(time.perf_counter() - t0, 3)
    stages["load_label_free_inputs"] = "PASS"

    # Generate synthetic labels stratified by institution at fixed 45% positive rate
    rng = np.random.default_rng(seed)
    syn_labels_r1 = np.zeros(len(inputs.subject_ids), dtype=np.int64)
    for site in sorted(set(inputs.institutions)):
        s_idx = np.where(inputs.institutions == site)[0]
        n_pos = max(1, int(round(0.45 * len(s_idx))))
        y_site = np.array([1] * n_pos + [0] * (len(s_idx) - n_pos), dtype=np.int64)
        rng.shuffle(y_site)
        syn_labels_r1[s_idx] = y_site

    sid_to_syn = {str(sid): int(y) for sid, y in zip(inputs.subject_ids, syn_labels_r1)}
    syn_labels_mr = np.array([sid_to_syn[str(sid)] for sid in inputs.multirun_subject_ids], dtype=np.int64)

    t1 = time.perf_counter()
    cohorts = build_cohorts(inputs, syn_labels_r1, syn_labels_mr)
    timings["build_cohorts_sec"] = round(time.perf_counter() - t1, 3)
    stages["build_cohorts"] = "PASS"

    use_workers = resolve_n_workers(n_workers)
    t2 = time.perf_counter()
    captured_warnings: List[str] = []
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        results = run_full_analysis(
            cohorts=cohorts,
            n_permutations=n_permutations,
            n_boot=n_boot,
            seed=seed,
            checkpoint_path=rehearsal_checkpoint_path,
            progress_fn=lambda msg: print(f"[rehearsal p={n_permutations}] {msg}", file=sys.stderr, flush=True),
            n_workers=use_workers,
        )
        for w in w_list:
            captured_warnings.append(f"{w.category.__name__}: {w.message}")
    phase_timings = results.pop("_phase_timings", {})
    timings["run_full_analysis_sec"] = round(time.perf_counter() - t2, 3)
    stages["run_full_analysis"] = "PASS"

    write_results_once(rehearsal_results_path, results)
    stages["write_rehearsal_results"] = "PASS"

    # macOS ru_maxrss is in bytes; Linux is in KiB (darwin check)
    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    ru_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    parent_mb = ru_self.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru_self.ru_maxrss / 1024.0
    child_max_mb = ru_child.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru_child.ru_maxrss / 1024.0

    warning_counts: Dict[str, int] = defaultdict(int)
    for w_str in captured_warnings:
        cat = w_str.split(":", 1)[0]
        warning_counts[cat] += 1

    sections_produced = {
        "top_level": list(results.keys()),
        "primary_analysis": list(results["primary_analysis"].keys()),
        "sensitivity_analyses": list(results["sensitivity_analyses"].keys()),
        "multirun_family_cfg13_16": list(results["multirun_family_cfg13_16"].keys()),
    }

    return {
        "rehearsal_status": "PASS",
        "stages": stages,
        "sections_produced": sections_produced,
        "n_permutations_rehearsal": n_permutations,
        "n_workers": use_workers,
        "warning_counts_by_category": dict(warning_counts),
        "total_warnings": len(captured_warnings),
        "peak_rss_mb": round(parent_mb, 2),
        "child_max_rss_mb": round(child_max_mb, 2),
        "timings_sec": timings,
        "phase_timings": phase_timings,
        "rehearsal_results_path": str(rehearsal_results_path),
        "rehearsal_checkpoint_path": str(rehearsal_checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory Analysis Runner (Guarded).")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check that all required inputs exist with expected shapes/counts WITHOUT reading labels.",
    )
    parser.add_argument(
        "--label-free-check",
        action="store_true",
        help="Run the real label-free loader on the archive (no labels, no models, no AUCs) and print counts/pass-fail.",
    )
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="Run full pipeline rehearsal on real label-free arrays with synthetic 45%-positive stratified labels.",
    )
    parser.add_argument(
        "--execute-real",
        action="store_true",
        help="Explicit permission to execute real cohort evaluation. Requires 'exploratory-frozen' tag.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_PERMUTATION_B,
        help=f"Number of within-site permutation iterations (default: {DEFAULT_PERMUTATION_B}).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help=f"Parallel worker processes for permutations/tangent means (0=auto capped <= {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("NeuroADHD/data/athena/results/exploratory_results.json"),
        help="Write-once output JSON path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("NeuroADHD/data/athena/results/exploratory_perm_checkpoint.json"),
        help="Null-only permutation checkpoint path for crash recovery.",
    )
    args = parser.parse_args()

    if args.preflight:
        summary = run_preflight()
        print("Preflight passed (no label columns read):", json.dumps(summary))
        return

    if args.label_free_check:
        report = run_label_free_check()
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(1)
        return

    if args.rehearsal:
        rehearsal_report = run_rehearsal(
            n_permutations=args.n_permutations,
            seed=DEFAULT_RANDOM_STATE,
            n_workers=args.n_workers,
        )
        print(json.dumps(rehearsal_report, indent=2))
        return

    enforce_exploratory_guards(execute_real=args.execute_real)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results file: {args.output}")

    data_root = Path("NeuroADHD/data/athena")
    inputs = load_label_free_inputs(data_root=data_root)
    phenotypes_tsv = data_root / "phenotypes" / "adhd200_preprocessed_phenotypics.tsv"
    labels_rest1 = load_labels_last(phenotypes_tsv, inputs.subject_ids)
    labels_multirun = load_labels_last(phenotypes_tsv, inputs.multirun_subject_ids)
    cohorts = build_cohorts(inputs, labels_rest1, labels_multirun)

    results = run_full_analysis(
        cohorts=cohorts,
        n_permutations=args.n_permutations,
        n_boot=DEFAULT_BOOTSTRAP_B,
        seed=DEFAULT_RANDOM_STATE,
        checkpoint_path=args.checkpoint,
        progress_fn=print,
        n_workers=args.n_workers,
    )
    results.pop("_phase_timings", None)
    write_results_once(args.output, results)
    print("Exploratory analysis complete. Results written.")


if __name__ == "__main__":
    main()
