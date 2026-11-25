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
import re
import subprocess
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
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
    Input: (T, K)
    Output: 1D array of shape (K*(K-1)//2,)
    """
    corr = np.corrcoef(ts, rowvar=False)
    k = ts.shape[1]
    triu_i, triu_j = np.triu_indices(k, k=1)
    r_edges = np.clip(corr[triu_i, triu_j], -1.0 + clip_eps, 1.0 - clip_eps)
    z_edges = np.arctanh(r_edges).astype(np.float64)
    if not np.all(np.isfinite(z_edges)):
        raise ValueError("Non-finite Fisher z edges encountered")
    return z_edges


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
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Add bias column
    M_tr_ext = np.column_stack([np.ones(n_train, dtype=np.float64), motion_train])
    M_te_ext = np.column_stack([np.ones(n_test, dtype=np.float64), motion_test])

    # Solve least squares for each edge: (K_motion+1, P_edges)
    coeffs, _, _, _ = np.linalg.lstsq(M_tr_ext, X_train, rcond=None)
    b = coeffs[0, :]
    w = coeffs[1:, :]

    X_tr_pred = M_tr_ext @ coeffs
    X_te_pred = M_te_ext @ coeffs

    X_tr_resid = X_train - X_tr_pred
    X_te_resid = X_test - X_te_pred

    return X_tr_resid, X_te_resid, w, b


# =====================================================================
# Model Factory & Feature Selection
# =====================================================================

def build_classifier(
    classifier_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ClassifierMixin:
    """Instantiate pre-declared classifier architecture."""
    if classifier_name == "rf":
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
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
        k_val = min(100, X_train.shape[1])
        selector = SelectKBest(f_classif, k=k_val)
        X_tr_sel = selector.fit_transform(X_train, y_train)
        X_te_sel = selector.transform(X_test)
        indices = selector.get_support(indices=True)
        return X_tr_sel, X_te_sel, indices
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


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, float, float]:
    """Compute point estimate and 95% bootstrap confidence interval."""
    point = float(metric_fn(y_true, y_scores))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return point, point, point

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        boots[b] = metric_fn(y_true[sample_idx], y_scores[sample_idx])

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
    diff_point = float(roc_auc_score(y_true, y_scores_a) - roc_auc_score(y_true, y_scores_b))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        diffs[b] = roc_auc_score(y_true[sample_idx], y_scores_a[sample_idx]) - roc_auc_score(
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
        y_test = labels[test_mask]

        if feature_kind == "pearson_fisher_z":
            X_train = pearson_z_edges[train_mask]
            X_test = pearson_z_edges[test_mask]
        elif feature_kind == "tangent_space":
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

        clf = build_classifier(classifier_name, random_state=random_state)
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

        s_m = StandardScaler().fit(motion_covariates[train_mask])
        m_tr = s_m.transform(motion_covariates[train_mask])
        m_te = s_m.transform(motion_covariates[test_mask])

        s_a = StandardScaler().fit(age_sex_covariates[train_mask])
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
) -> np.ndarray:
    """Apply feature selection and fit classifier on a connectivity fold, returning test probabilities."""
    X_tr_sel, X_te_sel, _ = apply_feature_selection(X_tr_conn, y_tr, X_te_conn, selection_mode)
    clf = build_classifier(classifier_name, random_state=random_state)
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

        sc_in = StandardScaler().fit(covariates[inner_tr_mask])
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
    sc_out = StandardScaler().fit(covariates[outer_train_mask])
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

    # Pre-extract tangent space and residualized features once per fold
    uses_tangent = any(fk == "tangent_space" for fk, _, _ in configs)
    uses_resid = any(fk == "motion_residualized" for fk, _, _ in configs)
    tangent_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    resid_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for test_site in unique_sites:
        train_mask = institutions != test_site
        test_mask = institutions == test_site
        if uses_tangent:
            ts_train = [timeseries_list[i] for i in np.where(train_mask)[0]]
            ts_test = [timeseries_list[i] for i in np.where(test_mask)[0]]
            X_tr_tang, X_te_tang, _ = fit_transform_tangent_space(ts_train, ts_test, vectorize=True)
            tangent_by_fold[test_site] = (X_tr_tang, X_te_tang)
        if uses_resid:
            X_tr_raw = pearson_z_edges[train_mask]
            X_te_raw = pearson_z_edges[test_mask]
            M_tr = motion_covariates[train_mask]
            M_te = motion_covariates[test_mask]
            X_tr_res, X_te_res, _, _ = fit_transform_motion_residualization(X_tr_raw, M_tr, X_te_raw, M_te)
            resid_by_fold[test_site] = (X_tr_res, X_te_res)

    for b in range(n_permutations):
        y_perm = labels.copy()
        for site in unique_sites:
            s_idx = np.where(institutions == site)[0]
            y_perm[s_idx] = rng.permutation(y_perm[s_idx])

        if b < len(completed):
            continue

        if progress_fn is not None:
            progress_fn(f"Test 1 permutation {b + 1}/{n_permutations}")

        perm_aucs: List[float] = []
        for feat_kind, sel_mode, clf_name in configs:
            try:
                probs = np.zeros(len(labels), dtype=np.float64)
                for test_site in unique_sites:
                    train_mask = institutions != test_site
                    test_mask = institutions == test_site
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

                    X_tr_sel, X_te_sel, _ = apply_feature_selection(X_tr, y_tr, X_te, sel_mode)
                    clf = build_classifier(clf_name, random_state=seed + b)
                    clf.fit(X_tr_sel, y_tr)
                    probs[test_mask] = clf.predict_proba(X_te_sel)[:, 1]

                perm_aucs.append(roc_auc_score(y_perm, probs))
            except Exception:
                # Failed configuration excluded from max statistic
                continue

        if not perm_aucs:
            raise RuntimeError("All configurations failed during Test 1 permutation.")
        null_max_aucs[b] = max(perm_aucs)
        save_permutation_checkpoint(checkpoint_path, "test1_null_max_aucs", null_max_aucs[: b + 1])

    p_val = float((1.0 + np.sum(null_max_aucs >= observed_max_auc)) / (1.0 + n_permutations))
    return p_val, null_max_aucs


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
) -> Tuple[float, np.ndarray]:
    """
    Test 2: Max-statistic permutation test for the Incremental Contrast (Late Fusion):
        delta_AUC = AUC(fused[motion+age+sex, connectivity]) - AUC(motion+age+sex)

    Null Specification:
    - Diagnostic labels (y) and non-imaging covariates (motion + age + sex) stay PAIRED and intact.
    - Only the connectivity feature rows (X) are permuted among subjects WITHIN each institution.
    - Outer & inner covariate-only scores (s_cov_oof, s_cov_test) depend only on intact covariates/labels
      and are precomputed once before the permutation loop.
    - Supports null-only checkpointing for crash recovery without exposing partial results.
    """
    rng = np.random.default_rng(seed)
    unique_sites = sorted(set(institutions))
    n = len(labels)
    baseline_auc = roc_auc_score(labels, baseline_probs)
    completed = load_permutation_checkpoint(checkpoint_path, "test2_null_max_deltas")
    null_max_deltas = np.empty(n_permutations, dtype=np.float64)
    for idx, val in enumerate(completed[:n_permutations]):
        null_max_deltas[idx] = val

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
            sc_in = StandardScaler().fit(covariates[inner_tr_mask])
            clf_in = LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed
            )
            clf_in.fit(sc_in.transform(covariates[inner_tr_mask]), labels[inner_tr_mask])
            s_cov_oof[inner_val_mask] = clf_in.predict_proba(sc_in.transform(covariates[inner_val_mask]))[:, 1]

        sc_out = StandardScaler().fit(covariates[outer_train_mask])
        clf_out = LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed
        )
        clf_out.fit(sc_out.transform(covariates[outer_train_mask]), labels[outer_train_mask])
        cov_oof_by_outer[test_site] = s_cov_oof
        cov_test_by_outer[test_site] = clf_out.predict_proba(sc_out.transform(covariates[outer_test_mask]))[:, 1]

    uses_tangent = any(fk == "tangent_space" for fk, _, _ in configs)

    # Precompute tangent space features ONCE per unique training site subset (5 4-site + 10 3-site = 15 splits).
    # Mathematically invariant to within-site connectivity row permutations because the training multiset is identical.
    tangent_all_by_subset: Dict[frozenset, np.ndarray] = {}
    if uses_tangent:
        subsets_to_precompute: set = set()
        for test_site in unique_sites:
            outer_sites = frozenset(s for s in unique_sites if s != test_site)
            subsets_to_precompute.add(outer_sites)
            for inner_val_site in outer_sites:
                inner_sites = frozenset(s for s in outer_sites if s != inner_val_site)
                subsets_to_precompute.add(inner_sites)
        for tr_sites in subsets_to_precompute:
            tr_mask = np.isin(institutions, list(tr_sites))
            ts_tr = [timeseries_list[i] for i in np.where(tr_mask)[0]]
            cm = ConnectivityMeasure(kind="tangent", vectorize=True)
            cm.fit(ts_tr)
            # Project all N subjects onto this split's tangent space once
            tangent_all_by_subset[tr_sites] = cm.transform(timeseries_list)

    for b in range(n_permutations):
        conn_perm_indices = np.arange(n)
        for site in unique_sites:
            s_idx = np.where(institutions == site)[0]
            conn_perm_indices[s_idx] = rng.permutation(s_idx)

        if b < len(completed):
            continue

        if progress_fn is not None:
            progress_fn(f"Test 2 permutation {b + 1}/{n_permutations}")

        pearson_perm = pearson_z_edges[conn_perm_indices]
        resid_split_cache: Dict[Tuple[frozenset, str], Tuple[np.ndarray, np.ndarray]] = {}

        def _get_split_feats(
            fk: str, tr_sites: frozenset, tr_mask: np.ndarray, te_mask: np.ndarray, te_key: str
        ) -> Tuple[np.ndarray, np.ndarray]:
            if fk == "pearson_fisher_z" or fk == "multi_run_avg":
                return pearson_perm[tr_mask], pearson_perm[te_mask]
            elif fk == "tangent_space":
                tang_perm = tangent_all_by_subset[tr_sites][conn_perm_indices]
                return tang_perm[tr_mask], tang_perm[te_mask]
            elif fk == "motion_residualized":
                cache_key = (tr_sites, te_key)
                if cache_key not in resid_split_cache:
                    resid_split_cache[cache_key] = fit_transform_motion_residualization(
                        pearson_perm[tr_mask], motion_covariates[tr_mask], pearson_perm[te_mask], motion_covariates[te_mask]
                    )[:2]
                return resid_split_cache[cache_key]
            else:
                raise ValueError(f"Unknown feature kind: {fk}")

        deltas: List[float] = []
        for feat_kind, sel_mode, clf_name in configs:
            try:
                fused_probs = np.zeros(n, dtype=np.float64)
                for test_site in unique_sites:
                    outer_train_mask = institutions != test_site
                    outer_test_mask = institutions == test_site
                    outer_tr_sites = frozenset(s for s in unique_sites if s != test_site)
                    inner_sites = [s for s in unique_sites if s != test_site]

                    s_conn_oof = np.zeros(n, dtype=np.float64)
                    if len(inner_sites) >= 2:
                        in_splits_conn = [
                            (outer_train_mask & (institutions != iv), institutions == iv, iv)
                            for iv in inner_sites
                        ]
                    else:
                        tr_idx = np.where(outer_train_mask)[0]
                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                        in_splits_conn = []
                        for f_i, (tr_rel, val_rel) in enumerate(skf.split(tr_idx, labels[tr_idx])):
                            m_tr = np.zeros(n, dtype=bool)
                            m_val = np.zeros(n, dtype=bool)
                            m_tr[tr_idx[tr_rel]] = True
                            m_val[tr_idx[val_rel]] = True
                            in_splits_conn.append((m_tr, m_val, f"cv{f_i}"))

                    for inner_tr_mask, inner_val_mask, iv_key in in_splits_conn:
                        inner_tr_sites = frozenset(s for s in outer_tr_sites if s != iv_key)
                        X_in_tr, X_in_val = _get_split_feats(
                            feat_kind, inner_tr_sites, inner_tr_mask, inner_val_mask, iv_key
                        )
                        s_conn_oof[inner_val_mask] = _fit_predict_connectivity_fold(
                            feat_kind, sel_mode, clf_name, X_in_tr, X_in_val, labels[inner_tr_mask], random_state=seed + b
                        )

                    s_cov_oof = cov_oof_by_outer[test_site]
                    meta_X_tr = np.column_stack([s_cov_oof[outer_train_mask], s_conn_oof[outer_train_mask]])
                    meta_clf = LogisticRegression(
                        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed + b
                    )
                    meta_clf.fit(meta_X_tr, labels[outer_train_mask])

                    X_out_tr, X_out_te = _get_split_feats(
                        feat_kind, outer_tr_sites, outer_train_mask, outer_test_mask, test_site
                    )
                    s_conn_test = _fit_predict_connectivity_fold(
                        feat_kind, sel_mode, clf_name, X_out_tr, X_out_te, labels[outer_train_mask], random_state=seed + b
                    )
                    meta_X_te = np.column_stack([cov_test_by_outer[test_site], s_conn_test])
                    fused_probs[outer_test_mask] = meta_clf.predict_proba(meta_X_te)[:, 1]

                comb_auc = roc_auc_score(labels, fused_probs)
                deltas.append(comb_auc - baseline_auc)
            except Exception:
                # Failed configuration excluded from max statistic
                continue

        if not deltas:
            raise RuntimeError("All configurations failed during Test 2 permutation.")
        null_max_deltas[b] = max(deltas)
        save_permutation_checkpoint(checkpoint_path, "test2_null_max_deltas", null_max_deltas[: b + 1])

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
    summary["valid_roi_keys"] = len(roi_meta)

    # 3. Check phenotype TSV non-label columns ONLY (never read DX or ADHD columns)
    with open(required_files["phenotypes_tsv"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        for req_col in ["ScanDir ID", "Site", "Gender", "Age"]:
            if req_col not in cols:
                raise ValueError(f"Preflight failed: missing column '{req_col}' in phenotypes TSV")
        n_pheno_rows = sum(1 for _ in reader)
    if n_pheno_rows < 900:
        raise ValueError(f"Preflight failed: unexpected phenotype row count {n_pheno_rows}")
    summary["phenotypes_rows"] = n_pheno_rows

    # 4. Check motion & matching inputs (dev_niak_fd_metrics.csv and NYU_motion.csv)
    with open(required_files["dev_fd_csv"], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fd_cols = reader.fieldnames or []
        for req_col in ["subject_id_norm", "institution", "mean_fd_mm", "max_disp_mm", "frames_cc200"]:
            if req_col not in fd_cols:
                raise ValueError(f"Preflight failed: missing column '{req_col}' in dev_niak_fd_metrics.csv")
        n_fd_rows = sum(1 for _ in reader)
    summary["dev_fd_rows"] = n_fd_rows

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

    return summary


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


def parse_cc200_1d_text(text: str) -> np.ndarray:
    """Parse a CC200 .1D timecourse file into a (T, 190) float64 array."""
    rows: List[List[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("File\t"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        rows.append([float(x) for x in parts[2:]])
    if not rows:
        raise ValueError("Empty CC200 .1D timecourse stream")
    return np.array(rows, dtype=np.float64)


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
            )

    phenotypes_tsv = data_root / "phenotypes" / "adhd200_preprocessed_phenotypics.tsv"
    dev_fd_csv = data_root / "motion" / "dev_niak_fd_metrics.csv"
    nyu_tar = data_root / "staging" / "intermediate_nyu.tar"
    cc200_archive = data_root / "staging" / "ADHD200_CC200_TCs_filtfix.tar.gz"

    # 1. Non-label phenotype columns ONLY (never touch DX or ADHD columns)
    pheno_cov: Dict[str, Tuple[str, float, float]] = {}
    with open(phenotypes_tsv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            site_code = str(row.get("Site", "")).strip()
            if site_code not in SITE_CODE_TO_INSTITUTION:
                continue
            sid = str(row.get("ScanDir ID", "")).strip().lstrip("0") or "0"
            age_str = str(row.get("Age", "")).strip()
            sex_str = str(row.get("Gender", "")).strip()
            try:
                age_val = float(age_str)
                sex_val = float(sex_str)
            except ValueError:
                continue
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

    # 3. Stream CC200 archive in memory (rest_1 + rest_2/rest_3 for NYU and OHSU)
    rest1_data: Dict[str, Tuple[str, np.ndarray, np.ndarray]] = {}
    multirun_runs: Dict[str, List[np.ndarray]] = defaultdict(list)

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
                ts = parse_cc200_1d_text(stream.read().decode("utf-8", errors="replace"))
                z_edges = timeseries_to_fisher_z(ts)
                if run_num == 1:
                    rest1_data[sid] = (inst, ts, z_edges)
                if inst in {"NYU", "OHSU"} and run_num in {1, 2, 3}:
                    multirun_runs[sid].append(z_edges)

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
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ts_obj = np.empty(len(ts_list), dtype=object)
        for i, ts in enumerate(ts_list):
            ts_obj[i] = ts.astype(np.float32)
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
) -> Dict[str, object]:
    """Evaluate baselines and all candidate configurations (late fusion + connectivity-only) on a cohort."""
    baselines_probs = evaluate_non_imaging_baselines(
        motion_covariates=cohort.motion_covariates,
        age_sex_covariates=cohort.age_sex_covariates,
        labels=cohort.labels,
        institutions=cohort.institutions,
        random_state=seed,
    )
    baselines_summary: Dict[str, object] = {}
    for b_name, b_probs in baselines_probs.items():
        auc_pt, auc_lo, auc_hi = stratified_bootstrap_ci(
            cohort.labels, b_probs, roc_auc_score, n_boot=n_boot, seed=seed
        )
        baselines_summary[b_name] = {"auc": auc_pt, "ci_lo": auc_lo, "ci_hi": auc_hi}

    base_cov_probs = baselines_probs["motion_age_sex"]
    unique_sites = sorted(set(cohort.institutions))

    # Precompute 15-split tangent space cache once per cohort if any config uses tangent_space
    tangent_cache: Optional[Dict[frozenset, np.ndarray]] = None
    if any(fk == "tangent_space" for _, fk, _, _ in configs) and len(unique_sites) >= 2:
        tangent_cache = {}
        subsets: Set[frozenset] = set()
        for ts_site in unique_sites:
            outer_s = frozenset(s for s in unique_sites if s != ts_site)
            subsets.add(outer_s)
            for iv_site in outer_s:
                subsets.add(frozenset(s for s in outer_s if s != iv_site))
        for tr_sites in subsets:
            if not tr_sites:
                continue
            tr_m = np.isin(cohort.institutions, list(tr_sites))
            ts_tr = [cohort.timeseries_list[i] for i in np.where(tr_m)[0]]
            cm = ConnectivityMeasure(kind="tangent", vectorize=True)
            cm.fit(ts_tr)
            tangent_cache[tr_sites] = cm.transform(cohort.timeseries_list)

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

            if fk == "tangent_space" and tangent_cache is not None:
                conn_probs = np.zeros(len(cohort.labels), dtype=np.float64)
                for ts_site in unique_sites:
                    tr_m = cohort.institutions != ts_site
                    te_m = cohort.institutions == ts_site
                    tang_all = tangent_cache[frozenset(s for s in unique_sites if s != ts_site)]
                    conn_probs[te_m] = _fit_predict_connectivity_fold(
                        fk, sel, clf_name, tang_all[tr_m], tang_all[te_m], cohort.labels[tr_m], random_state=seed
                    )
            else:
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

            per_site: Dict[str, Dict[str, float]] = {}
            for site in unique_sites:
                m = cohort.institutions == site
                s_fused = float(roc_auc_score(cohort.labels[m], fused_probs[m]))
                s_base = float(roc_auc_score(cohort.labels[m], base_cov_probs[m]))
                per_site[str(site)] = {"fused_auc": s_fused, "cov_auc": s_base, "delta_auc": s_fused - s_base}

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
) -> Dict[str, object]:
    """
    Execute the complete pre-specified exploratory analysis:
      1. Primary cohort (N=564): baselines, 12 configs, late fusion, Test 1 & Test 2 permutations,
         per-institution CIs, chosen configuration, 3-part conjunction, and outcome wording.
      2. Sensitivity analyses (N=620, 215-pair matched, 226-pair matched).
      3. Separate CFG_13-16 family (NYU + OHSU, 2-fold subject-grouped, Bonferroni 4 tests).
    Prints ONLY progress counters via progress_fn; never prints AUCs, p-values, or subject IDs.
    """
    primary_cohort = cohorts["primary_564"]
    prim_eval = _evaluate_cohort_grid(
        primary_cohort,
        PRIMARY_CONFIGS,
        n_boot=n_boot,
        seed=seed,
        failing_config_ids=failing_config_ids,
        progress_fn=progress_fn,
        progress_prefix="Primary N=564",
    )
    valid_tuples = [(fk, sel, clf) for _, fk, sel, clf in prim_eval["_internal"]["valid_configs"]]
    base_probs = prim_eval["_internal"]["baselines_probs"]
    chosen_id = str(prim_eval["chosen_config_id"])
    chosen_fused_probs = prim_eval["_internal"]["fused_probs_by_cfg"][chosen_id]

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
    )
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
    )

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
    outcome_wording = (
        "a hypothesis for a new cohort; NYU is not sealed, so this is not confirmation"
        if conjunction["is_interesting"]
        else "no evidence of incremental connectivity signal in this design"
    )

    del prim_eval["_internal"]
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
            s_eval = _evaluate_cohort_grid(
                cohorts[sens_key],
                PRIMARY_CONFIGS,
                n_boot=n_boot,
                seed=seed,
                failing_config_ids=failing_config_ids,
                progress_fn=progress_fn,
                progress_prefix=f"Sensitivity {sens_key}",
            )
            del s_eval["_internal"]
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
    )
    mr_base_probs = mr_eval["_internal"]["baselines_probs"]
    mr_bonf_alpha = 0.05 / 4.0
    mr_perm_p: Dict[str, float] = {}
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
        )
        mr_perm_p[cid] = p_cfg
    del mr_eval["_internal"]
    mr_eval["bonferroni_alpha"] = mr_bonf_alpha
    mr_eval["per_config_perm_p"] = mr_perm_p
    mr_eval["bonferroni_significant_configs"] = [cid for cid, pv in mr_perm_p.items() if pv < mr_bonf_alpha]

    return {
        "protocol_version": "exploratory_12_plus_4_late_fusion",
        "primary_analysis": prim_eval,
        "sensitivity_analyses": sensitivity,
        "multirun_family_cfg13_16": mr_eval,
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
    )
    write_results_once(output_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory Analysis Runner (Guarded).")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check that all required inputs exist with expected shapes/counts WITHOUT reading labels.",
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
    )
    write_results_once(args.output, results)
    print("Exploratory analysis complete. Results written.")


if __name__ == "__main__":
    main()
