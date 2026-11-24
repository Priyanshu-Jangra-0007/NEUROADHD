#!/usr/bin/env python3
"""
Exploratory Analysis Runner for Multi-Site Resting-State fMRI Connectivity in ADHD-200.

Implements the protocol in docs/exploratory/EXPLORATORY_PLAN_DRAFT.md:
- Strictly exploratory and hypothesis-generating.
- Strictly guarded: refuses to read real label files unless BOTH:
    1) --execute-real is passed on CLI, AND
    2) Git tag 'exploratory-frozen' is an ancestor of HEAD.
- 12 Primary LOIO configurations across 5 multi-class sites (Peking, KKI, NeuroIMAGE, OHSU, NYU).
- 4 Multi-Run configurations (NYU + OHSU sub-analysis, subject-grouped folds).
- Deterministic motion-matched evaluation (1:1 within-site matching on mean FD).
- Primary exploratory contrast: Motion+Age+Sex vs Motion+Age+Sex+Connectivity.
- Max-statistic within-institution permutation test repeating full configuration selection:
    1) Label-permutation max-AUC test for AUC > 0.5.
    2) Connectivity-permutation max-delta-AUC test for incremental contrast (covariates & labels stay paired).
- NEVER prints subject IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REQUIRED_FREEZE_TAG = "exploratory-frozen"
DEFAULT_RANDOM_STATE = 42
DEFAULT_BOOTSTRAP_B = 2000
DEFAULT_PERMUTATION_B = 200
FISHER_CLIP_EPS = 1e-7


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
) -> np.ndarray:
    """
    Evaluate LOIO predictions for Motion + Age + Sex + Connectivity.
    Returns: out_of_fold_probabilities
    """
    if unique_institutions is None:
        unique_institutions = sorted(set(institutions))

    n = len(labels)
    probs_out = np.zeros(n, dtype=np.float64)

    for test_site in unique_institutions:
        train_mask = institutions != test_site
        test_mask = institutions == test_site
        y_tr = labels[train_mask]

        # Standardize covariates strictly on training fold
        covariates = np.column_stack([motion_covariates, age_sex_covariates])
        s_cov = StandardScaler().fit(covariates[train_mask])
        cov_tr = s_cov.transform(covariates[train_mask])
        cov_te = s_cov.transform(covariates[test_mask])

        # Connectivity features
        if feature_kind == "pearson_fisher_z":
            X_tr = pearson_z_edges[train_mask]
            X_te = pearson_z_edges[test_mask]
        elif feature_kind == "tangent_space":
            ts_train = [timeseries_list[i] for i in np.where(train_mask)[0]]
            ts_test = [timeseries_list[i] for i in np.where(test_mask)[0]]
            X_tr, X_te, _ = fit_transform_tangent_space(ts_train, ts_test, vectorize=True)
        elif feature_kind == "motion_residualized":
            X_tr_raw = pearson_z_edges[train_mask]
            X_te_raw = pearson_z_edges[test_mask]
            M_tr = motion_covariates[train_mask]
            M_te = motion_covariates[test_mask]
            X_tr, X_te, _, _ = fit_transform_motion_residualization(X_tr_raw, M_tr, X_te_raw, M_te)
        else:
            X_tr = pearson_z_edges[train_mask]
            X_te = pearson_z_edges[test_mask]

        X_tr_sel, X_te_sel, _ = apply_feature_selection(X_tr, y_tr, X_te, selection_mode)

        # Concatenate standardized covariates and selected connectivity
        full_tr = np.column_stack([cov_tr, X_tr_sel])
        full_te = np.column_stack([cov_te, X_te_sel])

        clf = build_classifier(classifier_name, random_state=random_state)
        clf.fit(full_tr, y_tr)

        test_indices = np.where(test_mask)[0]
        probs_out[test_indices] = clf.predict_proba(full_te)[:, 1]

    return probs_out


# =====================================================================
# Permutation Tests (Label Permutation & Connectivity Permutation)
# =====================================================================

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
) -> Tuple[float, np.ndarray]:
    """
    Label-permutation max-AUC test for H0: AUC = 0.5.
    Permutes labels strictly within each institution, repeating full selection across configurations.
    Returns: (empirical_p_value, max_null_distribution)
    """
    rng = np.random.default_rng(seed)
    unique_sites = sorted(set(institutions))
    null_max_aucs = np.empty(n_permutations, dtype=np.float64)

    # Pre-extract tangent space and residualized features once per fold
    tangent_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    resid_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for test_site in unique_sites:
        train_mask = institutions != test_site
        test_mask = institutions == test_site
        ts_train = [timeseries_list[i] for i in np.where(train_mask)[0]]
        ts_test = [timeseries_list[i] for i in np.where(test_mask)[0]]
        X_tr_tang, X_te_tang, _ = fit_transform_tangent_space(ts_train, ts_test, vectorize=True)
        tangent_by_fold[test_site] = (X_tr_tang, X_te_tang)

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

        perm_aucs: List[float] = []
        for feat_kind, sel_mode, clf_name in configs:
            probs = np.zeros(len(labels), dtype=np.float64)
            for test_site in unique_sites:
                train_mask = institutions != test_site
                test_mask = institutions == test_site
                y_tr = y_perm[train_mask]

                if feat_kind == "pearson_fisher_z":
                    X_tr = pearson_z_edges[train_mask]
                    X_te = pearson_z_edges[test_mask]
                elif feat_kind == "tangent_space":
                    X_tr, X_te = tangent_by_fold[test_site]
                elif feat_kind == "motion_residualized":
                    X_tr, X_te = resid_by_fold[test_site]
                else:
                    X_tr = pearson_z_edges[train_mask]
                    X_te = pearson_z_edges[test_mask]

                X_tr_sel, X_te_sel, _ = apply_feature_selection(X_tr, y_tr, X_te, sel_mode)
                clf = build_classifier(clf_name, random_state=seed + b)
                clf.fit(X_tr_sel, y_tr)
                probs[test_mask] = clf.predict_proba(X_te_sel)[:, 1]

            perm_aucs.append(roc_auc_score(y_perm, probs))

        null_max_aucs[b] = max(perm_aucs)

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
) -> Tuple[float, np.ndarray]:
    """
    Max-statistic permutation test for the Incremental Contrast:
        delta_AUC = AUC(motion + age + sex + connectivity) - AUC(motion + age + sex)

    New Null Specification:
    - Diagnostic labels (y) and non-imaging covariates (motion + age + sex) stay PAIRED and intact.
    - Only the connectivity feature rows (X) are permuted among subjects WITHIN each institution.
    - Test statistic: max delta_AUC over the 12 candidate configurations.
    Returns: (empirical_p_value, null_max_delta_distribution)
    """
    rng = np.random.default_rng(seed)
    unique_sites = sorted(set(institutions))
    baseline_auc = roc_auc_score(labels, baseline_probs)
    null_max_deltas = np.empty(n_permutations, dtype=np.float64)

    # Standardize covariates once per LOIO fold
    cov_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    all_cov = np.column_stack([motion_covariates, age_sex_covariates])
    for test_site in unique_sites:
        train_mask = institutions != test_site
        test_mask = institutions == test_site
        s_cov = StandardScaler().fit(all_cov[train_mask])
        cov_by_fold[test_site] = (s_cov.transform(all_cov[train_mask]), s_cov.transform(all_cov[test_mask]))

    for b in range(n_permutations):
        # Permute connectivity rows strictly within each site
        conn_perm_indices = np.arange(len(labels))
        for site in unique_sites:
            s_idx = np.where(institutions == site)[0]
            conn_perm_indices[s_idx] = rng.permutation(s_idx)

        ts_perm = [timeseries_list[i] for i in conn_perm_indices]
        pearson_perm = pearson_z_edges[conn_perm_indices]

        # Tangent space on permuted timeseries
        tangent_perm_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        resid_perm_by_fold: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for test_site in unique_sites:
            train_mask = institutions != test_site
            test_mask = institutions == test_site
            ts_tr = [ts_perm[i] for i in np.where(train_mask)[0]]
            ts_te = [ts_perm[i] for i in np.where(test_mask)[0]]
            t_tr, t_te, _ = fit_transform_tangent_space(ts_tr, ts_te, vectorize=True)
            tangent_perm_by_fold[test_site] = (t_tr, t_te)

            X_tr_raw = pearson_perm[train_mask]
            X_te_raw = pearson_perm[test_mask]
            M_tr = motion_covariates[train_mask]
            M_te = motion_covariates[test_mask]
            r_tr, r_te, _, _ = fit_transform_motion_residualization(X_tr_raw, M_tr, X_te_raw, M_te)
            resid_perm_by_fold[test_site] = (r_tr, r_te)

        deltas: List[float] = []
        for feat_kind, sel_mode, clf_name in configs:
            comb_probs = np.zeros(len(labels), dtype=np.float64)
            for test_site in unique_sites:
                train_mask = institutions != test_site
                test_mask = institutions == test_site
                y_tr = labels[train_mask]
                cov_tr, cov_te = cov_by_fold[test_site]

                if feat_kind == "pearson_fisher_z":
                    X_tr = pearson_perm[train_mask]
                    X_te = pearson_perm[test_mask]
                elif feat_kind == "tangent_space":
                    X_tr, X_te = tangent_perm_by_fold[test_site]
                elif feat_kind == "motion_residualized":
                    X_tr, X_te = resid_perm_by_fold[test_site]
                else:
                    X_tr = pearson_perm[train_mask]
                    X_te = pearson_perm[test_mask]

                X_tr_sel, X_te_sel, _ = apply_feature_selection(X_tr, y_tr, X_te, sel_mode)
                full_tr = np.column_stack([cov_tr, X_tr_sel])
                full_te = np.column_stack([cov_te, X_te_sel])

                clf = build_classifier(clf_name, random_state=seed + b)
                clf.fit(full_tr, y_tr)
                comb_probs[test_mask] = clf.predict_proba(full_te)[:, 1]

            comb_auc = roc_auc_score(labels, comb_probs)
            deltas.append(comb_auc - baseline_auc)

        null_max_deltas[b] = max(deltas)

    p_val = float((1.0 + np.sum(null_max_deltas >= observed_max_delta_auc)) / (1.0 + n_permutations))
    return p_val, null_max_deltas


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory Analysis Runner (Guarded).")
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
    args = parser.parse_args()

    enforce_exploratory_guards(execute_real=args.execute_real)
    print("Guards passed. Real execution permitted under exploratory-frozen.")


if __name__ == "__main__":
    main()
