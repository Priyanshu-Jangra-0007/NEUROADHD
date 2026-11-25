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
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract connectivity features for a train/test split without touching test labels."""
    if feature_kind == "pearson_fisher_z" or feature_kind == "multi_run_avg":
        return pearson_z_edges[train_mask], pearson_z_edges[test_mask]
    elif feature_kind == "tangent_space":
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
    Evaluate outer LOIO predictions for Motion + Age + Sex + Connectivity via LATE FUSION:
    For each outer held-out institution `test_site`:
      1. Build inner 4-fold LOIO splits strictly within the 4 training institutions
         (no held-out institution ever touches the inner models or fusion weights).
      2. Compute inner out-of-fold probabilities `s_cov_oof` (covariate-only LR) and
         `s_conn_oof` (connectivity configuration) for all training-institution subjects.
      3. Fit a 2-parameter LogisticRegression meta-learner on `[s_cov_oof, s_conn_oof]`
         using training-institution labels only.
      4. Fit covariate-only LR and connectivity configuration on all 4 training institutions,
         predict `[s_cov_test, s_conn_test]` on `test_site`, and return meta-learner probabilities.
    """
    if unique_institutions is None:
        unique_institutions = sorted(set(institutions))

    n = len(labels)
    probs_out = np.zeros(n, dtype=np.float64)
    covariates = np.column_stack([motion_covariates, age_sex_covariates])

    for test_site in unique_institutions:
        outer_train_mask = institutions != test_site
        outer_test_mask = institutions == test_site
        inner_sites = [s for s in unique_institutions if s != test_site]

        # 1-2. Inner LOIO out-of-fold scores strictly within the 4 training institutions
        s_cov_oof = np.zeros(n, dtype=np.float64)
        s_conn_oof = np.zeros(n, dtype=np.float64)

        for inner_val_site in inner_sites:
            inner_tr_mask = outer_train_mask & (institutions != inner_val_site)
            inner_val_mask = institutions == inner_val_site
            y_in_tr = labels[inner_tr_mask]

            sc_in = StandardScaler().fit(covariates[inner_tr_mask])
            clf_cov_in = LogisticRegression(
                C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=random_state
            )
            clf_cov_in.fit(sc_in.transform(covariates[inner_tr_mask]), y_in_tr)
            s_cov_oof[inner_val_mask] = clf_cov_in.predict_proba(sc_in.transform(covariates[inner_val_mask]))[:, 1]

            X_in_tr, X_in_val = _build_conn_fold_features(
                feature_kind, inner_tr_mask, inner_val_mask, timeseries_list, pearson_z_edges, motion_covariates
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

        X_out_tr, X_out_te = _build_conn_fold_features(
            feature_kind, outer_train_mask, outer_test_mask, timeseries_list, pearson_z_edges, motion_covariates
        )
        s_conn_test = _fit_predict_connectivity_fold(
            feature_kind, selection_mode, classifier_name, X_out_tr, X_out_te, y_out_tr, random_state=random_state
        )

        meta_X_te = np.column_stack([s_cov_test, s_conn_test])
        probs_out[outer_test_mask] = meta_clf.predict_proba(meta_X_te)[:, 1]

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

        if b < len(completed):
            continue

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
        for inner_val_site in inner_sites:
            inner_tr_mask = outer_train_mask & (institutions != inner_val_site)
            inner_val_mask = institutions == inner_val_site
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
            fused_probs = np.zeros(n, dtype=np.float64)
            for test_site in unique_sites:
                outer_train_mask = institutions != test_site
                outer_test_mask = institutions == test_site
                outer_tr_sites = frozenset(s for s in unique_sites if s != test_site)
                inner_sites = [s for s in unique_sites if s != test_site]

                s_conn_oof = np.zeros(n, dtype=np.float64)
                for inner_val_site in inner_sites:
                    inner_tr_mask = outer_train_mask & (institutions != inner_val_site)
                    inner_val_mask = institutions == inner_val_site
                    inner_tr_sites = frozenset(s for s in outer_tr_sites if s != inner_val_site)
                    X_in_tr, X_in_val = _get_split_feats(
                        feat_kind, inner_tr_sites, inner_tr_mask, inner_val_mask, inner_val_site
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
    print("Guards passed. Real execution permitted under exploratory-frozen.")


if __name__ == "__main__":
    main()
