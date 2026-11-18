#!/usr/bin/env python3
"""
Study 2 Registered Development Analysis Runner (LOIO over KKI, NeuroIMAGE, OHSU, Peking).

Implements the exact locked protocol from docs/study2/PREREGISTRATION.md (SHA-256:
e227005c1386dddef3d8b265579f293483769c249cccf31d1cf7f8ec8e09e7ec) and the
co-primary sensitivity regime in docs/study2/DEVIATIONS.md Branch (b):

- Locked fMRI pipeline (refitted from scratch inside every LOIO training fold):
    1. SimpleImputer(strategy='median')
    2. VarianceThreshold(threshold=0.0)
    3. StandardScaler()
    4. SelectKBest(score_func=f_classif, k=100)
    5. RandomForestClassifier(n_estimators=50, max_depth=3,
                              class_weight='balanced', random_state=42)
- Non-imaging baselines (exact same LOIO folds):
    - Motion-Only: SimpleImputer -> StandardScaler -> LogisticRegression(C=1.0, class_weight='balanced')
      on [mean_fd_mm, max_disp_mm] (Branch b NIAK-derived volume-centered motion).
    - Age & Sex:   SimpleImputer -> StandardScaler -> LogisticRegression(C=1.0, class_weight='balanced')
      on [Age, Gender].
- Endpoints & Statistical Inference:
    - Pooled LOIO ROC-AUC (primary endpoint) & per-institution ROC-AUCs
    - Stratified bootstrap 95% CIs (B = 2,000)
    - Paired stratified bootstrap Delta-AUC vs Motion baseline and vs Age/Sex baseline (B = 2,000)
    - Within-institution permutation test (B = 1,000):
      p_perm = (1 + sum(AUC_b >= AUC_obs)) / (1 + B)
- Regimes:
    - Primary Motion Exclusion (N = 368)
    - Co-Primary No-Exclusion Sensitivity (N = 404)
    (Strict threshold N = 133 is descriptive only per DEVIATIONS.md Section 3.4.)

Strict Guardrails:
- Never touches NYU data.
- Never prints individual subject IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEV_INSTITUTIONS = ("KKI", "NeuroIMAGE", "OHSU", "Peking")
DEFAULT_K_BEST = 100
DEFAULT_BOOTSTRAP_B = 2000
DEFAULT_PERM_B = 1000
LOCKED_RANDOM_STATE = 42


def build_locked_fmri_pipeline(k_best: int = DEFAULT_K_BEST) -> Pipeline:
    """Construct the exact locked 4-step preprocessing + RF(50, depth=3, balanced) pipeline."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("var_thresh", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k_best)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=3,
                    class_weight="balanced",
                    random_state=LOCKED_RANDOM_STATE,
                ),
            ),
        ]
    )


def build_linear_baseline_pipeline() -> Pipeline:
    """Construct the locked logistic regression baseline (motion-only or age/sex)."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    random_state=LOCKED_RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )


def _fast_f_classif_topk(X_tr: np.ndarray, y_tr: np.ndarray, k: int) -> np.ndarray:
    """
    Exact vectorized binary-class ANOVA F-statistic (mathematically identical to
    sklearn.feature_selection.f_classif for binary y in {0, 1}) returning top-k indices.
    """
    m0 = y_tr == 0
    m1 = ~m0
    n0 = int(np.sum(m0))
    n1 = int(np.sum(m1))
    n = n0 + n1
    if n0 == 0 or n1 == 0 or n <= 2:
        return np.arange(min(k, X_tr.shape[1]))

    mean0 = np.mean(X_tr[m0], axis=0)
    mean1 = np.mean(X_tr[m1], axis=0)
    mean_all = (n0 * mean0 + n1 * mean1) / n

    ss_between = n0 * (mean0 - mean_all) ** 2 + n1 * (mean1 - mean_all) ** 2
    ss_within = np.sum((X_tr[m0] - mean0) ** 2, axis=0) + np.sum((X_tr[m1] - mean1) ** 2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_scores = (ss_between / 1.0) / (ss_within / (n - 2))
    f_scores = np.where(np.isfinite(f_scores), f_scores, -np.inf)

    k_eff = min(k, X_tr.shape[1])
    # Match sklearn SelectKBest stable sort ascending, taking last k_eff indices in ascending order
    top_idx = np.argsort(f_scores, kind="mergesort")[-k_eff:]
    return np.sort(top_idx)


def run_loio_oof(
    X: np.ndarray,
    y: np.ndarray,
    institutions: np.ndarray,
    pipeline_builder,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run Leave-One-Institution-Out (LOIO) CV, refitting pipeline_builder() strictly from
    scratch inside each training fold. Returns (oof_probs, per_institution_auc).
    """
    n = len(y)
    oof_probs = np.empty(n, dtype=np.float64)
    per_inst_auc: Dict[str, float] = {}
    unique_insts = sorted(set(str(s) for s in institutions))

    for inst in unique_insts:
        val_mask = institutions == inst
        tr_mask = ~val_mask
        model = pipeline_builder()
        model.fit(X[tr_mask], y[tr_mask])
        probs = model.predict_proba(X[val_mask])[:, 1]
        oof_probs[val_mask] = probs
        per_inst_auc[inst] = float(roc_auc_score(y[val_mask], probs))

    return oof_probs, per_inst_auc


def _stratified_bootstrap_indices(
    y: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one bootstrap sample preserving exact counts within each (stratum, class) cell."""
    chosen: List[np.ndarray] = []
    for st in sorted(set(str(s) for s in strata)):
        for cls in (0, 1):
            idx = np.where((strata == st) & (y == cls))[0]
            if len(idx) > 0:
                chosen.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(chosen)


def stratified_bootstrap_auc(
    y: np.ndarray,
    probs: np.ndarray,
    strata: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = LOCKED_RANDOM_STATE,
) -> Dict[str, float]:
    """Compute ROC-AUC point estimate and stratified bootstrap 95% CI."""
    point = float(roc_auc_score(y, probs))
    rng = np.random.default_rng(seed)
    boot_aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = _stratified_bootstrap_indices(y, strata, rng)
        boot_aucs[b] = float(roc_auc_score(y[idx], probs[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    return {"auc": point, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def paired_stratified_bootstrap_delta_auc(
    y: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    strata: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    seed: int = LOCKED_RANDOM_STATE,
) -> Dict[str, object]:
    """
    Compute paired stratified bootstrap 95% CI for Delta_AUC = AUC(a) - AUC(b).
    Outperformance holds iff ci_lo > 0.0 per PREREGISTRATION.md Section 6.
    """
    delta_point = float(roc_auc_score(y, probs_a) - roc_auc_score(y, probs_b))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = _stratified_bootstrap_indices(y, strata, rng)
        deltas[b] = float(
            roc_auc_score(y[idx], probs_a[idx]) - roc_auc_score(y[idx], probs_b[idx])
        )
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_auc": delta_point,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "outperforms_ci_gt_zero": bool(ci_lo > 0.0),
    }


def within_institution_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    institutions: np.ndarray,
    observed_auc: float,
    k_best: int = DEFAULT_K_BEST,
    n_perm: int = DEFAULT_PERM_B,
    seed: int = LOCKED_RANDOM_STATE,
) -> Dict[str, object]:
    """
    Non-parametric permutation test (B = 1,000 default) permuting labels strictly within
    each institution and refitting feature selection + locked RF inside every LOIO fold.

    Steps 1-3 (SimpleImputer, VarianceThreshold, StandardScaler) depend strictly on X_train
    (label-invariant) and are therefore fit once per LOIO fold on X_train alone with zero
    validation leakage; Steps 4-5 (ANOVA SelectKBest + RF) are refit from scratch for every
    permutation and fold.
    """
    unique_insts = sorted(set(str(s) for s in institutions))
    inst_indices = {inst: np.where(institutions == inst)[0] for inst in unique_insts}

    # Precompute strictly unsupervised (label-independent) fold transforms on X_train only
    fold_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for inst in unique_insts:
        val_mask = institutions == inst
        tr_mask = ~val_mask
        unsup = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("var_thresh", VarianceThreshold(threshold=0.0)),
                ("scaler", StandardScaler()),
            ]
        )
        X_tr_s = unsup.fit_transform(X[tr_mask])
        X_val_s = unsup.transform(X[val_mask])
        fold_cache.append((tr_mask, val_mask, X_tr_s, X_val_s))

    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_perm, dtype=np.float64)
    n = len(y)

    for b in range(n_perm):
        y_perm = np.empty_like(y)
        for inst in unique_insts:
            idx = inst_indices[inst]
            y_perm[idx] = rng.permutation(y[idx])

        oof_perm = np.empty(n, dtype=np.float64)
        for tr_mask, val_mask, X_tr_s, X_val_s in fold_cache:
            y_tr_p = y_perm[tr_mask]
            top_cols = _fast_f_classif_topk(X_tr_s, y_tr_p, k_best)
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                class_weight="balanced",
                random_state=LOCKED_RANDOM_STATE,
            )
            rf.fit(X_tr_s[:, top_cols], y_tr_p)
            oof_perm[val_mask] = rf.predict_proba(X_val_s[:, top_cols])[:, 1]

        null_aucs[b] = float(roc_auc_score(y_perm, oof_perm))

    exceed = int(np.sum(null_aucs >= observed_auc))
    p_perm = float((1 + exceed) / (1 + n_perm))
    return {
        "n_perm": int(n_perm),
        "observed_auc": float(observed_auc),
        "null_mean_auc": float(np.mean(null_aucs)),
        "null_sd_auc": float(np.std(null_aucs, ddof=1)) if n_perm > 1 else 0.0,
        "n_exceed": exceed,
        "p_perm": p_perm,
    }


def evaluate_development_regime(
    X_fmri: np.ndarray,
    X_motion: np.ndarray,
    X_age_sex: np.ndarray,
    y: np.ndarray,
    institutions: np.ndarray,
    k_best: int = DEFAULT_K_BEST,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    n_perm: int = DEFAULT_PERM_B,
    seed: int = LOCKED_RANDOM_STATE,
) -> Dict[str, object]:
    """Run complete registered development evaluation for one regime (N=368 or N=404)."""
    k_eff = min(k_best, X_fmri.shape[1])

    oof_fmri, per_inst_fmri = run_loio_oof(
        X_fmri, y, institutions, lambda: build_locked_fmri_pipeline(k_best=k_eff)
    )
    oof_motion, per_inst_motion = run_loio_oof(
        X_motion, y, institutions, build_linear_baseline_pipeline
    )
    oof_agesex, per_inst_agesex = run_loio_oof(
        X_age_sex, y, institutions, build_linear_baseline_pipeline
    )

    pooled_fmri = stratified_bootstrap_auc(y, oof_fmri, institutions, n_boot=n_boot, seed=seed)
    pooled_motion = stratified_bootstrap_auc(y, oof_motion, institutions, n_boot=n_boot, seed=seed)
    pooled_agesex = stratified_bootstrap_auc(y, oof_agesex, institutions, n_boot=n_boot, seed=seed)

    per_inst_ci: Dict[str, Dict[str, float]] = {}
    for inst in sorted(set(str(s) for s in institutions)):
        m = institutions == inst
        per_inst_ci[inst] = stratified_bootstrap_auc(
            y[m], oof_fmri[m], institutions[m], n_boot=n_boot, seed=seed
        )

    delta_vs_motion = paired_stratified_bootstrap_delta_auc(
        y, oof_fmri, oof_motion, institutions, n_boot=n_boot, seed=seed
    )
    delta_vs_agesex = paired_stratified_bootstrap_delta_auc(
        y, oof_fmri, oof_agesex, institutions, n_boot=n_boot, seed=seed
    )

    perm = within_institution_permutation_test(
        X_fmri,
        y,
        institutions,
        observed_auc=pooled_fmri["auc"],
        k_best=k_eff,
        n_perm=n_perm,
        seed=seed,
    )

    dev_success = bool((perm["p_perm"] < 0.05) and delta_vs_motion["outperforms_ci_gt_zero"])

    return {
        "n_samples": int(len(y)),
        "n_features": int(X_fmri.shape[1]),
        "k_selected": int(k_eff),
        "fmri_pooled": pooled_fmri,
        "fmri_per_institution": per_inst_ci,
        "motion_baseline_pooled": pooled_motion,
        "motion_baseline_per_institution": per_inst_motion,
        "age_sex_baseline_pooled": pooled_agesex,
        "age_sex_baseline_per_institution": per_inst_agesex,
        "delta_vs_motion": delta_vs_motion,
        "delta_vs_age_sex": delta_vs_agesex,
        "permutation_test": perm,
        "meets_dev_success_criteria": dev_success,
    }


def run_both_regimes_from_npz(
    features_dir: Path,
    n_boot: int = DEFAULT_BOOTSTRAP_B,
    n_perm: int = DEFAULT_PERM_B,
) -> Dict[str, object]:
    """Load frozen development .npz files (primary N=368 and no-exclusion N=404) and evaluate."""
    results: Dict[str, object] = {}
    for regime, fname, expected_n in [
        ("primary_368", "dev_cc200_fisher_z_primary368.npz", 368),
        ("no_exclusion_404", "dev_cc200_fisher_z_all404.npz", 404),
    ]:
        data = np.load(features_dir / fname, allow_pickle=False)
        X = data["X"]
        y = data["y"]
        inst = data["institutions"]
        motion_X = data["motion_X"]
        age_sex_X = data["age_sex_X"]
        if len(y) != expected_n:
            raise ValueError(f"Expected N={expected_n} in {fname}, found {len(y)}")
        found_insts = tuple(sorted(set(str(s) for s in inst)))
        if found_insts != DEV_INSTITUTIONS:
            raise ValueError(f"Expected institutions {DEV_INSTITUTIONS}, got {found_insts}")

        results[regime] = evaluate_development_regime(
            X, motion_X, age_sex_X, y, inst, k_best=DEFAULT_K_BEST, n_boot=n_boot, n_perm=n_perm
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Study 2 registered LOIO development analysis (requires --execute-real-dev)."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("NeuroADHD/data/athena/features"),
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--n-perm", type=int, default=DEFAULT_PERM_B)
    parser.add_argument(
        "--execute-real-dev",
        action="store_true",
        help="Explicit safety guard required to execute on real Study 2 development labels.",
    )
    args = parser.parse_args()
    if not args.execute_real_dev:
        raise SystemExit(
            "Refusing to run on real labels without --execute-real-dev. "
            "Run synthetic unit tests in tests/test_run_study2_dev_analysis.py instead."
        )

    res = run_both_regimes_from_npz(args.features_dir, n_boot=args.n_boot, n_perm=args.n_perm)
    text = json.dumps(res, indent=2)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
