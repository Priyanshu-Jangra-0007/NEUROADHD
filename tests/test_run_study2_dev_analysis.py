import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from scripts.run_study2_dev_analysis import (
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_K_BEST,
    DEFAULT_PERM_B,
    DEFAULT_RESULTS_JSON,
    DEV_INSTITUTIONS,
    LOCKED_RANDOM_STATE,
    _fast_f_classif_topk,
    build_locked_fmri_pipeline,
    evaluate_development_regime,
    run_both_regimes_from_npz,
    run_loio_oof,
)


def _make_synthetic_cohort(
    seed: int,
    signal_strength: float,
    n_per_site: int = 40,
    n_features: int = 200,
):
    rng = np.random.default_rng(seed)
    n_total = n_per_site * len(DEV_INSTITUTIONS)
    institutions = np.repeat(np.array(DEV_INSTITUTIONS, dtype="<U16"), n_per_site)
    y = np.tile(np.array([0] * (n_per_site // 2) + [1] * (n_per_site // 2), dtype=np.int64), len(DEV_INSTITUTIONS))

    X_fmri = rng.normal(0.0, 1.0, size=(n_total, n_features))
    if signal_strength > 0.0:
        # Plant consistent cross-institution signal on 25 features
        X_fmri[y == 1, :25] += signal_strength

    # Pure noise motion and age/sex baselines (no confounding signal)
    X_motion = rng.normal(0.2, 0.05, size=(n_total, 2))
    X_age_sex = np.column_stack(
        [
            rng.normal(11.0, 2.0, size=n_total),
            rng.integers(0, 2, size=n_total).astype(np.float64),
        ]
    )
    return X_fmri, X_motion, X_age_sex, y, institutions


class TestRunStudy2DevAnalysisSynthetic(unittest.TestCase):
    def test_fast_f_classif_matches_sklearn_exact(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(120, 300))
        y = rng.integers(0, 2, size=120)
        X[y == 1, :15] += 0.8

        sk_sel = SelectKBest(score_func=f_classif, k=100).fit(X, y)
        sk_cols = np.where(sk_sel.get_support())[0]
        fast_cols = _fast_f_classif_topk(X, y, k=100)
        np.testing.assert_array_equal(fast_cols, sk_cols)

    def test_planted_signal_synthetic_dataset(self):
        X_fmri, X_motion, X_age_sex, y, inst = _make_synthetic_cohort(
            seed=42, signal_strength=1.0, n_per_site=40, n_features=200
        )
        res = evaluate_development_regime(
            X_fmri,
            X_motion,
            X_age_sex,
            y,
            inst,
            k_best=100,
            n_boot=200,
            n_perm=50,
            seed=42,
        )
        # AUC must be clearly above 0.5 with significant permutation p and outperforming motion
        self.assertGreater(res["fmri_pooled"]["auc"], 0.85)
        self.assertGreater(res["fmri_pooled"]["ci_lo"], 0.75)
        for site_name in DEV_INSTITUTIONS:
            self.assertGreater(res["fmri_per_institution"][site_name]["auc"], 0.75)
        self.assertLess(res["permutation_test"]["p_perm"], 0.05)
        self.assertTrue(res["delta_vs_motion"]["outperforms_ci_gt_zero"])
        self.assertTrue(res["meets_dev_success_criteria"])

    def test_null_signal_synthetic_dataset(self):
        X_fmri, X_motion, X_age_sex, y, inst = _make_synthetic_cohort(
            seed=99, signal_strength=0.0, n_per_site=40, n_features=200
        )
        res = evaluate_development_regime(
            X_fmri,
            X_motion,
            X_age_sex,
            y,
            inst,
            k_best=100,
            n_boot=200,
            n_perm=100,
            seed=99,
        )
        auc = res["fmri_pooled"]["auc"]
        self.assertGreater(auc, 0.35)
        self.assertLess(auc, 0.65)
        # Permutation p must NOT be significant on pure noise
        self.assertGreater(res["permutation_test"]["p_perm"], 0.05)
        self.assertFalse(res["meets_dev_success_criteria"])

    def test_zero_leakage_across_loio_folds(self):
        X_fmri, _, _, y, inst = _make_synthetic_cohort(
            seed=7, signal_strength=0.8, n_per_site=30, n_features=150
        )
        # Hold out KKI; verify corrupting KKI features (+50,000) does not change training pipeline
        val_mask = inst == "KKI"
        tr_mask = ~val_mask

        pipe_clean = build_locked_fmri_pipeline(k_best=100)
        pipe_clean.fit(X_fmri[tr_mask], y[tr_mask])

        X_corrupted = X_fmri.copy()
        X_corrupted[val_mask] += 50000.0

        pipe_corrupt = build_locked_fmri_pipeline(k_best=100)
        pipe_corrupt.fit(X_corrupted[tr_mask], y[tr_mask])

        np.testing.assert_array_equal(
            pipe_clean.named_steps["imputer"].statistics_,
            pipe_corrupt.named_steps["imputer"].statistics_,
        )
        np.testing.assert_array_equal(
            pipe_clean.named_steps["scaler"].mean_,
            pipe_corrupt.named_steps["scaler"].mean_,
        )
        np.testing.assert_array_equal(
            pipe_clean.named_steps["scaler"].scale_,
            pipe_corrupt.named_steps["scaler"].scale_,
        )
        np.testing.assert_array_equal(
            pipe_clean.named_steps["selector"].scores_,
            pipe_corrupt.named_steps["selector"].scores_,
        )
        np.testing.assert_array_equal(
            pipe_clean.named_steps["selector"].get_support(),
            pipe_corrupt.named_steps["selector"].get_support(),
        )

        # Also check LOIO wrapper itself uses strictly training fold for KKI predictions
        oof_1, _ = run_loio_oof(
            X_fmri, y, inst, lambda: build_locked_fmri_pipeline(k_best=100)
        )
        # Flipping KKI labels y[val_mask] cannot change KKI's predicted probabilities!
        y_val_flipped = y.copy()
        y_val_flipped[val_mask] = 1 - y_val_flipped[val_mask]
        oof_2, _ = run_loio_oof(
            X_fmri, y_val_flipped, inst, lambda: build_locked_fmri_pipeline(k_best=100)
        )
        np.testing.assert_array_equal(oof_1[val_mask], oof_2[val_mask])

    def test_both_regimes_npz_h1_h2_h3_and_locked_defaults(self):
        # Verify locked protocol defaults and git-ignored default output path
        self.assertEqual(DEFAULT_K_BEST, 100)
        self.assertEqual(DEFAULT_BOOTSTRAP_B, 2000)
        self.assertEqual(DEFAULT_PERM_B, 1000)
        self.assertEqual(LOCKED_RANDOM_STATE, 42)
        self.assertIn("athena", DEFAULT_RESULTS_JSON.parts)

        # Build synthetic N=404 and N=368 .npz archives with planted signal
        rng = np.random.default_rng(2026)
        counts_404 = {"KKI": 83, "NeuroIMAGE": 48, "OHSU": 79, "Peking": 194}
        counts_368 = {"KKI": 79, "NeuroIMAGE": 42, "OHSU": 58, "Peking": 189}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for fname, counts in [
                ("dev_cc200_fisher_z_all404.npz", counts_404),
                ("dev_cc200_fisher_z_primary368.npz", counts_368),
            ]:
                inst_list, y_list = [], []
                for site_name, n_s in counts.items():
                    inst_list.extend([site_name] * n_s)
                    n_pos = n_s // 2
                    y_list.extend([0] * (n_s - n_pos) + [1] * n_pos)
                y_arr = np.array(y_list, dtype=np.int64)
                inst_arr = np.array(inst_list, dtype="<U16")
                n_tot = len(y_arr)
                X_arr = rng.normal(0.0, 1.0, size=(n_tot, 120))
                X_arr[y_arr == 1, :20] += 0.9
                m_arr = rng.normal(0.2, 0.05, size=(n_tot, 2))
                as_arr = np.column_stack(
                    [rng.normal(11.0, 2.0, size=n_tot), rng.integers(0, 2, size=n_tot).astype(np.float64)]
                )
                np.savez_compressed(
                    tmp_dir / fname,
                    X=X_arr,
                    y=y_arr,
                    institutions=inst_arr,
                    motion_X=m_arr,
                    age_sex_X=as_arr,
                )

            res = run_both_regimes_from_npz(tmp_dir, n_boot=50, n_perm=10)
            self.assertEqual(set(res.keys()), {"primary_368", "no_exclusion_404"})
            for key, expected_n in [("primary_368", 368), ("no_exclusion_404", 404)]:
                r = res[key]
                self.assertEqual(r["n_samples"], expected_n)
                # H1: pooled LOIO AUC + bootstrap CI + within-institution permutation
                self.assertGreater(r["fmri_pooled"]["auc"], 0.85)
                self.assertIn("ci_lo", r["fmri_pooled"])
                self.assertIn("ci_hi", r["fmri_pooled"])
                self.assertIn("p_perm", r["permutation_test"])
                # Per-institution AUCs for all 4 institutions
                self.assertEqual(tuple(sorted(r["fmri_per_institution"].keys())), DEV_INSTITUTIONS)
                # H2: paired bootstrap Delta AUC vs motion-only
                self.assertIn("delta_auc", r["delta_vs_motion"])
                self.assertTrue(r["delta_vs_motion"]["outperforms_ci_gt_zero"])
                # H3: paired stratified bootstrap Delta AUC vs age/sex
                self.assertIn("delta_auc", r["delta_vs_age_sex"])
                self.assertTrue(r["delta_vs_age_sex"]["outperforms_ci_gt_zero"])


if __name__ == "__main__":
    unittest.main()
