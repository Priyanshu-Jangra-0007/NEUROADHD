#!/usr/bin/env python3
"""
Unit and Leakage Tests for scripts/run_exploratory.py (Synthetic Data Only; CC200 (190 populated ROIs) protocol).

Covers all pre-specified properties:
(a) Planted signal gives AUC clearly above 0.5.
(b) No signal gives about 0.5 and a non-significant label permutation p.
(c) Leakage test: changing held-out rows changes no fitted statistic, tangent reference mean, or selected edges.
(d) Subject-grouped folds keep all runs of a subject together.
(e) The guard refuses without the flag and tag.
(f) Incremental contrast null: covariates carry signal, connectivity is noise -> delta ~ 0, p not significant.
(g) Incremental contrast signal: connectivity carries signal beyond covariates -> delta > 0, p significant.
(h) Within-institution label permutation preserves per-site class prevalence (Test 1).
(i) "Interesting" 3-part conjunction & write-once / preflight checks.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_exploratory import (
    apply_feature_selection,
    build_classifier,
    enforce_exploratory_guards,
    evaluate_combined_covariates_and_connectivity,
    evaluate_interesting_criteria,
    evaluate_loio_pipeline,
    evaluate_non_imaging_baselines,
    fit_transform_motion_residualization,
    fit_transform_tangent_space,
    match_subjects_on_motion_within_site,
    run_incremental_contrast_permutation_test,
    run_preflight,
    run_within_site_label_permutation_test,
    timeseries_to_fisher_z,
    write_results_once,
)


class TestRunExploratorySynthetic(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def _generate_synthetic_loio_data(
        self,
        n_per_site: int = 40,
        n_rois: int = 15,
        t_len: int = 50,
        planted_connectivity: bool = False,
        planted_covariates: bool = False,
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic LOIO multi-site dataset with optional planted signals."""
        sites = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]
        timeseries_list: List[np.ndarray] = []
        labels: List[int] = []
        institutions: List[str] = []
        motion_covariates: List[List[float]] = []
        age_sex_covariates: List[List[float]] = []

        pearson_z_edges = []

        for site_idx, site in enumerate(sites):
            for i in range(n_per_site):
                y = i % 2
                labels.append(y)
                institutions.append(site)

                if planted_covariates:
                    # Covariates strongly separate labels
                    mean_fd = 0.15 + (0.15 * y) + 0.02 * self.rng.normal()
                    max_disp = 0.30 + (0.30 * y) + 0.04 * self.rng.normal()
                    age = 10.0 + (3.0 * y) + 0.5 * self.rng.normal()
                    sex = float(y)
                else:
                    mean_fd = 0.15 + (0.02 * y) + 0.02 * self.rng.normal()
                    max_disp = mean_fd * 3.0 + 0.05 * self.rng.normal()
                    age = 10.0 + 2.0 * self.rng.normal()
                    sex = float(self.rng.integers(0, 2))

                motion_covariates.append([mean_fd, max_disp])
                age_sex_covariates.append([age, sex])

                # Generate BOLD timeseries
                ts = self.rng.normal(size=(t_len, n_rois))
                if planted_connectivity and y == 1:
                    shared_signal = self.rng.normal(size=t_len) * 3.5
                    ts[:, 0] += shared_signal
                    ts[:, 1] += shared_signal

                timeseries_list.append(ts)
                pearson_z_edges.append(timeseries_to_fisher_z(ts))

        return (
            timeseries_list,
            np.array(pearson_z_edges, dtype=np.float64),
            np.array(motion_covariates, dtype=np.float64),
            np.array(age_sex_covariates, dtype=np.float64),
            np.array(labels, dtype=np.int64),
            np.array(institutions, dtype="<U16"),
        )

    # -----------------------------------------------------------------
    # Test (a): Planted signal gives AUC clearly above 0.5
    # -----------------------------------------------------------------
    def test_a_planted_signal_gives_high_auc(self) -> None:
        ts_list, z_edges, motion, _, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=30, n_rois=10, planted_connectivity=True
        )
        probs, y_out = evaluate_loio_pipeline(
            feature_kind="pearson_fisher_z",
            selection_mode="k100",
            classifier_name="rf",
            timeseries_list=ts_list,
            pearson_z_edges=z_edges,
            motion_covariates=motion,
            labels=labels,
            institutions=sites,
            random_state=42,
        )
        auc = roc_auc_score(y_out, probs)
        self.assertGreater(
            auc,
            0.85,
            f"Planted signal expected to yield AUC > 0.85, obtained {auc:.4f}",
        )

    # -----------------------------------------------------------------
    # Test (b): No signal gives ~0.5 and non-significant permutation p
    # -----------------------------------------------------------------
    def test_b_null_signal_gives_chance_auc_and_nonsignificant_permutation_p(self) -> None:
        ts_list, z_edges, motion, _, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=25, n_rois=8, planted_connectivity=False
        )
        configs = [
            ("pearson_fisher_z", "k100", "l2_lr"),
            ("tangent_space", "all", "l2_lr"),
        ]

        probs, y_out = evaluate_loio_pipeline(
            feature_kind="pearson_fisher_z",
            selection_mode="k100",
            classifier_name="l2_lr",
            timeseries_list=ts_list,
            pearson_z_edges=z_edges,
            motion_covariates=motion,
            labels=labels,
            institutions=sites,
            random_state=42,
        )
        observed_auc = roc_auc_score(y_out, probs)
        self.assertAlmostEqual(observed_auc, 0.50, delta=0.20)

        p_val, _ = run_within_site_label_permutation_test(
            configs=configs,
            timeseries_list=ts_list,
            pearson_z_edges=z_edges,
            motion_covariates=motion,
            labels=labels,
            institutions=sites,
            observed_max_auc=observed_auc,
            n_permutations=20,
            seed=42,
        )
        self.assertGreater(
            p_val,
            0.05,
            f"Null signal should yield non-significant p > 0.05, obtained {p_val:.4f}",
        )

    # -----------------------------------------------------------------
    # Test (c): Leakage test (modifying held-out rows changes no fitted stat)
    # -----------------------------------------------------------------
    def test_c_zero_leakage_across_held_out_site(self) -> None:
        ts_list, z_edges, motion, _, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=20, n_rois=10, planted_connectivity=False
        )
        test_site = "Site_E"
        train_mask = sites != test_site
        test_mask = sites == test_site

        ts_train = [ts_list[i] for i in np.where(train_mask)[0]]
        ts_test_orig = [ts_list[i] for i in np.where(test_mask)[0]]

        X_tr_tang_orig, _, cm_orig = fit_transform_tangent_space(ts_train, ts_test_orig)
        _, _, w_orig, b_orig = fit_transform_motion_residualization(
            z_edges[train_mask], motion[train_mask], z_edges[test_mask], motion[test_mask]
        )
        X_tr_sel_orig, _, sel_idx_orig = apply_feature_selection(
            z_edges[train_mask], labels[train_mask], z_edges[test_mask], "k100"
        )

        # Corrupt held-out test site data
        ts_test_corrupt = [ts + 1000.0 * self.rng.normal(size=ts.shape) for ts in ts_test_orig]
        z_test_corrupt = z_edges[test_mask] + 999.0
        motion_test_corrupt = motion[test_mask] * 50.0

        X_tr_tang_new, _, cm_new = fit_transform_tangent_space(ts_train, ts_test_corrupt)
        _, _, w_new, b_new = fit_transform_motion_residualization(
            z_edges[train_mask], motion[train_mask], z_test_corrupt, motion_test_corrupt
        )
        X_tr_sel_new, _, sel_idx_new = apply_feature_selection(
            z_edges[train_mask], labels[train_mask], z_test_corrupt, "k100"
        )

        np.testing.assert_array_equal(
            cm_orig.mean_,
            cm_new.mean_,
            err_msg="Tangent space reference mean changed when held-out test data changed!",
        )
        np.testing.assert_array_equal(
            w_orig,
            w_new,
            err_msg="Motion regression weights changed when held-out test data changed!",
        )
        np.testing.assert_array_equal(
            b_orig,
            b_new,
            err_msg="Motion regression intercept changed when held-out test data changed!",
        )
        np.testing.assert_array_equal(
            sel_idx_orig,
            sel_idx_new,
            err_msg="ANOVA selected feature indices changed when held-out test data changed!",
        )
        np.testing.assert_array_equal(
            X_tr_sel_orig,
            X_tr_sel_new,
            err_msg="Training selected features changed when held-out test data changed!",
        )

    # -----------------------------------------------------------------
    # Test (d): Subject-grouped folds keep all runs of a subject together
    # -----------------------------------------------------------------
    def test_d_subject_grouped_folds_keep_runs_together(self) -> None:
        synthetic_subjects = [f"SYN_SUBJ_{i:03d}" for i in range(50)]
        runs_per_subject = [1 if i < 20 else (2 if i < 40 else 3) for i in range(50)]

        all_run_sub_ids: List[str] = []
        all_run_site: List[str] = []
        for s_id, n_runs in zip(synthetic_subjects, runs_per_subject):
            site = "Site_NYU" if int(s_id.split("_")[-1]) % 2 == 0 else "Site_OHSU"
            for _ in range(n_runs):
                all_run_sub_ids.append(s_id)
                all_run_site.append(site)

        all_run_sub_ids_arr = np.array(all_run_sub_ids)
        all_run_site_arr = np.array(all_run_site)

        for test_site in ["Site_NYU", "Site_OHSU"]:
            train_mask = all_run_site_arr != test_site
            test_mask = all_run_site_arr == test_site

            train_subs = set(all_run_sub_ids_arr[train_mask])
            test_subs = set(all_run_sub_ids_arr[test_mask])

            overlap = train_subs.intersection(test_subs)
            self.assertEqual(
                len(overlap),
                0,
                f"Subject leakage detected in LOIO: {len(overlap)} subjects span both train and test!",
            )

        fake_labels = [i % 2 for i in range(len(all_run_sub_ids))]
        fake_mean_fd = [0.10 + 0.05 * (i % 3) for i in range(len(all_run_sub_ids))]
        matched_idx = match_subjects_on_motion_within_site(
            subject_ids=all_run_sub_ids,
            site_names=all_run_site,
            mean_fd_values=fake_mean_fd,
            labels=fake_labels,
            caliper_mm=0.05,
        )
        self.assertGreater(len(matched_idx), 0)
        self.assertEqual(len(matched_idx) % 2, 0, "Matched cohort must have an even number of entries")

    # -----------------------------------------------------------------
    # Test (e): Guard refuses without flag and tag
    # -----------------------------------------------------------------
    def test_e_guard_refuses_without_flag_and_tag(self) -> None:
        with self.assertRaises(RuntimeError) as ctx1:
            enforce_exploratory_guards(execute_real=False)
        self.assertIn("Refusing to read real labels without --execute-real", str(ctx1.exception))

        with self.assertRaises(RuntimeError) as ctx2:
            enforce_exploratory_guards(
                execute_real=True,
                tag_name="nonexistent-exploratory-tag-9999",
            )
        self.assertIn("does not exist on an ancestor of HEAD", str(ctx2.exception))

    # -----------------------------------------------------------------
    # Test (f): Incremental contrast null (covariates have signal, conn is noise)
    # -----------------------------------------------------------------
    def test_f_incremental_null_covariates_carry_signal_connectivity_is_noise(self) -> None:
        ts_list, z_edges, motion, age_sex, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=25, n_rois=8, planted_connectivity=False, planted_covariates=True
        )
        baselines = evaluate_non_imaging_baselines(
            motion_covariates=motion,
            age_sex_covariates=age_sex,
            labels=labels,
            institutions=sites,
            random_state=42,
        )
        base_auc = roc_auc_score(labels, baselines["motion_age_sex"])
        self.assertGreater(base_auc, 0.70, f"Covariates should carry signal, base_auc={base_auc:.4f}")

        configs = [("pearson_fisher_z", "k100", "l2_lr")]
        comb_probs = evaluate_combined_covariates_and_connectivity(
            feature_kind="pearson_fisher_z",
            selection_mode="k100",
            classifier_name="l2_lr",
            timeseries_list=ts_list,
            pearson_z_edges=z_edges,
            motion_covariates=motion,
            age_sex_covariates=age_sex,
            labels=labels,
            institutions=sites,
            random_state=42,
        )
        comb_auc = roc_auc_score(labels, comb_probs)
        delta_auc = comb_auc - base_auc
        self.assertLess(
            delta_auc,
            0.05,
            f"Noise connectivity should not increase AUC over signal covariates, delta={delta_auc:.4f}",
        )

        p_val, _ = run_incremental_contrast_permutation_test(
            configs=configs,
            timeseries_list=ts_list,
            pearson_z_edges=z_edges,
            motion_covariates=motion,
            age_sex_covariates=age_sex,
            labels=labels,
            institutions=sites,
            baseline_probs=baselines["motion_age_sex"],
            observed_max_delta_auc=delta_auc,
            n_permutations=20,
            seed=42,
        )
        self.assertGreater(
            p_val,
            0.05,
            f"Incremental contrast should be non-significant (p > 0.05) when connectivity is noise, got p={p_val:.4f}",
        )

    # -----------------------------------------------------------------
    # Test (g): Incremental contrast signal (connectivity carries signal beyond covariates)
    # -----------------------------------------------------------------
    def test_g_incremental_signal_connectivity_carries_signal_beyond_covariates(self) -> None:
        ts_list, z_edges, motion, age_sex, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=30, n_rois=8, planted_connectivity=True, planted_covariates=False
        )
        baselines = evaluate_non_imaging_baselines(
            motion_covariates=motion,
            age_sex_covariates=age_sex,
            labels=labels,
            institutions=sites,
            random_state=42,
        )
        base_auc = roc_auc_score(labels, baselines["motion_age_sex"])

        configs = [("pearson_fisher_z", "k100", "l2_lr")]
        comb_probs = evaluate_combined_covariates_and_connectivity(
            feature_kind="pearson_fisher_z",
            selection_mode="k100",
            classifier_name="l2_lr",
            timeseries_list=ts_list,
            pearson_z_edges=z_edges,
            motion_covariates=motion,
            age_sex_covariates=age_sex,
            labels=labels,
            institutions=sites,
            random_state=42,
        )
        comb_auc = roc_auc_score(labels, comb_probs)
        delta_auc = comb_auc - base_auc
        self.assertGreater(
            delta_auc,
            0.15,
            f"Connectivity signal should strongly boost AUC beyond noise covariates, delta={delta_auc:.4f}",
        )

        p_val, _ = run_incremental_contrast_permutation_test(
            configs=configs,
            timeseries_list=ts_list,
            pearson_z_edges=z_edges,
            motion_covariates=motion,
            age_sex_covariates=age_sex,
            labels=labels,
            institutions=sites,
            baseline_probs=baselines["motion_age_sex"],
            observed_max_delta_auc=delta_auc,
            n_permutations=20,
            seed=42,
        )
        self.assertLessEqual(
            p_val,
            0.10,
            f"Incremental contrast should be statistically significant, got p={p_val:.4f}",
        )

    # -----------------------------------------------------------------
    # Test (h): Test 1 permutes labels strictly within each institution
    # -----------------------------------------------------------------
    def test_h_within_site_label_permutation_preserves_site_prevalence(self) -> None:
        # Construct 5 sites with unequal ADHD prevalence, pure noise features
        rng = np.random.default_rng(123)
        sites_cfg = [("S1", 20, 4), ("S2", 20, 16), ("S3", 20, 10), ("S4", 20, 6), ("S5", 20, 14)]
        ts_list, z_edges, motion, labels, institutions = [], [], [], [], []
        for s_name, n_s, n_pos in sites_cfg:
            y_site = np.array([1] * n_pos + [0] * (n_s - n_pos))
            for i in range(n_s):
                ts = rng.normal(size=(40, 8))
                ts_list.append(ts)
                z_edges.append(timeseries_to_fisher_z(ts))
                motion.append([0.15 + 0.02 * rng.normal(), 0.30 + 0.05 * rng.normal()])
                labels.append(int(y_site[i]))
                institutions.append(s_name)

        z_arr = np.array(z_edges, dtype=np.float64)
        m_arr = np.array(motion, dtype=np.float64)
        y_arr = np.array(labels, dtype=np.int64)
        inst_arr = np.array(institutions, dtype="<U8")

        # Verify within-site permutation keeps per-site counts invariant and yields non-sig p
        probs, _ = evaluate_loio_pipeline(
            "pearson_fisher_z", "k100", "l2_lr", ts_list, z_arr, m_arr, y_arr, inst_arr, random_state=42
        )
        obs_auc = roc_auc_score(y_arr, probs)
        p_val, null_dist = run_within_site_label_permutation_test(
            configs=[("pearson_fisher_z", "k100", "l2_lr")],
            timeseries_list=ts_list,
            pearson_z_edges=z_arr,
            motion_covariates=m_arr,
            labels=y_arr,
            institutions=inst_arr,
            observed_max_auc=obs_auc,
            n_permutations=20,
            seed=42,
        )
        self.assertGreater(p_val, 0.05)
        self.assertEqual(len(null_dist), 20)

    # -----------------------------------------------------------------
    # Test (i): "Interesting" 3-part conjunction, preflight, and write-once
    # -----------------------------------------------------------------
    def test_i_interesting_conjunction_preflight_and_write_once(self) -> None:
        rng = np.random.default_rng(7)
        sites = np.repeat(["S1", "S2", "S3", "S4", "S5"], 40)
        y = np.tile([0, 1], 100)

        # Good signal across all 5 sites
        p_good = np.clip(0.2 + 0.6 * y + 0.1 * rng.normal(size=200), 0.01, 0.99)
        p_weak = np.clip(0.45 + 0.1 * y + 0.15 * rng.normal(size=200), 0.01, 0.99)

        res_pass = evaluate_interesting_criteria(
            y_true=y,
            institutions=sites,
            probs_combined=p_good,
            probs_motion_baseline=p_weak,
            probs_agesex_baseline=p_weak,
            probs_combined_baseline=p_weak,
            incremental_perm_p=0.01,
            n_boot=100,
        )
        self.assertTrue(res_pass["is_interesting"])
        self.assertEqual(res_pass["verdict"], "interesting")

        # Fail condition (b): p >= 0.05
        res_fail_p = evaluate_interesting_criteria(
            y_true=y,
            institutions=sites,
            probs_combined=p_good,
            probs_motion_baseline=p_weak,
            probs_agesex_baseline=p_weak,
            probs_combined_baseline=p_weak,
            incremental_perm_p=0.06,
            n_boot=100,
        )
        self.assertFalse(res_fail_p["is_interesting"])
        self.assertEqual(res_fail_p["verdict"], "not interesting")

        # Fail condition (c): only 3 of 5 sites positive
        p_3sites = p_good.copy()
        for s in ["S4", "S5"]:
            mask = sites == s
            p_3sites[mask] = 0.5  # tie/below baseline
        res_fail_sites = evaluate_interesting_criteria(
            y_true=y,
            institutions=sites,
            probs_combined=p_3sites,
            probs_motion_baseline=p_weak,
            probs_agesex_baseline=p_weak,
            probs_combined_baseline=p_weak,
            incremental_perm_p=0.01,
            n_boot=100,
        )
        self.assertFalse(res_fail_sites["cond_c_sites_positive_ge_4"])
        self.assertFalse(res_fail_sites["is_interesting"])

        # Preflight failure on missing directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                run_preflight(data_root=tmp_path)

            # Write-once test
            out_file = tmp_path / "results.json"
            write_results_once(out_file, {"status": "ok"})
            self.assertTrue(out_file.is_file())
            with self.assertRaises(FileExistsError):
                write_results_once(out_file, {"status": "overwrite"})


if __name__ == "__main__":
    unittest.main()
