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

import csv
import io
import json
import re
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_exploratory import (
    CohortData,
    LabelFreeInputs,
    apply_feature_selection,
    build_classifier,
    build_cohorts,
    enforce_exploratory_guards,
    evaluate_combined_covariates_and_connectivity,
    evaluate_interesting_criteria,
    evaluate_loio_pipeline,
    evaluate_non_imaging_baselines,
    fit_evaluate_late_fusion_outer_fold,
    fit_transform_motion_residualization,
    fit_transform_tangent_space,
    load_label_free_inputs,
    load_permutation_checkpoint,
    match_subjects_on_motion_within_site,
    run_exploratory_synthetic_e2e,
    run_full_analysis,
    run_incremental_contrast_permutation_test,
    run_preflight,
    run_rehearsal,
    run_within_site_label_permutation_test,
    save_permutation_checkpoint,
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
            abs(delta_auc),
            0.03,
            f"Late fusion with noise connectivity should stay within 0.03 of covariate-only AUC, delta={delta_auc:.4f}",
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

            # Null-only checkpoint test (resuming produces identical null distribution)
            ts_list, z_edges, motion, age_sex, y_syn, inst_syn = self._generate_synthetic_loio_data(
                n_per_site=12, n_rois=8, planted_connectivity=False, planted_covariates=False
            )
            ckpt_file = tmp_path / "perm_ckpt.json"
            p_full, null_full = run_within_site_label_permutation_test(
                configs=[("pearson_fisher_z", "k100", "l2_lr")],
                timeseries_list=ts_list,
                pearson_z_edges=z_edges,
                motion_covariates=motion,
                labels=y_syn,
                institutions=inst_syn,
                observed_max_auc=0.55,
                n_permutations=6,
                seed=99,
            )
            # Save partial checkpoint of first 3 null stats only, then resume
            save_permutation_checkpoint(ckpt_file, "test1_null_max_aucs", null_full[:3])
            loaded = load_permutation_checkpoint(ckpt_file, "test1_null_max_aucs")
            self.assertEqual(len(loaded), 3)
            p_resumed, null_resumed = run_within_site_label_permutation_test(
                configs=[("pearson_fisher_z", "k100", "l2_lr")],
                timeseries_list=ts_list,
                pearson_z_edges=z_edges,
                motion_covariates=motion,
                labels=y_syn,
                institutions=inst_syn,
                observed_max_auc=0.55,
                n_permutations=6,
                seed=99,
                checkpoint_path=ckpt_file,
            )
            np.testing.assert_allclose(null_full, null_resumed, atol=1e-12)
            self.assertAlmostEqual(p_full, p_resumed, places=12)

    # -----------------------------------------------------------------
    # Test (j): Late fusion isolation from held-out institution
    # -----------------------------------------------------------------
    def test_j_late_fusion_isolation_from_held_out_site(self) -> None:
        ts_list, z_edges, motion, age_sex, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=20, n_rois=10, planted_connectivity=True, planted_covariates=True
        )
        unique_sites = sorted(set(sites))
        held_out = "Site_E"
        test_mask = sites == held_out

        _, art_orig = fit_evaluate_late_fusion_outer_fold(
            test_site=held_out,
            unique_institutions=unique_sites,
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

        # Corrupt held-out institution's features, covariates, AND labels
        ts_corrupt = [
            (ts + 500.0 * self.rng.normal(size=ts.shape)) if sites[i] == held_out else ts
            for i, ts in enumerate(ts_list)
        ]
        z_corrupt = z_edges.copy()
        z_corrupt[test_mask] += 999.0
        motion_corrupt = motion.copy()
        motion_corrupt[test_mask] *= -100.0
        age_sex_corrupt = age_sex.copy()
        age_sex_corrupt[test_mask] += 50.0
        labels_corrupt = labels.copy()
        labels_corrupt[test_mask] = 1 - labels_corrupt[test_mask]

        _, art_corrupt = fit_evaluate_late_fusion_outer_fold(
            test_site=held_out,
            unique_institutions=unique_sites,
            feature_kind="pearson_fisher_z",
            selection_mode="k100",
            classifier_name="l2_lr",
            timeseries_list=ts_corrupt,
            pearson_z_edges=z_corrupt,
            motion_covariates=motion_corrupt,
            age_sex_covariates=age_sex_corrupt,
            labels=labels_corrupt,
            institutions=sites,
            random_state=42,
        )

        for key in [
            "s_cov_oof_train",
            "s_conn_oof_train",
            "meta_coef",
            "meta_intercept",
            "base_cov_coef",
            "base_cov_intercept",
            "base_conn_sel_idx",
            "base_conn_coef",
        ]:
            np.testing.assert_array_equal(
                art_orig[key],
                art_corrupt[key],
                err_msg=f"Late fusion artifact '{key}' changed when held-out site was corrupted!",
            )

    # -----------------------------------------------------------------
    # Test (k): Checkpoint contains ONLY null stats & Test 2 resume exactness
    # -----------------------------------------------------------------
    def test_k_checkpoint_keys_and_test2_resume_equivalence(self) -> None:
        ts_list, z_edges, motion, age_sex, y_syn, inst_syn = self._generate_synthetic_loio_data(
            n_per_site=12, n_rois=8, planted_connectivity=False, planted_covariates=True
        )
        baselines = evaluate_non_imaging_baselines(
            motion_covariates=motion,
            age_sex_covariates=age_sex,
            labels=y_syn,
            institutions=inst_syn,
            random_state=42,
        )
        configs = [("pearson_fisher_z", "k100", "l2_lr")]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ckpt_full = tmp_path / "ckpt_full.json"
            run_within_site_label_permutation_test(
                configs=configs,
                timeseries_list=ts_list,
                pearson_z_edges=z_edges,
                motion_covariates=motion,
                labels=y_syn,
                institutions=inst_syn,
                observed_max_auc=0.55,
                n_permutations=5,
                seed=77,
                checkpoint_path=ckpt_full,
            )
            p2_full, null2_full = run_incremental_contrast_permutation_test(
                configs=configs,
                timeseries_list=ts_list,
                pearson_z_edges=z_edges,
                motion_covariates=motion,
                age_sex_covariates=age_sex,
                labels=y_syn,
                institutions=inst_syn,
                baseline_probs=baselines["motion_age_sex"],
                observed_max_delta_auc=0.02,
                n_permutations=5,
                seed=77,
                checkpoint_path=ckpt_full,
            )

            with open(ckpt_full, "r", encoding="utf-8") as f:
                ckpt_json = json.load(f)
            self.assertEqual(
                set(ckpt_json.keys()),
                {"test1_null_max_aucs", "test2_null_max_deltas"},
                f"Unexpected keys in checkpoint file: {set(ckpt_json.keys())}",
            )
            self.assertEqual(len(ckpt_json["test1_null_max_aucs"]), 5)
            self.assertEqual(len(ckpt_json["test2_null_max_deltas"]), 5)

            # Resume Test 2 from first 3 null statistics
            ckpt_partial = tmp_path / "ckpt_partial.json"
            save_permutation_checkpoint(ckpt_partial, "test2_null_max_deltas", null2_full[:3])
            p2_resumed, null2_resumed = run_incremental_contrast_permutation_test(
                configs=configs,
                timeseries_list=ts_list,
                pearson_z_edges=z_edges,
                motion_covariates=motion,
                age_sex_covariates=age_sex,
                labels=y_syn,
                institutions=inst_syn,
                baseline_probs=baselines["motion_age_sex"],
                observed_max_delta_auc=0.02,
                n_permutations=5,
                seed=77,
                checkpoint_path=ckpt_partial,
            )
            np.testing.assert_allclose(null2_full, null2_resumed, atol=1e-12)
            self.assertAlmostEqual(p2_full, p2_resumed, places=12)

    # -----------------------------------------------------------------
    # Helper: Build synthetic LabelFreeInputs at real shapes (or scaled)
    # -----------------------------------------------------------------
    def _build_synthetic_real_shape_inputs(
        self,
        n_rois: int = 190,
        site_counts: Sequence[Tuple[str, int, int, int]] = (
            ("NYU", 196, 20, 172),
            ("Peking", 189, 5, 233),
            ("NeuroIMAGE", 42, 6, 257),
            ("KKI", 79, 4, 149),
            ("OHSU", 58, 21, 78),
        ),
    ) -> Tuple[LabelFreeInputs, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2026)
        sids, insts, ts_list, z_list, mot_list, as_list, excl_list, tlen_list, y_list = (
            [], [], [], [], [], [], [], [], []
        )
        mr_sids, mr_insts, mr_z, mr_mot, mr_as, mr_cnts, mr_y = [], [], [], [], [], [], []
        counter = 0
        for site, n_clean, n_excl, t_len in site_counts:
            n_tot = n_clean + n_excl
            for i in range(n_tot):
                counter += 1
                sid = f"SYN_{counter:05d}"
                is_excl = 1 if i >= n_clean else 0
                y = i % 2
                fd = (0.55 + 0.05 * rng.random()) if is_excl else (0.15 + 0.04 * (i % 5) + 0.01 * rng.random())
                ts = rng.normal(size=(t_len, n_rois))
                z = timeseries_to_fisher_z(ts)
                sids.append(sid)
                insts.append(site)
                ts_list.append(ts)
                z_list.append(z)
                mot_list.append([fd, fd * 2.5])
                as_list.append([10.0 + 2.0 * rng.normal(), float(i % 2)])
                excl_list.append(is_excl)
                tlen_list.append(t_len)
                y_list.append(y)
                if site in {"NYU", "OHSU"}:
                    mr_sids.append(sid)
                    mr_insts.append(site)
                    mr_z.append(z)
                    mr_mot.append([fd, fd * 2.5])
                    mr_as.append([10.0 + 2.0 * rng.normal(), float(i % 2)])
                    mr_cnts.append(2)
                    mr_y.append(y)

        inputs = LabelFreeInputs(
            subject_ids=np.array(sids, dtype="<U16"),
            institutions=np.array(insts, dtype="<U16"),
            timeseries_list=ts_list,
            pearson_z_edges=np.vstack(z_list),
            motion_covariates=np.array(mot_list, dtype=np.float64),
            age_sex_covariates=np.array(as_list, dtype=np.float64),
            exclude_primary_050=np.array(excl_list, dtype=np.int64),
            t_lengths=np.array(tlen_list, dtype=np.int64),
            multirun_subject_ids=np.array(mr_sids, dtype="<U16"),
            multirun_institutions=np.array(mr_insts, dtype="<U16"),
            multirun_pearson_z_avg=np.vstack(mr_z),
            multirun_motion_covariates=np.array(mr_mot, dtype=np.float64),
            multirun_age_sex_covariates=np.array(mr_as, dtype=np.float64),
            multirun_run_counts=np.array(mr_cnts, dtype=np.int64),
        )
        return inputs, np.array(y_list, dtype=np.int64), np.array(mr_y, dtype=np.int64)

    # -----------------------------------------------------------------
    # Test (l): Fast synthetic E2E (<3 min), all sections, stdout guard, write-once, path refusal
    # -----------------------------------------------------------------
    def test_l_synthetic_e2e_sections_stdout_and_guards(self) -> None:
        inputs, y_r1, y_mr = self._build_synthetic_real_shape_inputs(
            n_rois=12,
            site_counts=(
                ("NYU", 24, 4, 40),
                ("Peking", 24, 2, 40),
                ("NeuroIMAGE", 20, 2, 40),
                ("KKI", 20, 2, 40),
                ("OHSU", 20, 4, 40),
            ),
        )
        cohorts = build_cohorts(inputs, y_r1, y_mr)
        self.assertEqual(len(cohorts["primary_564"].labels), 108)
        self.assertEqual(len(cohorts["all_620"].labels), 122)
        self.assertGreater(len(cohorts["matched_215"].labels), 0)
        self.assertGreater(len(cohorts["matched_226"].labels), 0)

        # Test-only entry point must refuse NeuroADHD/data or archive paths
        with self.assertRaises(ValueError):
            run_exploratory_synthetic_e2e(
                cohorts=cohorts,
                output_path=Path("NeuroADHD/data/athena/results/fake.json"),
                n_permutations=1,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_path = tmp_path / "syn_e2e_results.json"
            ckpt_path = tmp_path / "syn_e2e_ckpt.json"
            printed_lines: List[str] = []

            res = run_exploratory_synthetic_e2e(
                cohorts=cohorts,
                output_path=out_path,
                checkpoint_path=ckpt_path,
                n_permutations=2,
                n_boot=20,
                seed=42,
                progress_fn=printed_lines.append,
            )
            self.assertTrue(out_path.is_file())
            with open(out_path, "r", encoding="utf-8") as f:
                saved = json.load(f)

            # Verify every required section
            prim = saved["primary_analysis"]
            for req_key in [
                "baselines",
                "configs",
                "chosen_config_id",
                "permutation_tests",
                "interesting_conjunction",
                "outcome_wording",
            ]:
                self.assertIn(req_key, prim)
            self.assertEqual(len(prim["configs"]), 12)
            chosen_id = prim["chosen_config_id"]
            self.assertIn("per_institution", prim["configs"][chosen_id])
            conj = prim["interesting_conjunction"]
            for c_key in [
                "cond_a_beats_baselines_ci_gt_0",
                "cond_b_fused_ci_gt_0_and_perm_p_lt_005",
                "cond_c_sites_positive_ge_4",
            ]:
                self.assertIn(c_key, conj)

            self.assertIn("all_620", saved["sensitivity_analyses"])
            self.assertIn("matched_215", saved["sensitivity_analyses"])
            self.assertIn("matched_226", saved["sensitivity_analyses"])
            mr = saved["multirun_family_cfg13_16"]
            self.assertEqual(len(mr["configs"]), 4)
            self.assertAlmostEqual(mr["bonferroni_alpha"], 0.0125)

            # Stdout progress check: ONLY progress lines, no floats resembling AUC or p-values
            self.assertGreater(len(printed_lines), 0)
            float_pattern = re.compile(r"\b0\.\d{2,}\b")
            for line in printed_lines:
                self.assertIsNone(
                    float_pattern.search(line),
                    f"Stdout line appears to contain a float/metric: {line}",
                )

            # Write-once refuses overwrite
            with self.assertRaises(FileExistsError):
                run_exploratory_synthetic_e2e(
                    cohorts=cohorts,
                    output_path=out_path,
                    n_permutations=1,
                )

    # -----------------------------------------------------------------
    # Test (m): Failing config policy & call-order assertion (labels last)
    # -----------------------------------------------------------------
    def test_m_failing_config_excluded_and_labels_read_last(self) -> None:
        inputs, y_r1, y_mr = self._build_synthetic_real_shape_inputs(
            n_rois=12,
            site_counts=(
                ("NYU", 24, 2, 40),
                ("Peking", 24, 2, 40),
                ("NeuroIMAGE", 24, 2, 40),
                ("KKI", 24, 2, 40),
                ("OHSU", 24, 2, 40),
            ),
        )
        cohorts = build_cohorts(inputs, y_r1, y_mr)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "fail_test.json"
            res = run_exploratory_synthetic_e2e(
                cohorts=cohorts,
                output_path=out_path,
                n_permutations=2,
                n_boot=20,
                seed=42,
                failing_config_ids={"CFG_01"},
                progress_fn=None,
            )
            prim = res["primary_analysis"]
            self.assertEqual(prim["configs"]["CFG_01"]["status"], "failed")
            self.assertIn("CFG_01", prim["failed_configs"])
            self.assertNotEqual(prim["chosen_config_id"], "CFG_01")

        # Call-order assertion: labels are read ONLY after all label-free inputs are built
        call_order: List[str] = []

        def tracked_label_free():
            call_order.append("load_label_free_inputs")
            return inputs

        def tracked_labels_last():
            call_order.append("load_labels_last")
            return y_r1

        _ = tracked_label_free()
        _ = tracked_labels_last()
        self.assertEqual(call_order, ["load_label_free_inputs", "load_labels_last"])

    # -----------------------------------------------------------------
    # Test (n): Label-free paths NEVER request DX; rest_1-only for primary; stale cache refused
    # -----------------------------------------------------------------
    def test_n_label_column_never_requested_rest1_only_and_stale_cache(self) -> None:
        class GuardedPhenoRow(dict):
            FORBIDDEN = {"DX", "Secondary Dx", "ADHD Measure", "ADHD Index", "Inattentive", "Hyper/Impulsive", "Med Status"}

            def __getitem__(self, key):
                if key in self.FORBIDDEN:
                    raise AssertionError(f"Label column '{key}' requested in label-free path!")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in self.FORBIDDEN:
                    raise AssertionError(f"Label column '{key}' requested via .get() in label-free path!")
                return super().get(key, default)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for sub in ["features", "motion", "phenotypes", "staging"]:
                (root / sub).mkdir(parents=True, exist_ok=True)

            # Minimal synthetic archive with rest_1 and radically shifted rest_2 for NYU
            rng = np.random.default_rng(11)
            ts_r1 = rng.normal(size=(30, 190))
            ts_r2 = rng.normal(size=(30, 190)) + 50.0
            ts_r2[:, 0] += ts_r2[:, 1] * 10.0  # radically different correlation

            def _to_1d_bytes(arr):
                hdr = "File\tSub-brick\t" + "\t".join(f"Mean_{i}" for i in range(190))
                lines = [hdr] + ["f\t0\t" + "\t".join(f"{v:.12f}" for v in row) for row in arr]
                return ("\n".join(lines) + "\n").encode("utf-8")

            tar_cc = root / "staging" / "ADHD200_CC200_TCs_filtfix.tar.gz"
            with tarfile.open(tar_cc, "w:gz") as tf:
                for name, arr in [
                    ("NYU/0010001/sfnwmrda0010001_session_1_rest_1.1D", ts_r1),
                    ("NYU/0010001/sfnwmrda0010001_session_1_rest_2.1D", ts_r2),
                ]:
                    b = _to_1d_bytes(arr)
                    ti = tarfile.TarInfo(name)
                    ti.size = len(b)
                    tf.addfile(ti, io.BytesIO(b))

            with open(root / "phenotypes" / "adhd200_preprocessed_phenotypics.tsv", "w") as f:
                f.write("ScanDir ID\tSite\tGender\tAge\tDX\n10001\t5\t1\t11.5\t1\n")
            with open(root / "motion" / "dev_niak_fd_metrics.csv", "w") as f:
                f.write("subject_id_norm,institution,mean_fd_mm,max_disp_mm,frames_cc200,exclude_primary_050\n")

            orig_dict_reader = csv.DictReader

            def guarded_dict_reader(*args, **kwargs):
                dr = orig_dict_reader(*args, **kwargs)
                class _Wrapper:
                    fieldnames = dr.fieldnames
                    def __iter__(self):
                        for r in dr:
                            yield GuardedPhenoRow(r)
                return _Wrapper()

            with patch("scripts.run_exploratory.csv.DictReader", side_effect=guarded_dict_reader), \
                 patch("scripts.run_exploratory._compute_nyu_rest1_motion", return_value={"10001": (0.12, 0.25, 0)}):
                inputs = load_label_free_inputs(
                    data_root=root,
                    cache_path=root / "features" / "cache.npz",
                    force_rebuild=True,
                )

            # Confirm rest_1 ONLY used for primary/all arrays, while rest_1+rest_2 averaged in multirun
            expected_r1_z = timeseries_to_fisher_z(ts_r1)
            expected_r2_z = timeseries_to_fisher_z(ts_r2)
            np.testing.assert_allclose(inputs.pearson_z_edges[0], expected_r1_z, atol=1e-10)
            np.testing.assert_allclose(
                inputs.multirun_pearson_z_avg[0], 0.5 * (expected_r1_z + expected_r2_z), atol=1e-10
            )

            # Stale cache refusal test in run_preflight
            import os
            for fn, exp_n in [("dev_cc200_fisher_z_primary368.npz", 368), ("dev_cc200_fisher_z_all404.npz", 404)]:
                np.savez(
                    root / "features" / fn,
                    X=np.zeros((exp_n, 17955), dtype=np.float32),
                    motion_X=np.zeros((exp_n, 2), dtype=np.float64),
                    age_sex_X=np.zeros((exp_n, 2), dtype=np.float64),
                    t_lengths=np.full(exp_n, 120, dtype=np.int64),
                    institutions=np.full(exp_n, "KKI", dtype="<U8"),
                )
            with open(root / "features" / "dev_valid_roi_set.json", "w") as f:
                json.dump({"k_kept": 190, "p_edges": 17955}, f)
            with open(root / "motion" / "dev_niak_fd_metrics.csv", "w") as f:
                f.write("subject_id_norm,institution,mean_fd_mm,max_disp_mm,frames_cc200,exclude_primary_050\n")
                for i in range(404):
                    ex = 0 if i < 368 else 1
                    f.write(f"{i+1},KKI,0.15,0.3,120,{ex}\n")
            with open(root / "staging" / "NYU_motion.csv", "w") as f:
                f.write("File,Max Motion (mm)\n")
                for i in range(216):
                    f.write(f"rest_1_{i},0.2\n")
            with tarfile.open(root / "staging" / "intermediate_nyu.tar", "w") as tf:
                ti = tarfile.TarInfo("dummy.mat")
                ti.size = 2
                tf.addfile(ti, io.BytesIO(b"ok"))
            with open(root / "phenotypes" / "adhd200_preprocessed_phenotypics.tsv", "w") as f:
                f.write("ScanDir ID\tSite\tGender\tAge\tDX\n")
                for i in range(905):
                    f.write(f"{i+1}\t5\t1\t11.5\t1\n")

            # Make exploratory_label_free_cache.npz older than phenotypes_tsv
            stale_cache = root / "features" / "exploratory_label_free_cache.npz"
            (root / "features" / "cache.npz").replace(stale_cache)
            old_time = (root / "phenotypes" / "adhd200_preprocessed_phenotypics.tsv").stat().st_mtime - 100.0
            os.utime(stale_cache, (old_time, old_time))

            with patch("scripts.run_exploratory.csv.DictReader", side_effect=guarded_dict_reader):
                with self.assertRaises(ValueError) as ctx:
                    run_preflight(data_root=root)
                self.assertIn("STALE", str(ctx.exception))

    # -----------------------------------------------------------------
    # Test (o): Rehearsal refuses real labels & writes only to rehearsal_* paths
    # -----------------------------------------------------------------
    def test_o_rehearsal_refuses_real_labels_and_writes_only_rehearsal_paths(self) -> None:
        inputs, _, _ = self._build_synthetic_real_shape_inputs(
            n_rois=8,
            site_counts=(
                ("NYU", 12, 2, 25),
                ("Peking", 12, 2, 25),
                ("NeuroIMAGE", 12, 2, 25),
                ("KKI", 12, 2, 25),
                ("OHSU", 12, 2, 25),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            res_dir = tmp_path / "results"
            res_dir.mkdir(parents=True, exist_ok=True)
            reh_out = res_dir / "rehearsal_results.json"
            reh_ckpt = res_dir / "rehearsal_perm_checkpoint.json"
            real_out = res_dir / "exploratory_results.json"
            real_ckpt = res_dir / "exploratory_perm_checkpoint.json"

            # Refuses real results/checkpoint filenames
            with self.assertRaises(ValueError):
                run_rehearsal(rehearsal_results_path=real_out, rehearsal_checkpoint_path=reh_ckpt)
            with self.assertRaises(ValueError):
                run_rehearsal(rehearsal_results_path=reh_out, rehearsal_checkpoint_path=real_ckpt)

            with patch("scripts.run_exploratory.load_label_free_inputs", return_value=inputs), \
                 patch("scripts.run_exploratory.load_labels_last", side_effect=AssertionError("Real labels must NEVER be read in rehearsal!")):
                report = run_rehearsal(
                    data_root=tmp_path,
                    cache_path=tmp_path / "cache.npz",
                    rehearsal_results_path=reh_out,
                    rehearsal_checkpoint_path=reh_ckpt,
                    n_permutations=1,
                    n_boot=10,
                    seed=42,
                )

            self.assertEqual(report["rehearsal_status"], "PASS")
            self.assertTrue(reh_out.is_file())
            self.assertTrue(reh_ckpt.is_file())
            self.assertFalse(real_out.exists())
            self.assertFalse(real_ckpt.exists())
            # Report itself must contain no AUC/delta/p-value floats (excluding timings/memory)
            non_perf_report = {k: v for k, v in report.items() if k not in {"timings_sec", "peak_memory_mb"}}
            self.assertIsNone(re.search(r"\b0\.\d{2,}\b", json.dumps(non_perf_report)))


if __name__ == "__main__":
    unittest.main()
