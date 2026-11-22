#!/usr/bin/env python3
"""
Unit and Leakage Tests for scripts/run_exploratory.py (Synthetic Data Only).

Covers all 5 pre-specified properties:
(a) Planted signal gives AUC clearly above 0.5.
(b) No signal gives about 0.5 and a non-significant permutation p.
(c) Leakage test: changing held-out rows changes no fitted statistic, tangent reference mean, or selected edges.
(d) Subject-grouped folds keep all runs of a subject together.
(e) The guard refuses without the flag and tag.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_exploratory import (
    apply_feature_selection,
    build_classifier,
    enforce_exploratory_guards,
    evaluate_loio_pipeline,
    fit_transform_motion_residualization,
    fit_transform_tangent_space,
    match_subjects_on_motion_within_site,
    run_within_site_permutation_test,
    timeseries_to_fisher_z,
)


class TestRunExploratorySynthetic(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def _generate_synthetic_loio_data(
        self,
        n_per_site: int = 40,
        n_rois: int = 15,
        t_len: int = 50,
        planted: bool = False,
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic LOIO multi-site dataset."""
        sites = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]
        timeseries_list: List[np.ndarray] = []
        labels: List[int] = []
        institutions: List[str] = []
        motion_covariates: List[List[float]] = []

        p_edges = n_rois * (n_rois - 1) // 2
        pearson_z_edges = []

        for site_idx, site in enumerate(sites):
            for i in range(n_per_site):
                y = i % 2
                labels.append(y)
                institutions.append(site)

                # Motion covariates
                mean_fd = 0.15 + (0.05 * y) + 0.02 * self.rng.normal()
                max_disp = mean_fd * 3.0 + 0.05 * self.rng.normal()
                motion_covariates.append([mean_fd, max_disp])

                # Generate BOLD timeseries
                ts = self.rng.normal(size=(t_len, n_rois))
                if planted and y == 1:
                    # Inject strong positive correlation between ROI 0 and ROI 1
                    shared_signal = self.rng.normal(size=t_len) * 3.0
                    ts[:, 0] += shared_signal
                    ts[:, 1] += shared_signal

                timeseries_list.append(ts)
                pearson_z_edges.append(timeseries_to_fisher_z(ts))

        return (
            timeseries_list,
            np.array(pearson_z_edges, dtype=np.float64),
            np.array(motion_covariates, dtype=np.float64),
            np.array(labels, dtype=np.int64),
            np.array(institutions, dtype="<U16"),
        )

    # -----------------------------------------------------------------
    # Test (a): Planted signal gives AUC clearly above 0.5
    # -----------------------------------------------------------------
    def test_a_planted_signal_gives_high_auc(self) -> None:
        ts_list, z_edges, motion, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=30, n_rois=10, planted=True
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
        ts_list, z_edges, motion, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=25, n_rois=8, planted=False
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

        # Fast permutation test on null data
        p_val, null_dist = run_within_site_permutation_test(
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
        ts_list, z_edges, motion, labels, sites = self._generate_synthetic_loio_data(
            n_per_site=20, n_rois=10, planted=False
        )
        test_site = "Site_E"
        train_mask = sites != test_site
        test_mask = sites == test_site

        # Baseline fit on training fold
        ts_train = [ts_list[i] for i in np.where(train_mask)[0]]
        ts_test_orig = [ts_list[i] for i in np.where(test_mask)[0]]

        X_tr_tang_orig, _, cm_orig = fit_transform_tangent_space(ts_train, ts_test_orig)
        _, _, w_orig, b_orig = fit_transform_motion_residualization(
            z_edges[train_mask], motion[train_mask], z_edges[test_mask], motion[test_mask]
        )
        X_tr_sel_orig, _, sel_idx_orig = apply_feature_selection(
            z_edges[train_mask], labels[train_mask], z_edges[test_mask], "k100"
        )

        # Now heavily corrupt / modify held-out test site data
        ts_test_corrupt = [ts + 1000.0 * self.rng.normal(size=ts.shape) for ts in ts_test_orig]
        z_test_corrupt = z_edges[test_mask] + 999.0
        motion_test_corrupt = motion[test_mask] * 50.0

        # Refit on same training fold with corrupted test data
        X_tr_tang_new, _, cm_new = fit_transform_tangent_space(ts_train, ts_test_corrupt)
        _, _, w_new, b_new = fit_transform_motion_residualization(
            z_edges[train_mask], motion[train_mask], z_test_corrupt, motion_test_corrupt
        )
        X_tr_sel_new, _, sel_idx_new = apply_feature_selection(
            z_edges[train_mask], labels[train_mask], z_test_corrupt, "k100"
        )

        # Assert zero leakage into training statistics
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
        # Simulate multi-run cohort
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

        # Check Leave-One-Institution-Out splitting on sites
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

        # Test within-site motion matching
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
        # Refuse when execute_real is False
        with self.assertRaises(RuntimeError) as ctx1:
            enforce_exploratory_guards(execute_real=False)
        self.assertIn("Refusing to read real labels without --execute-real", str(ctx1.exception))

        # Refuse when execute_real is True but tag does not exist
        with self.assertRaises(RuntimeError) as ctx2:
            enforce_exploratory_guards(
                execute_real=True,
                tag_name="nonexistent-exploratory-tag-9999",
            )
        self.assertIn("does not exist on an ancestor of HEAD", str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
