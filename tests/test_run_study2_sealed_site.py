import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_study2_dev_analysis import build_locked_fmri_pipeline
from scripts.run_study2_sealed_site import (
    DEFAULT_SEALED_RESULTS_JSON,
    REQUIRED_FREEZE_TAG,
    classify_section_8_2_zone,
    enforce_sealed_guards,
    evaluate_sealed_regime,
    sealed_timeseries_to_fisher_z_edges,
)


def _make_dev_and_sealed(seed: int, signal_strength: float, n_dev: int = 160, n_sealed: int = 120, n_feat: int = 150):
    rng = np.random.default_rng(seed)
    y_dev = np.array([0] * (n_dev // 2) + [1] * (n_dev // 2), dtype=np.int64)
    y_sealed = np.array([0] * (n_sealed // 2) + [1] * (n_sealed // 2), dtype=np.int64)

    X_dev = rng.normal(0.0, 1.0, size=(n_dev, n_feat))
    X_sealed = rng.normal(0.0, 1.0, size=(n_sealed, n_feat))
    if signal_strength > 0.0:
        X_dev[y_dev == 1, :25] += signal_strength
        X_sealed[y_sealed == 1, :25] += signal_strength

    m_dev = rng.normal(0.2, 0.05, size=(n_dev, 2))
    m_sealed = rng.normal(0.2, 0.05, size=(n_sealed, 2))
    as_dev = np.column_stack([rng.normal(11.0, 2.0, n_dev), rng.integers(0, 2, n_dev).astype(float)])
    as_sealed = np.column_stack([rng.normal(11.0, 2.0, n_sealed), rng.integers(0, 2, n_sealed).astype(float)])
    return X_dev, m_dev, as_dev, y_dev, X_sealed, m_sealed, as_sealed, y_sealed


class TestRunStudy2SealedSiteSynthetic(unittest.TestCase):
    def test_a_planted_signal_site_gives_confirmed_zone(self):
        X_d, m_d, as_d, y_d, X_s, m_s, as_s, y_s = _make_dev_and_sealed(
            seed=42, signal_strength=1.0
        )
        res, probs = evaluate_sealed_regime(
            X_d, m_d, as_d, y_d, X_s, m_s, as_s, y_s, k_best=100, n_boot=300, seed=42
        )
        self.assertEqual(len(probs), len(y_s))
        self.assertGreater(res["fmri"]["ci_lo"], 0.50)
        self.assertEqual(res["fmri"]["outcome_zone"], "confirmed")
        self.assertIn("auc", res["descriptive_motion_baseline"])
        self.assertIn("auc", res["descriptive_age_sex_baseline"])

    def test_b_null_site_gives_inconclusive_or_no_evidence(self):
        X_d, m_d, as_d, y_d, X_s, m_s, as_s, y_s = _make_dev_and_sealed(
            seed=101, signal_strength=0.0
        )
        res, _ = evaluate_sealed_regime(
            X_d, m_d, as_d, y_d, X_s, m_s, as_s, y_s, k_best=100, n_boot=300, seed=42
        )
        self.assertIn(res["fmri"]["outcome_zone"], ("inconclusive", "no_evidence"))
        # Direct zone boundary unit checks
        self.assertEqual(classify_section_8_2_zone(0.65, 0.51), "confirmed")
        self.assertEqual(classify_section_8_2_zone(0.56, 0.49), "inconclusive")
        self.assertEqual(classify_section_8_2_zone(0.50, 0.42), "no_evidence")
        self.assertEqual(classify_section_8_2_zone(0.45, 0.37), "no_evidence")

    def test_c_sealed_labels_flipped_leave_predictions_bit_for_bit_identical(self):
        X_d, m_d, as_d, y_d, X_s, m_s, as_s, y_s = _make_dev_and_sealed(
            seed=7, signal_strength=0.8
        )
        _, probs_orig = evaluate_sealed_regime(
            X_d, m_d, as_d, y_d, X_s, m_s, as_s, y_s, k_best=100, n_boot=50, seed=42
        )
        y_s_flipped = 1 - y_s
        _, probs_flipped = evaluate_sealed_regime(
            X_d, m_d, as_d, y_d, X_s, m_s, as_s, y_s_flipped, k_best=100, n_boot=50, seed=42
        )
        np.testing.assert_array_equal(probs_orig, probs_flipped)

    def test_d_zero_fill_rule_for_missing_or_constant_roi_leaves_model_unchanged(self):
        rng = np.random.default_rng(19)
        dev_rois = ["Mean_1", "Mean_3", "Mean_4", "Mean_6"]  # K=4 -> P=6 edges
        t_len = 50
        # Subject 1: Mean_4 (index 2) is constant (std == 0)
        ts_const = rng.normal(size=(t_len, 4))
        ts_const[:, 2] = 7.0
        edges_const, n_bad_const = sealed_timeseries_to_fisher_z_edges(
            dev_rois, ts_const, dev_rois
        )
        self.assertEqual(n_bad_const, 1)
        self.assertEqual(edges_const.shape, (6,))
        # Upper triangle pairs for K=4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # Indices touching ROI 2 are indices 1, 3, 5 -> must all be strictly 0.0
        self.assertEqual(edges_const[1], 0.0)
        self.assertEqual(edges_const[3], 0.0)
        self.assertEqual(edges_const[5], 0.0)
        self.assertNotEqual(edges_const[0], 0.0)

        # Subject 2: Mean_6 is missing completely from subject's .1D header
        edges_miss, n_bad_miss = sealed_timeseries_to_fisher_z_edges(
            ["Mean_1", "Mean_3", "Mean_4"], rng.normal(size=(t_len, 3)), dev_rois
        )
        self.assertEqual(n_bad_miss, 1)
        # Edges touching ROI 3 are indices 2, 4, 5 -> must all be strictly 0.0
        self.assertEqual(edges_miss[2], 0.0)
        self.assertEqual(edges_miss[4], 0.0)
        self.assertEqual(edges_miss[5], 0.0)

        # Verify zero-filling a sealed subject's edges never alters the fitted dev model/selected features
        X_d, _, _, y_d, X_s, _, _, _ = _make_dev_and_sealed(seed=33, signal_strength=0.9, n_feat=6)
        pipe_clean = build_locked_fmri_pipeline(k_best=4).fit(X_d, y_d)
        support_before = pipe_clean.named_steps["selector"].get_support().copy()
        scores_before = pipe_clean.named_steps["selector"].scores_.copy()
        feat_imp_before = pipe_clean.named_steps["clf"].feature_importances_.copy()

        X_s_zeroed = X_s.copy()
        X_s_zeroed[0] = edges_const
        probs_scored = pipe_clean.predict_proba(X_s_zeroed)[:, 1]
        self.assertTrue(np.all(np.isfinite(probs_scored)))
        np.testing.assert_array_equal(pipe_clean.named_steps["selector"].get_support(), support_before)
        np.testing.assert_array_equal(pipe_clean.named_steps["selector"].scores_, scores_before)
        np.testing.assert_array_equal(pipe_clean.named_steps["clf"].feature_importances_, feat_imp_before)

    def test_e_refuses_without_flag_and_without_freeze_tag(self):
        self.assertIn("athena", DEFAULT_SEALED_RESULTS_JSON.parts)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out_json = repo / "sealed_out.json"
            subprocess.run(["git", "init"], cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "a.txt").write_text("init\n", encoding="utf-8")
            subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, stdout=subprocess.DEVNULL, check=True)

            # 1) Fails without --execute-real-sealed flag
            with self.assertRaises(RuntimeError):
                enforce_sealed_guards(execute_real_sealed=False, out_json=out_json, repo_root=repo)

            # 2) Fails with flag=True when freeze tag does NOT exist
            with self.assertRaises(RuntimeError):
                enforce_sealed_guards(execute_real_sealed=True, out_json=out_json, repo_root=repo)

            # 3) Create annotated freeze tag -> guard now passes
            subprocess.run(
                ["git", "tag", "-a", REQUIRED_FREEZE_TAG, "-m", "freeze"],
                cwd=repo,
                check=True,
            )
            enforce_sealed_guards(execute_real_sealed=True, out_json=out_json, repo_root=repo)

            # 4) Fails again if flag=False even when freeze tag exists
            with self.assertRaises(RuntimeError):
                enforce_sealed_guards(execute_real_sealed=False, out_json=out_json, repo_root=repo)

    def test_f_second_invocation_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out_json = repo / "study2_sealed_results.json"
            subprocess.run(["git", "init"], cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "a.txt").write_text("init\n", encoding="utf-8")
            subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, stdout=subprocess.DEVNULL, check=True)
            subprocess.run(["git", "tag", "-a", REQUIRED_FREEZE_TAG, "-m", "freeze"], cwd=repo, check=True)

            # First check succeeds before file exists
            enforce_sealed_guards(execute_real_sealed=True, out_json=out_json, repo_root=repo)

            # Simulate completed first write
            out_json.write_text('{"locked": true}\n', encoding="utf-8")

            # Second invocation MUST refuse to overwrite
            with self.assertRaises(FileExistsError):
                enforce_sealed_guards(execute_real_sealed=True, out_json=out_json, repo_root=repo)


if __name__ == "__main__":
    unittest.main()
