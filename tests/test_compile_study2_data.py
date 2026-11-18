"""Tests for NeuroADHD_ModelDashboard/backend/compile_study2_data.py."""

import json
import unittest
from pathlib import Path

from NeuroADHD_ModelDashboard.backend.compile_study2_data import (
    DOCS_STUDY2_DIR,
    FRONTEND_DATA_DIR,
    compile_study2_data,
)


class TestCompileStudy2Data(unittest.TestCase):
    def test_compiled_json_matches_tracked_artifact(self) -> None:
        fresh = compile_study2_data(DOCS_STUDY2_DIR)
        disk_path = FRONTEND_DATA_DIR / "study2Data.json"
        self.assertTrue(disk_path.is_file(), "study2Data.json must exist")
        on_disk = json.loads(disk_path.read_text(encoding="utf-8"))
        self.assertEqual(fresh, on_disk)

    def test_key_fields_and_unrounded_bounds(self) -> None:
        data = compile_study2_data(DOCS_STUDY2_DIR)
        self.assertEqual(data["development_primary"]["n_scans"], 368)
        self.assertFalse(data["development_primary"]["meets_section_8_1"])
        self.assertEqual(data["development_sensitivity"]["n_scans"], 404)
        self.assertFalse(data["development_sensitivity"]["meets_section_8_1"])
        self.assertEqual(data["sealed_nyu_primary"]["n_scans"], 196)
        self.assertEqual(data["sealed_nyu_primary"]["outcome_zone"], "inconclusive")
        self.assertEqual(data["sealed_nyu_sensitivity"]["n_scans"], 216)
        self.assertEqual(data["sealed_nyu_sensitivity"]["outcome_zone"], "inconclusive")
        self.assertIn("0.499996", data["sealed_nyu_sensitivity"]["fmri_ci_95_exact_display"])
        self.assertEqual(data["prior_exposure"]["dev_overlap_subjects"], 126)
        self.assertEqual(data["prior_exposure"]["nyu_overlap_subjects"], 6)
        self.assertEqual(data["development_primary_attrition"]["excluded_tdc_pct_display"], "5.0%")
        self.assertEqual(data["development_primary_attrition"]["excluded_adhd_pct_display"], "14.8%")
        self.assertEqual(data["development_primary_attrition"]["fisher_p_display"], "0.0011")
        self.assertEqual(data["sealed_nyu_attrition"]["excluded_tdc_pct_display"], "7.1%")
        self.assertEqual(data["sealed_nyu_attrition"]["excluded_adhd_pct_display"], "11.0%")
        self.assertEqual(data["sealed_nyu_attrition"]["fisher_p_display"], "0.3568")


if __name__ == "__main__":
    unittest.main()
