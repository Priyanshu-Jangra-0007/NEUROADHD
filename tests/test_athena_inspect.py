"""Unit test for scripts/athena_inspect.py using purely synthetic mock data."""

import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.athena_inspect import inspect_athena_directory


class TestAthenaInspectSynthetic(unittest.TestCase):
    """Verifies that athena_inspect accurately parses structure, counts, and enforces zero-ID policy."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.data_dir = self.base_path / "mock_athena"
        self.pheno_dir = self.base_path / "mock_pheno"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pheno_dir.mkdir(parents=True, exist_ok=True)

        # Generate 4 synthetic subjects
        self.synthetic_ids = ["9000001", "9000002", "9000003", "9000004"]

        # Create synthetic .1D time-series and motion files
        for sid in self.synthetic_ids:
            # 120 rows (timepoints) x 200 cols (ROIs)
            ts_file = self.data_dir / f"{sid}_rest_1_rois_cc200.1D"
            ts_content = "\n".join([" ".join(["0.123"] * 200) for _ in range(120)]) + "\n"
            ts_file.write_text(ts_content, encoding="utf-8")

            # 120 rows x 6 cols (AFNI motion)
            mot_file = self.data_dir / f"{sid}_rest_1_motion.1D"
            mot_content = "\n".join([" ".join(["0.010"] * 6) for _ in range(120)]) + "\n"
            mot_file.write_text(mot_content, encoding="utf-8")

        # Create mock phenotype CSV
        pheno_csv = self.pheno_dir / "Mock_phenotypic.csv"
        pheno_lines = [
            "ScanDir ID,Site,DX",
            f"{self.synthetic_ids[0]},KKI,0",      # TDC
            f"{self.synthetic_ids[1]},KKI,1",      # ADHD
            f"{self.synthetic_ids[2]},OHSU,0",     # TDC
            f"{self.synthetic_ids[3]},OHSU,2",     # ADHD
        ]
        pheno_csv.write_text("\n".join(pheno_lines) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inspection_counts_and_privacy_guardrail(self):
        """Verify summary counts, shape detection, and strict absence of raw subject IDs in output."""
        output_buffer = io.StringIO()
        result = inspect_athena_directory(self.data_dir, self.pheno_dir, output_stream=output_buffer)

        # 1. Verification of execution status
        self.assertEqual(result.get("status"), "SUCCESS")

        # 2. File and subject counts
        self.assertEqual(result["total_files"], 8)
        self.assertEqual(result["unique_ts_subjects"], 4)
        self.assertEqual(result["unique_motion_subjects"], 4)
        self.assertEqual(result["paired_subjects"], 4)

        # 3. Shape detection
        self.assertIn("120x200", result["time_series_shapes"])
        self.assertEqual(result["time_series_shapes"]["120x200"], 4)

        # 4. Phenotypic diagnosis counts
        self.assertEqual(result["total_tdc"], 2)
        self.assertEqual(result["total_adhd"], 2)
        self.assertEqual(result["total_other"], 0)

        # 5. Strict Zero-ID Privacy Guardrail check
        report_text = output_buffer.getvalue()
        for sid in self.synthetic_ids:
            self.assertNotIn(
                sid,
                report_text,
                f"Privacy violation: Subject identifier found in inspection report output!"
            )

        # Ensure masked pattern is present in report
        self.assertIn("<id>_rest_1_rois_cc200.1d", report_text.lower())
        self.assertIn("120 timepoints  x   200 rois", report_text.lower())

    def test_site_level_motion_csv_support(self):
        """Verify that consolidated site-level motion CSVs (e.g. KKI_motion.csv) are properly parsed."""
        # Create a fresh subdirectory with only time series and a single site motion CSV
        site_data_dir = self.base_path / "mock_site_athena"
        site_data_dir.mkdir(parents=True, exist_ok=True)

        for sid in self.synthetic_ids:
            ts_file = site_data_dir / f"{sid}_rest_1_rois_cc200.1D"
            ts_content = "\n".join([" ".join(["0.123"] * 200) for _ in range(120)]) + "\n"
            ts_file.write_text(ts_content, encoding="utf-8")

        # Consolidated site motion CSV
        site_mot_csv = site_data_dir / "KKI_motion.csv"
        csv_lines = [
            "ScanDir ID,TR,roll,pitch,yaw,dS,dL,dP",
            f"{self.synthetic_ids[0]},0,0.01,0.02,0.01,0.05,0.03,0.02",
            f"{self.synthetic_ids[1]},0,0.01,0.02,0.01,0.05,0.03,0.02",
            f"{self.synthetic_ids[2]},0,0.01,0.02,0.01,0.05,0.03,0.02",
            f"{self.synthetic_ids[3]},0,0.01,0.02,0.01,0.05,0.03,0.02",
        ]
        site_mot_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

        output_buffer = io.StringIO()
        result = inspect_athena_directory(site_data_dir, self.pheno_dir, output_stream=output_buffer)

        self.assertEqual(result.get("status"), "SUCCESS")
        self.assertEqual(result["unique_ts_subjects"], 4)
        self.assertEqual(result["unique_motion_subjects"], 4)
        self.assertEqual(result["paired_subjects"], 4)

        report_text = output_buffer.getvalue()
        for sid in self.synthetic_ids:
            self.assertNotIn(sid, report_text)


if __name__ == "__main__":
    unittest.main()
