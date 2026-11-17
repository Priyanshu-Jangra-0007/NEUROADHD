"""Unit test for scripts/athena_verify_acquisition.py using synthetic mock data."""

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.athena_verify_acquisition import (
    compute_sha256,
    inspect_motion_header,
    run_verification,
    verify_tar_integrity,
)


class TestAthenaVerifyAcquisition(unittest.TestCase):
    """Test suite for Athena acquisition verification tooling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.staging = self.base / "staging"
        self.dest = self.base / "dest"
        self.pheno = self.base / "pheno"

        self.staging.mkdir(parents=True, exist_ok=True)
        self.dest.mkdir(parents=True, exist_ok=True)
        self.pheno.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sha256_and_tar_integrity(self):
        """Verify checksum calculation and tar integrity checking."""
        test_file = self.staging / "test_file.txt"
        test_file.write_text("Hello Athena Verification", encoding="utf-8")
        h = compute_sha256(test_file)
        self.assertEqual(len(h), 64)

        # Create a mock tar.gz
        tar_path = self.staging / "ADHD200_CC200_TCs_filtfix.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(test_file, arcname="sample.txt")

        ok, count, msg = verify_tar_integrity(tar_path)
        self.assertTrue(ok)
        self.assertEqual(count, 1)
        self.assertEqual(msg, "OK")

    def test_motion_header_inspection(self):
        """Verify parsing and validation of representative motion headers."""
        # Framewise motion header
        mot_path = self.staging / "framewise_motion.csv"
        mot_path.write_text("ScanDir ID,TR,roll,pitch,yaw,dS,dL,dP\n9000001,0,0.1,0.2,0.1,0.01,0.02,0.03\n")

        header_info = inspect_motion_header(mot_path)
        self.assertTrue(header_info["is_valid"])
        self.assertTrue(header_info["is_framewise_timeseries"])
        self.assertFalse(header_info["is_summary_table"])
        self.assertTrue(header_info["has_subject_col"])
        self.assertTrue(header_info["has_frame_col"])
        self.assertTrue(header_info["has_6_motion_params"])

        # Summary motion header (e.g. actual KKI_motion.csv)
        summary_path = self.staging / "KKI_motion.csv"
        summary_path.write_text("File,Max Motion (mm),Max Motion Time Point, Max Rotation (degree), Max Rotation Time Point, Max X (mm), Max Y (mm), Max Z (mm), Max Roll (degree), Max Pitch (degree), Max Yaw (degree)\nrp_9000001_session_1_rest_1.1D,1.214,146,0.566,81,0.185,1.110,-0.436,0.406,0.499,0.145\n")
        summary_info = inspect_motion_header(summary_path)
        self.assertTrue(summary_info["is_valid"])
        self.assertTrue(summary_info["is_summary_table"])
        self.assertFalse(summary_info["is_framewise_timeseries"])
        self.assertTrue(summary_info["has_subject_col"])

    def test_run_verification_workflow_status(self):
        """Verify that run_verification correctly flags framewise FD as unresolved when only summary tables exist."""
        # Setup files in staging
        tar_path = self.staging / "ADHD200_CC200_TCs_filtfix.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            f = self.staging / "dummy.1D"
            f.write_text("1.0 2.0 3.0", encoding="utf-8")
            tar.add(f, arcname="sample.1D")

        # Setup required motion files with summary header
        for mf in ["KKI_motion.csv", "NeuroIMAGE_motion.csv", "NYU_motion.csv", "OHSU_motion.csv", "Peking_1_motion.csv", "Peking_2_motion.csv", "Peking_3_motion.csv"]:
            p = self.staging / mf
            p.write_text("File,Max Motion (mm),Max Motion Time Point, Max Rotation (degree), Max Rotation Time Point, Max X (mm), Max Y (mm), Max Z (mm), Max Roll (degree), Max Pitch (degree), Max Yaw (degree)\n")

        # Setup phenotype file in dest
        pheno_dir = self.dest / "phenotypes"
        pheno_dir.mkdir(parents=True, exist_ok=True)
        (pheno_dir / "adhd200_preprocessed_phenotypics.tsv").write_text("ScanDir ID\tDX\n")

        report = run_verification(self.staging, self.dest, self.pheno)
        self.assertEqual(report["archive_integrity"], "PASSED")
        self.assertEqual(report["file_acquisition"], "PASS")
        self.assertIn("VERIFIED", report["motion_summary_metadata"])
        self.assertIn("UNRESOLVED", report["framewise_fd_availability"])
        self.assertIn("NOT READY", report["motion_cohort_exclusion_status"])
        self.assertFalse(report["ready_for_preprocessing"])


if __name__ == "__main__":
    unittest.main()
