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
        tar_path = self.staging / "ADHD200_CC200TCs_filtfix.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(test_file, arcname="sample.txt")

        ok, count, msg = verify_tar_integrity(tar_path)
        self.assertTrue(ok)
        self.assertEqual(count, 1)
        self.assertEqual(msg, "OK")

    def test_motion_header_inspection(self):
        """Verify parsing and validation of representative motion headers."""
        mot_path = self.staging / "KKI_motion.csv"
        mot_path.write_text("ScanDir ID,TR,roll,pitch,yaw,dS,dL,dP\n9000001,0,0.1,0.2,0.1,0.01,0.02,0.03\n")

        header_info = inspect_motion_header(mot_path)
        self.assertTrue(header_info["is_valid"])
        self.assertTrue(header_info["has_subject_col"])
        self.assertTrue(header_info["has_frame_col"])
        self.assertTrue(header_info["has_6_motion_params"])
        self.assertEqual(header_info["rotational_units"], "Degrees (AFNI 3dvolreg standard: roll, pitch, yaw)")
        self.assertEqual(header_info["translational_units"], "Millimeters (AFNI 3dvolreg standard: dS, dL, dP)")


if __name__ == "__main__":
    unittest.main()
