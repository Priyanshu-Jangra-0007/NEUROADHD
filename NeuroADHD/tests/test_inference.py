"""Automated tests for single-subject fMRI inference pipeline."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict_subject import predict_single_subject


class TestSingleSubjectInference(unittest.TestCase):
    def setUp(self):
        self.sample_subject = (
            PROJECT_ROOT / "data/adhd200/raw/cc200/1017176_session_1/roi_CC200.csv"
        )

    def test_inference_on_existing_subject(self):
        """Test complete inference pipeline on an existing verified subject file."""
        self.assertTrue(self.sample_subject.is_file(), f"Sample subject missing: {self.sample_subject}")

        result = predict_single_subject(self.sample_subject)

        # Structure checks
        self.assertIn("input_validation", result)
        self.assertIn("prediction", result)
        self.assertIn("disclaimer", result)

        # Validation checks
        val = result["input_validation"]
        self.assertEqual(val["status"], "VALIDATED (PASS)")
        self.assertEqual(val["roi_count"], 200)
        self.assertEqual(val["features_extracted"], 19_900)
        self.assertEqual(val["features_selected"], 100)
        self.assertGreaterEqual(val["timepoints"], 50)

        # Prediction probability checks
        pred = result["prediction"]
        self.assertIn(pred["predicted_class"], [0, 1])
        self.assertTrue(0.0 <= pred["adhd_probability"] <= 1.0)
        self.assertTrue(0.0 <= pred["control_probability"] <= 1.0)
        self.assertAlmostEqual(
            pred["adhd_probability"] + pred["control_probability"], 1.0, places=4
        )
        self.assertIn(pred["class_label"], ["ADHD Pattern", "Typical Control Pattern"])

    def test_inference_rejects_nonexistent_file(self):
        """Test that missing file paths raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            predict_single_subject("/non/existent/path/to/roi_CC200.csv")

    def test_inference_rejects_invalid_roi_count(self):
        """Test that inputs with incorrect ROI row count (e.g. 50 instead of 200) are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_csv = Path(tmp_dir) / "roi_CC200_bad.csv"
            # Create a 50 ROI x 100 timepoints dummy table
            dummy = np.random.normal(size=(50, 100))
            df = pd.DataFrame(dummy)
            df.insert(0, "ROI", np.arange(1, 51))
            df.to_csv(bad_csv, index=False)

            with self.assertRaises(ValueError) as ctx:
                predict_single_subject(bad_csv)
            self.assertIn("expected 200 ROI rows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
