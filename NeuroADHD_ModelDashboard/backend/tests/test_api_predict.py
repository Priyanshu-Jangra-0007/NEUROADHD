"""Automated API integration tests for POST /api/predict/subject."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Path setup to import server
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from server import app


class TestPredictSubjectAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.workspace_root = BACKEND_DIR.parents[1]
        cls.valid_subject_csv = (
            cls.workspace_root
            / "NeuroADHD/data/adhd200/raw/cc200/1017176_session_1/roi_CC200.csv"
        )

    def test_predict_valid_csv_upload(self):
        """Test successful prediction with a real CC200 ROI time series."""
        self.assertTrue(
            self.valid_subject_csv.is_file(),
            f"Test file not found at: {self.valid_subject_csv}",
        )

        with open(self.valid_subject_csv, "rb") as f:
            files = {"file": ("roi_CC200.csv", f, "text/csv")}
            response = self.client.post("/api/predict/subject", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Structure assertions
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "roi_CC200.csv")
        self.assertIn("input_validation", data)
        self.assertIn("prediction", data)
        self.assertIn("disclaimer", data)

        # Validation assertions
        val = data["input_validation"]
        self.assertEqual(val["status"], "VALIDATED (PASS)")
        self.assertEqual(val["roi_count"], 200)
        self.assertEqual(val["features_extracted"], 19_900)
        self.assertEqual(val["features_selected"], 100)
        self.assertGreaterEqual(val["timepoints"], 50)

        # Prediction probability assertions
        pred = data["prediction"]
        self.assertIn(pred["predicted_class"], [0, 1])
        self.assertTrue(0.0 <= pred["adhd_probability"] <= 1.0)
        self.assertTrue(0.0 <= pred["control_probability"] <= 1.0)
        self.assertAlmostEqual(
            pred["adhd_probability"] + pred["control_probability"], 1.0, places=4
        )
        self.assertIn(pred["class_label"], ["ADHD Pattern", "Typical Control Pattern"])

        # Security check: verify no server file system paths are leaked
        self.assertNotIn("/Users/", response.text)
        self.assertNotIn("/tmp/", response.text)

    def test_predict_missing_file_payload(self):
        """Test that requests missing a file field return HTTP 422."""
        response = self.client.post("/api/predict/subject")
        self.assertEqual(response.status_code, 422)

    def test_predict_invalid_extension(self):
        """Test rejection of non-CSV file uploads (e.g. .bin, .exe)."""
        files = {"file": ("scan_image.bin", io.BytesIO(b"binary data"), "application/octet-stream")}
        response = self.client.post("/api/predict/subject", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid file type", response.json()["detail"])

    def test_predict_empty_file(self):
        """Test rejection of empty 0-byte CSV files."""
        files = {"file": ("roi_CC200.csv", io.BytesIO(b""), "text/csv")}
        response = self.client.post("/api/predict/subject", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_predict_incorrect_roi_dimensions(self):
        """Test rejection of CSV with wrong ROI row count (50 instead of 200)."""
        dummy = np.random.normal(size=(50, 100))
        df = pd.DataFrame(dummy)
        df.insert(0, "ROI", np.arange(1, 51))
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {"file": ("roi_CC200.csv", csv_buffer, "text/csv")}
        response = self.client.post("/api/predict/subject", files=files)

        self.assertEqual(response.status_code, 400)
        self.assertIn("expected 200 ROI rows", response.json()["detail"])

    def test_predict_malformed_csv_content(self):
        """Test rejection of non-numeric malformed CSV files."""
        malformed_bytes = b"header1,header2\ninvalid_text,random_text\n"
        files = {"file": ("roi_CC200.csv", io.BytesIO(malformed_bytes), "text/csv")}
        response = self.client.post("/api/predict/subject", files=files)

        self.assertEqual(response.status_code, 400)

    def test_predict_oversized_file(self):
        """Test rejection of files exceeding the 10 MB limit."""
        # Create a dummy buffer slightly over 10 MB (10.5 MB)
        oversized_data = io.BytesIO(b"0" * (11 * 1024 * 1024))
        files = {"file": ("roi_CC200.csv", oversized_data, "text/csv")}
        response = self.client.post("/api/predict/subject", files=files)

        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds maximum allowed limit", response.json()["detail"])

    def test_get_sample_subject(self):
        """Test GET /api/predict/sample returns 200 and text/csv content."""
        response = self.client.get("/api/predict/sample")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers.get("content-type", ""))
        self.assertGreater(len(response.text), 1000)
        lines = response.text.strip().split("\n")
        self.assertEqual(len(lines), 201)  # Header + 200 ROIs


if __name__ == "__main__":
    unittest.main()


