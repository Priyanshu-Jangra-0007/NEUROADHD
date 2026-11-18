import io
import math
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compute_athena_fd import (
    compute_power_fd,
    extract_6dof_params,
    load_dev_cc200_frame_counts,
    parse_octave_transf_text,
)


class TestComputeAthenaFD(unittest.TestCase):
    def test_parse_octave_transf_fortran_order(self):
        # Construct 2 identity-with-translation 4x4 matrices in Fortran (column-major) order
        m0 = np.eye(4)
        m0[:3, 3] = [1.0, -2.0, 3.0]
        m1 = np.eye(4)
        m1[:3, 3] = [1.5, -2.0, 3.25]
        arr = np.stack([m0, m1], axis=2)  # shape (4, 4, 2)
        flat = arr.reshape(-1, order="F")
        text = "# Created by Octave\n# name: transf\n# type: matrix\n# ndims: 3\n 4 4 2\n"
        text += "\n".join(f" {v}" for v in flat) + "\n"

        parsed = parse_octave_transf_text(text)
        self.assertEqual(parsed.shape, (4, 4, 2))
        np.testing.assert_allclose(parsed, arr, atol=1e-12)

    def test_extract_6dof_and_power_fd_exact(self):
        # Frame 0: identity
        # Frame 1: translate +0.3 mm on X and rotate +1.0 deg around Z (yaw)
        theta = math.radians(1.0)
        m0 = np.eye(4)
        m1 = np.eye(4)
        m1[0, 0] = math.cos(theta)
        m1[0, 1] = -math.sin(theta)
        m1[1, 0] = math.sin(theta)
        m1[1, 1] = math.cos(theta)
        m1[0, 3] = 0.30

        arr = np.stack([m0, m1], axis=2)
        params = extract_6dof_params(arr)
        self.assertEqual(params.shape, (2, 6))
        self.assertAlmostEqual(params[1, 2], 1.0, places=6)  # yaw = 1 deg
        self.assertAlmostEqual(params[1, 3], 0.30, places=6)  # tx = 0.3 mm

        fd = compute_power_fd(params, radius_mm=50.0)
        self.assertEqual(fd.shape, (2,))
        self.assertEqual(fd[0], 0.0)
        expected_fd1 = 0.30 + (50.0 * math.pi / 180.0) * 1.0
        self.assertAlmostEqual(fd[1], expected_fd1, places=6)

    def test_non_rigid_matrix_rejected(self):
        bad = np.eye(4)[:, :, None]
        bad[0, 0, 0] = 2.0  # scaling, det = 2
        with self.assertRaises(ValueError):
            extract_6dof_params(bad)

    def test_load_dev_cc200_skips_nyu_and_excluded_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc_path = Path(tmp) / "mock_cc200.tar.gz"
            with tarfile.open(arc_path, "w:gz") as tf:
                for member_name, body in [
                    ("KKI/0010001/sfnwmrda0010001_session_1_rest_1_cc200_TCs.1D", b"1 2\n3 4\n5 6\n"),
                    ("NYU/0020001/sfnwmrda0020001_session_1_rest_1_cc200_TCs.1D", b"9 9\n9 9\n"),
                    ("WashU/0030001/sfnwmrda0030001_session_1_rest_1_cc200_TCs.1D", b"1 1\n"),
                    ("Peking_2/0040001/sfnwmrda0040001_session_1_rest_1_cc200_TCs.1D", b"1 2\n3 4\n"),
                ]:
                    info = tarfile.TarInfo(name=member_name)
                    info.size = len(body)
                    tf.addfile(info, io.BytesIO(body))

            dev_map = load_dev_cc200_frame_counts(arc_path)
            self.assertEqual(len(dev_map), 2)
            self.assertIn(("KKI", "10001"), dev_map)
            self.assertIn(("Peking", "40001"), dev_map)
            self.assertEqual(dev_map[("KKI", "10001")], ("KKI", 3))
            self.assertEqual(dev_map[("Peking", "40001")], ("Peking_2", 2))
            # Verify NYU and WashU are completely absent
            self.assertTrue(all(k[0] not in ("NYU", "WashU") for k in dev_map))


if __name__ == "__main__":
    unittest.main()
