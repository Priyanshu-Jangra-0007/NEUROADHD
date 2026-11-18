import io
import math
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compute_athena_fd import (
    center_shift_transf,
    compute_power_fd,
    extract_6dof_params,
    load_dev_cc200_frame_counts,
    parse_octave_transf_text,
)


def _rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _make_se3(rot3: np.ndarray, trans3: list[float]) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot3
    m[:3, 3] = trans3
    return m


class TestComputeAthenaFD(unittest.TestCase):
    def test_parse_octave_transf_fortran_order(self):
        m0 = _make_se3(np.eye(3), [1.0, -2.0, 3.0])
        m1 = _make_se3(np.eye(3), [1.5, -2.0, 3.25])
        arr = np.stack([m0, m1], axis=2)
        flat = arr.reshape(-1, order="F")
        text = "# Created by Octave\n# name: transf\n# type: matrix\n# ndims: 3\n 4 4 2\n"
        text += "\n".join(f" {v}" for v in flat) + "\n"

        parsed = parse_octave_transf_text(text)
        self.assertEqual(parsed.shape, (4, 4, 2))
        np.testing.assert_allclose(parsed, arr, atol=1e-12)

    def test_1mm_translation_each_axis_gives_fd_1(self):
        for axis in range(3):
            shift = [0.0, 0.0, 0.0]
            shift[axis] = 1.0
            arr = np.stack([np.eye(4), _make_se3(np.eye(3), shift)], axis=2)
            fd = compute_power_fd(extract_6dof_params(arr), radius_mm=50.0)
            self.assertEqual(fd[0], 0.0)
            self.assertAlmostEqual(fd[1], 1.0, places=9)

    def test_1deg_rotation_each_axis_gives_087mm_on_50mm_sphere(self):
        expected_arc = (math.pi / 180.0) * 50.0  # 0.8726646259971648 mm
        for rot_fn in (_rot_x, _rot_y, _rot_z):
            arr = np.stack([np.eye(4), _make_se3(rot_fn(1.0), [0.0, 0.0, 0.0])], axis=2)
            fd = compute_power_fd(extract_6dof_params(arr), radius_mm=50.0)
            self.assertEqual(fd[0], 0.0)
            self.assertAlmostEqual(fd[1], expected_arc, places=6)
            self.assertAlmostEqual(fd[1], 0.872665, places=5)

    def test_first_frame_handling_and_zero_motion(self):
        # Constant non-identity pose across 5 frames must yield all-zero FD with FD[0] == 0.0
        pose = _make_se3(_rot_z(2.5) @ _rot_y(-1.2) @ _rot_x(0.8), [1.2, -0.7, 0.4])
        const_arr = np.stack([pose] * 5, axis=2)
        fd_const = compute_power_fd(extract_6dof_params(const_arr), radius_mm=50.0)
        self.assertEqual(fd_const.shape, (5,))
        self.assertEqual(fd_const[0], 0.0)
        np.testing.assert_allclose(fd_const, np.zeros(5), atol=1e-12)

        # Shift only on frame 1 relative to frame 0
        pose_next = _make_se3(_rot_z(2.5) @ _rot_y(-1.2) @ _rot_x(0.8), [1.7, -0.7, 0.4])
        arr = np.stack([pose, pose_next, pose_next], axis=2)
        fd = compute_power_fd(extract_6dof_params(arr), radius_mm=50.0)
        self.assertEqual(fd[0], 0.0)
        self.assertAlmostEqual(fd[1], 0.5, places=6)
        self.assertAlmostEqual(fd[2], 0.0, places=9)

    def test_matrix_direction_inverse_gives_same_fd(self):
        # 1) Pure translation trajectory: M^-1 has -t -> exact identical FD
        trans_seq = np.stack(
            [
                _make_se3(np.eye(3), [0.0, 0.0, 0.0]),
                _make_se3(np.eye(3), [0.4, -0.2, 0.1]),
                _make_se3(np.eye(3), [-0.1, 0.3, 0.2]),
            ],
            axis=2,
        )
        trans_inv = np.stack([np.linalg.inv(trans_seq[:, :, t]) for t in range(3)], axis=2)
        np.testing.assert_allclose(
            compute_power_fd(extract_6dof_params(trans_seq)),
            compute_power_fd(extract_6dof_params(trans_inv)),
            atol=1e-12,
        )

        # 2) Pure rotation trajectory about a single axis: M^-1 has R^T -> exact identical FD
        rot_seq = np.stack(
            [
                _make_se3(_rot_x(0.0), [0.0, 0.0, 0.0]),
                _make_se3(_rot_x(0.6), [0.0, 0.0, 0.0]),
                _make_se3(_rot_x(-0.4), [0.0, 0.0, 0.0]),
            ],
            axis=2,
        )
        rot_inv = np.stack([np.linalg.inv(rot_seq[:, :, t]) for t in range(3)], axis=2)
        np.testing.assert_allclose(
            compute_power_fd(extract_6dof_params(rot_seq)),
            compute_power_fd(extract_6dof_params(rot_inv)),
            atol=1e-12,
        )

        # 3) Combined realistic resting-state rigid-body trajectory (rotations < 1 deg, trans < 0.5 mm):
        # M^-1 has translation -R^T t, which equals -t to first order (O(theta * ||t||) < 0.01 mm).
        comb_seq = np.stack(
            [
                _make_se3(_rot_z(0.1) @ _rot_y(-0.2) @ _rot_x(0.3), [0.10, -0.15, 0.05]),
                _make_se3(_rot_z(0.4) @ _rot_y(0.1) @ _rot_x(-0.2), [0.35, -0.05, 0.20]),
                _make_se3(_rot_z(-0.2) @ _rot_y(0.3) @ _rot_x(0.1), [0.20, 0.15, -0.10]),
            ],
            axis=2,
        )
        comb_inv = np.stack([np.linalg.inv(comb_seq[:, :, t]) for t in range(3)], axis=2)
        fd_fwd = compute_power_fd(extract_6dof_params(comb_seq))
        fd_inv = compute_power_fd(extract_6dof_params(comb_inv))
        np.testing.assert_allclose(fd_fwd, fd_inv, atol=0.01)

    def test_volume_center_shift_removes_origin_lever_arm(self):
        # Pure 1-deg rotation about an off-origin brain center c = [4.1, 53.3, -10.7] mm
        # World matrix: M(x) = R(x - c) + c = R x + (I - R) c
        c = np.array([4.1, 53.3, -10.7], dtype=np.float64)
        rot = _rot_z(1.0)
        t_world = (np.eye(3) - rot) @ c
        m0 = np.eye(4)
        m1 = _make_se3(rot, t_world.tolist())
        arr_world = np.stack([m0, m1], axis=2)

        # Naive world-origin FD is inflated by |(I - R) c| (~1.0 mm extra!)
        fd_naive = compute_power_fd(extract_6dof_params(arr_world), radius_mm=50.0)
        expected_pure_rot = (math.pi / 180.0) * 50.0
        self.assertGreater(fd_naive[1], expected_pure_rot + 0.5)

        # After shifting to brain center c, spurious lever-arm translation vanishes exactly
        arr_centered = center_shift_transf(arr_world, c)
        fd_centered = compute_power_fd(extract_6dof_params(arr_centered), radius_mm=50.0)
        self.assertAlmostEqual(fd_centered[1], expected_pure_rot, places=6)

    def test_non_rigid_matrix_rejected(self):
        bad = np.eye(4)[:, :, None]
        bad[0, 0, 0] = 2.0
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
            self.assertTrue(all(k[0] not in ("NYU", "WashU") for k in dev_map))


if __name__ == "__main__":
    unittest.main()
