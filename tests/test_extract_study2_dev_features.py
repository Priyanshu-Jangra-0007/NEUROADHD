import math
import unittest

import numpy as np

from scripts.extract_study2_dev_features import (
    compute_valid_roi_mask,
    parse_athena_1d_timeseries,
    timeseries_to_fisher_z_edges,
)


class TestExtractStudy2DevFeatures(unittest.TestCase):
    def test_parse_athena_1d_timeseries(self):
        text = (
            "File\tSub-brick\tMean_1\tMean_3\tMean_4\n"
            "sfnwmrda.nii.gz\t0[?]\t1.0\t2.0\t3.0\n"
            "sfnwmrda.nii.gz\t1[?]\t4.0\t5.0\t6.0\n"
            "sfnwmrda.nii.gz\t2[?]\t7.0\t8.0\t9.0\n"
        )
        names, ts = parse_athena_1d_timeseries(text)
        self.assertEqual(names, ["Mean_1", "Mean_3", "Mean_4"])
        self.assertEqual(ts.shape, (3, 3))
        self.assertEqual(ts.dtype, np.float64)
        np.testing.assert_allclose(ts[1], [4.0, 5.0, 6.0])

    def test_compute_valid_roi_mask_drops_constant_or_nan_roi(self):
        s1 = np.array([[1.0, 5.0, 2.0, 0.1], [2.0, 5.0, 3.0, 0.2], [3.0, 5.0, 1.0, 0.3]])
        s2 = np.array([[1.0, 2.0, np.nan, 0.5], [2.0, 3.0, 1.0, 0.2], [3.0, 4.0, 2.0, 0.1]])
        mask = compute_valid_roi_mask([s1, s2])
        # ROI 0 ok, ROI 1 constant in s1 -> False, ROI 2 NaN in s2 -> False, ROI 3 ok
        np.testing.assert_array_equal(mask, np.array([True, False, False, True]))

    def test_timeseries_to_fisher_z_edges_exact(self):
        rng = np.random.default_rng(7)
        t = np.linspace(0, 10, 60)
        x0 = np.sin(t)
        x1 = 0.5 * x0 + 0.5 * rng.normal(size=60)
        x2 = -x0  # perfectly anti-correlated -> clipped before arctanh
        ts = np.column_stack([x0, x1, x2])
        edges = timeseries_to_fisher_z_edges(ts)
        self.assertEqual(edges.shape, (3,))  # 3*2/2 = 3
        self.assertTrue(np.all(np.isfinite(edges)))
        r01 = float(np.corrcoef(x0, x1)[0, 1])
        self.assertAlmostEqual(edges[0], math.atanh(r01), places=10)
        # Edge (0, 2) is clipped at -1 + 1e-7
        self.assertAlmostEqual(edges[1], math.atanh(-1.0 + 1e-7), places=6)


if __name__ == "__main__":
    unittest.main()
