import unittest

import numpy as np

from experiments.r22_normal.candidate import Detector


class MedianFilterTest(unittest.TestCase):
    def test_one_sample_spike_is_rejected_componentwise(self):
        got = Detector.robust_horizontal([[8, -12], [10, -19], [9, -7]])
        np.testing.assert_array_equal(got, [9, -12])

    def test_coherent_ramp_survives(self):
        got = Detector.robust_horizontal([[18, -20], [19, -22], [20, -24]])
        np.testing.assert_array_equal(got, [19, -22])

    def test_nonfinite_or_wrong_history_is_rejected(self):
        for value in ([[1, 2]], [[1, 2], [3, 4], [np.nan, 5]]):
            with self.assertRaises(ValueError):
                Detector.robust_horizontal(value)


if __name__ == '__main__':
    unittest.main()
