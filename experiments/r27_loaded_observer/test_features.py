import numpy as np
import unittest

from experiments.r27_loaded_observer.features import normal_scores


def rows(force):
    values = np.asarray(force, float)
    ticks = np.arange(len(values))
    return np.c_[ticks, ticks - 18, np.arange(len(values)) + 100,
                 np.zeros(len(values)), values]


class FeatureTests(unittest.TestCase):
    def test_constant_force_has_zero_change_scores(self):
        onset, sustain = normal_scores(rows(np.tile([1.0, -2.0, 3.0], (80, 1))))
        self.assertTrue(np.array_equal(onset, np.zeros_like(onset)))
        self.assertTrue(np.array_equal(sustain, np.zeros_like(sustain)))


    def test_known_ramp_has_expected_vector_norms(self):
        force = np.c_[np.arange(80), np.zeros((80, 2))]
        onset, sustain = normal_scores(rows(force))
        self.assertTrue(np.allclose(onset, 8.0))
        self.assertTrue(np.allclose(sustain, 32.0))


    def test_nonconsecutive_or_unstable_samples_are_ineligible(self):
        data = rows(np.zeros((40, 3)))
        data[:, 2] = 0
        onset, sustain = normal_scores(data, grip_stability=64)
        self.assertFalse(len(onset))
        self.assertFalse(len(sustain))
        with self.assertRaises(ValueError):
            normal_scores(np.zeros((4, 6)))


if __name__ == "__main__":
    unittest.main()
