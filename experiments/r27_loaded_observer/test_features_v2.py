import unittest

import numpy as np

from experiments.r27_loaded_observer.features_v2 import normal_scores


def rows(force, table=False):
    values = np.asarray(force, float)
    ticks = np.arange(len(values))
    return np.c_[ticks, ticks - 18, np.arange(len(values)) + 100,
                 np.zeros(len(values)), np.full(len(values), table), values]


class FeatureV2Tests(unittest.TestCase):
    def test_constant_eligible_force_is_zero(self):
        onset, sustain = normal_scores(rows(np.tile([1.0, 2.0, 3.0], (80, 1))))
        self.assertTrue(np.array_equal(onset, np.zeros_like(onset)))
        self.assertTrue(np.array_equal(sustain, np.zeros_like(sustain)))

    def test_any_table_eligibility_in_window_excludes_sample(self):
        data = rows(np.zeros((80, 3)))
        data[:, 4] = 1
        onset, sustain = normal_scores(data)
        self.assertFalse(len(onset))
        self.assertFalse(len(sustain))


if __name__ == "__main__":
    unittest.main()
