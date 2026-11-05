import unittest

import numpy as np

from tools.r4_cause_analysis import (
    active_ema,
    active_rolling_mean,
    collision_is_missed,
    consecutive_crossing,
    empirical_roc,
    rolling_mean,
    score_signals,
)


class TestR4CauseAnalysis(unittest.TestCase):
    def test_trailing_means_and_active_reset(self):
        values = np.array([1., 2., 3., 8., 10.])
        np.testing.assert_allclose(rolling_mean(values, 3), [1, 1.5, 2, 13/3, 7])
        active = np.array([1, 1, 0, 1, 1], dtype=bool)
        np.testing.assert_allclose(active_rolling_mean(values, active, 3),
                                   [1, 1.5, 0, 8, 9])
        np.testing.assert_allclose(active_ema(values, active, .5),
                                   [.5, 1.25, 0, 4, 7])

    def test_consecutive_crossing_is_strict_and_marks_run_completion(self):
        values = np.array([0., 2., 2., 2., 2., 1., 3., 3., 3., 3.])
        expected = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
        np.testing.assert_array_equal(consecutive_crossing(values, 1., 4), expected)

    def test_score_families_use_only_declared_arrays(self):
        raw = {
            "active": np.array([False, True, True]),
            "estimated_wrench": np.array([[3., 4., -2., 0., 0., 0.],
                                           [3., 4., 2., 0., 0., 0.],
                                           [0., 0., 4., 0., 0., 0.]]),
            "joint_residual": np.array([[3., 4.], [0., 5.], [0., 0.]]),
        }
        scores = score_signals(raw, np.zeros((3, 2)))
        self.assertEqual(set(scores), {"xy_raw_active", "xy_ema01_active",
                                       "xy_mean40_active", "joint_l2",
                                       "fz_positive", "fz_abs", "fz_hp40"})
        np.testing.assert_allclose(scores["xy_raw_active"], [0, 5, 0])
        np.testing.assert_allclose(scores["joint_l2"], [5, 5, 0])
        np.testing.assert_allclose(scores["fz_positive"], [0, 2, 4])

    def test_collision_outcome_alias_and_conflict(self):
        self.assertTrue(collision_is_missed({"collision_outcome": "missed",
                                             "detect_delay_ms": None,
                                             "alarm_before_collision": False}))
        self.assertFalse(collision_is_missed({"collision_outcome": "detected",
                                              "detect_delay_ms": 12.,
                                              "alarm_before_collision": False}))
        with self.assertRaises(AssertionError):
            collision_is_missed({"collision_outcome": "detected",
                                 "detect_delay_ms": None,
                                 "alarm_before_collision": False})

    def test_empirical_roc_strict_crossing_and_curve(self):
        curve = empirical_roc(np.array([1., 2.]), np.array([2., 3.]))
        self.assertGreaterEqual(len(curve["threshold"]), 4)
        self.assertEqual(curve["false_positive_rate"][0], 0.0)
        self.assertEqual(curve["true_positive_rate"][0], 0.0)
        self.assertEqual(curve["false_positive_rate"][-1], 1.0)
        self.assertEqual(curve["true_positive_rate"][-1], 1.0)
        self.assertTrue(np.all(np.diff(curve["false_positive_rate"]) >= 0))
        self.assertTrue(np.all(np.diff(curve["true_positive_rate"]) >= 0))


if __name__ == "__main__":
    unittest.main()
