import unittest
import tempfile

import numpy as np

from experiments.r19_close_pairs import assay
from experiments.r18_paired_impulses import scoring
from experiments.r19_same_tuple.compare import arm_delta, load_trace, save_trace


def arm(raw, credit, end):
    slots = {}
    for name in assay.SLOTS:
        slots[name] = {
            "raw_delays_ms": list(raw), "credited_delays_ms": list(credit),
            "attributable_delays_ms": list(credit), "high_at_onset": not bool(credit),
            "quiet_before_onset": bool(credit),
        }
    return {
        "slots": slots,
        "levels": {name: {"on_intervals_ms": [[1000, end]] if name == "both" else [],
                           "active_samples": end - 1000,
                           "high_at_end": False, "high_at_isolated_window_end": False}
                   for name in ("none", "first", "second", "both")},
        "false_s": {name: [] for name in ("none", "first", "second", "both")},
        "tool": {name: {"tool_present": False, "tool_detected": False,
                          "tool_prealarm": False, "tool_delay_ms": None}
                 for name in ("none", "first", "second", "both")},
    }


class SameTupleTest(unittest.TestCase):
    def test_delta_exposes_credit_and_fall_on_same_case(self):
        case = {"impacts": [[1.0, [1, 0, 0]], [1.2, [0, 1, 0]]]}
        result = arm_delta(arm([20], [], 1190), arm([20], [20], 1120), case)
        self.assertFalse(result["strict_all_four_V2"])
        self.assertTrue(result["strict_all_four_impulse"])
        self.assertEqual(result["slots"]["both_second"]["credited_delta"], 1)
        self.assertEqual(result["first_alarm_fall"]["impulse"]["clear_before_second_ms"], 80)

    def test_trace_round_trip_preserves_measured_finite_grid_flag(self):
        trace = {key: np.zeros(6000) for key in scoring.CAUSAL_KEYS}
        trace["t"] = np.arange(6000) * 0.001
        trace["f_ext_xyz_committed"] = np.zeros((6000, 3))
        trace["r18_diverged"] = False
        with tempfile.TemporaryDirectory() as folder:
            path = folder + "/trace.npz"
            save_trace(path, trace)
            loaded = load_trace(path)
        self.assertIs(loaded["r18_diverged"], False)
        self.assertTrue(scoring.finite_grid(loaded))


if __name__ == "__main__":
    unittest.main()
