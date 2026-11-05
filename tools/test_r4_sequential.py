"""Synthetic, simulation-free tests for the registered R4 detector kernel."""

from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

import numpy as np

from experiments.r4_sequential import SequentialMonitor
from tools import r4_build
from tools.production_audit import audit_text


ROOT = Path(__file__).resolve().parents[1]
SPECS = {row["id"]: row for row in
         json.loads((ROOT / "results/v2_protocol/PROTOCOL_R4.json").read_text())["candidate_grid"]}


def calibration(count=1, threshold=1.0):
    return {"location": [0.0] * count, "scale": [1.0] * count,
            "threshold": [float(threshold)] * count}


class SequentialMonitorTests(unittest.TestCase):
    def test_w40_startup_window_onset_clear_and_inactive_reset(self):
        monitor = SequentialMonitor(SPECS["W40_ACTIVE"], calibration(threshold=2.0))
        self.assertFalse(monitor.update(0.0, 1.0, True))
        self.assertEqual(monitor.statistic, 1.0)
        for index in range(3):
            hit = monitor.update(0.001 * (index + 1), 4.0, True)
        self.assertFalse(hit)
        self.assertAlmostEqual(monitor.statistic, 3.25)
        self.assertTrue(monitor.update(0.004, 4.0, True))
        for index in range(8):
            hit = monitor.update(0.005 + 0.001 * index, 0.0, True)
        self.assertTrue(hit)  # the 40-sample mean remains above the boundary
        self.assertFalse(monitor.update(0.02, 99.0, False))
        self.assertEqual(monitor.statistic, 0.0)
        self.assertFalse(monitor.detected)

    def test_w40_discards_the_oldest_sample_after_forty(self):
        monitor = SequentialMonitor(SPECS["W40_ACTIVE"],
                                    calibration(threshold=float("inf")))
        for index in range(40):
            monitor.update(index * .001, float(index + 1), True)
        self.assertAlmostEqual(monitor.statistic, 20.5)
        monitor.update(.04, 41.0, True)
        self.assertAlmostEqual(monitor.statistic, 21.5)

    def test_cusum_formula_onset_and_eight_clear_samples(self):
        monitor = SequentialMonitor(SPECS["CUSUM_ACTIVE_K05"],
                                    calibration(threshold=1.0))
        self.assertFalse(monitor.update(2.0, 1.0, True))
        self.assertAlmostEqual(monitor.statistic, 0.5)
        self.assertFalse(monitor.update(2.001, 1.0, True))
        self.assertAlmostEqual(monitor.statistic, 1.0)
        states = [monitor.update(2.002 + index * .001, 1.0, True)
                  for index in range(4)]
        self.assertEqual(states, [False, False, False, True])
        for index in range(14):
            state = monitor.update(2.01 + index * .001, 0.0, True)
        self.assertFalse(state)
        self.assertEqual(monitor.statistic, 0.0)

    def test_phase_cusum_is_ungated_and_resets_at_boundary(self):
        monitor = SequentialMonitor(SPECS["CUSUM_PHASE_K05"],
                                    calibration(5, threshold=10.0))
        monitor.update(1.699, 3.0, False)
        self.assertEqual(monitor.phase, 1)
        self.assertGreater(monitor.statistic, 0.0)
        monitor.update(1.7, 1.0, False)
        self.assertEqual(monitor.phase, 2)
        self.assertAlmostEqual(monitor.statistic, 0.5)
        monitor.update(3.9, 2.0, False)
        self.assertEqual(monitor.phase, 4)
        self.assertAlmostEqual(monitor.statistic, 1.5)

    def test_score_norm_is_direction_symmetric(self):
        spec = SPECS["CUSUM_ACTIVE_K10"]
        a = SequentialMonitor(spec, calibration(threshold=100.0))
        b = SequentialMonitor(spec, calibration(threshold=100.0))
        for index in range(5):
            self.assertEqual(a.update(index * .001, np.linalg.norm([3.0, 4.0]), True),
                             b.update(index * .001, np.linalg.norm([-3.0, -4.0]), True))
        self.assertEqual(a.statistic, b.statistic)

    def test_nonfinite_and_invalid_calibration_fail_closed(self):
        monitor = SequentialMonitor(SPECS["CUSUM_ACTIVE_K05"],
                                    calibration(threshold=float("inf")))
        monitor.update(0.0, 10.0, True)
        self.assertFalse(monitor.detected)
        self.assertFalse(monitor.update(.001, float("nan"), True))
        self.assertEqual(monitor.statistic, 0.0)
        with self.assertRaises(ValueError):
            SequentialMonitor(SPECS["CUSUM_PHASE_K05"], calibration(1))


class MaterializationTests(unittest.TestCase):
    def parameters(self):
        return {"servo": {"spec": SPECS["W40_ACTIVE"],
                           "calibration": calibration(threshold=7.0)},
                "torque": None}

    def test_materialization_is_deterministic_audited_and_preserves_r1_prefix(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            first = Path(folder) / "first.py"
            second = Path(folder) / "second.py"
            record1 = r4_build.materialize(self.parameters(), first)
            record2 = r4_build.materialize(self.parameters(), second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(record1["sha256"], record2["sha256"])
            r1 = (ROOT / "r1_detector.py").read_bytes()
            self.assertTrue(first.read_bytes().startswith(r1))
            self.assertEqual(first.read_text().count(
                "residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)"), 1)
            self.assertEqual(audit_text(first.read_text(), str(first))["status"], "pass")

    def test_mixed_fallback_materializes_and_imports(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            target = Path(folder) / "candidate.py"
            r4_build.materialize(self.parameters(), target)
            module_spec = importlib.util.spec_from_file_location("r4_candidate_test", target)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            self.assertIs(module.Detector, module.R4Detector)
            self.assertIsNone(module.R4_PARAMETERS["torque"])
            self.assertEqual(module.R4_PARAMETERS["servo"]["spec"],
                             SPECS["W40_ACTIVE"])


if __name__ == "__main__":
    unittest.main()
