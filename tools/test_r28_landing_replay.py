import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from experiments.r28_landing_replay import evaluator
from tools import r28_landing_replay as replay


class ResultComparisonTest(unittest.TestCase):
    def test_exact_result_is_exact(self):
        value = {"episodes": 18, "protection_passed": False}
        result = replay.compare_result(value, value, b"x", b"x")
        self.assertTrue(result["byte_exact"])
        self.assertTrue(result["physical_result_exact"])

    def test_any_physical_result_change_is_rejected(self):
        expected = {"tool_rows": [{"records": {"r28": {"latency": 10}}}]}
        actual = copy.deepcopy(expected)
        actual["tool_rows"][0]["records"]["r28"]["latency"] = 11
        result = replay.compare_result(actual, expected)
        self.assertFalse(result["physical_result_exact"])
        self.assertEqual(result["difference_paths"],
                         ["/tool_rows/0/records/r28/latency"])


class FixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((replay.ROOT / replay.PLAN).read_bytes())

    def test_frozen_candidate_and_preregistration_are_pinned(self):
        self.assertEqual(self.plan["status"], "FROZEN")
        self.assertEqual(self.plan["arms"]["r28"], replay.CANDIDATE)
        self.assertEqual(self.plan["candidate_freeze"]["candidate_sha256"],
                         replay.CANDIDATE_SHA256)
        self.assertEqual(self.plan["candidate_freeze"]["frozen_dependency_sha256"],
                         replay.DEPENDENCY_SHA256)
        self.assertEqual(self.plan["candidate_freeze"]["pulse_diagnostic_sha256"],
                         replay.DIAGNOSTIC_SHA256)
        self.assertEqual(self.plan["candidate_freeze"]["preregistration_sha256"],
                         replay.CANDIDATE_PLAN_SHA256)

    def test_tool_pair_differs_only_by_offset(self):
        tool, zero = self.plan["known_tool_pair"]
        a = dict(tool)
        b = dict(zero)
        for value in (a, b):
            value.pop("id")
            value.pop("offset")
        self.assertEqual(a, b)
        self.assertEqual(tool["offset"], [0.0, 0.0, 0.029])
        self.assertEqual(zero["offset"], [0.0, 0.0, 0.0])

    def test_positive_controls_are_mass0_and_mass8_y15(self):
        controls = self.plan["positive_controls"]
        self.assertEqual([row["mass"] for row in controls], [0.0, 8.0])
        self.assertTrue(all(row["impact"] == [1.42, [0.0, 15.0, 0.0]]
                            for row in controls))
        self.assertEqual(self.plan["runtime"]["episodes"], 24)


class EvaluatorPrimitiveTest(unittest.TestCase):
    def log(self):
        return {"t": np.arange(20) / 1000.0,
                "detect": np.array([False] * 5 + [True] * 3 + [False] * 12),
                "f_tool": np.array([0.0] * 3 + [2.0] * 6 + [0.0] * 11)}

    def test_real_tool_first_latency_is_reported(self):
        record = evaluator.tool_record(self.log(), 1.0, 300)
        self.assertTrue(record["present"])
        self.assertTrue(record["detected"])
        self.assertEqual(record["first_contact_tick"], 3)
        self.assertEqual(record["first_latency_ms"], 2)

    def test_event_record_requires_quiet_unique_attributable_rise(self):
        reference = self.log()
        reference["detect"] = np.zeros(20, dtype=bool)
        forced = self.log()
        scoring = {"event_window_ms": 10, "reference_twin_ms": 2, "quiet_ms": 2}
        record = evaluator.event_record(forced, reference, 3, scoring)
        self.assertTrue(record["strict"])
        self.assertEqual(record["delays_ms"], [2])


class BundleTest(unittest.TestCase):
    def test_zip_is_deterministic_kit_free_and_contains_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.zip"
            second = Path(folder) / "second.zip"
            a = replay.bundle(first)
            b = replay.bundle(second)
            self.assertEqual(a, b)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                self.assertTrue(all(name.startswith(replay.PREFIX + "/") for name in names))
                self.assertFalse(any("/sim/" in name or "/model/" in name for name in names))
                self.assertFalse(any("r27_loaded_holdout" in name or "r22_normal" in name
                                     for name in names))
                self.assertIn(replay.PREFIX + "/tools/test_r28_landing_replay.py", names)
                bundle = json.loads(archive.read(replay.PREFIX + "/bundle.json"))
                self.assertFalse(bundle["kit_included"])
                self.assertFalse(bundle["results_included"])
                self.assertEqual(bundle["claim"], "opt-in-research-not-validated-default")


if __name__ == "__main__":
    unittest.main()
