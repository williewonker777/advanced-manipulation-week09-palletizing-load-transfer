import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.r28_memory_cases import evaluator
from tools import r28_memory_cases as replay


class MemoryCaseDeliveryTest(unittest.TestCase):
    def test_summary_requires_both_y_and_exact_four_edge_removal(self):
        def trace(rises, strict=True):
            return {"none": {"rises": rises, "late_false_rises_s": [],
                             "physical_sha256": {"q": "same"}},
                    "force": {"rises": rises, "physical_sha256": {"q": "same"},
                              "force_exact": {"passed": True}},
                    "event": {"strict": strict}}

        tool_records = {name: {"none": {"rises": rises, "late_false_rises_s": [],
                                         "tool": {"detected": True}}}
                        for name, rises in {"c1": [1, 9], "r27_v3": [1, 3, 4, 5, 6, 9],
                                            "r28": [1, 9], "pulse_diagnostic": [1, 9]}.items()}
        quiet_records = {name: {"none": {"rises": [], "late_false_rises_s": [],
                                          "tool": {"detected": False}}}
                         for name in tool_records}
        positives = []
        for name in ("official_y15_mass0", "official_y15_mass8"):
            positives.append({"id": name, "integrity_gates": {"ok": True},
                              "records": {arm: trace([10]) for arm in tool_records}})
        result = {"tool_rows": [
            {"id": "known_5p25kg_29mm_tool", "records": tool_records,
             "integrity_gates": {"ok": True}},
            {"id": "known_5p25kg_friction_matched_zero_offset", "records": quiet_records,
             "integrity_gates": {"ok": True}},
        ], "positive_control_rows": positives}
        summary = evaluator.summarize(result)
        self.assertEqual(summary["removed_v3_tool_edges"], [3, 4, 5, 6])
        self.assertTrue(summary["core_question_passed"])

    def test_zip_is_deterministic_kit_free_and_contains_replay_test(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.zip"
            second = Path(folder) / "second.zip"
            self.assertEqual(replay.bundle(first), replay.bundle(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                self.assertTrue(all(name.startswith(replay.PREFIX + "/") for name in names))
                self.assertFalse(any("/sim/" in name or "/model/" in name for name in names))
                self.assertIn(replay.PREFIX + "/tools/test_r28_memory_cases.py", names)
                bundle = json.loads(archive.read(replay.PREFIX + "/bundle.json"))
                self.assertFalse(bundle["kit_included"])
                self.assertFalse(bundle["results_included"])


if __name__ == "__main__":
    unittest.main()
