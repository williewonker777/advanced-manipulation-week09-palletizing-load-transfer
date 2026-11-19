import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from tools import r24_transfer_replay as replay


class TransferReplayTests(unittest.TestCase):
    def test_closure_is_relative_pinned_and_excludes_kit(self):
        document = replay.closure()
        self.assertEqual(set(document["files"]),
                         set(document["overlay_files"]) | set(document["public_files"]))
        self.assertIn("experiments/r24_transfer/deadband.py", document["overlay_files"])
        self.assertIn("tools/r24_transfer_replay.py", document["public_files"])
        for name in document["files"]:
            path = PurePosixPath(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertFalse(name.startswith(("sim/", "model/")))
            self.assertNotEqual(name, "evaluate.py")

    def test_plan_predeclares_known_new_mechanics_and_signed_twins(self):
        plan = json.loads((replay.ROOT / "experiments/r24_transfer_replay/plan.json").read_text())
        self.assertEqual(plan["status"], "PREDECLARED_BEFORE_NEW_CASE_OUTCOMES")
        self.assertEqual(len(plan["known_replay"]), 3)
        self.assertEqual(len(plan["new_normal"]), 3)
        self.assertTrue(any(row["mechanics"]["table_solref_time_scale"] != 1.0
                            for row in plan["new_normal"]))
        forces = {tuple(row["force_n"]) for row in plan["late_impulses"]}
        self.assertEqual(forces, {(12.0, 0.0, 0.0), (-12.0, 0.0, 0.0),
                                  (0.0, 0.0, 16.0), (0.0, 0.0, -16.0)})

    def test_compact_keeps_inherited_false_and_true_failures(self):
        data = {"classification": "finite", "known_reproduction_passed": True,
                "new_challenge_strict_passed": False, "promotion_eligible": False,
                "known_replay": [], "new_normal": [],
                "late_impulses": [{"id": "x", "collision": [4.18, [1, 0, 0]],
                  "strict_pass": False, "input_integrity": True,
                  "arms": {name: {"all_rises_s": [1.0], "twin_all_rises_s": [1.0],
                          "genuine_rises_s": [], "first_genuine_delay_ms": None,
                          "forced_false_rises_s": [1.0], "twin_false_rises_s": [1.0],
                          "peak_signed_force_n": [1, 0, 0]} for name in ("r10", "r19", "r24")},
                  "candidate_vs": {name: {"gates": {"true_nonloss": False}, "passed": False}
                                   for name in ("r10", "r19")}}]}
        summary = replay.compact(data)
        self.assertEqual(summary["late_impulses"][0]["arms"]["r19"]["twin_false_rises_s"], [1.0])
        self.assertEqual(summary["failures"]["impulse_strict"][0]["r10"], ["true_nonloss"])
        self.assertEqual(summary["failures"]["inherited_true_detection"][0]
                         ["arms_without_genuine_event"], ["r10", "r19"])
        self.assertTrue(summary["failures"]["candidate_true_detection"][0]
                        ["candidate_missing_genuine_event"])

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); kit = root / "kit"; out = root / "out"
            kit.mkdir(); out.mkdir()
            with (patch.object(replay, "ROOT", root),
                  patch.object(replay, "validate_paths", side_effect=ValueError("fresh"))):
                with self.assertRaises(SystemExit):
                    replay.main(["--kit", str(kit), "--out", str(out)])
            self.assertTrue(out.is_dir())


if __name__ == "__main__":
    unittest.main()
