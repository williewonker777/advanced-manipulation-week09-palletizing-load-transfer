import importlib.util
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r26_signed_yaw_replay as replay


HAS_COURSE_KIT = importlib.util.find_spec("sim") is not None
if HAS_COURSE_KIT:
    from experiments.r26_yaw_portable import evaluator


@unittest.skipUnless(HAS_COURSE_KIT, "external course kit required")
class PortableEvaluatorTests(unittest.TestCase):
    def test_registration_is_fixed_disclosed_case_and_candidate(self):
        data = evaluator.registration()
        self.assertEqual(data["case"]["id"], "s25_long_transfer_signed")
        self.assertEqual(list(data["variants"]),
                         ["stock", "actuator_shift", "contact_shift"])
        self.assertEqual(data["candidate_sha256"],
                         "a0570079ad5ce8425e9fbdc291b20401a9056b3bca35824237d157ec37a73057")
        self.assertNotIn("independent", " ".join(data).lower())

    def test_twin_attribution_excludes_matched_background_edge(self):
        forced = np.zeros(40, dtype=bool)
        none = np.zeros(40, dtype=bool)
        forced[12:14] = True
        forced[20:22] = True
        none[20:22] = True
        scoring = {"quiet_ms": 5, "reference_twin_ms": 2}
        row = evaluator.event({"detect": forced}, {"detect": none}, 10, 30, scoring)
        self.assertEqual(row["raw_rise_indices"], [12, 20])
        self.assertEqual(row["attributable_rise_indices"], [12])
        self.assertEqual(row["duplicate_rises"], 0)
        self.assertFalse(row["strict"])

    def test_inclusive_intervals_and_rises(self):
        values = [False, True, True, False, True, False]
        self.assertEqual(evaluator.bool_intervals(values), [[1, 2], [4, 4]])
        self.assertEqual(evaluator.rise_indices(values), [1, 4])


class PortableReplayTests(unittest.TestCase):
    def test_no_default_run_and_fixed_command(self):
        catalog = replay.describe()
        self.assertFalse(catalog["default_run"])
        self.assertEqual(catalog["case"], "s25_long_transfer_signed")
        self.assertEqual(replay.command("python", "result.json"),
                         ["python", "-B", "-m",
                          "experiments.r26_yaw_portable.evaluator",
                          "--out", "result.json"])

    def test_closure_is_relative_pinned_and_kit_free(self):
        data = replay.closure()
        self.assertEqual(set(data["files"]),
                         set(data["overlay_files"]) | set(data["public_files"]))
        self.assertIn("experiments/r26_long/candidate.py", data["overlay_files"])
        self.assertIn("experiments/r25_holdout/runner.py", data["overlay_files"])
        for name in data["files"]:
            path = PurePosixPath(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertFalse(name.startswith(("sim/", "model/")))
            self.assertNotEqual(name, "evaluate.py")

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, out = root / "kit", root / "result"
            kit.mkdir()
            out.write_text("sentinel")
            with (patch.object(replay, "ROOT", root),
                  patch.object(replay.shared, "validate_paths",
                               side_effect=ValueError("fresh output required"))):
                with self.assertRaises(SystemExit):
                    replay.main(["replay", "--kit", str(kit), "--out", str(out)])
            self.assertEqual(out.read_text(), "sentinel")

    def test_result_contract_rejects_independent_label(self):
        result = {
            "schema": "r26-known-signed-yaw-diagnostic.v1",
            "classification":
                "Previously disclosed R25 targeted improvement, rejected on older duration guards; not validated for unopened R26 independent conditions.",
            "independent": True,
            "unopened_r26_conditions_used": False,
            "original_course_physics": True,
            "episodes": 12,
            "case": {"id": "s25_long_transfer_signed"},
            "rows": [{"variant": name, "passed": True} for name in
                     ("stock", "actuator_shift", "contact_shift")],
        }
        with self.assertRaises(ValueError):
            replay.validate_result(result)


if __name__ == "__main__":
    unittest.main()
