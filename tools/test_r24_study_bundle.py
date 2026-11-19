import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import r24_study_bundle as bundle


class StudyBundleTests(unittest.TestCase):
    def test_stage_is_complete_relative_and_kit_free(self):
        document = bundle.stage()
        self.assertIn("tools/r24_sensor_replay.py", document["files"])
        self.assertIn("tools/r24_transfer_replay.py", document["files"])
        self.assertIn("tools/r24_holdout_replay.py", document["files"])
        self.assertIn("results/r24/independent_lock.json", document["files"])
        for name in document["files"]:
            self.assertTrue(bundle.relative_name(name))
            self.assertFalse(name.startswith(("sim/", "model/")))
            self.assertNotEqual(name, "evaluate.py")

    def test_two_builds_are_byte_identical_and_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "a.zip"
            second = Path(folder) / "b.zip"
            bundle.build(first); bundle.build(second)
            self.assertEqual(hashlib.sha256(first.read_bytes()).digest(),
                             hashlib.sha256(second.read_bytes()).digest())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                receipt = json.loads(archive.read(bundle.PREFIX + "/receipt.json"))
                readme = archive.read(bundle.PREFIX + "/README.md").decode()
            self.assertEqual(names, sorted(names))
            self.assertEqual(receipt["selection_status"], "pending")
            self.assertIsNone(receipt["selected_arm"])
            self.assertFalse(receipt["independent_evaluation"])
            self.assertTrue(receipt["independent_replay_available"])
            self.assertIn("new challenge did not pass strictly", readme)
            self.assertIn("tools/r24_holdout_replay.py", readme)
            self.assertFalse(any(name.endswith(("/evaluate.py", "/sim", "/model"))
                                 for name in names))

    def test_valid_disclosure_is_explicit_research_only(self):
        selection = {
            "schema": "week09-r24-research-disclosure.v1",
            "selected_for_independent": "joint",
            "entrypoint": "experiments.r24_composed.detector:Detector",
            "exposed_choice_sha256": "a" * 64,
            "independent": {"verdict": "FAIL", "rows": 7, "failed_rows": 2,
                            "episodes": 14, "result_sha256": "b" * 64},
            "default_promoted": False,
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); selected = root / "selection.json"; out = root / "bundle.zip"
            selected.write_text(json.dumps(selection) + "\n")
            bundle.build(out, selected)
            with zipfile.ZipFile(out) as archive:
                receipt = json.loads(archive.read(bundle.PREFIX + "/receipt.json"))
                readme = archive.read(bundle.PREFIX + "/README.md").decode()
                included = archive.read(bundle.PREFIX + "/selection.json")
            self.assertEqual(receipt["selected_arm"], "joint")
            self.assertEqual(receipt["independent"]["failed_rows"], 2)
            self.assertEqual(receipt["selection_sha256"], hashlib.sha256(included).hexdigest())
            self.assertFalse(receipt["default_promoted"])
            self.assertIn("**FAIL**: 7 rows, 2 failed rows, and 14 episodes", readme)
            self.assertIn("research-only evidence, not qualification", readme)
            self.assertIn("--study target --arm joint", readme)

    def test_disclosure_cannot_call_failed_rows_a_pass(self):
        selection = {
            "schema": "week09-r24-research-disclosure.v1",
            "selected_for_independent": "dc",
            "entrypoint": "experiments.r24_loaded.dc:Detector",
            "exposed_choice_sha256": "a" * 64,
            "independent": {"verdict": "PASS", "rows": 1, "failed_rows": 1,
                            "episodes": 1, "result_sha256": "b" * 64},
            "default_promoted": False,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selection.json"
            path.write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                bundle.disclosure(path, bundle.stage())


if __name__ == "__main__":
    unittest.main()
