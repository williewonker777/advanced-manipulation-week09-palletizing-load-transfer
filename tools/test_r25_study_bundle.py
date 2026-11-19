import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import r25_study_bundle as bundle


class R25StudyBundleTests(unittest.TestCase):
    def test_stage_is_final_relative_and_kit_free(self):
        document = bundle.stage()
        self.assertEqual(document["selection"]["path"], "results/r25/selection.json")
        self.assertEqual(document["independent"]["replay_tool"],
                         "tools/r25_holdout_replay.py")
        for name in document["files"]:
            self.assertTrue(bundle.relative_name(name))
            self.assertFalse(name.startswith(("sim/", "model/")))
            self.assertNotEqual(name, "evaluate.py")

    def test_build_is_byte_identical_disclosed_and_kit_free(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "a.zip"; second = Path(folder) / "b.zip"
            bundle.build(first); bundle.build(second)
            self.assertEqual(hashlib.sha256(first.read_bytes()).digest(),
                             hashlib.sha256(second.read_bytes()).digest())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                receipt = json.loads(archive.read(bundle.PREFIX + "/receipt.json"))
                readme = archive.read(bundle.PREFIX + "/README.md").decode()
            self.assertEqual(names, sorted(names))
            self.assertEqual(receipt["selection_status"], "disclosed")
            self.assertEqual(receipt["selected_candidate"], "joint")
            self.assertTrue(receipt["independent_evaluation"])
            self.assertEqual(receipt["independent"]["verdict"], "FAIL")
            self.assertIn("**FAIL**", readme)
            self.assertFalse(any(name.endswith(("/evaluate.py", "/sim", "/model"))
                                 for name in names))

    def test_final_disclosure_reports_negative_outcome_without_qualification(self):
        document = bundle.stage()
        document = {**document, "files": {**document["files"],
                    "experiments/r25_composed/detector.py": "0" * 64}}
        selection = {
            "schema": "week09-r25-research-disclosure.v1",
            "selected_for_independent": "joint",
            "entrypoint": "experiments.r25_composed.detector:Detector",
            "exposed_choice_sha256": "a" * 64,
            "independent": {"verdict": "FAIL", "rows": 30, "failed_rows": 7,
                            "episodes": 198, "result_sha256": "b" * 64},
            "default_promoted": False,
        }
        loaded = bundle.disclosure(json.dumps(selection).encode(), document)
        text = bundle.readme(loaded)
        self.assertIn("**FAIL**: 30 rows, 7 failed rows, and 198 episodes", text)
        self.assertIn("not qualification or default promotion", text)
        self.assertIn("MuJoCo 3.17.0", text)
        self.assertIn("MuJoCo 3.11.0", text)
        self.assertIn("tools/r25_holdout_replay.py", text)

    def test_disclosure_rejects_pass_with_failed_rows(self):
        document = bundle.stage()
        selection = {
            "schema": "week09-r25-research-disclosure.v1",
            "selected_for_independent": "joint",
            "entrypoint": "experiments.r25_composed.detector:Detector",
            "exposed_choice_sha256": "a" * 64,
            "independent": {"verdict": "PASS", "rows": 1, "failed_rows": 1,
                            "episodes": 1, "result_sha256": "b" * 64},
            "default_promoted": False,
        }
        with self.assertRaises(ValueError):
            bundle.disclosure(json.dumps(selection).encode(), document)


if __name__ == "__main__":
    unittest.main()
