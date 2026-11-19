from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from tools import r25_replay as replay


class R25ReplayTests(unittest.TestCase):
    def test_public_studies_and_candidate_entrypoints_are_explicit(self):
        self.assertEqual(replay.COHORTS, ("target", "stress", "nearby", "r23", "r24"))
        self.assertEqual(replay.ARMS, {
            "stream": "experiments.r25_fragment.stream:Detector",
            "pickup": "experiments.r25_pickup.candidate:Detector",
            "joint": "experiments.r25_composed.detector:Detector",
        })
        command = replay.command("python", "target", "stream", 1)
        self.assertEqual(command[1:3], ["-B", "-m"])
        self.assertIn("experiments.r25_fragment.stream:Detector", command)

    def test_closure_is_relative_pinned_and_kit_free(self):
        document = replay.closure()
        self.assertEqual(set(document["files"]),
                         set(document["overlay_files"]) | set(document["public_files"]))
        self.assertIn("experiments/r25_fragment/study.py", document["overlay_files"])
        for name in document["files"]:
            path = PurePosixPath(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertFalse(name.startswith(("sim/", "model/")))
            self.assertNotEqual(name, "evaluate.py")
            self.assertNotIn("r25_holdout", name)

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); kit = root / "kit"; out = root / "result.json"
            kit.mkdir(); out.write_text("sentinel")
            with (patch.object(replay, "ROOT", root),
                  patch.object(replay.shared, "validate_paths", side_effect=ValueError("fresh"))):
                with self.assertRaises(SystemExit):
                    replay.main(["--kit", str(kit), "--out", str(out),
                                 "--study", "target", "--candidate", "stream"])
            self.assertEqual(out.read_text(), "sentinel")

    def test_scientific_regressions_are_reported_not_process_errors(self):
        study = {"schema": "r25-disclosed-original-simulator-comparison.v1",
                 "cohort": "target", "arms": [replay.ARMS["stream"]],
                 "independent": False, "episodes": 8,
                 "regressions": {replay.ARMS["stream"]: [["case", "stock", ["gate"]]]}}
        self.assertIs(replay.validate_study(study, "target", "stream"), study)


if __name__ == "__main__":
    unittest.main()
