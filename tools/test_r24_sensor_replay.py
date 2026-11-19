import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from tools import r24_sensor_replay as replay
from tools import r14_paired_events as original_matcher
from experiments.r24_portable import r14_paired_events_shim as restricted_matcher


class SensorReplayTests(unittest.TestCase):
    def test_study_and_arm_mapping_is_explicit(self):
        self.assertEqual(replay.STUDIES,
                         {"target": "negative", "nearby": "nearby", "disclosed": "disclosed"})
        self.assertEqual(replay.ARMS["dc"], "experiments.r24_loaded.dc:Detector")
        self.assertEqual(replay.ARMS["joint"], "experiments.r24_composed.detector:Detector")
        command = replay.command("python", "target", "dc", 1)
        self.assertEqual(command[1:3], ["-B", "-m"])
        self.assertIn("negative", command)

    def test_closure_is_pinned_relative_and_kit_free(self):
        document = replay.closure()
        self.assertEqual(set(document["files"]),
                         set(document["overlay_files"]) | set(document["public_files"]))
        self.assertIn("experiments/r24_loaded/dc.py", document["overlay_files"])
        self.assertIn("tools/r24_sensor_replay.py", document["public_files"])
        self.assertEqual(document["overlay_map"], {
            "tools/r14_paired_events.py":
                "experiments/r24_portable/r14_paired_events_shim.py",
        })
        for name in document["files"]:
            path = PurePosixPath(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertFalse(name.startswith(("sim/", "model/")))
            self.assertNotEqual(name, "evaluate.py")

    def test_restricted_false_event_matcher_is_source_compatible(self):
        cases = [([], []), ([0.1], [0.1]), ([0.11, 0.2], [0.1, 0.201]),
                 ([0.15], [0.1]), ([0.1, 0.11], [0.105])]
        for candidate, reference in cases:
            self.assertEqual(restricted_matcher.match_false_events(candidate, reference),
                             original_matcher.match_false_events(candidate, reference))

    def test_science_and_r23_manifests_are_enforced(self):
        document = replay.closure()
        science = replay.parse_sha256((replay.ROOT / replay.SCIENCE).read_bytes())
        self.assertEqual(science["experiments/r24_loaded/dc.py"],
                         document["files"]["experiments/r24_loaded/dc.py"])
        r23 = json.loads((replay.ROOT / replay.R23_FREE).read_text())
        self.assertEqual(r23["entrypoint"], "experiments.r23_detached.free:Detector")
        self.assertTrue(set(r23["sources"]) <= set(document["files"]))

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); kit = root / "kit"; out = root / "out.json"
            kit.mkdir(); out.write_text("sentinel")
            with (patch.object(replay, "ROOT", root),
                  patch.object(replay, "validate_paths", side_effect=ValueError("fresh"))):
                with self.assertRaises(SystemExit):
                    replay.main(["--kit", str(kit), "--out", str(out),
                                 "--study", "target"])
            self.assertEqual(out.read_text(), "sentinel")


if __name__ == "__main__":
    unittest.main()
