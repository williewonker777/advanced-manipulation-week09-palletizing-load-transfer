import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r24_holdout_replay as replay


class HoldoutReplayTests(unittest.TestCase):
    def make_lock(self, path):
        protocol = json.loads((replay.ROOT / replay.HOLDOUT[2]).read_bytes())
        names = ("experiments/r24_composed/__init__.py",
                 "experiments/r24_composed/detector.py",
                 "results/r24/exposed_choice.json")
        document = {
            "schema": "r24-candidate-lock.v1",
            "status": "LOCKED_BEFORE_HOLDOUT_OUTCOMES",
            "entrypoint": "experiments.r24_composed.detector:Detector",
            "plan_sha256": replay.digest(replay.ROOT / replay.HOLDOUT[1]),
            "protocol_sha256": replay.digest(replay.ROOT / replay.HOLDOUT[2]),
            "evaluator_sha256": replay.digest(replay.ROOT / replay.HOLDOUT[3]),
            "frozen_protocol_sources": protocol["source_pins"],
            "transitive_sources": {name: replay.digest(replay.ROOT / name) for name in names},
        }
        path.write_text(json.dumps(document))
        return document

    def test_commands_are_isolated_repeatable_and_worker_capped(self):
        one = replay.commands("python", False)
        two = replay.commands("python", True)
        self.assertEqual([name for _, name in one], ["first.log", "verify.log"])
        self.assertEqual([name for _, name in two],
                         ["first.log", "repeat.log", "verify.log"])
        self.assertTrue(all(command[1] == "-B" for command, _ in two))
        self.assertTrue(all("{jobs}" in command for command, _ in two[:-1]))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                replay.main(["--kit", "kit", "--lock", "lock", "--out", "out",
                             "--jobs", "3"])

    def test_absolute_strings_are_rejected_recursively(self):
        self.assertTrue(replay.contains_absolute_string({"x": ["/" + "absolute-sentinel"]}))
        self.assertFalse(replay.contains_absolute_string(
            {"x": ["experiments/r24_composed/detector.py"]}))

    def test_portable_lock_and_snapshot_cover_candidate_and_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock.json"
            document = self.make_lock(path)
            _, loaded = replay.read_lock(path)
            self.assertEqual(loaded, document)
            blobs, pins = replay.snapshot(loaded)
            self.assertTrue(set(document["transitive_sources"]) <= set(blobs))
            self.assertEqual(set(blobs), set(pins))
            self.assertIn("experiments/r24_holdout/evaluator.py", blobs)
            self.assertIn("experiments/r24_composed/detector.py", blobs)
            self.assertFalse(any(name.startswith(("sim/", "model/")) for name in blobs))

    def test_read_lock_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock.json"
            document = self.make_lock(path)
            name = next(iter(document["transitive_sources"]))
            document["transitive_sources"][name] = "0" * 64
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "source pin differs"):
                replay.read_lock(path)

    def test_main_never_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, out, candidate = root / "kit", root / "sentinel", root / "lock.json"
            kit.mkdir()
            out.mkdir()
            candidate.write_text("{}")
            with patch.object(replay.shared, "validate_paths", side_effect=ValueError("fresh")):
                with self.assertRaises(SystemExit):
                    replay.main(["--kit", str(kit), "--lock", str(candidate),
                                 "--out", str(out)])
            self.assertTrue(out.is_dir())


if __name__ == "__main__":
    unittest.main()
