import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r22_challenge_replay as replay


class ChallengeReplayTests(unittest.TestCase):
    LOCK = replay.ROOT / "results/r22/holdout_selected.lock.json"

    def test_commands_are_isolated_and_repeat_is_explicit(self):
        one = replay.commands("python", False)
        two = replay.commands("python", True)
        self.assertEqual([name for _, name in one], ["first.log", "verify.log"])
        self.assertEqual([name for _, name in two], ["first.log", "repeat.log", "verify.log"])
        self.assertTrue(all(command[1] == "-B" for command, _ in two))
        self.assertNotIn("--repeat", one[-1][0])
        self.assertIn("--repeat", two[-1][0])

    def test_absolute_strings_are_rejected_recursively(self):
        self.assertTrue(replay.contains_absolute_string({"x": ["/" + "absolute-sentinel"]}))
        self.assertFalse(replay.contains_absolute_string({"x": ["experiments/r22_light/candidate.py"]}))

    def test_current_archived_lock_is_portable_and_pinned(self):
        raw, document = replay.read_lock(self.LOCK)
        self.assertFalse(replay.contains_absolute_string(document))
        self.assertEqual(document["entrypoint"], "experiments.r22_light.candidate:Detector")
        self.assertEqual(replay.digest(raw), "c8b8a646dd3d4153436d8acc2f737be4c29b917b8897b6bdc40cb72828d0a0a1")

    def test_snapshot_covers_lock_and_sealed_protocol(self):
        _, document = replay.read_lock(self.LOCK)
        blobs, pins = replay.snapshot(document)
        self.assertTrue(set(document["candidate_sources"]) <= set(blobs))
        self.assertEqual(set(blobs), set(pins))
        self.assertIn("experiments/r22_challenge/evaluator.py", blobs)
        self.assertIn("experiments/r15_delayed_wrench/observer.py", blobs)
        self.assertFalse(any(name.startswith(("sim/", "model/")) for name in blobs))

    def test_read_lock_rejects_changed_source(self):
        data = json.loads(self.LOCK.read_bytes())
        name = next(iter(data["candidate_sources"]))
        data["candidate_sources"][name] = "0" * 64
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "lock.json"
            changed.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "source pin differs"):
                replay.read_lock(changed)

    def test_main_never_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, out, lock = root / "kit", root / "sentinel", root / "lock.json"
            kit.mkdir(); out.mkdir(); lock.write_text("{}")
            with patch.object(replay.shared, "validate_paths", side_effect=ValueError("fresh")):
                with self.assertRaises(SystemExit):
                    replay.main(["--kit", str(kit), "--lock", str(lock), "--out", str(out)])
            self.assertTrue(out.is_dir())


if __name__ == "__main__":
    unittest.main()
