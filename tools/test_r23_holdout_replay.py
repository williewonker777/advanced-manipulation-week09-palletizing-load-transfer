import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r23_holdout_replay as replay


class HoldoutReplayTests(unittest.TestCase):
    LOCK = replay.ROOT / "results/r23/independent.lock.json"

    def test_commands_are_isolated_repeatable_and_worker_capped(self):
        one = replay.commands("python", False)
        two = replay.commands("python", True)
        self.assertEqual([name for _, name in one], ["first.log", "verify.log"])
        self.assertEqual([name for _, name in two],
                         ["first.log", "repeat.log", "verify.log"])
        self.assertTrue(all(command[1] == "-B" for command, _ in two))
        self.assertNotIn("--repeat", one[-1][0])
        self.assertIn("--repeat", two[-1][0])
        self.assertTrue(all("{jobs}" in command for command, _ in two[:-1]))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                replay.main(["--kit", "kit", "--lock", "lock", "--out", "out",
                             "--jobs", "3"])

    def test_absolute_strings_are_rejected_recursively(self):
        self.assertTrue(replay.contains_absolute_string({"x": ["/" + "absolute-sentinel"]}))
        self.assertFalse(replay.contains_absolute_string(
            {"x": ["experiments/r23_detached/free.py"]}))

    def test_current_archived_lock_is_portable_and_pinned(self):
        raw, document = replay.read_lock(self.LOCK)
        self.assertFalse(replay.contains_absolute_string(document))
        self.assertEqual(document["entrypoint"],
                         "experiments.r23_detached.free:Detector")
        self.assertEqual(replay.digest(raw),
                         "bcfdeace274ae7f24a11d2d8d63824304c312b7a0dd447c55953995fe7ea22d8")

    def test_snapshot_covers_lock_and_protocol_without_kit_sources(self):
        _, document = replay.read_lock(self.LOCK)
        blobs, pins = replay.snapshot(document)
        self.assertTrue(set(document["transitive_sources"]) <= set(blobs))
        self.assertEqual(set(blobs), set(pins))
        self.assertIn("experiments/r23_holdout/evaluator.py", blobs)
        self.assertIn("experiments/r23_detached/free.py", blobs)
        self.assertIn("results/r23/exposed_choice.json", blobs)
        self.assertFalse(any(name.startswith(("sim/", "model/")) for name in blobs))

    def test_read_lock_rejects_changed_source(self):
        data = json.loads(self.LOCK.read_bytes())
        name = next(iter(data["transitive_sources"]))
        data["transitive_sources"][name] = "0" * 64
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "lock.json"
            changed.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "source pin differs"):
                replay.read_lock(changed)

    def test_main_never_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, out, lock = root / "kit", root / "sentinel", root / "lock.json"
            kit.mkdir()
            out.mkdir()
            lock.write_text("{}")
            with patch.object(replay.shared, "validate_paths", side_effect=ValueError("fresh")):
                with self.assertRaises(SystemExit):
                    replay.main(["--kit", str(kit), "--lock", str(lock),
                                 "--out", str(out)])
            self.assertTrue(out.is_dir())


if __name__ == "__main__":
    unittest.main()
