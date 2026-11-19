import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.r26_challenge_portable import support
from tools import r26_challenge_replay as replay


class PortableReplayContractTests(unittest.TestCase):
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
                replay.main(["--kit", "kit", "--selection", "choice",
                             "--lock", "lock", "--closure", "closure",
                             "--out", "out", "--jobs", "3"])

    def test_runtime_gate_requires_course_mujoco(self):
        fake_mujoco = type("Versioned", (), {"__version__": "3.11.0"})
        fake_numpy = type("Versioned", (), {"__version__": "2.5.4"})
        with patch.dict("sys.modules", {"mujoco": fake_mujoco, "numpy": fake_numpy}):
            with self.assertRaisesRegex(ValueError, "MuJoCo 3.17.0"):
                support.runtime_versions()

    def test_missing_or_stale_release_fails_before_candidate_import(self):
        with tempfile.TemporaryDirectory(dir=support.ROOT) as folder:
            root = Path(folder)
            paths = [root / name for name in ("selection.json", "lock.json", "closure.json")]
            for path in paths:
                path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "selection.*absent|stale"):
                support.read_release(*paths)

    def test_nonportable_contract_is_rejected_recursively(self):
        self.assertTrue(support.contains_absolute_string(
            {"source": ["/" + "machine-specific-sentinel"]}))
        self.assertFalse(support.contains_absolute_string(
            {"source": ["experiments/r26_candidate/detector.py"]}))

    def test_expected_result_requires_complete_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "expected.json"
            path.write_text(json.dumps({"schema": "wrong", "episodes": 156}))
            with self.assertRaisesRegex(ValueError, "contract differs"):
                replay.regular_expected(path)

    def test_integration_schemas_and_holdout_are_fixed(self):
        self.assertEqual(support.SELECTION_SCHEMA, "r26-finalized-exposed-choice.v1")
        self.assertEqual(support.LOCK_SCHEMA, "r26-candidate-lock.v1")
        self.assertEqual(support.CLOSURE_SCHEMA, "r26-independent-authored-closure.v1")
        self.assertEqual(len(support.HOLDOUT), 7)


if __name__ == "__main__":
    unittest.main()
