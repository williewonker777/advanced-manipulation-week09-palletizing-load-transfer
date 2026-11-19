"""R33 public replay selection, source closure and safe output contract."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from experiments.r33_vertical_replay import runner
from tools import r33_vertical_replay as replay


class RunnerContract(unittest.TestCase):
    def test_vertical_scopes_have_matched_no_force_inputs(self):
        for scope, collisions in (("smoke", 3), ("axial19", 19)):
            cases = runner.selected_cases(scope)
            self.assertEqual(len(cases), collisions * 2)
            self.assertEqual(sum(case["kind"] == "collision" for case in cases), collisions)
            refs = {runner.physical.reference_key(case) for case in cases if case["kind"] == "reference"}
            self.assertTrue(all(runner.physical.reference_key(case) in refs
                                for case in cases if case["kind"] == "collision"))
        self.assertEqual(len(runner.selected_cases("full-d4")), 385)

    def test_closure_members_are_regular_and_pinned(self):
        raw, files, overlay = replay.load()
        obj = json.loads(raw)
        self.assertFalse(obj["kit_included"])
        self.assertEqual(set(obj["files"]), set(files) - set(replay.parent.load()[1]) - {replay.parent.CLOSURE})
        self.assertNotIn("tools/r33_vertical_replay.py", overlay)
        self.assertTrue(set(replay.CORE) <= set(overlay))

    def test_source_bundle_is_deterministic_and_kit_free(self):
        with TemporaryDirectory() as folder:
            first, second = (Path(folder) / name for name in ("first.zip", "second.zip"))
            self.assertEqual(replay.bundle(first), replay.bundle(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = {name.removeprefix(replay.PREFIX + "/")
                         for name in archive.namelist()}
            self.assertFalse(any(Path(name).parts[0] in {"sim", "model"}
                                 or Path(name).name in {"run.py", "evaluate.py",
                                                       "servo_controller.py", "torque_controller.py",
                                                       "collision_detector.py"}
                                 for name in names))

    def test_non_destructive_output(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            occupied = root / "occupied"
            occupied.mkdir()
            args = type("Args", (), {"kit": root / "not-a-kit", "out": occupied,
                                      "scope": "smoke", "jobs": 1})()
            with self.assertRaises(ValueError):
                replay.compare(args)
            self.assertEqual(list(occupied.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
