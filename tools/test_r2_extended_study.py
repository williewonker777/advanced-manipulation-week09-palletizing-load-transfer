"""Pure builder and gate tests; never opens the real R2 holdout."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r2_extended_study as E
from tools import r2_study as R


class ExtendedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(E.PROTOCOL.read_text())
        cls.spec = next(item for item in R.specs() if item["id"] == "R1")

    def test_development_balance(self):
        rows = E.development_cases(self.protocol, self.spec, "servo")
        self.assertEqual(len(rows), 26)
        collisions = rows[2:]
        self.assertEqual({direction: sum(r["direction"] == direction for r in collisions)
                          for direction in self.protocol["development"]["directions"]},
                         {direction: 3 for direction in self.protocol["development"]["directions"]})
        self.assertEqual({force: sum(r["force_n"] == force for r in collisions)
                          for force in self.protocol["development"]["forces_N"]},
                         {9.: 12, 23.: 12})
        self.assertEqual({time: sum(r["collision"][0] == time for r in collisions)
                          for time in self.protocol["development"]["times_s"]},
                         {time: 8 for time in self.protocol["development"]["times_s"]})
        joint = {(force, mass): sum(r["force_n"] == force and r["mass"] == mass
                                    for r in collisions)
                 for force in self.protocol["development"]["forces_N"]
                 for mass in (1.35, 7.25)}
        self.assertEqual(set(joint.values()), {6})
        for plus, minus in (("+x", "-x"), ("+y", "-y"), ("+z", "-z"), ("+xy", "-xy")):
            signature = lambda direction: [(r["collision"][0], r["force_n"], r["mass"])
                                            for r in collisions if r["direction"] == direction]
            self.assertEqual(signature(plus), signature(minus))

    def test_d4_exact_shape(self):
        rows = E.d4_cases(self.spec, "torque")
        self.assertEqual(len(rows), 385)
        self.assertEqual(sum(r["kind"] == "reference" for r in rows), 189)
        self.assertEqual(sum(r["kind"] == "collision" for r in rows), 180)
        self.assertEqual(sum(r["kind"] == "offset" for r in rows), 16)

    def test_fake_holdout_design_only(self):
        # Deliberately not the registered real seed.
        rows = E.build_holdout_design(self.protocol, np.random.default_rng(12345))
        self.assertEqual(len(rows), 38)
        self.assertEqual(sum(r["kind"] == "reference" for r in rows), 4)
        self.assertEqual(sum(r["kind"] == "collision" for r in rows), 30)
        self.assertEqual(sum(r["kind"] == "offset" for r in rows), 4)
        collisions = [r for r in rows if r["kind"] == "collision"]
        self.assertEqual(sorted(r["force_n"] for r in collisions),
                         sorted(self.protocol["holdout"]["forces_N"] * 6))
        self.assertEqual(sorted(r["direction"] for r in collisions),
                         sorted(self.protocol["holdout"]["directions"] * 3))

    def test_freeze_required_before_marker_or_rng(self):
        with tempfile.TemporaryDirectory() as name:
            missing = Path(name) / "missing.json"
            marker = Path(name) / "marker.json"
            with self.assertRaises(FileNotFoundError):
                E.open_holdout(self.protocol, [self.spec], marker=marker, freeze_path=missing)
            self.assertFalse(marker.exists())

    def test_required_freeze_paths_follow_selected_families_and_model(self):
        c2 = next(item for item in R.specs() if item["family"] == "C2")
        c5 = next(item for item in R.specs() if item["family"] == "C5")
        paths = E._required_freeze_paths(
            [self.spec, c2, c5], {"ridge_model": "models/frozen.json"})
        self.assertIn("tools/r2_extended_study.py", paths)
        self.assertIn("sim/config.py", paths)
        self.assertIn("experiments/r2_contact.py", paths)
        self.assertIn("experiments/r2_prne.py", paths)
        self.assertIn("models/frozen.json", paths)
        self.assertNotIn("experiments/r2_robust.py", paths)

    def test_exclusive_marker_blocks_second_attempt(self):
        with tempfile.TemporaryDirectory() as name:
            marker = Path(name) / "opened.json"
            freeze = {"local_commit": "a" * 40}
            E._exclusive_marker(marker, freeze, [self.spec])
            with self.assertRaises(RuntimeError):
                E._exclusive_marker(marker, freeze, [self.spec])
            self.assertEqual(json.loads(marker.read_text())["batch_attempt"], 1)

    def test_selection_requires_exact_preregistered_spec(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "selection.json"
            path.write_text(json.dumps([self.spec]))
            self.assertEqual(E.load_selection(path), [self.spec])
            path.write_text(json.dumps([{**self.spec, "extra": 1}]))
            with self.assertRaises(ValueError):
                E.load_selection(path)

    def test_nominal_parity_comparator(self):
        log = {key: np.arange(3) for key in
               ("t", "e_mm", "q", "q_d", "tau", "f_table", "f_tool", "f_ext", "r", "detect")}
        result = {"rms_mm": 1., "ctrl_time_us": 7.}
        evidence = E._parity_result(result, dict(result), log, dict(log))
        self.assertTrue(evidence["passed"])
        changed = dict(log); changed["tau"] = np.arange(3) + 1
        self.assertFalse(E._parity_result(result, dict(result), log, changed)["passed"])


if __name__ == "__main__":
    unittest.main()
