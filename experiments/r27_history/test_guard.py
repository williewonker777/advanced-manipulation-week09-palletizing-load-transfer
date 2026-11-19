import json
from pathlib import Path
import tempfile
import unittest

from experiments.r24_loaded.d4 import load_cases
from experiments.r27_history import guard


class HistoryGuardTests(unittest.TestCase):
    def test_matrix_and_anchor_contract(self):
        cases = load_cases()
        self.assertEqual(len(cases), 385)
        self.assertEqual(guard.check.MATRIX, guard.MATRIX_SHA256)
        self.assertEqual(guard.d4_anchor_ids(cases), [0, 189, 249, 299, 309, 369, 384])

    def test_official_grid_and_displayed_cells(self):
        self.assertEqual(len(guard.official_tasks()), 29)
        cells = {"control": {str(n): ["x"] * 10 for n in (0, 4, 8)},
                 "collisions": [{"cells": ["x"] * 6} for _ in range(6)]}
        self.assertEqual(guard.displayed_cell_count(cells), 66)

    def test_candidate_route_rejects_loaded_holdout_and_non_r27(self):
        for name in ("experiments.r27_loaded_holdout.candidate:Detector",
                     "experiments.r26_yaw_acquisition.candidate:Detector",
                     "experiments.r27_any.candidate:Other"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                guard.source_path(name)

    def test_closure_fails_closed_for_loaded_holdout_member(self):
        candidate = "experiments.r27_history.__init__:Detector"
        candidate_sha = guard.sha(guard.ROOT / "experiments/r27_history/__init__.py")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "closure.json"
            path.write_text(json.dumps({
                "schema": guard.CLOSURE_SCHEMA, "entrypoint": candidate,
                "candidate_sha256": candidate_sha,
                "kit_included": False, "loaded_holdout_included": False,
                "files": {"experiments/r27_history/__init__.py": candidate_sha,
                          "experiments/r27_loaded_holdout/plan.json": "0" * 64},
                "overlay_files": ["experiments/r27_history/__init__.py",
                                  "experiments/r27_loaded_holdout/plan.json"]}))
            with self.assertRaises(ValueError):
                guard.load_candidate_closure(path, candidate, candidate_sha)


if __name__ == "__main__":
    unittest.main()
