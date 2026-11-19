import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from experiments.r30_g4_replay.study import contact_intervals, edge_attribution
from tools import r30_g4_replay as replay


class R30G4ReplayTest(unittest.TestCase):
    def test_contact_bounds_and_edge_attribution_are_inclusive(self):
        intervals = contact_intervals([0, 2, 3, 0, 0, 4, 0])
        self.assertEqual(intervals, [[1, 2], [5, 5]])
        self.assertEqual(
            [row["relative_to_tool"] for row in edge_attribution([0, 1, 4, 6], intervals)],
            ["before_first_contact", "during_contact", "between_contact_intervals",
             "after_last_contact"],
        )

    def test_scientific_difference_is_not_normalized(self):
        expected = {"episodes": 16, "rows": [{"records": {"unload": {"rises": [1353]}}}]}
        actual = copy.deepcopy(expected)
        actual["rows"][0]["records"]["unload"]["rises"] = [1318]
        comparison = replay.compare_result(actual, expected)
        self.assertFalse(comparison["scientific_exact"])
        self.assertEqual(comparison["difference_paths"],
                         ["/rows/0/records/unload/rises/0"])

    def test_bundle_is_deterministic_kit_result_and_unopened_free(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.zip"
            second = Path(folder) / "second.zip"
            self.assertEqual(replay.bundle(first), replay.bundle(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                self.assertTrue(all(name.startswith(replay.PREFIX + "/") for name in names))
                self.assertFalse(any("/sim/" in name or "/model/" in name
                                     or "r27_loaded_holdout" in name
                                     or "r28_landing_prospect" in name
                                     or "/fresh/" in name
                                     or "/result.json" in name for name in names))
                bundle = json.loads(archive.read(replay.PREFIX + "/bundle.json"))
                self.assertFalse(bundle["kit_included"])
                self.assertFalse(bundle["results_included"])
                self.assertFalse(bundle["unopened_conditions_included"])
                readme = archive.read(replay.PREFIX + "/README.md").decode()
                self.assertIn("rejected early-onset", readme)


if __name__ == "__main__":
    unittest.main()
