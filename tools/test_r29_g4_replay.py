import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import r29_g4_replay as replay


class R29G4ReplayTest(unittest.TestCase):
    def test_scientific_difference_is_never_normalized(self):
        expected = {"episodes": 54, "passed": False,
                    "positive_gates": [{"id": "negative-z", "absolute_miss": True}]}
        actual = copy.deepcopy(expected)
        actual["positive_gates"][0]["absolute_miss"] = False
        comparison = replay.compare_result(actual, expected)
        self.assertFalse(comparison["scientific_exact"])
        self.assertEqual(comparison["difference_paths"],
                         ["/positive_gates/0/absolute_miss"])

    def test_bundle_is_deterministic_kit_free_rejected_and_prospect_free(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.zip"
            second = Path(folder) / "second.zip"
            self.assertEqual(replay.bundle(first), replay.bundle(second))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                self.assertTrue(all(name.startswith(replay.PREFIX + "/") for name in names))
                self.assertFalse(any("/sim/" in name or "/model/" in name
                                     or "prospect" in name or "r22_normal" in name
                                     for name in names))
                bundle = json.loads(archive.read(replay.PREFIX + "/bundle.json"))
                self.assertFalse(bundle["kit_included"])
                self.assertFalse(bundle["results_included"])
                self.assertFalse(bundle["prospect_included"])
                readme = archive.read(replay.PREFIX + "/README.md").decode()
                self.assertIn("rejected research mechanism", readme)


if __name__ == "__main__":
    unittest.main()
