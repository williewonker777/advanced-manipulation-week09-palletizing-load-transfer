import json
from pathlib import Path
import tempfile
import unittest

from experiments.r15_early_com.stock_check import cases, ratios, summarize
from tools.r15_early_check import EARLY_SHA256, validate_early_calibration, validate_paths


class Tests(unittest.TestCase):
    def test_matrix_and_exposed_cases_are_fixed(self):
        rows = cases()
        self.assertEqual(len(rows), 22)
        self.assertEqual(sum(row["id"].startswith("ec") for row in rows), 20)
        self.assertEqual({row["id"] for row in rows[-2:]}, {"sh3", "sh5"})
        self.assertIsNone(rows[0]["offset"])
        self.assertEqual(rows[1]["offset"], [0.0, 0.0, 0.0])

    def test_negative_qualification_is_a_valid_summary(self):
        rows = [{"id": f"ec{i}", "passed": i != 1,
                 "gates": {"G4": True, "cost_r10": i != 1}} for i in range(20)]
        summary = summarize(rows)
        self.assertFalse(summary["qualified"])
        self.assertEqual(summary["failures"], [{"id": "ec1", "gates": ["cost_r10"]}])

    def test_cost_ratios_include_pick_and_place_steps(self):
        candidate = {"cost": {"pick_step": 2.0, "place_step": 3.0}}
        baseline = {"cost": {"pick_step": 1.0, "place_step": 2.0}}
        self.assertEqual(ratios(candidate, baseline), {"pick_step": 2.0, "place_step": 1.5})

    def test_bundled_calibration_is_exact_and_malformed_rejected(self):
        path = Path(__file__).resolve().parents[1] / "results/r15/early_stock_calibration.json"
        blob = path.read_bytes()
        self.assertEqual(__import__("hashlib").sha256(blob).hexdigest(), EARLY_SHA256)
        self.assertEqual(validate_early_calibration(blob)["schema"], "r15-early-com-calibration.v1")
        with self.assertRaises(ValueError):
            validate_early_calibration(blob + b" ")
        with self.assertRaises(ValueError):
            validate_early_calibration(json.dumps({"schema": "r15-early-com-calibration.v1"}).encode())

    def test_existing_output_and_symlink_calibration_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit = root / "kit"
            kit.mkdir(); (kit / "sim").mkdir(); (kit / "model").mkdir(); (kit / "evaluate.py").touch()
            height = root / "height.json"; height.write_text("{}")
            output = root / "out.json"; output.write_text("{}")
            with self.assertRaises(ValueError):
                validate_paths(kit, height, output)
            output.unlink(); link = root / "height-link.json"; link.symlink_to(height)
            with self.assertRaises(ValueError):
                validate_paths(kit, link, output)


if __name__ == "__main__":
    unittest.main()
