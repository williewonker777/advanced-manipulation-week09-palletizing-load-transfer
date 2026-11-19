"""Unit tests for the public R10 startup measurement tool."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from tools.r10_startup import (conservative_summary, measurement_order,
                               validate_inputs, _validated_measurement)


class StartupTests(unittest.TestCase):
    def test_direct_file_bootstraps_repository_imports_without_pythonpath(self):
        tool = Path(__file__).with_name("r10_startup.py").resolve()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as folder:
            # A structurally plausible but empty kit gets as far as the
            # checked-overlay code, from a true direct-file invocation.
            root = Path(folder)
            kit = root / "kit"
            for sub in ("sim", "model"):
                (kit / sub).mkdir(parents=True)
            (kit / "evaluate.py").touch()
            result = subprocess.run([sys.executable, str(tool), "--kit", str(kit),
                                     "--out", str(root / "result.json"), "--pairs", "1"],
                                    cwd=folder, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Kit integrity FAILED", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse((root / "result.json").exists())

    def test_alternating_order(self):
        self.assertEqual(measurement_order(4), [
            "FAST2", "current", "current", "FAST2",
            "FAST2", "current", "current", "FAST2"])
        for bad in (True, 0, -1, 33, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                measurement_order(bad)

    def test_conservative_extrema_arithmetic(self):
        rows = [
            {"variant": "FAST2", "headline_ms": 12.},
            {"variant": "current", "headline_ms": 5.},
            {"variant": "current", "headline_ms": 6.},
            {"variant": "FAST2", "headline_ms": 10.},
        ]
        result = conservative_summary(rows)
        self.assertEqual(result["FAST2"], {"minimum_ms": 10., "maximum_ms": 12.})
        self.assertEqual(result["current"], {"minimum_ms": 5., "maximum_ms": 6.})
        self.assertAlmostEqual(result["conservative_speedup_FAST2_over_current"], 10 / 6)
        with self.assertRaises(ValueError):
            conservative_summary(rows[:-1])
        with self.assertRaises(ValueError):
            conservative_summary([{"variant": "unknown", "headline_ms": 1.}])

    def test_worker_identity_and_headline_mismatch(self):
        record = {"variant": "FAST2", "detector_sha256": "d", "controller_sha256": "c",
                  "preimport_numeric_modules": [], "detector_import_ms": 2., "setup_ms": 9.,
                  "first_reset_ms": 3., "headline_ms": 5., "worker_total_ms": 14.}
        self.assertIs(_validated_measurement(record, "FAST2", "d", "c"), record)
        for field, value in (("variant", "current"), ("detector_sha256", "wrong"),
                             ("controller_sha256", "wrong"),
                             ("preimport_numeric_modules", ["numpy"]),
                             ("headline_ms", 6.), ("first_reset_ms", -1.)):
            changed = dict(record, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validated_measurement(changed, "FAST2", "d", "c")

    def test_input_validity_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kit, output_dir = root / "kit", root / "out"
            kit.mkdir(); output_dir.mkdir()
            (kit / "sim").mkdir(); (kit / "model").mkdir(); (kit / "evaluate.py").touch()
            out = output_dir / "new.json"
            resolved = validate_inputs(kit, out, 4, None)
            self.assertEqual(resolved[:2], (kit.resolve(), out.resolve()))
            if resolved[2]:
                self.assertEqual(validate_inputs(kit, out, 4, resolved[2][0])[2], resolved[2])
                with self.assertRaises(ValueError):
                    validate_inputs(kit, out, 4, max(resolved[2]) + 1)
            out.write_text("keep")
            with self.assertRaises(ValueError):
                validate_inputs(kit, out, 4, None)
            with self.assertRaises(ValueError):
                validate_inputs(kit, kit / "inside.json", 4, None)
            out.unlink(); (kit / "evaluate.py").unlink()
            with self.assertRaises(ValueError):
                validate_inputs(kit, out, 4, None)


if __name__ == "__main__":
    unittest.main()
