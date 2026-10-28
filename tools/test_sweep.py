"""Unit tests for tools.sweep without running MuJoCo episodes."""

import argparse
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import sweep


class MetricTests(unittest.TestCase):
    def test_control_windows_match_evaluate(self):
        t = np.arange(0.0, 5.0, 0.001)
        tau = np.zeros((len(t), 6))
        tau[1350:, 0] = 7.0
        tau[3700:, 1] = -9.0
        f_table = np.zeros(len(t))
        f_table[3800] = 123.0
        result = {
            "rms_mm": 1.0,
            "max_mm": 2.0,
            "saturation_pct": 3.0,
            "diverged": False,
            "false_alarm_times": [1.35, 1.599, 1.6, 3.7, 4.099, 4.1, 2.0],
        }
        metrics = sweep.evaluate_control_metrics(
            result, {"t": t, "tau": tau, "f_table": f_table}
        )
        self.assertEqual(metrics["torque_step_pick_nm"], 7.0)
        self.assertEqual(metrics["torque_step_place_nm"], 9.0)
        self.assertEqual(metrics["table_force_place_max_n"], 123.0)
        self.assertEqual(
            (metrics["false_alarms_pick"], metrics["false_alarms_place"], metrics["false_alarms_other"]),
            (2, 2, 3),
        )

    def test_collision_classification_precedence_and_reference_window(self):
        reference = np.zeros(5000, dtype=bool)
        reference[1440] = True
        alarm = {"alarm_before_collision": True, "detect_delay_ms": None, "false_alarms": 1}
        self.assertEqual(
            sweep.classify_collision(alarm, reference, 1.42)["collision_outcome"],
            "alarm_before_collision",
        )
        missed = {"alarm_before_collision": False, "detect_delay_ms": None, "false_alarms": 0}
        self.assertEqual(sweep.classify_collision(missed, reference, 1.42)["collision_outcome"], "missed")
        ambiguous = {"alarm_before_collision": False, "detect_delay_ms": 20.0, "false_alarms": 0}
        self.assertEqual(
            sweep.classify_collision(ambiguous, reference, 1.42)["collision_outcome"],
            "indistinguishable",
        )
        reference[:] = False
        self.assertEqual(
            sweep.classify_collision(ambiguous, reference, 1.42)["collision_outcome"], "detected"
        )
        for endpoint in (1420, 1460):
            with self.subTest(endpoint=endpoint):
                reference[:] = False
                reference[endpoint] = True
                self.assertEqual(
                    sweep.classify_collision(ambiguous, reference, 1.42)["collision_outcome"],
                    "indistinguishable",
                )
        reference[:] = False
        reference[1461] = True
        self.assertEqual(
            sweep.classify_collision(ambiguous, reference, 1.42)["collision_outcome"], "detected"
        )

    def test_classify_rows_uses_matching_mass_com_and_offset_reference(self):
        common = {"mass": 8.0, "com": [0.02, -0.01, 0.03], "pick_offset": None}
        rows = [
            {**common, "collision": None, "_detect": np.zeros(5000, dtype=bool)},
            {
                **common,
                "collision": [1.42, [0.0, 40.0, 0.0]],
                "_detect": np.zeros(5000, dtype=bool),
                "alarm_before_collision": False,
                "detect_delay_ms": 20.0,
                "false_alarms": 0,
                "tool_contact_delay_ms": None,
            },
        ]
        sweep.classify_rows(rows)
        self.assertEqual(rows[1]["collision_outcome"], "detected")
        self.assertNotIn("_detect", rows[0])


class TimingTests(unittest.TestCase):
    def test_wrappers_return_values_and_preserve_state(self):
        class Controller:
            def reset(self):
                self.count = 0

            def compute(self, value):
                self.count += 1
                return value + self.count

        class Detector:
            def reset(self):
                self.count = 0

            def update(self, value):
                self.count += 1
                return value, bool(value)

        timed_controller, timed_detector = sweep.timed_classes(Controller, Detector)
        controller, detector = timed_controller(), timed_detector()
        controller.reset()
        detector.reset()
        self.assertEqual(controller.compute(4), 5)
        self.assertEqual(detector.update(2), (2, True))
        self.assertEqual(controller.count, 1)
        self.assertEqual(detector.count, 1)
        self.assertEqual(len(controller._sweep_compute_seconds), 1)
        self.assertEqual(len(detector._sweep_update_seconds), 1)
        controller.reset()
        detector.reset()
        self.assertEqual(controller._sweep_compute_seconds, [])
        self.assertEqual(detector._sweep_update_seconds, [])
        self.assertEqual(controller.compute(4), 5)
        self.assertEqual(detector.update(0), (0, False))
        self.assertEqual(len(controller._sweep_compute_seconds), 1)
        self.assertEqual(len(detector._sweep_update_seconds), 1)

    def test_timing_stats(self):
        stats = sweep.timing_stats([1e-6, 2e-6, 3e-6])
        self.assertEqual(stats["samples"], 3)
        self.assertAlmostEqual(stats["median_us"], 2.0)
        self.assertAlmostEqual(stats["max_us"], 3.0)


class CliTests(unittest.TestCase):
    def test_rejects_module_paths_and_unsafe_output_names(self):
        parser = sweep.build_parser()
        for arguments in (
            ["--name", "../escape"],
            ["--name", "ok", "--controller", "../controller.py"],
            ["--name", "/tmp/escape"],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(arguments)

    def test_rejects_five_workers(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sweep.build_parser().parse_args(["--name", "ok", "--workers", "5"])

    def test_generic_cross_product_and_baseline_shape(self):
        args = argparse.Namespace(
            controller="servo_controller",
            detector="collision_detector",
            pick_offset=None,
            baseline=False,
            masses=(0.0, 8.0),
            com=[sweep.DEFAULT_COM, (0.0, 0.0, 0.0)],
            collision=[(2.0, (1.0, 2.0, 3.0))],
        )
        self.assertEqual(len(sweep.build_cases(args)), 8)
        args.baseline = True
        self.assertEqual(len(sweep.build_cases(args)), 27)

    def test_collision_time_must_be_inside_episode(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            sweep.parse_collision(f"{sweep.T.T_END},0,1,0")

    def test_baseline_rejects_even_default_valued_overrides(self):
        for override in (
            ["--masses", "0,4,8"],
            ["--com", "0.02,-0.01,0.03"],
            ["--pick-offset", "0,0,0"],
            ["--collision", "1.42,0,40,0"],
        ):
            with self.subTest(override=override), self.assertRaises(SystemExit):
                sweep.main(["--name", "not-created", "--baseline", *override])

    def test_low_memory_refuses_before_integrity_repeat_or_cases(self):
        with (
            mock.patch.object(sweep, "available_memory_bytes", return_value=5 * 1024**3),
            mock.patch.object(sweep, "check_integrity") as integrity,
            mock.patch.object(sweep, "build_cases") as build_cases,
            self.assertRaises(SystemExit),
        ):
            sweep.main(["--name", "not-created", "--workers", "1"])
        integrity.assert_not_called()
        build_cases.assert_not_called()

    def test_output_symlinks_cannot_escape_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            outside = Path(temporary) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "results").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                sweep.safe_output_dir("safe-name", root)

    def test_result_file_symlinks_are_never_followed(self):
        metadata = {"name": "test", "controller": "c", "detector": "d", "workers": 1}
        for filename in ("results.json", "results.csv", "SUMMARY.md"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "root"
                output_dir = root / "results" / "sweeps" / "test"
                output_dir.mkdir(parents=True)
                outside = Path(temporary) / "outside.txt"
                outside.write_text("preserve", encoding="utf-8")
                (output_dir / filename).symlink_to(outside)
                with self.assertRaises(ValueError):
                    sweep.write_outputs(output_dir, metadata, [], root=root)
                self.assertEqual(outside.read_text(encoding="utf-8"), "preserve")
                self.assertTrue((output_dir / filename).is_symlink())

    def test_write_revalidates_parent_after_safe_path_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            sweeps = root / "results" / "sweeps"
            sweeps.mkdir(parents=True)
            output_dir = sweep.safe_output_dir("test", root)
            outside = Path(temporary) / "outside"
            outside.mkdir()
            sweeps.rmdir()
            sweeps.symlink_to(outside, target_is_directory=True)
            metadata = {"name": "test", "controller": "c", "detector": "d", "workers": 1}
            with self.assertRaises(ValueError):
                sweep.write_outputs(output_dir, metadata, [], root=root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_atomic_outputs_are_regular_and_replaceable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            output_dir = sweep.safe_output_dir("test", root)
            metadata = {"name": "test", "controller": "c", "detector": "d", "workers": 1}
            sweep.write_outputs(output_dir, metadata, [], root=root)
            sweep.write_outputs(output_dir, metadata, [], root=root)
            for filename in ("results.json", "results.csv", "SUMMARY.md"):
                target = output_dir / filename
                self.assertTrue(target.is_file())
                self.assertFalse(target.is_symlink())

    def test_clean_import_without_bytecode_env_creates_no_sim_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "kit"
            copy.mkdir()
            for name in ("sim", "model", "tools"):
                shutil.copytree(
                    sweep.ROOT / name,
                    copy / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            environment = os.environ.copy()
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-c", "import tools.sweep"],
                cwd=copy,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((copy / "sim" / "__pycache__").exists())

    def test_tampered_sim_fails_integrity_before_sim_executes(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "kit"
            copy.mkdir()
            for name in ("sim", "model", "tools"):
                shutil.copytree(
                    sweep.ROOT / name,
                    copy / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            marker = Path(temporary) / "sim-imported.txt"
            with (copy / "sim" / "__init__.py").open("a", encoding="utf-8") as stream:
                stream.write(f"\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n")
            environment = os.environ.copy()
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-c", "import tools.sweep"],
                cwd=copy,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Kit integrity FAILED", result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((copy / "sim" / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
