"""Independent contract tests for the R12 virtual-force audit."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r12_force_audit as audit
from tools import r12_normal_force as force


class ZeroNominal:
    elastic = np.zeros(6, dtype=bool)
    K = np.ones(6)

    def inverse_dynamics(self, q, velocity, acceleration):
        del q, velocity, acceleration
        return np.zeros(6)


class PhysicsIndexContractTest(unittest.TestCase):
    @patch.object(force, "vertical_displacement",
                  return_value=np.array([1., 0., 0., 0., 0., 0.]))
    def test_exact_plant_filter_delay_and_fir_center_is_eighteen_solver_steps(self, _):
        """A non-affine trace distinguishes the exact label from +/- one sample."""
        dt, hz = .001, 300.
        observer = force.NormalForce(
            ZeroNominal(), ZeroNominal(), dt=dt,
            viscous_spring=np.ones(6), jts_hz=hz,
        )
        observer.bias = np.zeros(6)
        rng = np.random.default_rng(12017)
        truth = rng.normal(size=760)
        direction = np.array([1., 0., 0., 0., 0., 0.])

        # Exact sim/plant.py sensor recurrence: filter current step torque, but
        # return the previous filter state. A residual of +truth uses torque
        # -truth because the nominal model is zero here.
        alpha = 1. - np.exp(-2. * np.pi * hz * dt)
        filtered = output = None
        checked = 0
        for i in range(len(truth)):
            current_torque = np.zeros(6) if i == 0 else -truth[i - 1] * direction
            filtered = (current_torque.copy() if filtered is None else
                        filtered + alpha * (current_torque - filtered))
            raw = filtered.copy() if output is None else output
            output = filtered.copy()
            packet = observer.update(i * dt, np.zeros(6), raw, True)
            if packet is None:
                continue
            start, stop = i - (force.N + 2), i - 2
            expected = float(force.KERNEL @ truth[start:stop])
            self.assertAlmostEqual(packet.vertical_n, expected, places=10)
            self.assertAlmostEqual(packet.effective_time, (i - force.LAG) * dt, places=14)
            self.assertEqual(force.LAG, 18)
            # A random trace makes either adjacent label observably different.
            if start > 0:
                self.assertNotAlmostEqual(
                    expected, float(force.KERNEL @ truth[start - 1:stop - 1]), places=7)
                self.assertNotAlmostEqual(
                    expected, float(force.KERNEL @ truth[start + 1:stop + 1]), places=7)
            checked += 1
        self.assertGreater(checked, 700)

    def test_assess_rejects_even_subsample_sensor_solver_clock_disagreement(self):
        samples = [(i * .001, np.zeros(6), np.zeros(6), False, np.nan)
                   for i in range(6000)]
        bad = list(samples[100])
        bad[0] += 2e-10
        samples[100] = tuple(bad)
        log = {"t": np.arange(6000) * .001, "f_table": np.ones(6000)}

        def physical(_log, _mass, _release, _dt):
            return {"support": {"samples": 20}}

        with self.assertRaisesRegex(ValueError, "sensor and evaluator clocks differ"):
            audit.assess(samples, log, 4.0, (0., 0., 0.), physical, force)


class AuditCliContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.kit = self.root / "kit"
        self.candidate = self.root / "candidate"
        self.output_parent = self.root / "output"
        for directory in (self.kit, self.candidate, self.output_parent):
            directory.mkdir()
        for name in ("servo", "torque", "detector"):
            (self.candidate / f"my_{name}.py").write_text(f"# frozen {name}\n")

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, output, candidate=None):
        argv = [
            "r12_force_audit.py", "--kit", str(self.kit),
            "--candidate-dir", str(candidate or self.candidate),
            "--out", str(output),
        ]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            audit.main()

    @staticmethod
    def rows():
        design = json.loads((audit.ROOT / "results/r11/placement_plan.json").read_text())
        return [{"id": case["id"], "status": "measured", "score": {}}
                for case in design["cases"]]

    def mocks(self):
        executed = []

        def prepare(_kit, overlay, _prefix, mode, **kwargs):
            self.assertTrue(kwargs["archived_r9"])
            (overlay / f"{mode}_controller.py").write_text("# reference\n")
            (overlay / "collision_detector.py").write_text("# reference\n")
            (overlay / "tools").mkdir()
            (overlay / "tools/kit_manifest.sha256").write_bytes(
                (audit.ROOT / "tools/kit_manifest.sha256").read_bytes()
            )
            return "reference-controller", "reference-detector"

        def run(_command, **kwargs):
            self.assertNotIn("PYTHONPATH", kwargs["env"])
            self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            mode = _command[-1]
            overlay = Path(kwargs["cwd"])
            controller = overlay / f"{mode}_controller.py"
            detector = overlay / "collision_detector.py"
            self.assertEqual(controller.read_bytes(),
                             (self.candidate / f"my_{mode}.py").read_bytes())
            self.assertEqual(detector.read_bytes(),
                             (self.candidate / "my_detector.py").read_bytes())
            for required in ("design.json", "r12_normal_force.py", "geometry_eval.py", "audit.py"):
                self.assertTrue((overlay / required).is_file())
            executed.append(mode)
            return type("Completed", (), {"stdout": json.dumps(self.rows())})()

        return prepare, run, executed

    def test_valid_assembly_pins_executed_candidate_and_preserves_inputs(self):
        output = self.output_parent / "audit.json"
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.candidate.iterdir()}
        prepare, run, executed = self.mocks()
        with patch("tools.validate_submission.prepare_overlay", side_effect=prepare), \
                patch.object(audit.subprocess, "run", side_effect=run):
            self.invoke(output)
        result = json.loads(output.read_text())
        self.assertEqual(executed, ["servo", "torque"])
        self.assertEqual(result["schema"], "r12-virtual-load-audit.v2")
        self.assertEqual(result["solver_step_lag_ms"], force.LAG)
        self.assertEqual(result["state_time_lag_ms"], force.STATE_LAG)
        self.assertIn("not a contact classifier", result["scope"])
        self.assertEqual(set(result["modes"]), {"servo", "torque"})
        for name in ("servo", "torque", "detector"):
            self.assertEqual(result["pins"][name], before[f"my_{name}.py"])
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in self.candidate.iterdir()}
        self.assertEqual(after, before)

    def test_snapshot_bytes_not_later_mutated_origin_are_installed_and_pinned(self):
        origin = self.candidate / "my_servo.py"
        original = origin.read_bytes()
        blobs, pins = audit.snapshot_files({"servo": origin})
        origin.write_text("# mutation after snapshot\n")
        destination = self.output_parent / "servo_controller.py"
        audit.install_snapshot(blobs["servo"], pins["servo"], destination)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(audit.digest(destination), pins["servo"])
        self.assertNotEqual(audit.digest(origin), pins["servo"])

    def test_existing_inside_kit_missing_parent_and_symlink_sources_are_rejected(self):
        existing = self.output_parent / "existing.json"
        existing.write_text("preserve")
        invalid = (existing, self.kit / "inside.json", self.root / "missing" / "result.json")
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(SystemExit):
                self.invoke(output)
        self.assertEqual(existing.read_text(), "preserve")
        self.assertFalse((self.kit / "inside.json").exists())

        target = self.root / "outside.py"
        target.write_text("# changed source\n")
        (self.candidate / "my_detector.py").unlink()
        (self.candidate / "my_detector.py").symlink_to(target)
        output = self.output_parent / "symlink.json"
        with self.assertRaises(SystemExit):
            self.invoke(output)
        self.assertFalse(output.exists())

        dangling = self.output_parent / "dangling.json"
        dangling.symlink_to(self.root / "absent.json")
        with self.assertRaises(SystemExit):
            self.invoke(dangling)
        self.assertTrue(dangling.is_symlink())

    def test_kit_integrity_or_child_failure_never_creates_output(self):
        output = self.output_parent / "audit.json"
        with patch("tools.validate_submission.prepare_overlay",
                   side_effect=ValueError("original kit manifest differs")), \
                self.assertRaisesRegex(ValueError, "kit manifest"):
            self.invoke(output)
        self.assertFalse(output.exists())

        prepare, _, _ = self.mocks()
        with patch("tools.validate_submission.prepare_overlay", side_effect=prepare), \
                patch.object(audit.subprocess, "run",
                             side_effect=subprocess.CalledProcessError(2, ["python"])), \
                self.assertRaises(subprocess.CalledProcessError):
            self.invoke(output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
