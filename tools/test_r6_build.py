"""Regression and byte-contract tests for the bounded R6 assembly fix."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sim.drive import Drive
from sim.nominal import Nominal
from tools import r6_build


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_nominal_payload(instance):
    instance.nom = Nominal()
    instance.nom_box = Nominal()
    instance.box_mass = 4.0
    instance.box_com = (0.02, -0.01, 0.03)
    instance.nom_box.set_payload(instance.box_mass, instance.box_com)


def real_controller(mode):
    filename = "my_servo.py" if mode == "servo" else "my_torque.py"
    module = load_module(ROOT / filename, "r6_controller_" + mode)
    controller = module.Controller()
    set_nominal_payload(controller)
    if mode == "servo":
        controller.drive = Drive()
    controller.reset()
    return controller


def reset_detector(module, controller_marker):
    detector = module.Detector()
    set_nominal_payload(detector)
    if controller_marker == "absent":
        pass
    elif controller_marker == "missing":
        detector.ctrl = object()
    elif controller_marker in ("servo", "torque"):
        detector.ctrl = real_controller(controller_marker)
    else:
        detector.ctrl = type("Controller", (), {"MODE": controller_marker})()
    detector.reset()
    return detector


class R6AssemblyTests(unittest.TestCase):
    def test_reset_mode_matrix(self):
        # First prove the nominal fixture itself against unchanged R1.
        r1_module = load_module(ROOT / "r1_detector.py", "r6_r1_sanity")
        reset_detector(r1_module, "absent")
        cases = (
            ("A", {"servo"}, "servo", True),
            ("A", {"servo"}, None, False),
            ("A", {"servo"}, "unknown", False),
            ("A", {"servo"}, "mod", False),
            ("A", {"servo"}, "missing", False),
            ("A", {"servo"}, "absent", False),
            ("A", {"servo"}, "torque", False),
            ("B", {"torque"}, "torque", True),
            ("B", {"torque"}, None, False),
            ("B", {"torque"}, "unknown", False),
            ("B", {"torque"}, "mod", False),
            ("B", {"torque"}, "missing", False),
            ("B", {"torque"}, "absent", False),
            ("B", {"torque"}, "servo", False),
            ("AB", {"servo", "torque"}, "servo", True),
            ("AB", {"servo", "torque"}, "torque", True),
            ("AB", {"servo", "torque"}, None, False),
            ("AB", {"servo", "torque"}, "unknown", False),
            ("AB", {"servo", "torque"}, "mod", False),
            ("AB", {"servo", "torque"}, "missing", False),
            ("AB", {"servo", "torque"}, "absent", False),
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder = Path(folder)
            modules = {}
            for name, modes, _, _ in cases:
                if name not in modules:
                    path = folder / (name + ".py")
                    path.write_bytes(r6_build.assembly_bytes(modes))
                    modules[name] = load_module(path, "r6_matrix_" + name)
            for name, _, controller, monitor_enabled in cases:
                detector = reset_detector(modules[name], controller)
                self.assertIs(monitor_enabled, detector._r5_monitor is not None,
                              (name, controller))

        empty = r6_build.assembly_bytes(set())
        self.assertEqual((ROOT / "r1_detector.py").read_bytes(), empty)

    def test_production_candidate_is_only_literal_fix(self):
        frozen = (ROOT / "experiments/r5_final/A_STATE.py").read_bytes()
        expected = frozen.replace(
            b"_R5_ENABLED_MODES = 'servo'\n",
            b"_R5_ENABLED_MODES = ('servo',)\n",
        )
        self.assertEqual(1, frozen.count(b"_R5_ENABLED_MODES = 'servo'\n"))
        self.assertEqual(expected, r6_build.assembly_bytes({"servo"}))
        self.assertEqual((ROOT / "experiments/r5_generated/SCALE_STATE.py").read_bytes(),
                         r6_build.assembly_bytes({"servo", "torque"}))

    def test_materialization_refuses_different_existing_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            target = Path(folder) / "owned.py"
            r6_build._write_new_or_equal(target, b"first")
            r6_build._write_new_or_equal(target, b"first")
            with self.assertRaises(RuntimeError):
                r6_build._write_new_or_equal(target, b"different")

    def test_build_record_pins_inputs_models_tools_outputs_and_diff(self):
        record = json.loads((ROOT / "results/v2_protocol/R6_BUILD.json").read_text())
        self.assertEqual(8, record["model_count"])
        self.assertEqual(8, len(record["models"]))
        self.assertTrue(record["prefix_proof"]["exact_R1_prefix"])
        self.assertTrue(record["diff_proof"]["all_other_bytes_preserved"])
        self.assertEqual(0, record["weights_or_threshold_updates"])
        for key in ("builder", "regression_test"):
            row = record[key]
            self.assertEqual(row["sha256"], hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest())
        for row in record["generated"]:
            self.assertEqual(row["sha256"], hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest())

    def test_cli_check_is_nondestructive_and_deterministic(self):
        paths = [ROOT / "experiments/r6_final/A_STATE.py",
                 ROOT / "experiments/r6_final/R1.py",
                 ROOT / "results/v2_protocol/R6_BUILD.json"]
        before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths}
        command = [sys.executable, str(ROOT / "tools/r6_build.py"), "--check"]
        first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(0, first.returncode, first.stdout + first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(before, {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths})


if __name__ == "__main__":
    unittest.main()
