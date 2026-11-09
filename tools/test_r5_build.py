"""Static and import tests for the self-contained R5 detector materialization."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r5_build
from tools.production_audit import audit_text


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "experiments/r5_generated"
CONFIGS = ("SCALE_PHASE", "SCALE_STATE", "AR20_FULL", "AR50_FULL")


class BuildTests(unittest.TestCase):
    def test_check_verifies_registered_generated_bundle(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/r5_build.py"), "--check"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_exactly_four_standalone_modules_are_generated(self):
        self.assertEqual(sorted(f"{name}.py" for name in CONFIGS),
                         sorted(path.name for path in GENERATED.glob("*.py")))

    def test_generated_modules_preserve_exact_r1_prefix_and_pass_audit(self):
        prefix = (ROOT / "r1_detector.py").read_bytes()
        for name in CONFIGS:
            with self.subTest(name=name):
                path = GENERATED / f"{name}.py"
                self.assertTrue(path.read_bytes().startswith(prefix))
                self.assertEqual("pass", audit_text(path.read_text(), path.name)["status"])

    def test_generated_modules_do_not_import_private_runtime_helpers(self):
        for name in CONFIGS:
            tree = ast.parse((GENERATED / f"{name}.py").read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):imports.append(node.module or "")
            self.assertFalse(any(value.startswith(("experiments", "tools", "r1_detector")) for value in imports), imports)

    def test_build_record_hashes_match_generated_and_frozen_models(self):
        record = json.loads((ROOT / "results/v2_protocol/R5_BUILD.json").read_text())
        generated = {row["config"]: row for row in record["generated"]}
        models = {(row["config"], row["mode"]): row for row in record["models"]}
        for name in CONFIGS:
            path = GENERATED / f"{name}.py"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             generated[name]["sha256"])
            for mode in ("servo", "torque"):
                model = ROOT / f"models/r5/{mode}_{name}.json"
                self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(),
                                 models[(name, mode)]["sha256"])

    def test_materialization_is_deterministic(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            first = Path(folder) / "first.py"
            second = Path(folder) / "second.py"
            payloads, _ = r5_build._registered_payloads("SCALE_STATE")
            one = r5_build.materialize(payloads, first)
            two = r5_build.materialize(payloads, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(one["sha256"], two["sha256"])

    def test_generated_detector_has_one_parent_update_and_owned_active_capture(self):
        text = (GENERATED / "SCALE_STATE.py").read_text()
        self.assertEqual(1, text.count("super().update(t, th, thd, tau_m, tau_s, grip)"))
        self.assertNotIn("self.normal_model.update =", text)
        self.assertIn("class _R5TrackedNormalResidualModel", text)

    def test_generated_detector_never_writes_the_shared_packet(self):
        text = (GENERATED / "SCALE_STATE.py").read_text()
        wrapper = text[text.index("_R1Detector = Detector"):]
        tree = ast.parse(wrapper)
        writes = []
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "packet":
                    writes.append(target.attr)
        self.assertEqual([], writes)

    def test_generated_detector_calls_parent_once_and_leaves_packet_unchanged(self):
        path = GENERATED / "SCALE_STATE.py"
        spec = importlib.util.spec_from_file_location("r5_generated_runtime_test", path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        packet = type("Packet", (), {})()
        packet.r_joint = np.arange(6, dtype=float)
        packet.last_wrench = np.arange(6, dtype=float) * .1
        before = (packet.r_joint.copy(), packet.last_wrench.copy())
        calls = []

        def base_reset(detector):
            detector.ctrl = type("Controller", (), {"MODE": "servo"})()
            detector.normal_model = object()
            detector.box_mass = 3.
            detector.box_com = (0., 0., .03)
            detector._shared = packet

        def base_update(detector, t, th, thd, tau_m, tau_s, grip):
            calls.append(float(t))
            detector.normal_model.update(t, detector.box_mass, detector.box_com,
                                         packet.last_wrench, True)
            return np.zeros(6), False

        detector = module.Detector.__new__(module.Detector)
        with patch.object(module._R1Detector, "reset", base_reset), \
                patch.object(module._R1Detector, "update", base_update):
            detector.reset()
            detector.update(2., *[np.zeros(6) for _ in range(4)], True)
        self.assertEqual([2.], calls)
        np.testing.assert_array_equal(before[0], packet.r_joint)
        np.testing.assert_array_equal(before[1], packet.last_wrench)


if __name__ == "__main__":
    unittest.main()
