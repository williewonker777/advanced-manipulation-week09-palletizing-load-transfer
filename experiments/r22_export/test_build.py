import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

import my_servo
import my_torque
from experiments.r22_export import build
from sim.nominal import Nominal


def load(path):
    spec = importlib.util.spec_from_file_location("r22_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured(module, controller_type):
    ctrl = controller_type(); ctrl.nom = Nominal(); ctrl.nom_box = Nominal()
    ctrl.box_mass = 4.; ctrl.box_com = (.02, -.01, .03)
    ctrl.nom_box.set_payload(ctrl.box_mass, ctrl.box_com); ctrl.reset()
    detector = module.Detector(); detector.nom = Nominal(); detector.nom_box = Nominal()
    detector.box_mass = ctrl.box_mass; detector.box_com = ctrl.box_com
    detector.nom_box.set_payload(detector.box_mass, detector.box_com)
    detector.ctrl = ctrl; detector.reset()
    return detector


class BuildTest(unittest.TestCase):
    def test_both_variants_are_single_file_and_pin_every_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ("core", "augmented", "guarded", "light"):
                out = Path(tmp) / (variant + ".py")
                receipt = build.build(out, variant)
                self.assertEqual(receipt["frozen_sources_sha256"], build.PINS)
                self.assertEqual(out.read_bytes()[:len(build.R10.read_bytes())], build.R10.read_bytes())
                self.assertEqual(load(out).Detector.SOURCE, "r10+r22-" + variant + "-research")

    def test_no_private_runtime_import_or_file_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ("core", "augmented", "guarded", "light"):
                out = Path(tmp) / (variant + ".py"); build.build(out, variant)
                imported = []
                for node in ast.walk(ast.parse(out.read_text())):
                    if isinstance(node, ast.Import): imported.extend(n.name for n in node.names)
                    elif isinstance(node, ast.ImportFrom): imported.append(node.module or "")
                self.assertFalse(any(name in ("json", "pathlib", "my_detector")
                                     or name.startswith("experiments") for name in imported))
                self.assertNotIn("read_bytes", out.read_text())
                self.assertNotIn("open(", out.read_text())

    def test_extension_enabled_only_for_registered_torque(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "core.py"; build.build(out, "core"); module = load(out)
            self.assertTrue(configured(module, my_torque.Controller).r22_enabled)
            self.assertFalse(configured(module, my_servo.Controller).r22_enabled)

    def test_existing_output_and_changed_variant_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.py"; out.write_text("sentinel")
            with self.assertRaises(ValueError): build.build(out, "core")
            self.assertEqual(out.read_text(), "sentinel")
            with self.assertRaises(ValueError): build.build(Path(tmp) / "new.py", "unknown")


if __name__ == "__main__":
    unittest.main()
