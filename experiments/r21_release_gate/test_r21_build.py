import ast
import importlib.util
import tempfile
from pathlib import Path
import unittest

from experiments.r21_release_gate import build
from sim.nominal import Nominal
import my_torque


def loaded(path):
    spec = importlib.util.spec_from_file_location("standalone_r21", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_detector(module, mass, com):
    ctrl = my_torque.Controller()
    ctrl.nom = Nominal(); ctrl.nom_box = Nominal()
    ctrl.box_mass = mass; ctrl.box_com = com
    ctrl.nom_box.set_payload(mass, com); ctrl.reset()
    det = module.Detector()
    det.nom = Nominal(); det.nom_box = Nominal()
    det.box_mass = mass; det.box_com = com
    det.nom_box.set_payload(mass, com); det.ctrl = ctrl; det.reset()
    return det


class R21StandaloneBuildTest(unittest.TestCase):
    def test_R21_build_embeds_model_without_private_runtime_import_or_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "collision_detector.py"
            receipt = build.build(out, variant="r21")
            imports = []
            tree = ast.parse(out.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(n == "json" or n == "pathlib" or n == "my_detector"
                                 or n.startswith("experiments") for n in imports))
            self.assertEqual(receipt["schema"], "r21-observer-standalone-build.v1")
            self.assertEqual(receipt["r21_candidate_sha256"],
                             "1f03ed4918fb279783813c8f1f769ee3a4476b1d973a765b2c5d81ce81e9e3e8")
            self.assertEqual(receipt["r21_model_sha256"],
                             "499b2577f659dff86bcc665fffc385adddadfe21824ef0e06c7d848caa49c6bf")

    def test_R21_model_is_active_only_inside_frozen_payload_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "collision_detector.py"
            build.build(out, variant="r21")
            module = loaded(out)
            supported = reset_detector(module, 4., (0.01, 0., 0.01))
            unsupported_mass = reset_detector(module, 8.1, (0.01, 0., 0.01))
            unsupported_com = reset_detector(module, 4., (0.031, 0., 0.01))
            self.assertTrue(supported.pick_supported)
            self.assertFalse(unsupported_mass.pick_supported)
            self.assertFalse(unsupported_com.pick_supported)
            self.assertTrue(unsupported_mass.horizontal_enabled)
            self.assertTrue(unsupported_com.horizontal_enabled)


if __name__ == "__main__":
    unittest.main()
