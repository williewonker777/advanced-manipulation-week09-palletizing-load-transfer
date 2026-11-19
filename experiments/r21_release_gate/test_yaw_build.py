import ast
import importlib.util
import tempfile
from pathlib import Path
import unittest

import my_servo
import my_torque
from experiments.r21_release_gate import build
from sim.nominal import Nominal


def loaded(path):
    spec = importlib.util.spec_from_file_location("standalone_yaw", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured(module, controller_type):
    ctrl = controller_type()
    ctrl.nom = Nominal(); ctrl.nom_box = Nominal()
    ctrl.box_mass = 4.; ctrl.box_com = (.02, -.01, .03)
    ctrl.nom_box.set_payload(ctrl.box_mass, ctrl.box_com); ctrl.reset()
    det = module.Detector()
    det.nom = Nominal(); det.nom_box = Nominal()
    det.box_mass = ctrl.box_mass; det.box_com = ctrl.box_com
    det.nom_box.set_payload(det.box_mass, det.box_com); det.ctrl = ctrl; det.reset()
    return det


class YawStandaloneBuildTest(unittest.TestCase):
    def test_yaw_build_pins_every_frozen_source_and_embeds_2_by_60_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "collision_detector.py"
            receipt = build.build(out, variant="yaw")
            self.assertEqual(receipt["schema"], "r21-yaw-standalone-build.v1")
            self.assertEqual(receipt["yaw_sources_sha256"], {
                "state.py": "f04d46e93c55e04aa869050fe43c895951f3a970e3729b01e8d94744532f3a9f",
                "detector.py": "69e1e884f393a17751546165554f994c19b0cb127830233fba1f2329481c87cb",
                "frozen.py": "71a75f131ea6d3aae94f9acf40c42abe92d9123000d86e685f8bb5f4a00a4342",
                "coarse.py": "f3a8a4202e638346dfa6a7994b056cbf509be4f335fb09c76d6657d47abcafe9",
                "normal_bins.json": "5f22f640eb8824f764ffeec7ecbef7ba23df02a874547b93b8ddc72925e25c15",
            })
            module = loaded(out)
            self.assertEqual(module._R21_YAW_BOUNDS.shape, (2, 60))

    def test_yaw_build_has_no_private_runtime_import_or_file_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "collision_detector.py"
            build.build(out, variant="yaw")
            imported = []
            for node in ast.walk(ast.parse(out.read_text())):
                if isinstance(node, ast.Import):
                    imported.extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            self.assertFalse(any(name == "json" or name == "pathlib" or name == "my_detector"
                                 or name.startswith("experiments") for name in imported))
            self.assertNotIn("open(", out.read_text())
            self.assertNotIn("read_bytes", out.read_text())

    def test_yaw_is_enabled_only_for_recognized_R10_torque_shared_estimator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "collision_detector.py"
            build.build(out, variant="yaw")
            module = loaded(out)
            self.assertTrue(configured(module, my_torque.Controller).yaw_enabled)
            self.assertFalse(configured(module, my_servo.Controller).yaw_enabled)

    def test_yaw_is_disabled_for_unregistered_controller_with_lookalike_estimator(self):
        class UnregisteredTorque(my_torque.Controller):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "collision_detector.py"
            build.build(out, variant="yaw")
            self.assertFalse(configured(loaded(out), UnregisteredTorque).yaw_enabled)


if __name__ == "__main__":
    unittest.main()
