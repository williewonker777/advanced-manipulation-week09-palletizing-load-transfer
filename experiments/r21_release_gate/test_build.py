import ast
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest


from experiments.r21_release_gate import build
from sim.nominal import Nominal


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location("standalone_r20_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(detector, controller):
    detector.nom = Nominal()
    detector.nom_box = Nominal()
    detector.box_mass = 4.0
    detector.box_com = (0.02, -0.01, 0.03)
    detector.nom_box.set_payload(detector.box_mass, detector.box_com)
    detector.ctrl = controller
    detector.reset()


class StandaloneBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "collision_detector.py"
        self.receipt = build.build(self.output)
        self.module = load(self.output)

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_has_no_private_or_sibling_runtime_import(self):
        imported = []
        for node in ast.walk(ast.parse(self.output.read_text())):
            if isinstance(node, ast.Import):
                imported.extend(name.name for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(any(name == "my_detector" or name.startswith("experiments")
                             for name in imported))
        self.assertEqual(self.output.read_bytes()[:len((ROOT / "my_detector.py").read_bytes())],
                         (ROOT / "my_detector.py").read_bytes())

    def test_build_receipt_pins_both_authored_inputs(self):
        self.assertEqual(set(self.receipt), {
            "schema", "r10_detector_sha256", "r20_branch_sha256",
            "standalone_sha256", "lines"})
        self.assertEqual(self.receipt["schema"], "r21-r20-standalone-build.v1")
        self.assertGreater(self.receipt["lines"], 1800)

    def test_unknown_torque_mode_with_incomplete_shared_core_falls_back(self):
        fake = SimpleNamespace(MODE="torque", shared_estimator=SimpleNamespace(
            API="icpbl.shared-estimator.v1"))
        detector = self.module.Detector()
        prepare(detector, fake)
        self.assertFalse(detector.horizontal_enabled)
        self.assertIs(detector._shared, detector)

    def test_current_r10_torque_shared_contract_enables_branch(self):
        import my_torque
        controller = my_torque.Controller()
        controller.nom = Nominal()
        controller.nom_box = Nominal()
        controller.box_mass = 4.0
        controller.box_com = (0.02, -0.01, 0.03)
        controller.nom_box.set_payload(controller.box_mass, controller.box_com)
        controller.reset()
        detector = self.module.Detector()
        prepare(detector, controller)
        self.assertTrue(detector.horizontal_enabled)
        self.assertIs(detector._shared, controller.shared_estimator)

    def test_servo_shared_contract_retains_r10_without_horizontal_branch(self):
        import my_servo
        controller = my_servo.Controller()
        controller.nom = Nominal()
        controller.nom_box = Nominal()
        controller.box_mass = 4.0
        controller.box_com = (0.02, -0.01, 0.03)
        controller.nom_box.set_payload(controller.box_mass, controller.box_com)
        controller.reset()
        detector = self.module.Detector()
        prepare(detector, controller)
        self.assertFalse(detector.horizontal_enabled)
        self.assertIs(detector._shared, controller.shared_estimator)


if __name__ == "__main__":
    unittest.main()
