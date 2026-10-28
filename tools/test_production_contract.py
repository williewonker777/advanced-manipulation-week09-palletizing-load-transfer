"""Production modules must run with only their documented public interfaces.

No Plant is constructed; these are interface/reset checks, not performance
claims. Import under unrelated names verifies no own-file/module dependency.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity
check_integrity(ROOT)
from sim import trajectory as T
from sim.drive import Drive
from sim.nominal import Nominal


def module_under_unrelated_name(filename):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(f"isolated_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def models(instance, mass=4.):
    instance.nom = Nominal()
    instance.nom_box = Nominal()
    instance.box_mass = mass
    instance.box_com = (0.02, -0.01, 0.03)
    instance.nom_box.set_payload(mass, instance.box_com)


class ProductionContractTests(unittest.TestCase):
    def test_controllers_without_any_detector(self):
        for filename in ("my_servo.py", "my_torque.py"):
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / filename).is_file(), "D2 production file missing")
                cls = module_under_unrelated_name(filename).Controller
                ctrl = cls()
                models(ctrl)
                if getattr(ctrl, "MODE", "torque") == "servo":
                    ctrl.drive = Drive()
                ctrl.reset()
                q, qd, qdd = T.reference(0.)
                tau = ctrl.nom.gravity(q)
                theta = q.copy()
                theta[ctrl.nom.elastic] += tau[ctrl.nom.elastic] / ctrl.nom.K[ctrl.nom.elastic]
                expected = (4, 6) if getattr(ctrl, "MODE", "torque") == "servo" else (6,)
                first = np.asarray(ctrl.compute(0., theta, np.zeros(6), tau, 0, q, qd, qdd))
                self.assertEqual(first.shape, expected)
                self.assertTrue(np.isfinite(first).all())
                ctrl.reset()
                repeated = np.asarray(ctrl.compute(0., theta, np.zeros(6), tau, 0, q, qd, qdd))
                np.testing.assert_allclose(repeated, first, atol=1e-10)

    def test_detector_missing_or_unrelated_controller(self):
        self.assertTrue((ROOT / "my_detector.py").is_file(), "D3 production file missing")
        cls = module_under_unrelated_name("my_detector.py").Detector
        for with_controller in (False, True):
            with self.subTest(with_controller=with_controller):
                detector = cls()
                models(detector)
                if with_controller:
                    detector.ctrl = object()  # no controller-specific fields exist
                detector.reset()
                q = T.reference(0.)[0]
                tau = detector.nom.gravity(q)
                theta = q.copy()
                elastic = detector.nom.elastic
                theta[elastic] += tau[elastic] / detector.nom.K[elastic]
                for k in range(3):
                    residual, alarm = detector.update(k * .001, theta.copy(), np.zeros(6),
                                                       tau.copy(), tau.copy(), 0)
                    self.assertEqual(np.asarray(residual).shape, (6,))
                    self.assertTrue(np.isfinite(residual).all())
                    self.assertIsInstance(alarm, (bool, np.bool_))


if __name__ == "__main__":
    unittest.main()
