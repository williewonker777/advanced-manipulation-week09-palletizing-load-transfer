import importlib
import unittest
from pathlib import Path
import numpy as np
from tools.sync_shared_core import extract, ROOT, FILES
from tools.test_production_contract import models
from sim import trajectory as T


class SharedEstimatorTests(unittest.TestCase):
    def test_embedded_classes_are_text_identical(self):
        reference, _ = extract((ROOT/'tools/shared_estimator_core.py').read_text())
        for name in FILES:
            self.assertEqual(extract((ROOT/name).read_text())[0], reference)

    def test_once_per_timestamp_for_controller_and_detector(self):
        from my_detector import Detector
        for mode in ('my_servo', 'my_torque'):
            ctrl = importlib.import_module(mode).Controller()
            models(ctrl)
            ctrl.reset()
            detector = Detector()
            models(detector)
            detector.ctrl = ctrl
            detector.reset()
            self.assertIs(detector._shared, ctrl.shared_estimator)
            for k in range(3):
                t = k*.001
                q, qd, qdd = T.reference(t)
                tau = ctrl.nom.gravity(q)
                th = q + ctrl.inv_k*tau
                detector.update(t, th, np.zeros(6), np.full(6, 1e6), tau, 0)
                ctrl.compute(t, th, np.zeros(6), tau, 0, q, qd, qdd)
                self.assertEqual(ctrl.shared_estimator.sample_count, k+1)


if __name__ == '__main__':
    unittest.main()
