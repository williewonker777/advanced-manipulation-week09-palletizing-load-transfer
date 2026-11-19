import unittest

import numpy as np

import r1_servo
from experiments.r9_saturation.feedback.governor import Controller
from sim import config as C, trajectory as T
from sim.nominal import Nominal


def configured(controller_class, mass=2.0, com=(0.011, -0.007, 0.021)):
    controller = controller_class()
    controller.nom, controller.nom_box = Nominal(), Nominal()
    controller.nom_box.set_payload(mass, com)
    controller.box_mass, controller.box_com = mass, com
    controller.reset()
    return controller


class GovernorTest(unittest.TestCase):
    def test_low_demand_first_command_is_bit_exact_r1(self):
        baseline, candidate = configured(r1_servo.Controller), configured(Controller)
        q, qd, qdd = T.reference(0.0)
        tau_s = baseline.nom.gravity(q)
        expected = baseline.compute(0.0, q, np.zeros(6), tau_s, 0, q, qd, qdd)
        actual = candidate.compute(0.0, q, np.zeros(6), tau_s, 0, q, qd, qdd)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(candidate.feedback_active)

    def test_soft_barrier_has_physical_reserve_without_changing_limit(self):
        candidate = configured(Controller, 8.0, (-0.03, 0.02, -0.02))
        raw = 346.0
        excess = raw - candidate.SOFT_START_NM
        safe = candidate.SOFT_START_NM + candidate.SOFT_BAND_NM * np.tanh(
            excess / candidate.SOFT_BAND_NM)
        self.assertLess(safe, C.TAU_MAX[1] - 18.0 + 1.0)
        self.assertEqual(candidate.SOFT_START_NM + candidate.SOFT_BAND_NM, 328.0)
        np.testing.assert_array_equal(candidate.nom.m.actuator_ctrlrange,
                                      Nominal().m.actuator_ctrlrange)


if __name__ == "__main__":
    unittest.main()
