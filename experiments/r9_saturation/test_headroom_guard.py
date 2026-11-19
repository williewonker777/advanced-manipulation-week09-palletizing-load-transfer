import unittest

import numpy as np

import r1_servo
from experiments.r9_saturation.headroom_guard import Controller
from sim import trajectory as T
from sim.nominal import Nominal


def configured(mass, com):
    controller = Controller()
    controller.nom = Nominal()
    controller.nom_box = Nominal()
    controller.nom_box.set_payload(mass, com)
    controller.box_mass, controller.box_com = mass, com
    controller.reset()
    return controller


class HeadroomGuardTest(unittest.TestCase):
    def test_only_minimum_critical_advance_is_removed(self):
        clip = configured(8.0, (-0.03, 0.02, -0.02))
        near = configured(8.0, (-0.03, 0.02, 0.015))
        weak_event = configured(8.0, (0.03, 0.0, -0.02))
        self.assertEqual(clip.selected_endpoint_delay, -0.004)
        self.assertEqual(near.selected_endpoint_delay, -0.004)
        self.assertEqual(weak_event.selected_endpoint_delay, -0.005)

    def test_unaffected_reference_is_exact_r1(self):
        candidate = configured(8.0, (0.03, 0.0, -0.02))
        baseline = r1_servo.Controller()
        baseline.nom, baseline.nom_box = candidate.nom, candidate.nom_box
        baseline.box_mass, baseline.box_com = candidate.box_mass, candidate.box_com
        baseline.reset()
        for t in (3.74, 3.80, 3.835, 3.88, 3.95):
            for actual, expected in zip(candidate._reference(t, *T.reference(t)),
                                        baseline._reference(t, *T.reference(t))):
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(candidate.nom.m.actuator_ctrlrange,
                                      Nominal().m.actuator_ctrlrange)


if __name__ == "__main__":
    unittest.main()
