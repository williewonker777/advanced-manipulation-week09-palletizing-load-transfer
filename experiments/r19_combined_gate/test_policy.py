import unittest
from unittest import mock

import my_detector
from experiments.r17_candidate.policy import Controller
from experiments.r19_combined_gate import policy
from experiments.r19_event_state.physical_detector import Detector as PhysicalEvent


class StubImpl:
    def reset(self):
        self.update = lambda *args: (None, False)


class PolicySelectionTest(unittest.TestCase):
    def wrapper(self, registered=True, loaded_policy=True, mass=4.0):
        wrapper = policy.Detector()
        ctrl = Controller.__new__(Controller) if registered else object()
        if registered:
            ctrl._active = object()
            ctrl.loaded_policy = loaded_policy
        wrapper.ctrl = ctrl
        wrapper.box_mass = mass
        wrapper.box_com = (0.0, 0.0, 0.0)
        return wrapper

    def select(self, wrapper, in_range):
        physical, baseline = StubImpl(), StubImpl()
        with (mock.patch.object(policy, "supported_payload", return_value=in_range),
              mock.patch.object(policy, "PhysicalEvent", return_value=physical),
              mock.patch.object(my_detector, "Detector", return_value=baseline),
              mock.patch.object(policy, "attach")):
            wrapper.reset()
        return wrapper._active, physical, baseline

    def test_registered_supported_payload_selects_immutable_v2(self):
        wrapper = self.wrapper(registered=True, loaded_policy=True)
        selected, physical, _ = self.select(wrapper, True)
        self.assertIs(selected, physical)

    def test_zero_and_outside_domain_select_exact_r10_detector(self):
        for mass, in_range in ((0.0, False), (8.15, False)):
            wrapper = self.wrapper(registered=True, loaded_policy=False, mass=mass)
            selected, _, baseline = self.select(wrapper, in_range)
            self.assertIs(selected, baseline)

    def test_plain_or_servo_unregistered_controller_selects_exact_r10_detector(self):
        wrapper = self.wrapper(registered=False)
        selected, _, baseline = self.select(wrapper, True)
        self.assertIs(selected, baseline)

    def test_dependency_identity_is_fixed(self):
        self.assertIs(policy.Controller, Controller)
        self.assertIs(policy.PhysicalEvent, PhysicalEvent)


if __name__ == "__main__":
    unittest.main()
