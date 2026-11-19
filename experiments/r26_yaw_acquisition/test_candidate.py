import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.r25_composed.detector import Detector as R25
from experiments.r26_yaw_acquisition.candidate import Detector, YawReacquisitionMixin


class CandidateTests(unittest.TestCase):
    def detector(self):
        item = object.__new__(Detector)
        item.yaw_acquisition_identity = __import__(
            "experiments.r26_long.candidate", fromlist=["SignedYawOwnership"]
        ).SignedYawOwnership()
        item.yaw_acquisition_episode_active = False
        item.yaw_reacquisition_consumed = False
        item.yaw_reacquisition_count = 0
        item.yaw_reacquisition_provenance_complete = True
        item._yaw_acquisition_previous_active = False
        item._yaw_acquisition_gap_certified = False
        item.yaw_active = False
        item.yaw_value = 3.0
        item.pick_completion_active = False
        item.pick_event_active = False
        item.r8_base_detect = False
        item.r8_extra = False
        item.horizontal_detected = False
        item._shared = SimpleNamespace(detected=False)
        item.normal_model = SimpleNamespace(detected=False, last_active=True,
                                            last_error=np.array([2.0, 0.0]))
        item.light_witness = SimpleNamespace(state=SimpleNamespace(active=False))
        item.detached = SimpleNamespace(evidence=SimpleNamespace(active=False))
        item.continuation = SimpleNamespace(bridge=False)
        return item

    def update(self, item, active, value=3.0, report=None):
        def inherited(this, *args):
            this.yaw_active = active
            this.yaw_value = value
            return np.zeros(6), active if report is None else report

        with mock.patch.object(R25, "update", autospec=True, side_effect=inherited):
            return YawReacquisitionMixin.update(
                item, 1.0, None, None, None, None, True)[1]

    def prime_owned_event(self, item):
        for _ in range(4):
            self.assertTrue(self.update(item, True))
        self.assertFalse(self.update(item, False, value=3.0, report=False))
        self.assertTrue(item._yaw_acquisition_gap_certified)

    def test_native_gap_is_never_bridged_and_same_yaw_reacquisition_is_consumed(self):
        item = self.detector()
        self.prime_owned_event(item)
        self.assertFalse(self.update(item, True))
        self.assertTrue(item.yaw_reacquisition_consumed)
        self.assertFalse(self.update(item, True))
        self.assertEqual(item.yaw_reacquisition_count, 1)

    def test_independent_continuation_route_is_never_suppressed(self):
        item = self.detector()
        self.prime_owned_event(item)
        item.continuation.bridge = True
        self.assertTrue(self.update(item, True, report=True))

    def test_reversal_is_not_consumed(self):
        item = self.detector()
        self.prime_owned_event(item)
        self.assertTrue(self.update(item, True, value=-3.0))
        self.assertFalse(item.yaw_reacquisition_consumed)

    def test_unknown_route_fails_open(self):
        item = self.detector()
        self.prime_owned_event(item)
        del item.continuation
        self.assertTrue(self.update(item, True))
        self.assertFalse(item.yaw_reacquisition_provenance_complete)


if __name__ == "__main__":
    unittest.main()
