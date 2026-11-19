import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.r24_composed.detector import Detector as R24Joint
from experiments.r25_pickup.candidate import Detector, PickupCompletionMixin, signed_completion


class SignedCompletionTest(unittest.TestCase):
    def test_requires_same_sign_nonincreasing_finite_momentum(self):
        self.assertEqual(signed_completion(3.0, 2.0), 1)
        self.assertEqual(signed_completion(-3.0, -2.0), -1)
        for before, after in ((3.0, 4.0), (3.0, -2.0), (0.0, 0.0),
                              (float("nan"), 1.0)):
            self.assertEqual(signed_completion(before, after), 0)


class CompletionMixinTest(unittest.TestCase):
    def detector(self):
        item = object.__new__(Detector)
        item.yaw_value = -3.0
        item.yaw_active = True
        item.yaw = SimpleNamespace(pick_continuity_selected=None)
        item.bounds = np.array([2.0, 2.5])
        item._shared = SimpleNamespace(detected=False)
        item.normal_model = None
        item.horizontal_detected = False
        item.light_witness = SimpleNamespace(state=SimpleNamespace(active=False))
        item.detached = SimpleNamespace(evidence=SimpleNamespace(active=False))
        item.pick_event_active = False
        item.pick_completion_active = False
        item.pick_completion_decided = False
        item.pick_completion_sign = 0
        item.pick_completion_selected = False
        item.pick_completion_reason = None
        return item

    def update(self, item, *, after=-2.0, active=True, selected=True, raw=True):
        def inherited(this, *args):
            this.yaw_value = after
            this.yaw_active = active
            this.yaw.pick_continuity_selected = selected
            this.pick_event_active = bool(selected)
            return np.zeros(6), raw

        with mock.patch.object(R24Joint, "update", autospec=True, side_effect=inherited):
            return PickupCompletionMixin.update(
                item, 1.4, None, None, None, None, True)[1]

    def test_confident_pick_completes_only_consumed_yaw_route(self):
        item = self.detector()
        self.assertFalse(self.update(item))
        self.assertTrue(item.pick_completion_selected)
        self.assertFalse(item.pick_event_active)

    def test_other_route_remains_live_during_consumed_yaw(self):
        item = self.detector()
        item._shared.detected = True
        self.assertTrue(self.update(item))

    def test_growing_or_reversed_switch_falls_back_to_r24(self):
        for after in (-4.0, 2.0):
            item = self.detector()
            self.assertTrue(self.update(item, after=after))
            self.assertFalse(item.pick_completion_selected)

    def test_above_bound_opposite_yaw_is_not_consumed(self):
        item = self.detector()
        self.assertFalse(self.update(item))
        item.yaw.pick_continuity_selected = True
        item.yaw_value = -2.0
        self.assertTrue(self.update(item, after=3.0, selected=True, raw=True))
        self.assertFalse(item.pick_completion_active)


if __name__ == "__main__":
    unittest.main()
