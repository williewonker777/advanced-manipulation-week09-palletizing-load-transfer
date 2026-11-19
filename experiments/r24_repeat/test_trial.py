import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.r22_guarded.candidate import GuardedReleaseBalance
from experiments.r23_detached.free import Detector as Free
from experiments.r24_repeat.trial import Detector, PickupContinuousBalance


class PickupBalanceTest(unittest.TestCase):
    def prime(self, active):
        state = PickupContinuousBalance(2.0)
        row = np.linspace(1.0, 2.0, 6)
        for tick in range(80):
            th = np.linspace(0.0, 0.05, 6) * tick
            state.feed(th, np.zeros(6), row, False)
        state.pre_pick_yaw_active = active
        value = state.feed(np.linspace(0.0, 0.05, 6) * 80,
                           np.zeros(6), row * 1.4, True)
        return state, value

    def test_active_pick_compensates_model_momentum_without_reset(self):
        state, value = self.prime(True)
        self.assertTrue(state.pick_continuity_selected)
        self.assertTrue(state.grip)
        self.assertIsNotNone(value)

    def test_quiet_pick_retains_inherited_reset(self):
        state, value = self.prime(False)
        self.assertFalse(state.pick_continuity_selected)
        self.assertTrue(state.grip)
        self.assertIsNone(value)

    def test_state_is_guarded_release_extension(self):
        self.assertTrue(issubclass(PickupContinuousBalance, GuardedReleaseBalance))


class EventFusionTest(unittest.TestCase):
    def detector(self):
        item = object.__new__(Detector)
        item.yaw_active = True
        item.yaw = SimpleNamespace(pre_pick_yaw_active=False,
                                   pick_continuity_selected=True)
        item._shared = SimpleNamespace(last_score=0.7, last_fast_score=0.0,
                                       release_persistence=8)
        item.pick_event_active = False
        item.pick_event_quiet = 0
        item.pick_continuity_consumed = False
        return item

    def feed(self, item, raw=False):
        with mock.patch.object(Free, "update", return_value=(np.zeros(6), raw)):
            return Detector.update(item, 1.4, None, None, None, None, True)[1]

    def test_existing_sensor_suspicion_bridges_only_selected_pick_event(self):
        item = self.detector()
        self.assertTrue(self.feed(item))
        self.assertTrue(item.pick_continuity_consumed)
        item.yaw.pick_continuity_selected = True
        item._shared.last_score = 0.0
        for _ in range(7):
            self.assertTrue(self.feed(item))
        self.assertFalse(self.feed(item))

    def test_consumed_selection_cannot_restart_finished_event(self):
        item = self.detector()
        self.feed(item)
        item._shared.last_score = 0.0
        for _ in range(8):
            result = self.feed(item)
        self.assertFalse(result)
        self.assertFalse(self.feed(item))


if __name__ == "__main__":
    unittest.main()
