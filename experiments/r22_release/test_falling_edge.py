import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.r22_release import falling_edge


class EventShapeTests(unittest.TestCase):
    def detector(self):
        item = object.__new__(falling_edge.Detector)
        item.release_event_active = False
        item.release_event_peak = 0.0
        item.release_fall_count = 0
        item._release_yaw_was_active = False
        item.yaw_active = False
        item.yaw_value = None
        item._shared = SimpleNamespace(detected=False)
        item.horizontal_detected = False
        return item

    def feed(self, item, active, value):
        def underlying(_self, _t):
            item.yaw_active, item.yaw_value = active, value
            return None, active
        with patch.object(falling_edge.Continuous, "update", underlying):
            return item.update(0.0)[1]

    def test_falling_edge_ends_without_rearming_inside_same_excursion(self):
        item = self.detector()
        outputs = [self.feed(item, True, value) for value in (10, 11, 9, 8, 7, 12)]
        self.assertEqual(outputs, [True, True, True, True, False, False])

    def test_new_excursion_rearms_after_underlying_quiet(self):
        item = self.detector()
        for value in (10, 8, 7, 6):
            self.feed(item, True, value)
        self.assertFalse(self.feed(item, False, 0))
        self.assertTrue(self.feed(item, True, 5))

    def test_baseline_alarm_is_never_shortened(self):
        item = self.detector()
        item._shared.detected = True
        self.assertTrue(self.feed(item, False, 0))
