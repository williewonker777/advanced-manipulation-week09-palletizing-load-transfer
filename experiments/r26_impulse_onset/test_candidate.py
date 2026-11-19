import unittest

import numpy as np

from experiments.r26_impulse_onset.event import OnsetEvent


LEVEL = {"horizontal": 4.0, "positive_vertical": 4.0, "negative_vertical": 3.0}
CHANGE = {"horizontal": 5.0, "positive_vertical": 4.0, "negative_vertical": 4.0}


class OnsetEventTests(unittest.TestCase):
    def event(self):
        return OnsetEvent(LEVEL, CHANGE, lag=2, pulse_samples=3)

    def feed(self, event, force, **kw):
        defaults = dict(slow=np.zeros(3), fast=np.zeros(3), cf=False,
                        slow_count=0, fast_count=0, learned=False)
        defaults.update(kw)
        return event.feed(np.asarray(force, float), **defaults)

    def test_signed_rise_emits_native_length_transient_and_sustained_level(self):
        event = self.event()
        self.feed(event, [0, 0, 0])
        self.feed(event, [0, 0, 1])
        extra, masked = self.feed(event, [0, 0, 6])
        self.assertTrue(extra)
        self.assertFalse(masked)
        self.assertTrue(event.onset)
        self.assertTrue(event.sustained)
        self.assertAlmostEqual(event.signed_change_n, 6.0)
        self.assertTrue(self.feed(event, [0, 0, 6])[0])
        self.assertTrue(self.feed(event, [0, 0, 6])[0])
        self.assertFalse(self.feed(event, [0, 0, 6])[0])

    def test_change_below_floor_does_not_trigger(self):
        event = self.event()
        for z in (2, 2, 4.5, 4.6):
            extra, _ = self.feed(event, [0, 0, z])
            self.assertFalse(extra)

    def test_learned_source_is_never_masked(self):
        event = self.event()
        self.feed(event, [0, 0, 0])
        self.feed(event, [0, 0, 1])
        self.feed(event, [0, 0, 6])
        _, masked = self.feed(event, [0, 0, 6], cf=True, slow=[0, 0, 8],
                              slow_count=4, learned=True)
        self.assertFalse(masked)

    def test_direction_mismatch_does_not_mask_counterfactual(self):
        event = self.event()
        self.feed(event, [0, 0, 0])
        self.feed(event, [0, 0, 1])
        self.feed(event, [0, 0, 6])
        _, masked = self.feed(event, [0, 0, 6], cf=True, slow=[0, 0, -8], slow_count=4)
        self.assertFalse(masked)

    def test_unload_rearms_without_clock_blackout(self):
        event = self.event()
        for force in ([0, 0, 0], [0, 0, 1], [0, 0, 6], [0, 0, 0],
                      [0, 0, 0], [0, 0, -1]):
            self.feed(event, force)
        extra, _ = self.feed(event, [0, 0, -6])
        self.assertTrue(extra)
        self.assertTrue(event.onset)


if __name__ == "__main__":
    unittest.main()
