import unittest

import numpy as np

from experiments.r26_weak.lifecycle import PulseLifecycle


def feed(gate, scores, incumbent=()):
    owned = set(incumbent)
    return [gate.feed(np.array([score, 0.0]), score, True, i in owned)
            for i, score in enumerate(scores)]


class PulseLifecycleTest(unittest.TestCase):
    def test_unowned_pulse_releases_only_on_trailing_decay(self):
        gate = PulseLifecycle(onset=5, release_samples=2, hold=3, rearm=2)
        outputs = feed(gate, [0, 6, 7, 8, 7, 3, 2, 0, 0])
        self.assertEqual(outputs, [False, False, False, False, False,
                                   False, True, True, True])

    def test_incumbent_anywhere_in_life_consumes_optional_shot(self):
        gate = PulseLifecycle(onset=5, release_samples=2, hold=3)
        self.assertFalse(any(feed(gate, [6, 7, 8, 7, 3, 2, 0], incumbent=(3,))))

    def test_unconfirmed_long_excursion_is_discarded_not_time_released(self):
        gate = PulseLifecycle(onset=5, maximum_episode=4, release_samples=2, hold=3)
        self.assertFalse(any(feed(gate, [6, 7, 8, 8, 8, 8, 8, 8])))

    def test_ineligible_episode_never_starts(self):
        gate = PulseLifecycle(onset=5)
        self.assertFalse(gate.feed(np.array([9.0, 0.0]), 9.0, False, False))
        self.assertEqual(gate.phase, "quiet")


if __name__ == "__main__":
    unittest.main()
