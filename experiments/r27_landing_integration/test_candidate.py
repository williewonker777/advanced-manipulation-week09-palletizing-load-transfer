import unittest

import numpy as np

from experiments.r27_landing_integration.candidate import GripForceEpisode


class GripForceEpisodeTest(unittest.TestCase):
    def test_requires_arming_and_two_consecutive_gripped_samples(self):
        event = GripForceEpisode(10.0, 0.8, 2, 3)
        self.assertFalse(event.feed(np.array([0.0, 0.0, 0.0]), False, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertTrue(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertTrue(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertTrue(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, False))

    def test_incumbent_owns_episode_and_no_rearm_without_physical_release(self):
        event = GripForceEpisode(10.0, 0.8, 2, 3)
        event.feed(np.zeros(3), False, False)
        event.feed(np.array([0.0, 11.0, 0.0]), True, True)
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, True))
        for _ in range(5):
            self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertFalse(event.feed(np.zeros(3), True, False))
        self.assertFalse(event.feed(np.array([11.0, 0.0, 0.0]), True, False))
        self.assertTrue(event.feed(np.array([11.0, 0.0, 0.0]), True, False))


if __name__ == "__main__":
    unittest.main()
