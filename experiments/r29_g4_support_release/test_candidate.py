import unittest

import numpy as np

from experiments.r29_g4_support_release.candidate import Detector


class CandidateStateTest(unittest.TestCase):
    def item(self):
        value = Detector.__new__(Detector)
        value.release_bounds = {"horizontal": 4.0, "positive_vertical": 3.0,
                                "negative_vertical": 2.0}
        return value

    def test_signed_classifier_requires_negative_vertical_dominance(self):
        item = self.item()
        self.assertEqual(item._ratios(np.array([1.0, 0.0, -3.0])),
                         (0.25, 1.5, "negative_vertical"))
        self.assertEqual(item._ratios(np.array([5.0, 0.0, -1.0])),
                         (1.25, 0.5, "negative_vertical"))

    def test_unavailable_or_nonfinite_vector_fails_open(self):
        item = self.item()
        self.assertIsNone(item._ratios(None))
        self.assertIsNone(item._ratios(np.array([0.0, np.nan, 0.0])))


if __name__ == "__main__":
    unittest.main()
