import unittest

from experiments.r20_horizontal.candidate_detector import Detector


class CandidateLogicTest(unittest.TestCase):
    def test_threshold_is_direction_neutral(self):
        threshold = Detector.threshold(1.50)
        self.assertEqual(threshold, Detector.threshold(1.50))
        self.assertGreater(threshold, 18.0150567392)

    def test_public_phase_windows_do_not_name_collision_times(self):
        self.assertIsNone(Detector.threshold(1.20))
        self.assertIsNotNone(Detector.threshold(1.50))
        self.assertIsNone(Detector.threshold(2.00))
        self.assertIsNone(Detector.threshold(4.20))


if __name__ == "__main__":
    unittest.main()
