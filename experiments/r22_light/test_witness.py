import unittest
from experiments.r22_light.witness import LightWitness, eligible, lift_stop


class TrustedPortTests(unittest.TestCase):
    def test_valid_only_with_known_light_nonnegative_mass(self):
        for m in (0, .4563, 1):self.assertTrue(eligible(m))
        for m in (-1, float('nan'), 1.3,8):self.assertFalse(eligible(m))

    def test_heavy_arm_needs_no_auxiliary_nominal_or_clock(self):
        w=LightWitness(8,None,None)
        self.assertFalse(w.feed(1.5,None,None,True))
        self.assertIsNone(w.observer)

    def test_segment_end_comes_from_trajectory_not_impact_time(self):
        self.assertEqual(lift_stop([(0,None),(1.1,None),(1.86,None),(2.1,None)],1.1),1.86)
