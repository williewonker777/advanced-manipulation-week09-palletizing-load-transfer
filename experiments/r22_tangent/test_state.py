import math
import unittest

from experiments.r22_tangent.state import Evidence, load_class


class TangentialEvidenceTests(unittest.TestCase):
    def test_load_groups_follow_actual_carried_mass(self):
        self.assertEqual([load_class(v,True) for v in (0, .35, 1, 1.1, 4, 5.5, 8)], [0,1,1,2,2,3,3])
        self.assertEqual(load_class(8,False),0)
        for v in (-1,8.01,math.nan):
            self.assertIsNone(load_class(v,True))

    def test_clear_on_missing_or_load_switch(self):
        e=Evidence()
        flags=[e.feed([4.,3.],0,4.9) for _ in range(14)]
        self.assertEqual(flags,[False]*13+[True])
        self.assertFalse(e.feed(None,0,4.9))
        self.assertIsNone(e.score)
        for _ in range(18):e.feed([4.,3.],0,4.9)
        self.assertFalse(e.feed([4.,3.],1,4.9))
        self.assertIsNone(e.score)

    def test_average_signed_force_then_take_magnitude(self):
        e=Evidence()
        for i in range(40):
            self.assertFalse(e.feed([10.*(-1)**i,0.],0,1.))
        self.assertEqual(e.score,0)
        with self.assertRaises(ValueError):e.feed([float('nan'),0],0,1.)
