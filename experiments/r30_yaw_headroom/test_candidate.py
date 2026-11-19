import unittest
from experiments.r30_yaw_headroom.candidate import has_yaw_reserve


class HeadroomTest(unittest.TestCase):
    def test_pending_component_and_measured_moment_both_matter(self):
        d=((-1.,2.),(-.8,2.1))
        self.assertTrue(has_yaw_reserve(-1.,3.,d,(1.,0.)))
        self.assertFalse(has_yaw_reserve(-2.,3.,d,(1.,0.)))
        self.assertFalse(has_yaw_reserve(-3.1,3.,d,(1.,0.)))
        self.assertTrue(has_yaw_reserve(2.8,3.,d,(1.,0.)))

    def test_one_instrument_is_enough_to_defer(self):
        self.assertFalse(has_yaw_reserve(2.,3.,((.1,2.),(1.1,3.)),(1.,0.)))

    def test_invalid_tie_and_unseen_sources_stay_native(self):
        for v,b,d in ((None,3,((0,1),(0,1))),(0,float('inf'),((0,1),(0,1))),
                      (0,3,((1,1),(1,1))),(0,3,((0,1),(0,-1)))):
            self.assertFalse(has_yaw_reserve(v,b,d,(1.,0.)))
