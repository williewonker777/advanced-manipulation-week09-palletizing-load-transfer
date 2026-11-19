import unittest
import numpy as np
from experiments.r30_source_association.candidate import MeasuredCoordinates, prefer_planar_notice


class Ownership(unittest.TestCase):
    def test_equal_energy_has_no_preference_and_axial_gives_way(self):
        self.assertFalse(prefer_planar_notice([[1,1],[1,1]],[1,0]))
        self.assertFalse(prefer_planar_notice([[4,1],[5,0]],[1,0]))
        self.assertTrue(prefer_planar_notice([[.2,1],[-.1,3]],[1,0]))

    def test_disagreement_or_invalid_has_no_optional_notice(self):
        for change,col in (([[.2,-3],[.1,4]],[1,0]),([[2,3],[2,1]],[1,0]),
                           ([[1,2],[1,2]],[0,0]),(None,[1,0]),([[1,2],[np.nan,2]],[1,0]),
                           ([[1,2],[1,2]],None)):
            self.assertFalse(prefer_planar_notice(change,col))

    def test_rigid_rotation_reversal_and_column_scaling_invariance(self):
        for angle in (0,.7,2.1):
            a=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            d=np.array([[.2,1.],[-.1,3.]])@a.T
            for sign in (-1.,1.):
                self.assertTrue(prefer_planar_notice(sign*d,3*a[:,0]))

    def test_coords_reset_on_unknown(self):
        c=MeasuredCoordinates()
        for _ in range(45):v=c.feed(np.zeros(6),True)
        np.testing.assert_array_equal(v,c.last)
        self.assertIsNone(c.feed(np.ones(6),False))
        self.assertIsNone(c.last)
