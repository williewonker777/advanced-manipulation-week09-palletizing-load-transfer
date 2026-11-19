import unittest
import numpy as np

from experiments.r11_velocity_ilc.controller import basis


class ProfileTests(unittest.TestCase):
    def test_c2_compact_support(self):
        start,end=1.,1.3
        for j,center in enumerate([.51,.71,.91,1.09]):
            for edge in [center-.26,center+.26]:
                np.testing.assert_allclose(basis(start+(end-start)*edge,start,end)[:,j],0.,atol=1.e-10)
        np.testing.assert_array_equal(basis([.8,1.6],start,end), np.zeros((2,3,4)))
        with self.assertRaises(ValueError):
            basis([1.],1.,.9)

    def test_independent_finite_differences(self):
        t=1.25101; eps=1.e-7
        b=basis(t,1.,1.3); plus=basis(t+eps,1.,1.3);minus=basis(t-eps,1.,1.3)
        np.testing.assert_allclose((plus[:2]-minus[:2])/(2*eps),b[1:],rtol=3.e-8,atol=2.e-6)
        # Fit and deploy use identical column order in scalar and vector paths.
        np.testing.assert_array_equal(b,basis(np.array([t]),1.,1.3)[0])
