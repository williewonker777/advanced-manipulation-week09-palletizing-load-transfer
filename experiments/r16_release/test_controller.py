import unittest
from unittest.mock import patch
import numpy as np
from experiments.r16_release import controller as c


class MotionTests(unittest.TestCase):
    def test_boundary_c2_and_size(self):
        start,r,stop=3.86,3.9,3.98
        fn=lambda t: np.asarray(c.release_motion(t,start,r,stop-r))
        np.testing.assert_array_equal(fn(start), [0.,0.,0.])
        np.testing.assert_allclose(fn(r), [.0001962,.00981,0.],atol=1e-13)
        np.testing.assert_array_equal(fn(stop), [0.,0.,0.])
        for t in (start,r,stop):
            for sign in (-1,1):
                np.testing.assert_allclose(fn(t+sign*1e-9),fn(t),atol=1e-6)
        values=np.array([fn(t) for t in np.linspace(start,stop,241)])
        self.assertLess(np.max(np.abs(values[:,0])),.0005)
        self.assertLess(np.max(np.abs(values[:81,2])),.05*9.81)

    def test_derivatives(self):
        f=lambda t: np.asarray(c.release_motion(t,3.86,3.9,.08))
        for t in np.linspace(3.861,3.979,37):
            deriv=(f(t+1e-7)-f(t-1e-7))/(2e-7)
            np.testing.assert_allclose(deriv[:2],f(t)[1:],rtol=1e-7,atol=1e-9)

    def test_no_assumed_contact_or_short_horizon_and_rank_fallback(self):
        q=tuple(np.arange(6.)+k for k in range(3))
        for mass,armed,t in ((0.,True,3.86),(5.,False,3.86),(5.,True,3.885),(5.,True,3.81),(5.,True,3.87)):
            ctrl=c.Controller.__new__(c.Controller)
            ctrl.box_mass,ctrl.hybrid_active,ctrl.relief_start=mass,armed,None
            ctrl._reference_at=lambda _:q
            ctrl._flange_pose_jacobian=lambda _:(None,None,np.zeros((6,6)))
            with patch.object(c.Early,'_reference',return_value=q):
                self.assertIs(ctrl._reference(t)[0],q[0])
                self.assertIsNone(ctrl.relief_start)

if __name__=='__main__':
    unittest.main()
