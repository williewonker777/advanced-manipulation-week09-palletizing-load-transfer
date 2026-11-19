import unittest
import numpy as np
from sim import trajectory
from sim.nominal import Nominal
from experiments.r11_datum.controller import frame
from experiments.r14_airborne import probe


class ProbeTests(unittest.TestCase):
    def test_offgrid_derivatives_and_unchanged_pose_position(self):
        h=1e-6;nom=Nominal()
        for t in (2.04357,2.22539,2.68154,3.22741):
            dv=probe.correction(t);a=probe.correction(t-h);b=probe.correction(t+h)
            np.testing.assert_allclose((b[:2]-a[:2])/(2*h),dv[1:],rtol=1e-5,atol=2e-7)
            q=trajectory.reference(t)[0];p,r=frame(nom,q);p2,r2=frame(nom,q+dv[0])
            np.testing.assert_allclose(p,p2,rtol=0,atol=1e-10)
            rel=r.T@r2
            self.assertAlmostEqual(np.arctan2(rel[0,2],rel[0,0]),probe.angle(t),places=9)

    def test_endpoints_and_outside_window(self):
        for s in (0.,probe.START-.002,probe.START,probe.STOP,probe.STOP+.002,5.99):
            self.assertFalse(np.any(probe.correction(s)))
        for s in (probe.START+1e-7,probe.STOP-1e-7):
            self.assertLess(np.linalg.norm(probe.correction(s)),1e-6)
        self.assertFalse(probe.table().flags.writeable)


if __name__=='__main__':unittest.main()
