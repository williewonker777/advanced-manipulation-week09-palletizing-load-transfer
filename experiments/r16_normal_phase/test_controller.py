import unittest
from unittest.mock import patch
import numpy as np
from sim.nominal import Nominal
from experiments.r16_normal_phase import controller as c


class PhaseTests(unittest.TestCase):
    def test_lead_bounded_and_no_reversal(self):
        t=np.linspace(3.58,3.92,3001)
        d=np.array([c.phase_lead(v,3.6,3.9,.02) for v in t])
        self.assertGreaterEqual(float(d.min()),0.)
        self.assertAlmostEqual(float(d.max()),.010)
        self.assertTrue(np.all(np.diff(t+d)>0.))
        self.assertTrue(np.all(d[(t<=3.6)|(t>=3.9)]==0.))

    def test_regular_grid_exact_endpoint_and_empty_fallback(self):
        q=tuple(np.ones(6)*k for k in range(3))
        x=c.Controller.__new__(c.Controller);x._correction=None
        with patch.object(c.Early,'_reference',return_value=q):
            self.assertIs(x._reference(3.82)[0],q[0])
            x._first,x._last=3600,3900;x._correction=(np.ones((301,6)),)*3
            for t in (1.,3.599,3.600,3.900,3.901,4.):
                self.assertIs(x._reference(t)[0],q[0])
            np.testing.assert_allclose(x._reference(3.8)[0],q[0]+1.)

    def test_tabulated_fk_and_bound(self):
        x=c.Controller({},{}); x.nom=Nominal();x.nom_box=Nominal();x.box_mass=5.;x.box_com=(.01,0.,.02)
        x.nom_box.set_payload(x.box_mass,x.box_com)
        x.reset()
        self.assertGreater(x.phase_max_shift_m,.001)
        self.assertLessEqual(x.phase_max_shift_m,.002)
        self.assertLess(x.phase_ik_error_m,5e-9)
        delta=x._correction[0];kin=Nominal()
        for t in (3.720,3.800,3.855,3.880):
            q=x._base_reference(t)[0];p=kin.tcp_position(q)
            v=kin.tcp_position(q+delta[round(t/.001)-3600])-p
            self.assertLess(np.linalg.norm(v[:2]),5e-9)
            self.assertLessEqual(abs(v[2]),.002+5e-9)

if __name__=='__main__':
    unittest.main()
