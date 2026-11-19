import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.r24_loaded.control import subtract_aligned
from experiments.r24_loaded.dc import MODEL, changed, measurements
from experiments.r24_loaded.drive import DriveEstimate
from experiments.r24_loaded.response_state import ConeHistory, SupportHistory


class SupportTest(unittest.TestCase):
    def test_signed_credits_cannot_create_or_reverse_components(self):
        for value in (-5., 0., 5.):
            for credit in (-8., -1., 0., 1., 8., float('nan')):
                y=subtract_aligned(value,credit)
                self.assertLessEqual(abs(y),abs(value));self.assertGreaterEqual(y*value,0.)
                if (np.isfinite(credit) and value*credit<=0) or not np.isfinite(credit):self.assertEqual(y,value)

    def test_pressure_band_state_matches_direct_past_only_convolution(self):
        z=SupportHistory();window=np.zeros(40)
        for x in np.random.default_rng(414).uniform(0,80,180):
            window=np.r_[x,window[:-1]]
            np.testing.assert_allclose(z.push(x),window.reshape(5,8).mean(axis=1),rtol=1e-13,atol=1e-12)
        cold=SupportHistory()
        np.testing.assert_array_equal(cold.push(0.),np.zeros(5))

    def test_cone_requires_measured_and_projected_sign_agreement(self):
        h=ConeHistory(1)
        self.assertEqual(h.signed_credit(-2.,3.,1.),0.)
        self.assertEqual(h.signed_credit(-2.,-3.,1.),-1.)
        self.assertEqual(h.signed_credit(2.,.5,2.),.5)

    def test_dc_and_slope_must_identify_same_joint_and_same_sign(self):
        lo=np.full((2,3),.99);hi=np.full((2,3),1.01);ok=np.ones(3,bool)
        self.assertFalse(changed(np.array([[.98,1,1],[1,.98,1]]),ok,lo,hi))
        self.assertFalse(changed(np.array([[.98,1,1],[1.02,1,1]]),ok,lo,hi))
        self.assertTrue(changed(np.array([[.98,1,1],[.98,1,1]]),ok,lo,hi))
        self.assertFalse(changed(np.array([[.98,1,1],[.98,1,1]]),np.array([False,True,True]),lo,hi))
        self.assertFalse(changed(None,None,lo,hi))

    def test_bias_affects_dc_but_is_not_itself_a_slope_change(self):
        est=SimpleNamespace(x=np.tile([20.,-40.,0.],(20,1)),y=np.tile([24.,-40.,0.],(20,1)),gain=np.ones(3))
        val,valid=measurements(est);np.testing.assert_allclose(val[1],[1.2,1.,0.])
        np.testing.assert_array_equal(valid,[True,True,False])
        self.assertFalse(changed(val,valid,np.full((2,3),.98),np.full((2,3),1.02)))

    def test_early_grip_abstains_and_never_uses_later_impacts_to_commission(self):
        z=DriveEstimate()
        for _ in range(48):z.feed(np.zeros(6),np.zeros(6),np.zeros(6),False)
        z.feed(np.ones(6),np.ones(6),np.ones(6),True)
        for _ in range(1100):z.feed(np.ones(6),np.ones(6)*1e4,np.ones(6),False)
        self.assertTrue(z.closed);self.assertIsNone(z.gain)

    def test_frozen_nominal_gain_records_are_inside_their_own_bounds(self):
        m=json.loads(Path(MODEL).read_bytes());lo,hi=np.array(m['low']),np.array(m['high'])
        self.assertEqual(len(m['fit_records']),29)
        for r in m['fit_records']:
            self.assertFalse(changed(np.array(r['estimates']),np.array(r['eligible']),lo,hi))


if __name__=='__main__':unittest.main()
