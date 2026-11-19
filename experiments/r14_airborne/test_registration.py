import unittest
import numpy as np
from tools import r13_grasp_estimator as old
from experiments.r14_airborne import registration as reg


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(92)
        n = len(old.CENTRES)
        self.jac = self.rng.normal(0., .11, size=(n,6,3))
        self.stiff = self.rng.normal(size=(n,6))
        self.bias = self.rng.normal(0.,.02,size=(n,6))
        self.noise = np.ones(6)*.1
        self.fits = [self.feature(m) for m in (3.,4.,5.,6.,7.,8.)]
        self.model = reg.calibrate(self.fits, self.noise)

    def feature(self, mass, residual=0., prior_shift=0.):
        mass=float(mass)
        return dict(mass=mass, y=(1.+.04*mass)*self.bias-self.jac[:,:,2]*residual,
                    jac=self.jac.copy(), stiffness=self.stiff.copy(), vertical=np.array([0.,0.,-1.]),
                    prior_bottom_mm=-.7+.04*mass+prior_shift,prior_com_mm=np.zeros(3))

    def test_moving_latch_cancels_apparent_vertical_correction(self):
        p=reg.decide(self.feature(5.5,residual=5.,prior_shift=-4.8),self.model)
        self.assertAlmostEqual(p['com_residual_height_mm'],5.,places=4)
        self.assertAlmostEqual(p['extra_height_mm'],.2,places=4)
        self.assertFalse(p['accept'])
        self.assertEqual(p['applied_mm'],0.)

    def test_real_bottom_change_neglected_by_abstention_is_not_counted_as_a_gain(self):
        p=reg.decide(self.feature(5.5,residual=-5.), self.model)
        self.assertTrue(p['accept'])
        self.assertAlmostEqual(p['applied_mm'],-5.,places=4)
        outside=reg.decide(self.feature(9.5,residual=-5.),self.model)
        self.assertFalse(outside['accept'])
        self.assertEqual(outside['applied_mm'],0.)

    def test_scaled_gram_and_information_match_independent_augmented_solve(self):
        f=self.feature(4.4,2.,-.55)
        mod=old.fit_normal(self.fits,self.noise)
        co,blocks,info=reg._raw(f,mod)
        pt,fr,ic=old._one(f,mod,np.ones(len(old.CENTRES),bool))
        np.testing.assert_allclose(co[:3],pt,rtol=1e-8,atol=1e-9)
        self.assertAlmostEqual(co[3],fr,places=8)
        self.assertAlmostEqual(info,ic,places=8)
        for i,b in enumerate(blocks):
            disp,_,_=old._one(f,mod,(old.CENTRES//80)%5!=i)
            self.assertAlmostEqual(b,float(f['vertical']@disp),places=8)

    def test_invalid_orientation_and_guard_inputs(self):
        f=self.feature(5.);f['vertical']=np.zeros(3)
        with self.assertRaises(ValueError):reg.decide(f,self.model)
        with self.assertRaises(ValueError):reg.decide(self.feature(5.),{})
        with self.assertRaises(ValueError):reg.calibrate(self.fits[:4],self.noise)


if __name__=='__main__':unittest.main()
