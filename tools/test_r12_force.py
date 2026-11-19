import unittest
from unittest.mock import patch
import numpy as np

from tools import r12_normal_force as force
from tools.r12_force_audit import errors, targets


class Nominal:
    elastic = np.zeros(6, dtype=bool)
    K = np.ones(6)
    def inverse_dynamics(self,q,v,acc):
        return 2.*acc + .4*v


class ForceTests(unittest.TestCase):
    def setUp(self):
        self.s = force.NormalForce(Nominal(),Nominal(),dt=.001,viscous_spring=np.ones(6),
                                   jts_hz=300.)

    def test_kernel_and_step_time_alignment(self):
        self.assertAlmostEqual(float(force.KERNEL.sum()),1.)
        self.assertEqual((force.KERNEL[0],force.KERNEL[-1]),(0.,0.))
        np.testing.assert_allclose(force.KERNEL,force.KERNEL[::-1])
        pattern=np.arange(620.,dtype=float)
        ids=np.array([200,480])
        lagged,now=targets(pattern,ids,force.KERNEL,force.LAG)
        np.testing.assert_allclose(lagged,ids-18)
        np.testing.assert_array_equal(now,ids)

    def test_missing_packet_never_counts_as_a_success(self):
        self.assertEqual(errors([.1,np.nan],[.1,0]),{'available':1,'required':2,'bias':None,'rmse':None,'mae':None})
        with self.assertRaises(ValueError):errors([1.],[np.nan])
        with self.assertRaises(ValueError):targets(np.zeros(6),np.array([1]),force.KERNEL,force.LAG)

    @patch.object(force,'vertical_displacement',return_value=np.array([0.,0.,1.,0.,0.,0.]))
    def test_known_dynamics_signed_wrench_bias_delay_and_reset(self,_):
        bias=np.array([.4,-.25,1.3,.1,.2,-.3])
        one=np.array([0.,0.,1.,0.,0.,0.])
        # Analytic sinusoid-free constant acceleration and exact static sensor bias.
        for k in range(700):
            t=.001*k
            q=.5*t*t*one
            # At t, JTS reports the previous angle's torque.
            t_prev=max(0.,t-.001)
            load=0. if k<501 else -4.
            measured=2.*one+.4*t_prev*one + bias - load*one
            v=self.s.update(t,q,measured,k>=500)
            if k<532:self.assertIsNone(v)
            if k>=610:
                self.assertAlmostEqual(v.vertical_n,-4.,places=5)
                self.assertAlmostEqual(v.time-v.effective_time,.018)
        self.assertIsNone(self.s.update(.700,.5*.7**2*one,measured,False))

    @patch.object(force,'vertical_displacement',return_value=None)
    def test_singularity_or_absent_calibration_does_not_become_zero(self,_):
        for k in range(720):
            self.assertIsNone(self.s.update(k*.001,np.zeros(6),np.zeros(6),k>=500))
        other=force.NormalForce(Nominal(),Nominal(),dt=.001,viscous_spring=np.ones(6),jts_hz=300.)
        for k in range(600):
            self.assertIsNone(other.update(k*.001,np.zeros(6),np.zeros(6),True))

    def test_reordered_grid_and_invalid_inputs(self):
        self.s.update(0.,np.zeros(6),np.zeros(6),False)
        with self.assertRaises(ValueError):self.s.update(.002,np.zeros(6),np.zeros(6),False)
        with self.assertRaises(ValueError):self.s.update(.001,np.zeros(4),np.zeros(6),False)
        with self.assertRaises(ValueError):self.s.update(.001,np.zeros(6),np.full(6,np.inf),False)
