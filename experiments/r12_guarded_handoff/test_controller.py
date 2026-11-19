import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

import r1_torque
from experiments.r12_guarded_handoff.controller import Controller, normal_part, transfer_weight
from sim import trajectory as T


class HandoffTests(unittest.TestCase):
    def test_preserves_tangent_and_does_not_amplify_position_torque(self):
        j=np.array([0.,.4,-.37,.2,.32,0.])
        x=np.array([1.,-2.,4.,6.,3.,-1.])
        n=normal_part(j,x)
        extra=x-.8*n
        self.assertAlmostEqual(float(j@(x-n)),0.)
        self.assertAlmostEqual(float(j@extra),.2*float(j@x))
        self.assertLessEqual(np.linalg.norm(extra),np.linalg.norm(x))
        np.testing.assert_array_equal(normal_part(np.zeros(6),x),np.zeros(6))

    def test_absent_and_late_support_inherit_baseline_across_release(self):
        obj=Controller()
        obj.arm_time=None;obj.box_mass=6.;obj.shared_estimator=SimpleNamespace(lambda_support_valid=True,lambda_support=.1)
        with patch.object(r1_torque.Controller,'_estimate_load',return_value=.73):
            for t in [T.T_RELEASE-.019, T.T_RELEASE-.001,T.T_RELEASE,T.T_RELEASE+.015]:
                self.assertEqual(obj._estimate_load(t,None,None,None,t<T.T_RELEASE),.73)
                self.assertIsNone(obj.arm_time)
        self.assertEqual(transfer_weight(T.T_RELEASE,None,T.T_RELEASE),0.)

    def test_timely_support_is_continuous_and_returns_to_r1(self):
        obj=Controller()
        obj.arm_time=None;obj.box_mass=6.;obj.shared_estimator=SimpleNamespace(lambda_support_valid=True,lambda_support=.1)
        with patch.object(r1_torque.Controller,'_estimate_load',return_value=.7):
            release=T.T_RELEASE
            self.assertEqual(obj._estimate_load(release-.035,None,None,None,True),.7)
            self.assertAlmostEqual(obj._estimate_load(release-1e-6,None,None,None,True),.45)
            self.assertAlmostEqual(obj._estimate_load(release,None,None,None,False),.45)
            self.assertAlmostEqual(obj._estimate_load(release+.060,None,None,None,False),.7)
            self.assertEqual(obj.blend,0.)
