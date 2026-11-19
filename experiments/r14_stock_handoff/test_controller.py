import unittest
import hashlib
import numpy as np
import mujoco

from sim import config as C
from sim.nominal import Nominal
from experiments.r14_stock_handoff.controller import Controller, height_weight, _rotation_vector, RAMP_START, RAMP_END, FLAT_END, TAPER_END
from experiments.r14_stock_handoff.detector import _load, DEFAULT_COEFFICIENTS


def pose(nom,q):
    nom.d.qpos[:]=q; mujoco.mj_kinematics(nom.m,nom.d)
    b=nom.m.body('link_6').id
    return nom.d.xpos[b].copy(),nom.d.xmat[b].reshape(3,3).copy()


class StubStream:
    def finish(self): return {'x':1}


class Tests(unittest.TestCase):
    def test_c2(self):
        self.assertEqual([height_weight(t) for t in (RAMP_START,RAMP_END,FLAT_END,TAPER_END)],[0.,1.,1.,0.])
        h=1e-5
        for t in (RAMP_START,RAMP_END,FLAT_END,TAPER_END):
            self.assertLess(abs((height_weight(t+h)-height_weight(t-h))/(2*h)),2e-5)
    def test_decision_negates_measured_height(self):
        c=Controller.__new__(Controller);c.registration_finished=False;c.registration_stream=StubStream()
        c._decide_fn=lambda f,m:{'extra_height_mm':-12.,'half_width_mm':1.,'accept':True}
        c.registration_calibration={};c.registration_result=None;c.registration_accepted=False;c.registration_finish_t=None;c.registered_height_m=0.
        c._finish_registration(RAMP_START)
        self.assertEqual(c.registered_height_m,.008)
    def test_full_pose_manufactured(self):
        c=Controller.__new__(Controller);c.nom=Nominal();c.dt=float(C.DT);c.ik_max_position_error_m=0.;c.ik_max_orientation_error_rad=0.
        for t in (3.79,3.90):
            q=c._base_reference(t)[0];p0,r0=pose(c.nom,q)
            for dz in (-.008,.008):
                q1=c._full_pose(q,dz);p1,r1=pose(c.nom,q1)
                self.assertLess(np.max(np.abs((p1-p0)-[0,0,dz])),5e-9)
                self.assertLess(np.max(np.abs(r1-r0)),1e-8)
    def test_rotation_error_has_newton_sign(self):
        c=Controller.__new__(Controller);c.nom=Nominal();c.dt=float(C.DT)
        q=c._base_reference(3.90)[0];_,target,jac=c._flange_pose_jacobian(q)
        dq=np.array([2e-5,-3e-5,1e-5,4e-5,-2e-5,3e-5])
        _,current,_=c._flange_pose_jacobian(q+dq)
        error=_rotation_vector(target,current)
        self.assertLess(np.max(np.abs(error + jac[3:]@dq)),2e-8)

    def test_coefficients_match_evaluated_arrays_without_private_archive(self):
        # Digests calculated independently from the originally evaluated arrays,
        # not from the public JSON's own metadata. Use explicit byte order.
        arrays = _load(DEFAULT_COEFFICIENTS)[:2]
        expected = (
            ('wrench_coef', '442c9dd6f08fa2b6a864910562a7219be96a6ca62a293fd5b6c69e633bf3ae9a'),
            ('fast_coef', 'a33d83cb0199b8bdeb45aa70f10aa9990a0453e8ee083c19775964348fbea5da'),
        )
        for value, (_, digest) in zip(arrays, expected):
            self.assertEqual(value.shape, (6000, 6, 2))
            self.assertEqual(hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest(), digest)
        private = DEFAULT_COEFFICIENTS.parents[2]/'experiments/r12_counterfactual/model.npz'
        if private.is_file():
            with np.load(private, allow_pickle=False) as original:
                for value, (key, _) in zip(arrays, expected):
                    np.testing.assert_array_equal(value, original[key])

if __name__=='__main__': unittest.main()
