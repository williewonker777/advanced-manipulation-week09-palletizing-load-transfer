"""Stock-only registered world-z handoff with fixed flange orientation.

The control law is the frozen R12 hybrid.  Registration only changes the
reference supplied to that law; a rejected decision or nonpositive payload is
an exact zero-action path.  Calibration is an explicit caller-owned dictionary.
"""
from __future__ import annotations

import mujoco
import numpy as np

import r1_torque
from sim import config as C
from sim import trajectory as T
from experiments.r12_hybrid.controller import Controller as Hybrid
from experiments.r12_hybrid.controller import smoothstep, scale_vertical
from experiments.r14_airborne.registration import decide as default_decide
from experiments.r14_online.stream import Stream as DefaultStream

RAMP_START = 3.55
RAMP_END = 3.79
FLAT_END = T.T_RELEASE + .025
TAPER_END = 4.24
MAX_HEIGHT_MM = 8.0


def _c2(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x**3 * (10.0 + x * (-15.0 + 6.0*x))


def height_weight(t):
    t = float(t)
    if t <= RAMP_START or t >= TAPER_END:
        return 0.0
    if t < RAMP_END:
        return _c2((t-RAMP_START)/(RAMP_END-RAMP_START))
    if t <= FLAT_END:
        return 1.0
    return 1.0-_c2((t-FLAT_END)/(TAPER_END-FLAT_END))


def clamp_height_mm(value):
    return float(np.clip(float(value), -MAX_HEIGHT_MM, MAX_HEIGHT_MM))


def _rotation_vector(target, current):
    delta = target @ current.T
    vector = .5*np.array([delta[2,1]-delta[1,2], delta[0,2]-delta[2,0],
                         delta[1,0]-delta[0,1]])
    sine = float(np.linalg.norm(vector))
    cosine = float(np.clip((np.trace(delta)-1.)*.5, -1., 1.))
    angle = float(np.arctan2(sine, cosine))
    return vector if sine < 1e-14 else vector*(angle/sine)


class Controller(Hybrid):
    """Frozen R12 hybrid plus a causal, fixed-orientation registered reference."""
    def __init__(self, calibration=None, stream_factory=None, decide_fn=None):
        self.registration_calibration = calibration
        self._stream_factory = stream_factory
        self._decide_fn = decide_fn

    def reset(self):
        super().reset()
        self.registration_finished = False
        self.registration_result = None
        self.registration_accepted = False
        self.registration_finish_t = None
        self.registered_height_m = 0.0
        self.registration_stream = None
        self.ik_max_position_error_m = 0.0
        self.ik_max_orientation_error_rad = 0.0
        if float(self.box_mass) <= 0.0:
            self.registration_finished = True
            self.registration_result = {'accept': False, 'reason': 'nonpositive_payload'}
            return
        if not isinstance(self.registration_calibration, dict):
            raise ValueError('explicit stock torque registration calibration required')
        factory = self._stream_factory or DefaultStream
        self.registration_stream = factory(float(self.box_mass), tuple(self.box_com))

    def _finish_registration(self, t):
        if self.registration_finished or t < RAMP_START:
            return
        features = self.registration_stream.finish()
        decision = (self._decide_fn or default_decide)(features, self.registration_calibration)
        if not isinstance(decision, dict) or not {'extra_height_mm','accept','half_width_mm'} <= set(decision):
            raise ValueError('invalid registration decision')
        values = np.asarray([decision['extra_height_mm'], decision['half_width_mm']], float)
        if not np.all(np.isfinite(values)) or values[1] < 0.0:
            raise ValueError('invalid registration values')
        self.registration_result = dict(decision)
        self.registration_accepted = bool(decision['accept'])
        self.registered_height_m = (-.001*clamp_height_mm(values[0])
                                    if self.registration_accepted else 0.0)
        self.registration_finished = True
        self.registration_finish_t = float(t)

    def _base_reference(self, t):
        return r1_torque.Controller._reference(self, float(t))

    def _flange_pose_jacobian(self, q):
        self.nom.d.qpos[:] = np.asarray(q, float)
        mujoco.mj_forward(self.nom.m, self.nom.d)
        body = self.nom.m.body('link_6').id
        position = self.nom.d.xpos[body].copy()
        rotation = self.nom.d.xmat[body].reshape(3,3).copy()
        jacp = np.zeros((3, self.nom.m.nv))
        jacr = np.zeros_like(jacp)
        mujoco.mj_jacBody(self.nom.m, self.nom.d, jacp, jacr, body)
        return position, rotation, np.vstack((jacp, jacr))

    def _full_pose(self, base_q, dz):
        if dz == 0.0:
            return base_q
        target_p, target_r, _ = self._flange_pose_jacobian(base_q)
        target_p = target_p + np.array([0.0, 0.0, float(dz)])
        q = np.asarray(base_q, float).copy()
        for _ in range(8):
            p, r, jac = self._flange_pose_jacobian(q)
            error = np.r_[target_p-p, _rotation_vector(target_r, r)]
            if np.linalg.norm(error[:3], ord=np.inf) < 2e-11 and np.linalg.norm(error[3:], ord=np.inf) < 2e-11:
                break
            q += np.linalg.solve(jac.T@jac + 1e-12*np.eye(6), jac.T@error)
        p, r, _ = self._flange_pose_jacobian(q)
        pe = float(np.linalg.norm(target_p-p, ord=np.inf))
        re = float(np.linalg.norm(_rotation_vector(target_r, r)))
        self.ik_max_position_error_m = max(self.ik_max_position_error_m, pe)
        self.ik_max_orientation_error_rad = max(self.ik_max_orientation_error_rad, re)
        if pe > 5e-9 or re > 5e-9:
            raise RuntimeError('full-pose IK did not converge')
        return q

    def _reference_at(self, t):
        q, qd, qdd = self._base_reference(t)
        weight = height_weight(t)
        if self.registered_height_m == 0.0 or weight == 0.0:
            return q, qd, qdd
        return self._full_pose(q, self.registered_height_m*weight), qd, qdd

    def _reference(self, t):
        q, qd, qdd = self._reference_at(float(t))
        if self.registered_height_m == 0.0 or height_weight(t) == 0.0:
            return q, qd, qdd
        dt = float(C.DT)
        qm = self._reference_at(max(0.0, float(t)-dt))[0]
        qp = self._reference_at(min(T.T_END, float(t)+dt))[0]
        return q, (qp-qm)/(2.*dt), (qp-2.*q+qm)/(dt*dt)

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        if not self.registration_finished:
            self.registration_stream.observe(float(t), np.asarray(th,float).copy(),
                                             np.asarray(tau_s,float).copy(), int(grip))
            self._finish_registration(float(t))
        # This is the pinned R12 hybrid compute, with only its reference source
        # replaced by the fixed-orientation reference above.
        q_ref, qd_ref, qdd_ref = self._reference(t)
        base = self._estimate_load(t, th, thd, tau_s, grip)
        if self.UNLOAD_START < t < T.T_RELEASE:
            s = np.clip((t-self.UNLOAD_START)/(T.T_RELEASE-self.UNLOAD_START),0,1)
            base = min(base, 1-self.UNLOAD_AMOUNT*smoothstep(s))
        alpha = self._alpha(t); p = self.parameters
        load = (1-alpha)*base + alpha*p.robot_support_fraction
        if t >= T.T_RELEASE:
            load = alpha*p.robot_support_fraction
        self.hybrid_alpha=float(alpha); self.max_alpha=max(self.max_alpha,float(alpha))
        self.commanded_load=float(load)
        self.normal_position_scale=float(1-alpha*(1-p.normal_position_scale))
        self.minimum_normal_position_scale_seen=min(self.minimum_normal_position_scale_seen,self.normal_position_scale)
        if T.T_RELEASE-.0015 <= t < T.T_RELEASE: self.pre_release_load=float(load)
        if t >= T.T_RELEASE and self.release_load is None: self.release_load=float(load)
        tau_empty=self.nom.inverse_dynamics(q_ref,qd_ref,qdd_ref)
        tau_full=self.nom_box.inverse_dynamics(q_ref,qd_ref,qdd_ref)
        tau_link=tau_empty+load*(tau_full-tau_empty); theta=q_ref+self.inv_k*tau_link
        self.v_motor += .2*(thd-self.v_motor)
        if self.tau_link_prev is None:
            theta_dot,theta_ddot=qd_ref.copy(),qdd_ref.copy()
        else:
            raw=(tau_link-self.tau_link_prev)/self.dt; old=self.tau_dot.copy()
            self.tau_dot += .04*(raw-self.tau_dot)
            theta_dot=np.clip(qd_ref+self.inv_k*self.tau_dot,qd_ref-.35,qd_ref+.35)
            theta_ddot=np.clip(qdd_ref+self.inv_k*(self.tau_dot-old)/self.dt,-25.,25.)
        self.tau_link_prev=tau_link.copy()
        friction=C.VISCOUS*theta_dot+C.COULOMB*np.tanh(theta_dot/C.FRICTION_EPS)
        ff=self.nom.B*theta_ddot+tau_link+friction
        position=self.nom.B*(self.wn**2*(theta-th))
        damping=self.nom.B*(2*self.wn*(theta_dot-self.v_motor))
        jz=self.nom.jacobian(q_ref)[2]
        position=scale_vertical(position,jz,self.normal_position_scale)
        return ff+position+damping+self.k_tau*(tau_s-tau_link)
