"""Bounded, friction-aware holding impedance; motion controller is unchanged."""
import numpy as np
from sim import config as C
from r1_torque import Controller as MotionController


class Controller(MotionController):
    def reset(self):
        super().reset()
        self._last_hold_q = None
        self._quiet_count = 0
        self._hold_blend = 0.
        self._hold_k = np.zeros(6)
        self._hold_d = np.zeros(6)
        self._hold_ready = False

    def _hold_gains(self, q, grip):
        # Small-speed friction slope for a smoothed Coulomb law. Bounds on
        # the assumed drag are design margins, not measured plant parameters.
        nominal = self.nom_box if grip else self.nom
        inertia = nominal.B + np.diag(nominal.mass_matrix(q))
        rate = -np.log(.05) / .5
        kp, kd = self.nom.B*self.wn**2, 2*self.wn*self.nom.B
        lower_drag = .7*C.VISCOUS
        upper_drag = 1.3*(C.VISCOUS + C.COULOMB/C.FRICTION_EPS)
        # In each decoupled local model, move the quadratic s = z-rate.
        # Nonnegative shifted coefficients are a sufficient model check,
        # not a stability proof for the full nonlinear/coupled robot.
        self._hold_d = np.maximum(2*rate*inertia - kd - lower_drag, 0.)
        required = rate*(kd+self._hold_d+upper_drag) - rate**2*inertia - kp
        self._hold_k = np.maximum(0., required)
        self._hold_k[self.elastic] = np.minimum(
            self._hold_k[self.elastic], .10 * self.nom.K[self.elastic])
        self._hold_ready = True

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        still = (not np.any(qd_d) and not np.any(qdd_d) and self._last_hold_q is not None
                 and np.array_equal(q_d, self._last_hold_q))
        self._last_hold_q = np.array(q_d, copy=True)
        self._quiet_count = self._quiet_count + 1 if still else 0
        if self._quiet_count == 5:
            self._hold_gains(q_d, grip)
        step = self.dt*self.wn/3
        if self._quiet_count >= 5:
            self._hold_blend = min(1., self._hold_blend + step)
        else:
            self._hold_blend = max(0., self._hold_blend - step)
        if self._hold_blend == 0. or not self._hold_ready:
            return command
        z = self._hold_blend
        weight = z**3*(10. + z*(-15. + 6.*z))
        theta = q_d + self.inv_k*self.tau_link_prev
        feedback = self._hold_k*(theta-th) + self._hold_d*(qd_d-self.v_motor)
        # Bound only the added restoring effort (one nominal friction budget).
        # Never change the original actuator limits or suppress its feedback.
        return command + weight*C.COULOMB*np.tanh(feedback / C.COULOMB)
