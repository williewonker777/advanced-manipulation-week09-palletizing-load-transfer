"""R1 plus a bounded motor-equilibrium capture for sustained ungripped hold."""
from __future__ import annotations

import numpy as np

from r1_torque import Controller as MotionController
from sim import config as C


QUIET_SAMPLES = 5
CAPTURE_MIN_DURATION_S = 0.20
CAPTURE_MAX_DURATION_S = 0.60
CAPTURE_DURATION_STEP_S = 0.01
CAPTURE_SPEED_MAX = 0.12
CAPTURE_ACCELERATION_FRICTION_FRACTION = 0.50


def _coefficients(start, velocity, goal, duration):
    """Quintic Hermite coefficients with zero initial acceleration and final v/a."""
    delta = goal - start
    return np.stack((
        start,
        velocity,
        np.zeros(6),
        (10.0*delta - 6.0*velocity*duration)/duration**3,
        (-15.0*delta + 8.0*velocity*duration)/duration**4,
        (6.0*delta - 3.0*velocity*duration)/duration**5,
    ))


def _evaluate(coefficients, elapsed):
    powers = np.array([1.0, elapsed, elapsed**2, elapsed**3, elapsed**4, elapsed**5])
    first = np.array([0.0, 1.0, 2.0*elapsed, 3.0*elapsed**2,
                      4.0*elapsed**3, 5.0*elapsed**4])
    second = np.array([0.0, 0.0, 2.0, 6.0*elapsed, 12.0*elapsed**2,
                       20.0*elapsed**3])
    return powers@coefficients, first@coefficients, second@coefficients


class Controller(MotionController):
    """Capture the measured motor state onto the nominal stationary equilibrium."""

    def reset(self):
        super().reset()
        self._last_public_q = None
        self._quiet_samples = 0
        self._capture = None
        self.capture_sizing_valid = None

    def _capture_gains(self, q):
        inertia = self.nom.B + np.diag(self.nom.mass_matrix(q))
        rate = -np.log(0.05)/0.5
        base_k = self.nom.B*self.wn**2
        base_d = 2.0*self.wn*self.nom.B
        lower_drag = 0.7*C.VISCOUS
        upper_drag = 1.3*(C.VISCOUS + C.COULOMB/C.FRICTION_EPS)
        damping = np.maximum(2.0*rate*inertia - base_d - lower_drag, 0.0)
        stiffness = np.maximum(
            rate*(base_d+damping+upper_drag) - rate**2*inertia - base_k, 0.0)
        stiffness[self.elastic] = np.minimum(
            stiffness[self.elastic], 0.10*self.nom.K[self.elastic])
        return stiffness, damping

    def _begin_capture(self, th, q_d):
        zeros = np.zeros(6)
        tau_link = self.nom.inverse_dynamics(q_d, zeros, zeros)
        goal = q_d + self.inv_k*tau_link
        velocity = np.clip(self.v_motor, -CAPTURE_SPEED_MAX, CAPTURE_SPEED_MAX)
        acceleration_limit = (CAPTURE_ACCELERATION_FRICTION_FRACTION
                              * C.COULOMB/self.nom.B)
        duration = CAPTURE_MIN_DURATION_S
        valid = False
        while duration <= CAPTURE_MAX_DURATION_S + 1e-12:
            coefficients = _coefficients(th, velocity, goal, duration)
            grid = np.linspace(0.0, duration, 101)
            values = [_evaluate(coefficients, sample) for sample in grid]
            peak_velocity = np.max(np.abs([value[1] for value in values]), axis=0)
            peak_acceleration = np.max(np.abs([value[2] for value in values]), axis=0)
            if (np.all(peak_velocity <= CAPTURE_SPEED_MAX + 1e-12)
                    and np.all(peak_acceleration <= acceleration_limit + 1e-12)):
                valid = True
                break
            duration += CAPTURE_DURATION_STEP_S
        self.capture_sizing_valid = valid
        if not valid:
            return None
        stiffness, damping = self._capture_gains(q_d)
        entry_pd = self.nom.B*(self.wn**2*(goal-th)-2.0*self.wn*self.v_motor)
        return {
            "coefficients": coefficients,
            "duration": duration,
            "goal": goal,
            "stiffness": stiffness,
            "damping": damping,
            "correction_limit": np.abs(entry_pd)+C.COULOMB,
        }

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        still = (not grip and not np.any(qd_d) and not np.any(qdd_d)
                 and self._last_public_q is not None
                 and np.array_equal(q_d, self._last_public_q))
        self._last_public_q = np.array(q_d, copy=True)
        self._quiet_samples = self._quiet_samples + 1 if still else 0
        if not still:
            self._capture = None
            self.capture_sizing_valid = None
            return command
        if self._quiet_samples == QUIET_SAMPLES:
            self._capture = self._begin_capture(th, q_d)
        if self._capture is None:
            return command

        elapsed = min((self._quiet_samples-QUIET_SAMPLES)*self.dt,
                      self._capture["duration"])
        theta, theta_dot, theta_ddot = _evaluate(
            self._capture["coefficients"], elapsed)
        goal = self._capture["goal"]

        # Replace the stationary R1 motor reference by the virtual settling
        # reference.  Subtracting the R1 restoring term expected on that path
        # prevents double-counting the force that R1 already supplies.
        path_feedforward = (self.nom.B*theta_ddot + C.VISCOUS*theta_dot
                            + C.COULOMB*np.tanh(theta_dot/C.FRICTION_EPS))
        r1_restoring_on_path = self.nom.B*(
            self.wn**2*(goal-theta)-2.0*self.wn*theta_dot)
        substitution = path_feedforward-r1_restoring_on_path
        blend_duration = 1.0/self.wn
        phase = min(elapsed/blend_duration, 1.0)
        blend = phase**3*(10.0+phase*(-15.0+6.0*phase))
        substitution = np.clip(blend*substitution,
                               -self._capture["correction_limit"],
                               self._capture["correction_limit"])

        # The extra disturbance reserve is referenced to the capture path, so
        # inherited tracking lag does not consume it at capture entry.
        feedback = (self._capture["stiffness"]*(theta-th)
                    + self._capture["damping"]*(theta_dot-self.v_motor))
        reserve = C.COULOMB*np.tanh(feedback/C.COULOMB)
        return command + substitution + reserve
