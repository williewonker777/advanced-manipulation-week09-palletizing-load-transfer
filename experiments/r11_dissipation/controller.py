"""R1 plus a passive-range, memoryless measured-speed friction compensator."""
from __future__ import annotations

import numpy as np

from r1_torque import Controller as MotionController
from sim import config as C


QUIET_SAMPLES = 5
MIN_FRICTION_SCALE = 0.65
COMPENSATION_FRACTION = 0.50
COMPENSATION_SPEED_MAX = C.FRICTION_EPS
RESIDUAL_FRICTION_FRACTION = MIN_FRICTION_SCALE-COMPENSATION_FRACTION


def safe_compensation_speed(raw_speed, filtered_speed):
    """Return a same-sign speed no larger than either public speed estimate."""
    raw_speed = np.asarray(raw_speed)
    filtered_speed = np.asarray(filtered_speed)
    same_sign = raw_speed*filtered_speed > 0.0
    magnitude = np.minimum(np.minimum(np.abs(raw_speed), np.abs(filtered_speed)),
                           COMPENSATION_SPEED_MAX)
    return np.where(same_sign, np.sign(raw_speed)*magnitude, 0.0)


class Controller(MotionController):
    """Partially cancel kinetic friction while retaining positive net dissipation."""

    def reset(self):
        super().reset()
        self._last_public_q = None
        self._quiet_samples = 0
        self.compensated_axis_samples = 0
        self.max_abs_injection = np.zeros(6)
        self.max_injected_power_w = 0.0
        self.injected_energy_j = 0.0
        self.nominal_viscous_energy_j = 0.0
        self.nominal_coulomb_energy_j = 0.0
        self.min_nominal_residual_power_w = np.inf

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        still = (not grip and not np.any(qd_d) and not np.any(qdd_d)
                 and self._last_public_q is not None
                 and np.array_equal(q_d, self._last_public_q))
        self._last_public_q = np.array(q_d, copy=True)
        self._quiet_samples = self._quiet_samples + 1 if still else 0
        if self._quiet_samples < QUIET_SAMPLES:
            return command

        velocity = safe_compensation_speed(thd, self.v_motor)
        nominal_friction = (C.VISCOUS*velocity
                            + C.COULOMB*np.tanh(velocity/C.FRICTION_EPS))
        injection = COMPENSATION_FRACTION*nominal_friction
        injected_power = float(np.sum(injection*thd))
        viscous_power = float(np.sum(C.VISCOUS*thd**2))
        coulomb_power = float(np.sum(
            C.COULOMB*np.tanh(thd/C.FRICTION_EPS)*thd))
        self.compensated_axis_samples += int(np.count_nonzero(velocity))
        self.max_abs_injection = np.maximum(self.max_abs_injection,
                                            np.abs(injection))
        self.max_injected_power_w = max(self.max_injected_power_w,
                                        injected_power)
        self.injected_energy_j += injected_power*self.dt
        self.nominal_viscous_energy_j += viscous_power*self.dt
        self.nominal_coulomb_energy_j += coulomb_power*self.dt
        self.min_nominal_residual_power_w = min(
            self.min_nominal_residual_power_w,
            viscous_power+coulomb_power-injected_power)
        return command+injection
