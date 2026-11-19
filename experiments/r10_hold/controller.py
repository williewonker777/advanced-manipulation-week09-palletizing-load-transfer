"""Motion R1 with a bounded transition to a stiffer stationary impedance.

No perturbation or contact labels enter this controller. A motion waypoint at
zero instantaneous speed is not a sustained constant-pose command.
"""
import numpy as np

from r1_torque import Controller as MotionController


class Controller(MotionController):
    def reset(self):
        super().reset()
        self._motion_wn = self.wn
        frequencies = np.sqrt(self.nom.K[self.elastic] / self.nom.B[self.elastic])
        self._hold_wn = max(self.wn, min(2 * self.wn, .2 * float(np.min(frequencies))))
        self._ramp_duration = 3 / self.wn
        self._steady_samples = 0
        self._last_target = None
        self._hold_mix = 0.

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        still = (not np.any(qd_d) and not np.any(qdd_d) and self._last_target is not None
                 and np.array_equal(q_d, self._last_target))
        self._last_target = np.array(q_d, copy=True)
        self._steady_samples = self._steady_samples + 1 if still else 0
        step = self.dt / self._ramp_duration
        if self._steady_samples >= 5:
            self._hold_mix = min(1., self._hold_mix + step)
        else:
            self._hold_mix = max(0., self._hold_mix - step)
        z = self._hold_mix
        self.wn = self._motion_wn + (self._hold_wn-self._motion_wn) * (z**3*(10+z*(-15+6*z)))
        return super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
