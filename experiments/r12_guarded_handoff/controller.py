"""Simulation-only position compliance for the world-vertical torque component.

Unlike the initial prototype this preserves the original torque law whenever
support has not armed it, including *after* release. A late observation cannot
jump an unfinished blend to 1 when the weld opens.
"""
import numpy as np

import r1_torque
from sim import trajectory as T


def curve(x):
    if x <= 0.:
        return 0.
    if x >= 1.:
        return 1.
    return x**3*(10.+x*(-15.+6.*x))


def transfer_weight(t, started, release, rise=.020, recover=.040):
    if started is None:
        return 0.
    if t < release:
        return curve((t-started)/rise)
    # Both value and first two derivatives meet because arming is prohibited
    # unless enough time remains to finish the C2 rise.
    return curve((release-started)/rise)*(1.-curve((t-release)/recover))


def normal_part(jz, position):
    squared = float(jz @ jz)
    if squared <= 1e-12:
        return np.zeros_like(position)
    return jz * (float(jz @ position)/squared)


class Controller(r1_torque.Controller):
    start_before_release = .120
    rise = .020
    recover = .040
    target_carried = .45
    normal_stiffness = .20
    support_threshold = .05

    def reset(self):
        super().reset()
        self.arm_time = None
        self.blend = 0.
        self._qr = None

    def _reference(self, t):
        q = super()._reference(t)
        self._qr = q[0]
        return q

    def _estimate_load(self, t, th, thd, tau_s, grip):
        base = super()._estimate_load(t, th, thd, tau_s, grip)
        release = T.T_RELEASE
        estimator = self.shared_estimator
        if (self.arm_time is None and self.box_mass > 0 and grip
                and release-self.start_before_release <= t <= release-self.rise
                and estimator.lambda_support_valid
                and 1.-estimator.lambda_support >= self.support_threshold):
            self.arm_time = float(t)
        self.blend = transfer_weight(t, self.arm_time, release, self.rise, self.recover)
        if not self.blend:
            return base
        if self.UNLOAD_START < t < release:
            base = min(base,1.-self.UNLOAD_AMOUNT*curve((t-self.UNLOAD_START)/(release-self.UNLOAD_START)))
        return (1.-self.blend)*base + self.blend*self.target_carried

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        if not self.blend:
            return command
        motor_goal = self._qr + self.inv_k*self.tau_link_prev
        position_torque = self.nom.B*(self.wn**2*(motor_goal-th))
        jz = self.nom.jacobian(self._qr)[2]
        return command - self.blend*(1.-self.normal_stiffness)*normal_part(jz,position_torque)
