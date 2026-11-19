"""Minimum-distortion scalar time reparameterization of servo R1.

The spatial path is unchanged.  A C2 endpoint-matched scalar clock advances
slightly early, slows through the nominal J2 demand peak, and is exactly back
on R1 twenty milliseconds before release.  Reset-time bisection chooses the
smallest clock shift whose full-payload nominal motor feedforward respects a
fixed physical headroom constraint.
"""
from __future__ import annotations

import numpy as np

import r1_servo
from sim import config as C, trajectory as T


class Controller(r1_servo.Controller):
    """R1 with a preview-constrained smooth approach clock."""

    CLOCK_START = 3.60
    CLOCK_END = 3.88
    MAX_SHIFT_S = 0.004
    # 346 - 12 Nm measured normal drive-correction envelope - 10 Nm reserve.
    PLANNED_J2_FF_LIMIT_NM = 324.0
    PREVIEW_START = 3.40
    PREVIEW_END = 3.89
    BISECTION_STEPS = 20

    @classmethod
    def clock(cls, t, amplitude):
        """Return warped time and its first two derivatives."""
        t = float(t)
        amplitude = float(amplitude)
        if t <= cls.CLOCK_START or t >= cls.CLOCK_END or amplitude == 0.0:
            return t, 1.0, 0.0
        duration = cls.CLOCK_END - cls.CLOCK_START
        x = (t - cls.CLOCK_START) / duration
        # Unit-height C2 bump: b, b', b'' vanish as required at endpoints.
        b = 64.0 * (x**3 - 3.0*x**4 + 3.0*x**5 - x**6)
        bx = 64.0 * (3.0*x**2 - 12.0*x**3 + 15.0*x**4 - 6.0*x**5)
        bxx = 64.0 * (6.0*x - 36.0*x**2 + 60.0*x**3 - 30.0*x**4)
        return (t + amplitude*b,
                1.0 + amplitude*bx/duration,
                amplitude*bxx/duration**2)

    def _warped_reference(self, t, amplitude):
        warped, rate, acceleration = self.clock(t, amplitude)
        q, qd, qdd = super()._reference(warped, *T.reference(warped))
        return q, qd*rate, qdd*rate**2 + qd*acceleration

    def _planned_j2_ff_peak(self, amplitude):
        previous_link = None
        filtered_link_rate = np.zeros(6)
        peak = 0.0
        for t in np.arange(self.PREVIEW_START,
                           self.PREVIEW_END + 0.5*self.dt, self.dt):
            q, qd, qdd = self._warped_reference(float(t), amplitude)
            link = self.nom_box.inverse_dynamics(q, qd, qdd)
            if previous_link is None:
                theta_dot, theta_ddot = qd.copy(), qdd.copy()
            else:
                raw_rate = (link - previous_link) / self.dt
                old_rate = filtered_link_rate.copy()
                filtered_link_rate += 0.04 * (raw_rate - filtered_link_rate)
                theta_dot = np.clip(qd + self.inv_k*filtered_link_rate,
                                    qd - 0.35, qd + 0.35)
                theta_ddot = np.clip(
                    qdd + self.inv_k*(filtered_link_rate - old_rate)/self.dt,
                    -25.0, 25.0)
            friction = (C.VISCOUS*theta_dot
                        + C.COULOMB*np.tanh(theta_dot/C.FRICTION_EPS))
            feedforward = self.nom.B*theta_ddot + link + friction
            peak = max(peak, float(abs(feedforward[1])))
            previous_link = link
        return peak

    def reset(self):
        super().reset()
        zero_peak = self._planned_j2_ff_peak(0.0)
        if zero_peak <= self.PLANNED_J2_FF_LIMIT_NM:
            amplitude = 0.0
        else:
            maximum_peak = self._planned_j2_ff_peak(self.MAX_SHIFT_S)
            if maximum_peak > self.PLANNED_J2_FF_LIMIT_NM:
                amplitude = self.MAX_SHIFT_S
            else:
                low, high = 0.0, self.MAX_SHIFT_S
                for _ in range(self.BISECTION_STEPS):
                    middle = 0.5*(low + high)
                    if self._planned_j2_ff_peak(middle) <= self.PLANNED_J2_FF_LIMIT_NM:
                        high = middle
                    else:
                        low = middle
                amplitude = high
        self.clock_amplitude_s = float(amplitude)
        self.clock_zero_peak_nm = float(zero_peak)
        self.clock_selected_peak_nm = float(self._planned_j2_ff_peak(amplitude))
        samples = [self.clock(t, amplitude)[1]
                   for t in np.linspace(self.CLOCK_START, self.CLOCK_END, 1001)]
        self.clock_rate_min = float(min(samples))
        self.clock_rate_max = float(max(samples))
        self.clock_active = bool(amplitude > 0.0)

    def _reference(self, t, q_d, qd_d, qdd_d):
        del q_d, qd_d, qdd_d
        return self._warped_reference(float(t), self.clock_amplitude_s)
