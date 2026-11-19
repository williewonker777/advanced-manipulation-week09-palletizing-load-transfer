"""Minimum-intervention endpoint advance chosen from planned J2 headroom.

At reset, the controller predicts its own full-payload motor feedforward over
the smooth R1 release-approach profile.  It retains the largest endpoint
advance whose planned J2 feedforward leaves a fixed reserve for the unchanged
inner drive correction.  Runtime command arithmetic remains inherited R1.
"""
import numpy as np

import r1_servo
from sim import config as C
from sim import trajectory as T


class Controller(r1_servo.Controller):
    """R1 with reset-time feedforward-headroom selection."""

    # Observed root-cause attribution found 11.19--11.77 Nm of normal drive
    # correction at the two limiting peaks.  Round upward rather than fitting
    # the actuator threshold exactly.
    DRIVE_CORRECTION_RESERVE_NM = 12.5
    DELAY_CANDIDATES = (-0.005, -0.004, -0.003, -0.002, -0.001, 0.0)
    PLAN_START = T.T_RELEASE - 0.5
    PLAN_END = T.T_RELEASE - 0.06

    def _planned_j2_ff_peak(self, endpoint_delay):
        original = self.ENDPOINT_DELAY
        self.ENDPOINT_DELAY = float(endpoint_delay)
        previous_link = None
        filtered_link_rate = np.zeros(6)
        peak = 0.0
        try:
            for t in np.arange(self.PLAN_START, self.PLAN_END + 0.5 * self.dt, self.dt):
                q_ref, qd_ref, qdd_ref = self._reference(float(t), *T.reference(float(t)))
                link = self.nom_box.inverse_dynamics(q_ref, qd_ref, qdd_ref)
                if previous_link is None:
                    theta_dot, theta_ddot = qd_ref.copy(), qdd_ref.copy()
                else:
                    raw_rate = (link - previous_link) / self.dt
                    old_rate = filtered_link_rate.copy()
                    filtered_link_rate += 0.04 * (raw_rate - filtered_link_rate)
                    theta_dot = np.clip(qd_ref + self.inv_k * filtered_link_rate,
                                        qd_ref - 0.35, qd_ref + 0.35)
                    theta_ddot = np.clip(qdd_ref + self.inv_k *
                                         (filtered_link_rate - old_rate) / self.dt,
                                         -25.0, 25.0)
                friction = (C.VISCOUS * theta_dot +
                            C.COULOMB * np.tanh(theta_dot / C.FRICTION_EPS))
                feedforward = self.nom.B * theta_ddot + link + friction
                peak = max(peak, float(abs(feedforward[1])))
                previous_link = link
        finally:
            self.ENDPOINT_DELAY = original
        return peak

    def reset(self):
        self.ENDPOINT_DELAY = r1_servo.Controller.ENDPOINT_DELAY
        super().reset()
        limit = float(C.TAU_MAX[1] - self.DRIVE_CORRECTION_RESERVE_NM)
        self.planned_j2_peaks_nm = {}
        selected = self.DELAY_CANDIDATES[-1]
        for delay in self.DELAY_CANDIDATES:
            peak = self._planned_j2_ff_peak(delay)
            self.planned_j2_peaks_nm[str(delay)] = peak
            if peak <= limit:
                selected = delay
                break
        self.ENDPOINT_DELAY = float(selected)
        self.selected_endpoint_delay = float(selected)
        self.endpoint_advance_guarded = bool(
            selected != r1_servo.Controller.ENDPOINT_DELAY
        )
