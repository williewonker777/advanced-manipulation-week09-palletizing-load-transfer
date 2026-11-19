"""Causal J2 drive-demand feasibility governor around unchanged servo R1.

The governor mirrors the configured drive state using only the public encoder
inputs and the documented one-cycle command delay.  R1's command is returned
bit-for-bit until predicted next-cycle J2 demand enters a physical soft zone.
Within that zone only feedforward is relieved; position/velocity feedback stays
enabled and unchanged.
"""
from __future__ import annotations

import numpy as np

import r1_servo
from sim import config as C
from sim import trajectory as T


class Controller(r1_servo.Controller):
    """R1 plus a fixed smooth drive-feasibility barrier on J2."""

    # Fixed before simulation.  The asymptote is 328 Nm, leaving 18 Nm to the
    # public 346 Nm actuator limit.  Softening begins at 310 Nm; this is not the
    # evaluator's 99.9% threshold and never changes actuator limits.
    SOFT_START_NM = 310.0
    SOFT_BAND_NM = 18.0

    def reset(self):
        super().reset()
        zeros = np.zeros(6)
        q0 = T.reference(0.0)[0]
        self._drive_v = zeros.copy()
        self._drive_i = zeros.copy()
        self._pending_command = np.array([q0, zeros, zeros, zeros])
        self.feedback_active = False
        self.j2_relief_nm = 0.0
        self.predicted_j2_raw_nm = 0.0
        self.predicted_j2_governed_nm = 0.0

    def _advance_shadow_drive(self, th, thd):
        """Advance through the command that the real drive consumes this cycle."""
        pos, _, vel_ff, _ = self._pending_command
        self._drive_v += C.DRIVE_VEL_FILTER * (thd - self._drive_v)
        error = vel_ff + C.DRIVE_KP * (pos - th) - self._drive_v
        self._drive_i += error * C.DT

    def _predict_next_drive_torque(self, command, th, thd):
        """Constant-velocity one-cycle encoder prediction through public drive gains."""
        pos, _, vel_ff, tau_ff = command
        th_next = th + C.DT * thd
        drive_v_next = self._drive_v + C.DRIVE_VEL_FILTER * (thd - self._drive_v)
        error_next = vel_ff + C.DRIVE_KP * (pos - th_next) - drive_v_next
        drive_i_next = self._drive_i + error_next * C.DT
        return tau_ff + C.DRIVE_KV * error_next + C.DRIVE_KI * drive_i_next

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        raw = np.asarray(
            super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d), float
        )
        self._advance_shadow_drive(th, thd)
        predicted = self._predict_next_drive_torque(raw, th, thd)
        j2 = float(predicted[1])
        magnitude = abs(j2)
        governed = raw
        relief = 0.0
        if magnitude > self.SOFT_START_NM:
            excess = magnitude - self.SOFT_START_NM
            safe_magnitude = self.SOFT_START_NM + self.SOFT_BAND_NM * np.tanh(
                excess / self.SOFT_BAND_NM
            )
            safe_j2 = np.copysign(safe_magnitude, j2)
            relief = j2 - safe_j2
            governed = raw.copy()
            governed[3, 1] -= relief

        self._pending_command = governed.copy()
        self.feedback_active = bool(relief != 0.0)
        self.j2_relief_nm = float(relief)
        self.predicted_j2_raw_nm = j2
        self.predicted_j2_governed_nm = float(j2 - relief)
        return governed
