"""Support-deadband passive pitch damping for physical load transfer."""
from __future__ import annotations

import numpy as np

from experiments.r19_load_transfer.controller import Controller as Transfer
from experiments.r23_transfer.controller import (
    GRAVITY_M_S2,
    MOMENT_WEIGHT_FRACTION,
    REFERENCE_LEVER_M,
    stopping_moment,
)
from experiments.r23_transfer.decoupled import normal_acceleration_nullspace
from sim import config as C


MINIMUM_SAFE_SUPPORT = 0.25
ESTABLISHED_SUPPORT = 0.50


def c2_progress(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def support_deadband_weight(table_support_fraction, handoff_fraction):
    """C2 admission across the pre-existing quarter-to-half-weight band."""
    values = np.asarray([table_support_fraction, handoff_fraction], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("finite support weights required")
    support = float(np.clip(table_support_fraction, 0.0, 1.0))
    progress = (support - MINIMUM_SAFE_SUPPORT) / (ESTABLISHED_SUPPORT - MINIMUM_SAFE_SUPPORT)
    return c2_progress(progress) * float(np.clip(handoff_fraction, 0.0, 1.0))


class Controller(Transfer):
    """R19 plus a passive pitch damper behind a physical support deadband."""

    def reset(self):
        super().reset()
        self.support_damping_samples = 0
        self.support_damping_impulse_nms = 0.0
        self.support_damping_max_moment_nm = 0.0
        self.support_damping_max_delta_nm = 0.0
        self.support_damping_max_normal_after = 0.0
        self.support_damping_trace = []

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = np.asarray(super().compute(t, th, thd, tau_s, grip,
                                             q_d, qd_d, qdd_d), dtype=float)
        packet = self.shared_estimator
        eligible = (self.box_mass > 0.0 and bool(grip) and self.hybrid_active
                    and packet.lambda_support_valid and self.hybrid_alpha > 0.0)
        if not eligible:
            return command
        q = np.asarray(packet.q_prev, dtype=float)
        velocity = np.asarray(packet.v, dtype=float)
        jacobian, _, _ = packet._point_jacobian(q)
        jy = np.asarray(jacobian[4], dtype=float)
        jz = np.asarray(jacobian[2], dtype=float)
        mass = np.asarray(self.nom_box.mass_matrix(q), dtype=float) + np.diag(self.nom_box.B)
        compliance = float(jy @ np.linalg.solve(mass, jy))
        if not np.isfinite(compliance) or compliance <= 0.0:
            raise ValueError("invalid world-y operational compliance")
        effective_inertia = 1.0 / compliance
        angular_velocity = float(jy @ velocity)
        table_support = float(np.clip(1.0 - packet.lambda_support, 0.0, 1.0))
        weight = support_deadband_weight(table_support, self.hybrid_alpha)
        limit = MOMENT_WEIGHT_FRACTION * self.box_mass * GRAVITY_M_S2 * REFERENCE_LEVER_M
        moment = weight * stopping_moment(effective_inertia, angular_velocity, limit)
        raw_delta = moment * jy
        delta, normal_after, _ = normal_acceleration_nullspace(raw_delta, mass, jz)
        output = command + delta
        self.support_damping_samples += 1
        self.support_damping_impulse_nms += abs(moment) * C.DT
        self.support_damping_max_moment_nm = max(self.support_damping_max_moment_nm,
                                                 abs(moment))
        self.support_damping_max_delta_nm = max(self.support_damping_max_delta_nm,
                                                float(np.max(np.abs(delta))))
        self.support_damping_max_normal_after = max(
            self.support_damping_max_normal_after, abs(normal_after))
        if len(self.support_damping_trace) < 256:
            self.support_damping_trace.append({
                "t": float(t), "omega_y_rad_s": angular_velocity,
                "table_support_fraction": table_support,
                "handoff_fraction": float(self.hybrid_alpha),
                "support_weight": weight, "moment_nm": moment,
                "hard_limit_nm": limit,
                "max_joint_delta_nm": float(np.max(np.abs(delta))),
                "normal_acceleration_after_m_s2": normal_after,
            })
        return output
