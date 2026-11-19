"""Pre-release bottom-centre pitch damping for physical load transfer."""
from __future__ import annotations

import numpy as np

from experiments.r19_load_transfer.controller import Controller as Transfer
from sim import config as C


STOPPING_HORIZON_S = 0.040
MOMENT_WEIGHT_FRACTION = 0.03
REFERENCE_LEVER_M = 0.10
GRAVITY_M_S2 = 9.81


def stopping_moment(effective_inertia, angular_velocity, limit):
    values = np.asarray([effective_inertia, angular_velocity, limit], dtype=float)
    if not np.isfinite(values).all() or effective_inertia <= 0.0 or limit < 0.0:
        raise ValueError("finite positive inertia and nonnegative limit required")
    raw = -float(effective_inertia)*float(angular_velocity)/STOPPING_HORIZON_S
    return float(np.clip(raw, -float(limit), float(limit)))


class Controller(Transfer):
    """R19 plus a bounded physical world-y angular stopping moment."""

    def reset(self):
        super().reset()
        self.pitch_damping_samples = 0
        self.pitch_damping_impulse_nms = 0.0
        self.pitch_damping_max_moment_nm = 0.0
        self.pitch_damping_max_delta_nm = 0.0
        self.pitch_damping_trace = []

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
        mass = np.asarray(self.nom_box.mass_matrix(q), dtype=float) + np.diag(self.nom_box.B)
        compliance = float(jy @ np.linalg.solve(mass, jy))
        if not np.isfinite(compliance) or compliance <= 0.0:
            raise ValueError("invalid world-y operational compliance")
        effective_inertia = 1.0/compliance
        angular_velocity = float(jy @ velocity)
        limit = MOMENT_WEIGHT_FRACTION*self.box_mass*GRAVITY_M_S2*REFERENCE_LEVER_M
        moment = stopping_moment(effective_inertia, angular_velocity, limit)
        delta = moment*jy
        output = command + delta
        self.pitch_damping_samples += 1
        self.pitch_damping_impulse_nms += abs(moment)*C.DT
        self.pitch_damping_max_moment_nm = max(self.pitch_damping_max_moment_nm, abs(moment))
        self.pitch_damping_max_delta_nm = max(self.pitch_damping_max_delta_nm,
                                               float(np.max(np.abs(delta))))
        if len(self.pitch_damping_trace) < 256:
            self.pitch_damping_trace.append({
                "t": float(t), "omega_y_rad_s": angular_velocity,
                "effective_inertia_kgm2": effective_inertia,
                "moment_nm": moment, "limit_nm": limit,
                "max_joint_delta_nm": float(np.max(np.abs(delta))),
            })
        return output
