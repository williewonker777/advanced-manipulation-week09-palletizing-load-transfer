"""Pitch damping projected out of estimated bottom-contact normal acceleration."""
from __future__ import annotations

import numpy as np

from experiments.r19_load_transfer.controller import Controller as Transfer
from experiments.r23_transfer.controller import (
    GRAVITY_M_S2,
    MOMENT_WEIGHT_FRACTION,
    REFERENCE_LEVER_M,
    stopping_moment,
)
from sim import config as C


def normal_acceleration_nullspace(torque, mass_matrix, normal_jacobian):
    """Minimum Euclidean torque change satisfying Jz M^-1 delta == 0."""
    torque = np.asarray(torque, dtype=float)
    mass_matrix = np.asarray(mass_matrix, dtype=float)
    normal_jacobian = np.asarray(normal_jacobian, dtype=float)
    if torque.shape != (6,) or mass_matrix.shape != (6, 6) or normal_jacobian.shape != (6,):
        raise ValueError("six-axis torque, mass matrix, and normal Jacobian required")
    if not (np.isfinite(torque).all() and np.isfinite(mass_matrix).all()
            and np.isfinite(normal_jacobian).all()):
        raise ValueError("finite projection inputs required")
    row = np.linalg.solve(mass_matrix.T, normal_jacobian)
    scale = float(row @ row)
    if not np.isfinite(scale) or scale <= 1e-16:
        raise ValueError("ill-conditioned contact-normal acceleration row")
    projected = torque - row*float(row @ torque)/scale
    residual = float(normal_jacobian @ np.linalg.solve(mass_matrix, projected))
    return projected, residual, float(normal_jacobian @ np.linalg.solve(mass_matrix, torque))


class Controller(Transfer):
    """Frozen phase-one pitch law with exact estimated normal-acceleration decoupling."""

    def reset(self):
        super().reset()
        self.decoupled_samples = 0
        self.decoupled_impulse_nms = 0.0
        self.decoupled_max_moment_nm = 0.0
        self.decoupled_max_delta_nm = 0.0
        self.decoupled_max_normal_before = 0.0
        self.decoupled_max_normal_after = 0.0
        self.decoupled_trace = []

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
        effective_inertia = 1.0/compliance
        angular_velocity = float(jy @ velocity)
        limit = MOMENT_WEIGHT_FRACTION*self.box_mass*GRAVITY_M_S2*REFERENCE_LEVER_M
        moment = stopping_moment(effective_inertia, angular_velocity, limit)
        raw_delta = moment*jy
        delta, normal_after, normal_before = normal_acceleration_nullspace(raw_delta, mass, jz)
        output = command + delta
        self.decoupled_samples += 1
        self.decoupled_impulse_nms += abs(moment)*C.DT
        self.decoupled_max_moment_nm = max(self.decoupled_max_moment_nm, abs(moment))
        self.decoupled_max_delta_nm = max(self.decoupled_max_delta_nm, float(np.max(np.abs(delta))))
        self.decoupled_max_normal_before = max(self.decoupled_max_normal_before, abs(normal_before))
        self.decoupled_max_normal_after = max(self.decoupled_max_normal_after, abs(normal_after))
        if len(self.decoupled_trace) < 256:
            self.decoupled_trace.append({
                "t": float(t), "omega_y_rad_s": angular_velocity,
                "effective_inertia_kgm2": effective_inertia, "moment_nm": moment,
                "limit_nm": limit, "max_joint_delta_nm": float(np.max(np.abs(delta))),
                "normal_acceleration_before_m_s2": normal_before,
                "normal_acceleration_after_m_s2": normal_after,
            })
        return output
