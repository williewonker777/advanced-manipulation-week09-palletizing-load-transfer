"""Finite-difference continuous linearization of an unmodified MuJoCo plant.

The caller supplies an original ``MjModel`` and snapshot ``MjData``.  Every
acceleration evaluation allocates/uses private data copied from that snapshot;
the caller's model, data, solver settings, and integrator are never changed.
Motor torque is injected through ``qfrc_applied`` at ``tau_indices`` after the
public smooth Coulomb law (and any explicitly requested *extra* viscous law) is
subtracted.  Model-native joint damping remains active and must not be repeated
as extra viscous friction.

Position perturbations use ``mj_integratePos`` and their effective tangent
distance is measured with ``mj_differentiatePos``, so free/ball joints are
handled in the model's nv-dimensional tangent coordinates.  The result is the
full continuous state linearization

    A = [[0, I], [d(qacc)/dq, d(qacc)/dv]],  B = [[0], [d(qacc)/du]].

This captures MuJoCo's active local contact/constraint response directly.  Do
not add an independent Kelvin-Voigt contact term to the returned matrices; that
would double count contact.  As with any finite difference at unilateral mode
boundaries, step sensitivity is evidence to inspect, not a stability proof.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from tools.r2_linear import exact_zoh


@dataclass(frozen=True)
class MuJoCoPlantLinearization:
    """Full nv-coordinate continuous linearization plus convergence evidence."""

    A: np.ndarray
    B: np.ndarray
    acceleration: np.ndarray
    dacc_dq: np.ndarray
    dacc_dv: np.ndarray
    dacc_du: np.ndarray
    diagnostics: dict


def _vector(value, size, name):
    out = np.asarray(value, dtype=float)
    if out.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _indices(value, nv):
    out = np.asarray(value, dtype=int)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("tau_indices must be a nonempty one-dimensional sequence")
    if np.any(out < 0) or np.any(out >= nv) or np.unique(out).size != out.size:
        raise ValueError("tau_indices must be unique valid dof indices")
    return out


def _private_acceleration(
    model,
    source_data,
    qpos,
    qvel,
    torque,
    tau_indices,
    viscous,
    coulomb,
    friction_eps,
    work_data,
):
    """Evaluate qacc using only private MjData and the explicit motor input."""
    mujoco.mj_copyData(work_data, model, source_data)
    work_data.qpos[:] = qpos
    work_data.qvel[:] = qvel
    work_data.qfrc_applied[:] = 0.0
    if model.nu:
        # This helper's input channel is explicit generalized motor torque.
        # Copied actuator controls would otherwise create a second input path.
        work_data.ctrl[:] = 0.0
    motor_velocity = qvel[tau_indices]
    friction = viscous * motor_velocity + coulomb * np.tanh(
        motor_velocity / friction_eps
    )
    work_data.qfrc_applied[tau_indices] = torque - friction
    mujoco.mj_forward(model, work_data)
    return work_data.qacc.copy()


def _linearize_at_steps(
    model,
    source_data,
    torque,
    tau_indices,
    viscous,
    coulomb,
    friction_eps,
    position_step,
    velocity_step,
    input_step,
):
    nv, nq, nu = model.nv, model.nq, torque.size
    q0, v0 = source_data.qpos.copy(), source_data.qvel.copy()
    work = mujoco.MjData(model)

    def acceleration(qpos, qvel, motor_torque):
        return _private_acceleration(
            model,
            source_data,
            qpos,
            qvel,
            motor_torque,
            tau_indices,
            viscous,
            coulomb,
            friction_eps,
            work,
        )

    base = acceleration(q0, v0, torque)
    Aq, Av, Bu = np.empty((nv, nv)), np.empty((nv, nv)), np.empty((nv, nu))
    effective_position_steps = np.empty(nv)
    for column in range(nv):
        tangent = np.zeros(nv)
        tangent[column] = 1.0
        qplus, qminus = q0.copy(), q0.copy()
        mujoco.mj_integratePos(model, qplus, tangent, position_step)
        mujoco.mj_integratePos(model, qminus, tangent, -position_step)
        difference = np.zeros(nv)
        mujoco.mj_differentiatePos(model, difference, 1.0, qminus, qplus)
        denominator = difference[column]
        if abs(denominator) <= np.finfo(float).eps:
            raise RuntimeError(f"position perturbation vanished at dof {column}")
        effective_position_steps[column] = denominator
        Aq[:, column] = (
            acceleration(qplus, v0, torque) - acceleration(qminus, v0, torque)
        ) / denominator

    for column in range(nv):
        h = velocity_step * max(1.0, abs(v0[column]))
        delta = np.zeros(nv)
        delta[column] = h
        Av[:, column] = (
            acceleration(q0, v0 + delta, torque)
            - acceleration(q0, v0 - delta, torque)
        ) / (2.0 * h)

    for column in range(nu):
        h = input_step * max(1.0, abs(torque[column]))
        delta = np.zeros(nu)
        delta[column] = h
        Bu[:, column] = (
            acceleration(q0, v0, torque + delta)
            - acceleration(q0, v0, torque - delta)
        ) / (2.0 * h)
    return base, Aq, Av, Bu, effective_position_steps


def linearize_mujoco_acceleration(
    model,
    data,
    torque,
    tau_indices,
    *,
    viscous=None,
    coulomb,
    friction_eps=0.02,
    position_step=1e-6,
    velocity_step=1e-6,
    input_step=1e-5,
    sensitivity_ratio=0.5,
    convergence_tolerance=1e-3,
):
    """Linearize actual MuJoCo acceleration and compare h against h/2.

    ``data`` is read-only.  Existing equality activation, contacts, mocap,
    external Cartesian forces, and other snapshot state are copied into private
    data on every evaluation.  Existing generalized applied force and actuator
    control are replaced by this helper's explicit motor torque channel.
    """
    if data.qpos.shape != (model.nq,) or data.qvel.shape != (model.nv,):
        raise ValueError("data does not match model dimensions")
    indices = _indices(tau_indices, model.nv)
    torque = _vector(torque, indices.size, "torque")
    viscous = (
        np.zeros(indices.size)
        if viscous is None
        else _vector(viscous, indices.size, "viscous")
    )
    coulomb = _vector(coulomb, indices.size, "coulomb")
    if friction_eps <= 0.0:
        raise ValueError("friction_eps must be positive")
    if min(position_step, velocity_step, input_step) <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    if not 0.0 < sensitivity_ratio < 1.0:
        raise ValueError("sensitivity_ratio must be in (0,1)")

    coarse = _linearize_at_steps(
        model,
        data,
        torque,
        indices,
        viscous,
        coulomb,
        friction_eps,
        position_step,
        velocity_step,
        input_step,
    )
    fine = _linearize_at_steps(
        model,
        data,
        torque,
        indices,
        viscous,
        coulomb,
        friction_eps,
        position_step * sensitivity_ratio,
        velocity_step * sensitivity_ratio,
        input_step * sensitivity_ratio,
    )
    base, Aq, Av, Bu, effective_steps = fine

    def relative(first, second):
        return float(
            np.linalg.norm(first - second) / max(1.0, np.linalg.norm(second))
        )

    changes = {
        "dacc_dq": relative(coarse[1], Aq),
        "dacc_dv": relative(coarse[2], Av),
        "dacc_du": relative(coarse[3], Bu),
    }
    A = np.block([[np.zeros((model.nv, model.nv)), np.eye(model.nv)], [Aq, Av]])
    B = np.vstack((np.zeros((model.nv, indices.size)), Bu))
    diagnostics = {
        "position_step": float(position_step * sensitivity_ratio),
        "velocity_step": float(velocity_step * sensitivity_ratio),
        "input_step": float(input_step * sensitivity_ratio),
        "sensitivity_ratio": float(sensitivity_ratio),
        "relative_step_change": changes,
        "max_relative_step_change": max(changes.values()),
        "step_converged": bool(max(changes.values()) <= convergence_tolerance),
        "convergence_tolerance": float(convergence_tolerance),
        "effective_position_diameters": effective_steps,
        "qualification": (
            "local fixed-contact-mode finite difference; inspect sensitivity near unilateral boundaries"
        ),
    }
    return MuJoCoPlantLinearization(
        A=A,
        B=B,
        acceleration=base,
        dacc_dq=Aq,
        dacc_dv=Av,
        dacc_du=Bu,
        diagnostics=diagnostics,
    )


__all__ = ["MuJoCoPlantLinearization", "linearize_mujoco_acceleration", "exact_zoh"]
