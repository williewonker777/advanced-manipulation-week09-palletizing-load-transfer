#!/usr/bin/env python3
"""Restored-state sampled stability analysis for the R2 controllers.

This tool never advances a simulator episode.  It restores an owned MuJoCo
plant at a recorded P0 pre-step state, finite-differences its continuous
acceleration on private ``MjData``, applies an exact 1 ms ZOH, and closes that
plant around the frozen R1 controller including the command queue, servo drive
(architecture A), JTS filter/delay, sampled encoder velocity, and controller
filters/observers.

The result is a local, fixed-contact-mode numerical analysis.  It is not a
nonlinear, hybrid, robust, or hard-real-time stability proof.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from sim import config as C, trajectory as T
from sim.nominal import Nominal
from sim.plant import Plant
from sim.drive import Drive
from tools.r2_linear import closed_loop_poles, exact_zoh, linearize_discrete_map
from tools.r2_plant_fd import linearize_mujoco_acceleration

TIMES = (2.5, 3.75, 3.89)
MASSES = (0.85, 7.7)
COM = (0.013, -0.009, 0.019)

# Only numerical memory which can affect a later command is a linearization
# state.  Calibration constants, clocks, counters and alarm bookkeeping are
# restored before every evaluation but deliberately are not pole-producing.
SHARED_FIELDS = (
    "r_joint", "integral", "mass_prev", "q_prev", "v", "motor_v",
    "deflection_prev", "deflection_v", "corrected_deflection_prev",
    "fast_wrench_lp", "load_q_prev", "load_v", "lambda_ramp",
    "lambda_direct", "lambda_support", "lambda_fused", "load_fraction",
)
CONTROLLER_FIELDS = {
    "servo": ("load_fraction", "tau_link_prev", "tau_dot"),
    "torque": ("v_motor", "load_fraction", "tau_link_prev", "tau_dot"),
}


def _flat(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _copy_value(value: Any) -> Any:
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return copy.deepcopy(value)


def _restore_public_snapshot(obj: Any, values: dict[str, Any]) -> None:
    """Restore captured public controller memory, excluding model handles."""
    for name, value in values.items():
        if name in {"box_mass", "box_com", "dt", "elastic", "inv_k",
                    "bottom_offset", "table_heights", "force_threshold",
                    "moment_threshold", "fast_force_threshold",
                    "fast_moment_threshold", "joint_threshold"}:
            # These are constants and were already constructed from the same
            # public mass/COM/config.  Reassigning is harmless but unnecessary.
            continue
        if hasattr(obj, name):
            setattr(obj, name, _copy_value(value))


class ControllerMemory:
    """Pack/restore the explicit R1 numeric controller state whitelist."""

    def __init__(self, architecture: str, snapshot: dict[str, Any], *,
                 extra_controller_fields: tuple[str, ...] = (),
                 extra_shared_fields: tuple[str, ...] = ()):
        self.architecture = architecture
        self.controller_snapshot = snapshot["controller"]
        self.shared_snapshot = snapshot["shared_estimator"]
        self.entries: list[tuple[str, str, tuple[int, ...], slice]] = []
        start = 0
        for owner, names, source in (
            ("controller", CONTROLLER_FIELDS[architecture] + tuple(extra_controller_fields),
             self.controller_snapshot),
            ("shared", SHARED_FIELDS + tuple(extra_shared_fields), self.shared_snapshot),
        ):
            for name in names:
                value = source[name]
                if value is None:
                    raise ValueError(f"uninitialized state {owner}.{name}")
                array = np.asarray(value, dtype=float)
                stop = start + array.size
                self.entries.append((owner, name, array.shape, slice(start, stop)))
                start = stop
        self.size = start

    @classmethod
    def for_candidate(cls, architecture: str, snapshot: dict[str, Any], candidate: Any,
                      *, time_s: float | None = None, grip: bool | None = None):
        """Construct from optional candidate-declared dynamic state names.

        Candidate controllers may expose ``R2_STABILITY_STATE`` and their
        shared estimator may expose the same attribute.  This keeps candidate
        observer/integrator arrays explicit instead of silently omitting them.
        Their captured snapshot must contain each declared name.
        """
        selector = getattr(candidate, "r2_stability_state", None)
        if callable(selector):
            controller_fields = tuple(selector(time_s=time_s, grip=grip))
        elif hasattr(candidate, "family"):
            # Current contact candidate has phase-specific memory.  Frozen,
            # unused states are parameters of that mode, not unit-pole states.
            active = bool(grip and time_s is not None and 3.6 < time_s < T.T_RELEASE)
            if candidate.family == "C1":
                controller_fields = ("feedback_force",) if active else ()
            elif candidate.family == "C2" and candidate.MODE == "servo":
                controller_fields = ("normal_x", "normal_v")
            elif candidate.family == "C2":
                controller_fields = ("feedback_force",) if active else ()
            else:
                controller_fields = ()
        else:
            controller_fields = tuple(getattr(candidate, "R2_STABILITY_STATE", ()))
        controller_fields = tuple(
            name for name in controller_fields
            if not isinstance(getattr(candidate, name), (bool, np.bool_))
        )
        shared = getattr(candidate, "shared_estimator", None)
        shared_fields = tuple(getattr(shared, "R2_STABILITY_STATE", ()))
        return cls(architecture, snapshot,
                   extra_controller_fields=controller_fields,
                   extra_shared_fields=shared_fields)

    def pack_snapshot(self) -> np.ndarray:
        values = []
        for owner, name, _shape, _span in self.entries:
            source = self.controller_snapshot if owner == "controller" else self.shared_snapshot
            values.append(_flat(source[name]))
        return np.concatenate(values)

    def restore(self, controller: Any, vector: np.ndarray, time_s: float) -> None:
        _restore_public_snapshot(controller, self.controller_snapshot)
        _restore_public_snapshot(controller.shared_estimator, self.shared_snapshot)
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError("controller memory vector has wrong size")
        for owner, name, shape, span in self.entries:
            target = controller if owner == "controller" else controller.shared_estimator
            value = vector[span].reshape(shape)
            setattr(target, name, float(value) if shape == () else value.copy())
        # Exactly one estimator update is permitted for each evaluated map.
        controller.shared_estimator.last_update_t = float(time_s - C.DT)

    def pack_object(self, controller: Any) -> np.ndarray:
        values = []
        for owner, name, _shape, _span in self.entries:
            target = controller if owner == "controller" else controller.shared_estimator
            values.append(_flat(getattr(target, name)))
        return np.concatenate(values)

    def schema(self) -> list[dict[str, Any]]:
        return [
            {"owner": owner, "name": name, "shape": list(shape),
             "start": span.start, "stop": span.stop}
            for owner, name, shape, span in self.entries
        ]


def make_controller(architecture: str, mass: float, com: tuple[float, ...],
                    controller_class=None):
    if controller_class is None:
        controller_class = importlib.import_module(f"r1_{architecture}").Controller
    controller = controller_class()
    controller.nom = Nominal()
    controller.nom_box = Nominal()
    controller.nom_box.set_payload(mass, com)
    controller.box_mass, controller.box_com = float(mass), tuple(com)
    controller.reset()
    return controller


def _capture_candidate_snapshot(controller: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    captured = copy.deepcopy(baseline)
    for name in tuple(captured["controller"]) + tuple(
            getattr(controller, "R2_STABILITY_STATE", ())):
        if hasattr(controller, name):
            value = getattr(controller, name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            captured["controller"][name] = copy.deepcopy(value)
    shared = controller.shared_estimator
    for name in tuple(captured["shared_estimator"]) + tuple(
            getattr(shared, "R2_STABILITY_STATE", ())):
        if hasattr(shared, name):
            value = getattr(shared, name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            captured["shared_estimator"][name] = copy.deepcopy(value)
    return captured


def replay_candidate_operating_point(controller_class, architecture: str,
                                     source: dict[str, Any], requested_time: float,
                                     private_dir: Path):
    """Replay permitted recorded observations to initialize candidate memory.

    The physical trajectory remains the frozen R1 P0 trajectory.  Candidate
    commands and drive/queue states are reconstructed causally but never sent to
    a simulator.  This is therefore a forced-trajectory local analysis, not a
    candidate episode or candidate equilibrium.
    """
    snapshot = min(source["snapshots"], key=lambda row: abs(row["time_s"] - requested_time))
    target_index = int(snapshot["sample_index"])
    mass, com = float(source["mass_kg"]), tuple(source["com_m"])
    controller = make_controller(architecture, mass, com, controller_class)
    raw_path = private_dir / source["source_raw_npz"]
    with np.load(raw_path) as raw:
        obs_t = np.asarray(raw["obs_t"])
        encoder = np.asarray(raw["encoder"])
        encoder_velocity = np.asarray(raw["encoder_velocity"])
        jts = np.asarray(raw["jts"])
        grip = np.asarray(raw["grip"])
    if target_index <= 0 or target_index >= len(obs_t):
        raise ValueError("candidate replay target outside recorded observations")
    q0 = T.reference(0.0)[0]
    pending = (np.array([q0, np.zeros(6), np.zeros(6), np.zeros(6)])
               if architecture == "servo" else np.zeros(6))
    drive = Drive()
    previous_applied = np.zeros(6)
    for index in range(target_index):
        t = float(obs_t[index])
        qd, qv, qa = T.reference(t)
        output = np.asarray(controller.compute(
            t, encoder[index].copy(), encoder_velocity[index].copy(),
            jts[index].copy(), int(grip[index]), qd, qv, qa), float)
        if architecture == "servo":
            applied = drive.torque(pending, encoder[index], encoder_velocity[index])
        else:
            applied = pending
        previous_applied = np.clip(applied, -C.TAU_MAX, C.TAU_MAX)
        pending = output
    encoder_previous = encoder[target_index - 1].copy()
    return {
        "memory_snapshot": _capture_candidate_snapshot(
            controller, snapshot["before_compute"]
        ),
        "pending_command": pending.copy(),
        "previous_applied_torque": previous_applied.copy(),
        "drive_velocity_state": drive.v.copy(),
        "drive_integral_state": drive.I.copy(),
        "encoder_previous": encoder_previous,
        "qualification": (
            "candidate memory/queue replayed causally on frozen R1 permitted observations; "
            "physical trajectory not regenerated"
        ),
    }


def restore_plant(snapshot: dict[str, Any], mass: float, com: tuple[float, ...],
                  grasp_weld_relation: dict[str, Any] | None = None):
    physical = snapshot["before_compute"]["physical_before_step"]
    plant = Plant(T.reference(0.0)[0], mass, com)
    qpos = np.asarray(physical["qpos"], float)
    qpos_previous = np.asarray(physical["qpos_previous"], float)
    qvel = np.empty(plant.m.nv)
    mujoco.mj_differentiatePos(plant.m, qvel, C.DT, qpos_previous, qpos)
    plant.d.qpos[:] = qpos
    plant.d.qvel[:] = qvel
    plant.d.time = float(snapshot["time_s"])
    mujoco.mj_forward(plant.m, plant.d)
    if snapshot["input"]["grip"]:
        # Activate through the normal API, then restore the relative transform
        # captured at the actual grasp cycle.  The current moving pose contains
        # small compliant weld error and must not be mistaken for the anchor.
        plant.grip(True)
        if grasp_weld_relation is not None:
            plant.m.eq_data[0, 0:3] = 0.0
            plant.m.eq_data[0, 3:6] = np.asarray(
                grasp_weld_relation["translation_in_flange_m"], float
            )
            plant.m.eq_data[0, 6:10] = np.asarray(
                grasp_weld_relation["relative_quaternion_wxyz"], float
            )
            plant.m.eq_data[0, 10] = 1.0
        plant.d.qpos[:] = qpos
        plant.d.qvel[:] = qvel
        plant.d.time = float(snapshot["time_s"])
        mujoco.mj_forward(plant.m, plant.d)
    return plant


def true_torque_linear_map(plant: Plant, mechanics,
                           sampled_coulomb_derivative: np.ndarray):
    """Return d(tau_true)/d([dq,dv]) and d(tau_true)/d(u_previous)."""
    nv = plant.m.nv
    Cx = np.zeros((6, 2 * nv))
    Cu = np.zeros((6, 6))
    # Elastic transmitted torque is -(K*phi + D*phi_dot).
    for local, axis in enumerate(plant.E):
        Cx[axis, plant.ph_v[local]] = -C.K[axis]
        Cx[axis, nv + plant.ph_v[local]] = -C.D[axis]
    # Rigid sensor is link-side torque u-f(v)-B*qdd.  Native model viscous
    # damping is already included in mechanics.dacc_dv; add only the explicit
    # derivative of Coulomb and the documented viscous link-side term.
    friction_d = C.VISCOUS + sampled_coulomb_derivative
    effective_dacc_dv = mechanics.dacc_dv.copy()
    effective_dacc_dv[:, plant.th_v] -= (
        mechanics.dacc_du @ np.diag(sampled_coulomb_derivative)
    )
    for axis in plant.Rg:
        dof = int(plant.th_v[axis])
        Cx[axis, nv + dof] -= friction_d[axis]
        Cx[axis, :nv] -= C.B[axis] * mechanics.dacc_dq[dof]
        Cx[axis, nv:] -= C.B[axis] * effective_dacc_dv[dof]
        Cu[axis] = -C.B[axis] * mechanics.dacc_du[dof]
        Cu[axis, axis] += 1.0
    return Cx, Cu


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [{"real": float(z.real), "imag": float(z.imag),
                     "magnitude": float(abs(z))} for z in value]
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def analyze_snapshot(architecture: str, source: dict[str, Any], requested_time: float,
                     *, controller_class=None, operating_point=None,
                     candidate_label: str = "R1"):
    snapshots = source["snapshots"]
    snapshot = min(snapshots, key=lambda row: abs(row["time_s"] - requested_time))
    if abs(snapshot["time_s"] - requested_time) > 0.5 * C.DT:
        raise ValueError("requested snapshot not found")
    mass, com = float(source["mass_kg"]), tuple(source["com_m"])
    before = snapshot["before_compute"]
    act = before["actuation"]
    plant = restore_plant(snapshot, mass, com, source.get("grasp_weld_relation"))
    if operating_point is None:
        memory_snapshot = before
        pending_override = np.asarray(act["pending_command"], float)
        previous = np.asarray(act["previous_applied_torque"], float)
        vf_override = np.asarray(act["mirrored_drive_velocity_state"], float)
        int_override = np.asarray(act["mirrored_drive_integral_state"], float)
        encprev_override = np.asarray(act["encoder_previous"], float)
        operating_qualification = "recorded frozen-R1 controller/actuation state"
    else:
        memory_snapshot = operating_point["memory_snapshot"]
        pending_override = np.asarray(operating_point["pending_command"], float)
        previous = np.asarray(operating_point["previous_applied_torque"], float)
        vf_override = np.asarray(operating_point["drive_velocity_state"], float)
        int_override = np.asarray(operating_point["drive_integral_state"], float)
        encprev_override = np.asarray(operating_point["encoder_previous"], float)
        operating_qualification = operating_point["qualification"]

    # Current-cycle applied torque is produced from the old queued command and
    # current encoder sample.  Compute it before mechanical linearization.
    base_input = snapshot["input"]
    th0 = np.asarray(base_input["th"], float)
    sampled_thd0 = (th0 - encprev_override) / C.DT
    if architecture == "servo":
        vf_after0 = vf_override + C.DRIVE_VEL_FILTER * (sampled_thd0 - vf_override)
        ev0 = (pending_override[2] + C.DRIVE_KP * (pending_override[0] - th0)
               - vf_after0)
        int_after0 = int_override + C.DT * ev0
        applied_now = np.clip(
            pending_override[3] + C.DRIVE_KV * ev0 + C.DRIVE_KI * int_after0,
            -C.TAU_MAX, C.TAU_MAX,
        )
    else:
        vf_after0, int_after0 = np.zeros(6), np.zeros(6)
        applied_now = np.clip(pending_override, -C.TAU_MAX, C.TAU_MAX)
    # Plant.step samples smooth Coulomb once before RK4 and holds it for the
    # whole 1 ms interval.  Therefore it is a state-dependent held input, not
    # continuous damping.  Native model viscosity remains continuous.
    mechanics = linearize_mujoco_acceleration(
        plant.m, plant.d, applied_now, plant.th_v,
        coulomb=np.zeros(6), friction_eps=C.FRICTION_EPS,
    )
    sensor_mechanics = linearize_mujoco_acceleration(
        plant.m, plant.d, previous, plant.th_v,
        coulomb=np.zeros(6), friction_eps=C.FRICTION_EPS,
    )
    Ad, Bd = exact_zoh(mechanics.A, mechanics.B, C.DT)
    thv_actual = np.asarray(plant.d.qvel[plant.th_v], float)
    sampled_coulomb_derivative = (
        C.COULOMB / C.FRICTION_EPS
        / np.cosh(thv_actual / C.FRICTION_EPS) ** 2
    )
    velocity_selector = np.zeros((6, 2 * plant.m.nv))
    velocity_selector[np.arange(6), plant.m.nv + plant.th_v] = 1.0
    Ad -= Bd @ np.diag(sampled_coulomb_derivative) @ velocity_selector
    Ctrue, Dtrue = true_torque_linear_map(
        plant, sensor_mechanics, sampled_coulomb_derivative
    )

    controller = make_controller(architecture, mass, com, controller_class)
    time_s = float(snapshot["time_s"])
    grip = int(base_input["grip"])
    memory = (ControllerMemory.for_candidate(
                  architecture, memory_snapshot, controller,
                  time_s=time_s, grip=bool(grip)
              )
              if controller_class is not None else ControllerMemory(architecture, memory_snapshot))
    xc0 = memory.pack_snapshot()
    tau_s0 = np.asarray(base_input["tau_s"], float)
    encprev0 = encprev_override
    pending0 = pending_override
    vf0, int0 = vf_override, int_override
    qd0 = np.asarray(base_input["q_d"], float)
    qdd0 = np.asarray(base_input["qdd_d"], float)
    qvel0 = np.asarray(base_input["qd_d"], float)
    nv = plant.m.nv
    th_dofs = np.asarray(plant.th_v, int)
    alpha = 1.0 - np.exp(-2.0 * np.pi * C.JTS_CUTOFF * C.DT)

    command_size = pending0.size
    sizes = {
        "mechanics": 2 * nv,
        "controller": memory.size,
        "pending": command_size,
        "drive_v": 6 if architecture == "servo" else 0,
        "drive_i": 6 if architecture == "servo" else 0,
        "jts_filter": 6,
        "jts_delay": 6,
        "applied_previous": 6,
        "encoder_previous": 6,
    }
    spans: dict[str, slice] = {}
    cursor = 0
    for name, size in sizes.items():
        spans[name] = slice(cursor, cursor + size)
        cursor += size
    x0 = np.zeros(cursor)
    x0[spans["controller"]] = xc0
    # Controller memory is absolute; all other states are perturbations.

    # Candidate output at the restored pre-compute state is the affine command
    # origin; only perturbations about it enter the feedback Jacobian.
    memory.restore(controller, xc0, time_s)
    computed_output0 = np.asarray(controller.compute(
        time_s, th0, sampled_thd0, tau_s0, grip, qd0, qvel0, qdd0
    ), float)
    output0 = (np.asarray(snapshot["output"], float)
               if operating_point is None else computed_output0)
    baseline_new_applied = applied_now.copy()

    def state_map(flat: np.ndarray, command_scales: np.ndarray | None = None):
        flat = np.asarray(flat, float)
        dx = flat[spans["mechanics"]]
        dq, dv = dx[:nv], dx[nv:]
        xc = flat[spans["controller"]]
        pending_delta = flat[spans["pending"]].reshape(pending0.shape)
        jts_f = flat[spans["jts_filter"]]
        jts_delay = flat[spans["jts_delay"]]
        applied_prev_delta = flat[spans["applied_previous"]]
        encprev_delta = flat[spans["encoder_previous"]]
        th_delta = dq[th_dofs]
        sampled_thd = (th0 + th_delta - (encprev0 + encprev_delta)) / C.DT

        memory.restore(controller, xc, time_s)
        args = (time_s, th0 + th_delta, sampled_thd, tau_s0 + jts_delay,
                grip, qd0, qvel0, qdd0)
        command = np.asarray(controller.compute(*args), float)
        command_delta = command - output0
        if command_scales is not None:
            scales = np.asarray(command_scales, float)
            command_delta = command_delta * (scales if architecture == "torque" else scales[None, :])
        xc_next = memory.pack_object(controller)

        true_delta = Ctrue @ dx + Dtrue @ applied_prev_delta
        jts_f_next = jts_f + alpha * (true_delta - jts_f)
        jts_delay_next = jts_f_next.copy()

        if architecture == "torque":
            applied_delta = pending_delta.reshape(6)
            drive_v_next = drive_i_next = np.empty(0)
        else:
            drive_v = flat[spans["drive_v"]]
            drive_i = flat[spans["drive_i"]]
            vf_abs = vf0 + drive_v
            int_abs = int0 + drive_i
            cmd_abs = pending0 + pending_delta
            vf_next_abs = vf_abs + C.DRIVE_VEL_FILTER * (sampled_thd - vf_abs)
            ev = cmd_abs[2] + C.DRIVE_KP * (cmd_abs[0] - (th0 + th_delta)) - vf_next_abs
            int_next_abs = int_abs + C.DT * ev
            torque_abs = cmd_abs[3] + C.DRIVE_KV * ev + C.DRIVE_KI * int_next_abs
            torque_abs = np.clip(torque_abs, -C.TAU_MAX, C.TAU_MAX)
            applied_delta = torque_abs - baseline_new_applied
            drive_v_next = vf_next_abs - vf_after0
            drive_i_next = int_next_abs - int_after0

        mechanics_next = Ad @ dx + Bd @ applied_delta
        out = np.empty_like(flat)
        out[spans["mechanics"]] = mechanics_next
        out[spans["controller"]] = xc_next
        out[spans["pending"]] = command_delta.reshape(-1)
        if architecture == "servo":
            out[spans["drive_v"]] = drive_v_next
            out[spans["drive_i"]] = drive_i_next
        out[spans["jts_filter"]] = jts_f_next
        out[spans["jts_delay"]] = jts_delay_next
        out[spans["applied_previous"]] = applied_delta
        out[spans["encoder_previous"]] = th_delta
        return out

    # Absolute controller memory makes x0 nonzero.  Verify the recorded command
    # and that every non-memory perturbation remains at its local origin.
    memory.restore(controller, xc0, time_s)
    replay = np.asarray(controller.compute(time_s, th0, sampled_thd0,
                                            tau_s0, grip, qd0, qvel0, qdd0), float)
    replay_error = float(np.max(np.abs(replay - output0)))
    if replay_error > 1e-9:
        raise RuntimeError(f"controller snapshot replay error {replay_error:g}")

    affine_origin = state_map(x0)
    nonmemory_spans = tuple(name for name in spans if name != "controller")
    affine_nonmemory_max = max(
        float(np.max(np.abs(affine_origin[spans[name]]))) if sizes[name] else 0.0
        for name in nonmemory_spans
    )
    linear = linearize_discrete_map(state_map, x0, state_step=2e-6)
    poles = closed_loop_poles(linear.A)

    # Scaling the six command-sensitivity rows is an inexpensive exact operation
    # on the local Jacobian.  It is explicitly not called a classical gain margin.
    pending_rows = np.arange(spans["pending"].start, spans["pending"].stop)
    row_axis = np.tile(np.arange(6), 4) if architecture == "servo" else np.arange(6)

    def scaled_matrix(scales):
        matrix = linear.A.copy()
        matrix[pending_rows] *= np.asarray(scales, float)[row_axis, None]
        return matrix

    def boundary(axis: int | None, upper=16.0, iterations=45):
        def stable(scale):
            scales = np.ones(6)
            if axis is None:
                scales[:] = scale
            else:
                scales[axis] = scale
            return closed_loop_poles(scaled_matrix(scales))["stable"]
        result = {"lower_zero_stable": bool(stable(0.0)), "upper_test": upper,
                  "upper_test_stable": bool(stable(upper))}
        if not poles["stable"]:
            result["status"] = "nominal_unstable_or_marginal"
            return result
        if stable(upper):
            result.update(status="stable_to_tested_limit", upper_stable=upper)
            return result
        lo, hi = 1.0, upper
        for _ in range(iterations):
            mid = (lo + hi) / 2.0
            if stable(mid):
                lo = mid
            else:
                hi = mid
        result.update(status="sampled_boundary", upper_stable=lo, upper_unstable=hi)
        return result

    return {
        "candidate": candidate_label,
        "architecture": architecture,
        "mass_kg": mass,
        "com_m": list(com),
        "time_s": time_s,
        "grip": bool(grip),
        "state_size": int(cursor),
        "state_blocks": {name: {"start": span.start, "stop": span.stop}
                         for name, span in spans.items()},
        "controller_state_schema": memory.schema(),
        "controller_replay_max_abs_error": replay_error,
        "operating_point_qualification": operating_qualification,
        "moving_point_affine_noncontroller_max_abs": affine_nonmemory_max,
        "mechanical_fd": {
            "step_input": mechanics.diagnostics,
            "previous_applied_input_for_jts": sensor_mechanics.diagnostics,
            "max_relative_step_change": max(
                mechanics.diagnostics["max_relative_step_change"],
                sensor_mechanics.diagnostics["max_relative_step_change"],
            ),
        },
        "friction_discretization": (
            "model-native viscosity is continuous; smooth Coulomb is sampled from the "
            "pre-step motor velocity and held for the exact 1ms ZOH, matching Plant.step"
        ),
        "weld_restoration": {
            "active": bool(snapshot["input"]["grip"]),
            "source": "private observed P0 grasp-relative pose; activated through owned Plant.grip API",
            "translation_m": plant.m.eq_data[0, 3:6].copy() if plant.has_box else [],
            "relative_quaternion_wxyz": plant.m.eq_data[0, 6:10].copy() if plant.has_box else [],
        },
        "closed_loop": {
            "spectral_radius": poles["spectral_radius"],
            "stable_strict": poles["stable"],
            "protocol_radius_pass_lt_1_plus_1e_minus_6": bool(
                poles["spectral_radius"] < 1.0 + 1e-6
            ),
            "poles": poles["poles"],
            "near_unit_count_abs_ge_0_999": int(np.count_nonzero(np.abs(poles["poles"]) >= 0.999)),
            "jacobian_step_diagnostics": linear.diagnostics,
        },
        "local_command_sensitivity_scale": {
            "scalar": boundary(None),
            "per_axis": [{"axis": axis + 1, **boundary(axis)} for axis in range(6)],
            "feedforward_margin": "N/A: affine feed-forward does not alter local poles",
            "qualification": "sampled command-Jacobian row scaling, not a classical SISO gain margin or robust stability proof",
        },
        "qualification": (
            "elastic actual MuJoCo fixed-mode mechanics; exact 1 ms ZOH; one-cycle command and JTS delays; "
            "unquantized local encoder perturbations; recorded moving operating point; unilateral/contact and clip branch changes excluded"
        ),
    }


def run(private_dir: Path, output: Path):
    analyses = []
    for architecture in ("servo", "torque"):
        for mass in MASSES:
            source = json.loads((private_dir / f"{architecture}_{mass:g}kg_stability_snapshots.json").read_text())
            for time_s in TIMES:
                analyses.append(analyze_snapshot(architecture, source, time_s))
    radii = [row["closed_loop"]["spectral_radius"] for row in analyses]
    result = {
        "schema": "R2_R1_STABILITY.v1",
        "method": "restored frozen-R1 controller + actual elastic MuJoCo local FD + exact 1ms ZOH + explicit one-cycle delays",
        "episodes_run": 0,
        "representative_grid": {"architectures": ["servo", "torque"],
                                "masses_kg": list(MASSES), "times_s": list(TIMES),
                                "com_m": list(COM)},
        "analyses": analyses,
        "summary": {
            "strict_stable_cases": int(sum(row["closed_loop"]["stable_strict"] for row in analyses)),
            "protocol_radius_pass_cases": int(sum(
                row["closed_loop"]["protocol_radius_pass_lt_1_plus_1e_minus_6"]
                for row in analyses
            )),
            "total_cases": len(analyses),
            "spectral_radius_min": float(min(radii)),
            "spectral_radius_max": float(max(radii)),
            "non_strict_cases": [
                {"architecture": row["architecture"], "mass_kg": row["mass_kg"],
                 "time_s": row["time_s"],
                 "spectral_radius": row["closed_loop"]["spectral_radius"]}
                for row in analyses if not row["closed_loop"]["stable_strict"]
            ],
            "protocol_radius_fail_cases": [
                {"architecture": row["architecture"], "mass_kg": row["mass_kg"],
                 "time_s": row["time_s"],
                 "spectral_radius": row["closed_loop"]["spectral_radius"]}
                for row in analyses
                if not row["closed_loop"]["protocol_radius_pass_lt_1_plus_1e_minus_6"]
            ],
        },
        "global_scope_warning": "finite local fixed-mode checks are not a nonlinear/hybrid or robust guarantee",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def run_candidate_specs(specs: list[dict[str, Any]], private_dir: Path,
                        output: Path, *, append: bool = False):
    """Analyze preregistered candidate specs without advancing an episode.

    Spec form: ``{"module":"experiments.r2_contact", "architecture":"servo",
    "params":{...}, "label":"C1-A-.85-.25", "masses_kg":[.85,7.7],
    "times_s":[2.5,3.75,3.89]}``.
    """
    analyses = []
    module_hashes: dict[str, str] = {}
    for spec in specs:
        module = importlib.import_module(spec["module"])
        module_path = Path(module.__file__).resolve()
        module_hashes[spec["module"]] = hashlib.sha256(module_path.read_bytes()).hexdigest()
        architecture = spec["architecture"]
        params = spec["params"]
        controller_class = module.make_controller(architecture, params)
        label = spec.get("label", f"{spec['module']}:{architecture}:{params}")
        for mass in spec.get("masses_kg", MASSES):
            source = json.loads(
                (private_dir / f"{architecture}_{float(mass):g}kg_stability_snapshots.json").read_text()
            )
            for time_s in spec.get("times_s", TIMES):
                point = replay_candidate_operating_point(
                    controller_class, architecture, source, float(time_s), private_dir
                )
                row = analyze_snapshot(
                    architecture, source, float(time_s),
                    controller_class=controller_class,
                    operating_point=point,
                    candidate_label=label,
                )
                row["module"] = spec["module"]
                row["module_sha256"] = module_hashes[spec["module"]]
                row["parameters"] = params
                analyses.append(row)
    if append and output.exists():
        existing = json.loads(output.read_text())
        if existing.get("schema") != "R2_CANDIDATE_STABILITY.v1":
            raise ValueError("append target has incompatible schema")
        specs = existing["specs"] + specs
        analyses = existing["analyses"] + analyses
        old_hashes = existing.get("module_sha256", {})
        for name, digest in module_hashes.items():
            if name in old_hashes and old_hashes[name] != digest:
                raise ValueError(f"cannot append changed candidate module {name}")
        module_hashes = old_hashes | module_hashes
    result = {
        "schema": "R2_CANDIDATE_STABILITY.v1",
        "method": (
            "candidate memory/queue observed-input replay on frozen R1 physical points; "
            "actual elastic MuJoCo local FD; exact 1ms ZOH; explicit delays"
        ),
        "episodes_run": 0,
        "specs": specs,
        "module_sha256": module_hashes,
        "analyses": analyses,
        "summary": {
            "strict_stable_cases": int(sum(row["closed_loop"]["stable_strict"] for row in analyses)),
            "protocol_radius_pass_cases": int(sum(
                row["closed_loop"]["protocol_radius_pass_lt_1_plus_1e_minus_6"]
                for row in analyses
            )),
            "total_cases": len(analyses),
            "non_strict_cases": [
                {key: row[key] for key in ("candidate", "architecture", "mass_kg", "time_s")}
                | {"spectral_radius": row["closed_loop"]["spectral_radius"]}
                for row in analyses if not row["closed_loop"]["stable_strict"]
            ],
            "protocol_radius_fail_cases": [
                {key: row[key] for key in ("candidate", "architecture", "mass_kg", "time_s")}
                | {"spectral_radius": row["closed_loop"]["spectral_radius"]}
                for row in analyses
                if not row["closed_loop"]["protocol_radius_pass_lt_1_plus_1e_minus_6"]
            ],
        },
        "scope_warning": (
            "forced-trajectory fixed-mode local checks; candidate physical operating points "
            "must be rechecked if adopted; not a nonlinear/hybrid/robust proof"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def compact_candidate_result(source: Path, output: Path):
    """Write a portable all-case stability ledger without raw state schemas."""
    data = json.loads(source.read_text())
    rows = []
    columns = (
        "candidate", "architecture", "mass_kg", "time_s", "spectral_radius",
        "dominant_real", "dominant_imag", "strict_lt_one",
        "protocol_lt_one_plus_1e_minus_6", "near_unit_count_abs_ge_0_999",
        "scalar_lower_zero_stable", "scalar_upper_status", "scalar_upper_stable",
        "worst_axis_upper_stable", "mechanical_fd_relative_change",
        "closed_loop_fd_relative_change",
    )
    for row in data["analyses"]:
        poles = row["closed_loop"]["poles"]
        dominant = max(poles, key=lambda pole: pole["magnitude"])
        margins = row["local_command_sensitivity_scale"]
        scalar = margins["scalar"]
        axis_upper = []
        for axis in margins["per_axis"]:
            value = axis.get("upper_stable", axis.get("gain"))
            if value is not None:
                axis_upper.append(float(value))
        rows.append([
            row["candidate"], row["architecture"], row["mass_kg"], row["time_s"],
            row["closed_loop"]["spectral_radius"], dominant["real"], dominant["imag"],
            row["closed_loop"]["stable_strict"],
            row["closed_loop"]["protocol_radius_pass_lt_1_plus_1e_minus_6"],
            row["closed_loop"]["near_unit_count_abs_ge_0_999"],
            scalar["lower_zero_stable"], scalar["status"],
            scalar.get("upper_stable", scalar.get("gain")),
            min(axis_upper) if axis_upper else None,
            row["mechanical_fd"]["max_relative_step_change"],
            row["closed_loop"]["jacobian_step_diagnostics"]["A_relative_step_change"],
        ])
    index = {name: i for i, name in enumerate(columns)}
    by_candidate = []
    for label in dict.fromkeys(row[index["candidate"]] for row in rows):
        group = [row for row in rows if row[index["candidate"]] == label]
        by_candidate.append({
            "candidate": label, "cases": len(group),
            "protocol_pass_cases": sum(
                row[index["protocol_lt_one_plus_1e_minus_6"]] for row in group
            ),
            "strict_cases": sum(row[index["strict_lt_one"]] for row in group),
            "worst_spectral_radius": max(row[index["spectral_radius"]] for row in group),
        })
    compact = {
        "schema": "R2_CANDIDATE_STABILITY_COMPACT.v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "module_sha256": data["module_sha256"],
        "protocol_radius_gate": "spectral_radius < 1 + 1e-6",
        "strict_label": "spectral_radius < 1; exact-neutral modes are marginal, not strict stable",
        "margin_scope": (
            "local structured perturbation-to-command feedback-Jacobian scaling; "
            "not a classical SISO or universal MIMO robust margin; affine feed-forward N/A"
        ),
        "uncertainty": (
            "forced frozen-R1 physical points, fixed contact mode and unquantized local "
            "perturbations; baseline B already fails at 7.7kg/2.5s; finite checks are not guarantees"
        ),
        "summary": {
            "total_cases": data["summary"]["total_cases"],
            "strict_stable_cases": data["summary"]["strict_stable_cases"],
            "protocol_radius_pass_cases": data["summary"]["protocol_radius_pass_cases"],
        },
        "by_candidate": by_candidate,
        "case_columns": list(columns),
        "all_cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, separators=(",", ":")) + "\n")
    return compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-dir", type=Path,
        help="local private P0 snapshot/raw directory (never copied to output)",
    )
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/v2_protocol/R2_R1_STABILITY.json")
    parser.add_argument(
        "--candidate-specs", type=Path,
        help="JSON list of candidate module/architecture/params specs",
    )
    parser.add_argument("--append", action="store_true",
                        help="append candidate analyses to a compatible output")
    parser.add_argument("--compact-from", type=Path,
                        help="write a portable compact ledger from a full candidate result")
    args = parser.parse_args()
    if args.compact_from:
        result = compact_candidate_result(args.compact_from, args.output)
        print("compact", len(result["all_cases"]), args.output)
        return
    if args.private_dir is None:
        parser.error("--private-dir is required unless --compact-from is used")
    if args.candidate_specs:
        specs = json.loads(args.candidate_specs.read_text())
        if not isinstance(specs, list):
            raise ValueError("candidate specs file must contain a JSON list")
        result = run_candidate_specs(
            specs, args.private_dir, args.output, append=args.append
        )
    else:
        result = run(args.private_dir, args.output)
    for row in result["analyses"]:
        print(row["architecture"], row["mass_kg"], row["time_s"],
              row["closed_loop"]["spectral_radius"], row["closed_loop"]["stable_strict"])


if __name__ == "__main__":
    main()
