#!/usr/bin/env python3
"""Stored-observation snapshot capture and local stability orchestration for R3.

No function in this module advances a simulator episode.  Controller and,
optionally, detector state are replayed only from logged observations.  MuJoCo
``qpos`` is used after the replay solely to restore the physical pre-step point
for :func:`tools.r2_stability.analyze_snapshot`.

Two source kinds are explicit:

``forced_r1``
    A candidate is replayed on a frozen R1 observation/physics trace.  Its
    predicted torque need not equal the R1 torque.
``actual_candidate``
    The controller which generated the trace is replayed and every selected
    operating point is rejected unless its applied torque matches the raw log
    within 1e-9 N m.

The resulting pole checks are fixed-phase, fixed-contact local checks.  They
are not nonlinear, LTV, hybrid, passivity, or robust-stability proofs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from sim import config as C, trajectory as T
from sim.nominal import Nominal
from sim.plant import Plant
from tools.r2_p0 import (
    COM as R2_P0_COM,
    mirrored_servo_drive,
    numeric_state,
    weld_relation_from_logged_qpos,
)
from tools.r2_stability import _jsonable, analyze_snapshot, make_controller

ACTUAL_TORQUE_TOLERANCE = 1e-9
PLACE_WINDOW_S = (3.7, 4.1)
FD_CONVERGENCE_TOLERANCE = 1e-3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _raw_time(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    for name in ("obs_t", "t"):
        if name in raw:
            return np.asarray(raw[name], float)
    raise KeyError("raw archive needs obs_t or t")


def _raw_applied_torque(raw: Mapping[str, np.ndarray], key: str | None) -> np.ndarray:
    names = (key,) if key else ("tau", "applied_torque", "logtau")
    for name in names:
        if name and name in raw:
            value = np.asarray(raw[name], float)
            if value.ndim == 2 and value.shape[1] == 6:
                return value
    raise KeyError(f"raw archive needs a six-axis applied torque array; tried {names}")


def _controller_instance(controller_class, architecture: str, mass: float,
                         com: tuple[float, ...]):
    return make_controller(architecture, mass, com, controller_class)


def make_detector(detector_class, controller: Any, mass: float,
                  com: tuple[float, ...]):
    """Construct a detector exactly as ``sim.runner.run_episode`` does."""
    detector = detector_class()
    detector.nom, detector.nom_box = Nominal(), Nominal()
    detector.nom_box.set_payload(mass, com)
    detector.ctrl = controller
    detector.box_mass, detector.box_com = float(mass), tuple(com)
    if hasattr(detector, "reset"):
        detector.reset()
    return detector


def _weld_relation(qpos: np.ndarray, mass: float,
                   com: tuple[float, ...]) -> dict[str, Any]:
    """Reconstruct the grasp relation in separate data, never controller input."""
    plant = Plant(T.reference(0.0)[0], mass, com)
    data = mujoco.MjData(plant.m)
    data.qpos[:] = qpos
    mujoco.mj_forward(plant.m, data)
    flange = plant.m.body("link_6").id
    box = plant.m.body("box_free").id
    inverse, relative = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(inverse, data.xquat[flange])
    mujoco.mju_mulQuat(relative, inverse, data.xquat[box])
    rotation = data.xmat[flange].reshape(3, 3)
    return {
        "source": "raw qpos immediately before grasp, reconstructed post hoc",
        "translation_in_flange_m": (
            rotation.T @ (data.xpos[box] - data.xpos[flange])
        ).tolist(),
        "relative_quaternion_wxyz": relative.tolist(),
    }


def actual_peak_indices(raw: Mapping[str, np.ndarray], *,
                        applied_torque_key: str | None = None,
                        place_window_s: tuple[float, float] = PLACE_WINDOW_S
                        ) -> dict[str, int]:
    """Return deterministic TCP-maximum and place torque-step sample indices."""
    time_s = _raw_time(raw)
    if "e_mm" not in raw:
        raise KeyError("actual peak selection needs e_mm")
    tcp_index = int(np.nanargmax(np.asarray(raw["e_mm"], float)))
    torque = _raw_applied_torque(raw, applied_torque_key)
    step = np.max(np.abs(np.diff(torque, axis=0)), axis=1)
    candidates = np.flatnonzero(
        (time_s[1:] >= place_window_s[0]) & (time_s[1:] < place_window_s[1])
    )
    if not candidates.size:
        raise ValueError("place window has no raw samples")
    place_index = int(candidates[np.argmax(step[candidates])] + 1)
    return {"actual_tcp_max": tcp_index, "actual_place_torque_step": place_index}


def requested_indices(time_s: np.ndarray, requested_times: Iterable[float], *,
                      extra_indices: Mapping[str, int] | None = None
                      ) -> dict[int, dict[str, Any]]:
    """Resolve exact 1 ms samples and merge duplicate requested/peak points."""
    time_s = np.asarray(time_s, float)
    selected: dict[int, dict[str, Any]] = {}
    for requested in requested_times:
        requested = float(requested)
        index = int(np.argmin(np.abs(time_s - requested)))
        if abs(float(time_s[index]) - requested) > 0.5 * C.DT + 1e-12:
            raise ValueError(f"requested time {requested:g} has no 1 ms sample")
        row = selected.setdefault(index, {"requested_times_s": [], "peak_tags": []})
        row["requested_times_s"].append(requested)
    for tag, index_value in (extra_indices or {}).items():
        index = int(index_value)
        if not 0 <= index < len(time_s):
            raise IndexError(f"peak index {tag}={index} outside raw trace")
        selected.setdefault(index, {"requested_times_s": [], "peak_tags": []})[
            "peak_tags"
        ].append(str(tag))
    return dict(sorted(selected.items()))


def capture_snapshots(
    raw_path: Path,
    architecture: str,
    mass_kg: float,
    com_m: Sequence[float],
    requested_times_s: Iterable[float],
    controller_class,
    *,
    source_kind: str,
    candidate_label: str,
    detector_class=None,
    include_actual_peaks: bool = False,
    applied_torque_key: str | None = None,
    output: Path | None = None,
    controller_builder: Callable[..., Any] = _controller_instance,
    detector_builder: Callable[..., Any] = make_detector,
    weld_relation: Mapping[str, Any] | None = None,
    place_window_s: tuple[float, float] = PLACE_WINDOW_S,
) -> dict[str, Any]:
    """Replay a stored trace and capture all requested pre-compute states.

    ``source_kind`` must be ``forced_r1`` or ``actual_candidate``.  The latter
    enforces exact causal actuation replay at *all* cycles, not just snapshots.
    """
    if source_kind not in {"forced_r1", "actual_candidate"}:
        raise ValueError("source_kind must be forced_r1 or actual_candidate")
    if architecture not in {"servo", "torque"}:
        raise ValueError("architecture must be servo or torque")
    raw_path = Path(raw_path)
    raw = _materialize_npz(raw_path)
    time_s = _raw_time(raw)
    required = ("encoder", "encoder_velocity", "jts", "grip", "qpos")
    missing = [name for name in required if name not in raw]
    if missing:
        raise KeyError(f"raw archive missing {missing}")
    n = len(time_s)
    if any(len(np.asarray(raw[name])) != n for name in required):
        raise ValueError("raw observation arrays have inconsistent lengths")
    applied_raw = _raw_applied_torque(raw, applied_torque_key)
    if len(applied_raw) != n:
        raise ValueError("raw applied torque has inconsistent length")

    extras = (actual_peak_indices(raw, applied_torque_key=applied_torque_key,
                                 place_window_s=place_window_s)
              if include_actual_peaks else {})
    targets = requested_indices(time_s, requested_times_s, extra_indices=extras)
    if not targets:
        raise ValueError("at least one requested or actual-peak point is required")

    com = tuple(float(value) for value in com_m)
    controller = controller_builder(controller_class, architecture, float(mass_kg), com)
    detector = (detector_builder(detector_class, controller, float(mass_kg), com)
                if detector_class is not None else None)
    q0 = T.reference(0.0)[0]
    pending = (np.array([q0, np.zeros(6), np.zeros(6), np.zeros(6)])
               if architecture == "servo" else np.zeros(6))
    drive_v = np.zeros(6)
    drive_i = np.zeros(6)
    previous_applied = np.zeros(6)
    snapshots: list[dict[str, Any]] = []
    maximum_error = 0.0

    for index, t_value in enumerate(time_s):
        t = float(t_value)
        th = np.asarray(raw["encoder"][index], float)
        thd = np.asarray(raw["encoder_velocity"][index], float)
        tau_s = np.asarray(raw["jts"][index], float)
        grip = bool(raw["grip"][index])
        before = None
        if index in targets:
            physical_current = np.asarray(raw["qpos"][max(index - 1, 0)], float)
            physical_previous = np.asarray(raw["qpos"][max(index - 2, 0)], float)
            before = {
                "controller": numeric_state(controller),
                "shared_estimator": numeric_state(controller.shared_estimator),
                "actuation": {
                    "pending_command": np.asarray(pending, float).tolist(),
                    "previous_applied_torque": previous_applied.tolist(),
                    "mirrored_drive_velocity_state": drive_v.tolist(),
                    "mirrored_drive_integral_state": drive_i.tolist(),
                    "encoder_previous": np.asarray(
                        raw["encoder"][max(index - 1, 0)], float
                    ).tolist(),
                },
                "physical_before_step": {
                    "qpos": physical_current.tolist(),
                    "qpos_previous": physical_previous.tolist(),
                    "qpos_finite_difference_per_s": (
                        (physical_current - physical_previous) / C.DT
                    ).tolist(),
                    "note": "qpos[k-1]/qpos[k-2] are post-step logs used only for posthoc physical FD restoration",
                },
            }

        # Runner order is detector -> controller -> old queued command applied.
        # Capture precedes both calls because r2_stability restores last_update_t
        # to t-dt and evaluates the shared update once inside controller.compute.
        # Replaying the detector here still reproduces the live shared-cache
        # behavior and all state accumulated through the previous cycle.
        if detector is not None:
            detector_tau = (np.asarray(raw["motor_torque"][index], float)
                            if "motor_torque" in raw else previous_applied)
            detector.update(t, th.copy(), thd.copy(), detector_tau.copy(),
                            tau_s.copy(), int(grip))

        q_d, qd_d, qdd_d = T.reference(t)
        output_command = np.asarray(controller.compute(
            t, th.copy(), thd.copy(), tau_s.copy(), int(grip),
            q_d.copy(), qd_d.copy(), qdd_d.copy(),
        ), float)
        expected_shape = (4, 6) if architecture == "servo" else (6,)
        if output_command.shape != expected_shape or not np.all(np.isfinite(output_command)):
            raise ValueError(f"invalid controller output at {t:.3f}s: {output_command.shape}")
        if architecture == "servo":
            predicted, next_drive_v, next_drive_i = mirrored_servo_drive(
                pending, th, thd, drive_v, drive_i
            )
        else:
            predicted = np.asarray(pending, float)
            next_drive_v, next_drive_i = drive_v, drive_i
        predicted = np.clip(predicted, -C.TAU_MAX, C.TAU_MAX)
        error = float(np.max(np.abs(predicted - applied_raw[index])))
        maximum_error = max(maximum_error, error)
        if source_kind == "actual_candidate" and error > ACTUAL_TORQUE_TOLERANCE:
            raise AssertionError(
                f"actual candidate actuation mismatch at index {index} t={t:.3f}: {error:g}"
            )

        if before is not None:
            selection = targets[index]
            snapshots.append({
                "requested_time_s": (selection["requested_times_s"][0]
                                     if selection["requested_times_s"] else t),
                "requested_times_s": selection["requested_times_s"],
                "peak_tags": selection["peak_tags"],
                "sample_index": index,
                "time_s": t,
                "before_compute": before,
                "input": {
                    "th": th.tolist(), "thd": thd.tolist(),
                    "tau_s": tau_s.tolist(), "grip": grip,
                    "q_d": q_d.tolist(), "qd_d": qd_d.tolist(),
                    "qdd_d": qdd_d.tolist(),
                },
                "output": output_command.tolist(),
                "after_compute": {
                    "controller": numeric_state(controller),
                    "shared_estimator": numeric_state(controller.shared_estimator),
                    "actuation": {
                        "predicted_applied_torque": predicted.tolist(),
                        "raw_applied_torque": applied_raw[index].tolist(),
                        "torque_replay_max_abs_error": error,
                        "next_pending_command": output_command.tolist(),
                        "mirrored_drive_velocity_state": next_drive_v.tolist(),
                        "mirrored_drive_integral_state": next_drive_i.tolist(),
                    },
                },
            })
        previous_applied = predicted.copy()
        pending = output_command.copy()
        drive_v, drive_i = next_drive_v.copy(), next_drive_i.copy()

    if weld_relation is None:
        grasp_index = int(np.searchsorted(time_s, T.T_GRASP))
        grasp_qpos = np.asarray(raw["qpos"][max(grasp_index - 1, 0)], float)
        weld_relation = (
            weld_relation_from_logged_qpos(grasp_qpos, float(mass_kg))
            if np.array_equal(np.asarray(com), np.asarray(R2_P0_COM)) else
            _weld_relation(grasp_qpos, float(mass_kg), com)
        )
    payload = {
        "schema": "R3_STABILITY_SNAPSHOTS.v1",
        "source_kind": source_kind,
        "architecture": architecture,
        "candidate": candidate_label,
        "mass_kg": float(mass_kg),
        "com_m": list(com),
        "source_raw_name": raw_path.name,
        "source_raw_sha256": _sha256(raw_path),
        "episodes_run": 0,
        "replay_statement": (
            "controller/detector replay from stored encoder, encoder velocity, JTS, grip, and previous motor torque only; qpos is posthoc physical restoration only"
        ),
        "actuation_replay": {
            "maximum_applied_torque_abs_error_nm": maximum_error,
            "assertion_tolerance_nm": (ACTUAL_TORQUE_TOLERANCE
                                       if source_kind == "actual_candidate" else None),
            "exact_match_required": source_kind == "actual_candidate",
            "one_cycle_queue": True,
            "servo_drive": "caller-owned reconstruction using public Drive equations",
        },
        "grasp_weld_relation": copy.deepcopy(dict(weld_relation)),
        "snapshots": snapshots,
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return payload


def quintic_weight(value: float, start: float, full: float) -> float:
    """C2 0/.5/1 schedule used for the registered B mass qualification."""
    if not full > start:
        raise ValueError("full mass must exceed start mass")
    value = float(value)
    if value <= start:
        return 0.0
    if value >= full:
        return 1.0
    x = float((value - start) / (full - start))
    if abs(x - 0.5) <= 4.0 * np.finfo(float).eps:
        return 0.5
    return float(10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5)


def qualify_mass_schedule(start: float, full: float, *, delta: float = 0.001
                          ) -> dict[str, Any]:
    """Record knot weights and one-sided numerical derivatives without physics."""
    midpoint = 0.5 * (start + full)
    points = (start - delta, start, start + delta,
              midpoint, full - delta, full, full + delta)
    weights = {f"{point:.6f}": quintic_weight(point, start, full)
               for point in points}
    derivative = lambda a, b: (
        quintic_weight(b, start, full) - quintic_weight(a, start, full)
    ) / (b - a)
    return {
        "method": "registered clipped quintic schedule; mathematical branch qualification only",
        "start_kg": float(start), "midpoint_kg": midpoint, "full_kg": float(full),
        "delta_kg": float(delta), "weights": weights,
        "one_sided_derivative_per_kg": {
            "start_left": derivative(start - delta, start),
            "start_right": derivative(start, start + delta),
            "full_left": derivative(full - delta, full),
            "full_right": derivative(full, full + delta),
        },
        "endpoint_midpoint_exact": {
            "start": quintic_weight(start, start, full),
            "midpoint": quintic_weight(midpoint, start, full),
            "full": quintic_weight(full, start, full),
        },
    }


def _resolve_class(spec: Mapping[str, Any]):
    module = importlib.import_module(str(spec["module"]))
    architecture = str(spec["architecture"])
    params = copy.deepcopy(dict(spec.get("params", {})))
    is_r1 = spec.get("id") == "R1" or params.get("id") == "R1"
    if is_r1:
        controller_class = module.Controller
    else:
        controller_class = module.make_controller(architecture, params)
    if (architecture == "torque"
            and "r3_reference_integral" in tuple(
                getattr(controller_class, "R2_STABILITY_STATE", ())
            )):
        original_class = controller_class

        class StabilityStateController(original_class):
            """Analysis-only selector; runtime compute remains inherited."""

            def r2_stability_state(self, time_s, grip):
                del grip  # fixed discrete grip is not a pole-producing state
                if time_s is None or float(self._r3_alpha(float(time_s))) == 0.0:
                    return ()
                return ("r3_reference_integral",)

        StabilityStateController.__name__ = original_class.__name__
        StabilityStateController.__qualname__ = original_class.__qualname__
        controller_class = StabilityStateController
    return module, controller_class


def fd_convergence(row: Mapping[str, Any]) -> dict[str, Any]:
    """Qualify both mechanical inputs and the closed-map h/h2 comparison."""
    mechanical = row["mechanical_fd"]
    step_input = mechanical["step_input"]
    previous_input = mechanical["previous_applied_input_for_jts"]
    closed = row["closed_loop"]["jacobian_step_diagnostics"]
    closed_a = float(closed["A_relative_step_change"])
    closed_b_value = closed.get("B_relative_step_change")
    closed_b = None if closed_b_value is None else float(closed_b_value)
    checks = {
        "mechanical_step_input": bool(step_input.get("step_converged", False)),
        "mechanical_previous_applied_input_for_jts": bool(
            previous_input.get("step_converged", False)
        ),
        "closed_map_A": bool(
            np.isfinite(closed_a) and closed_a <= FD_CONVERGENCE_TOLERANCE
        ),
        "closed_map_B": bool(
            closed_b is None
            or (np.isfinite(closed_b) and closed_b <= FD_CONVERGENCE_TOLERANCE)
        ),
    }
    return {
        "pass": all(checks.values()),
        "relative_change_tolerance": FD_CONVERGENCE_TOLERANCE,
        "checks": checks,
        "relative_step_change": {
            "mechanical_step_input": float(step_input["max_relative_step_change"]),
            "mechanical_previous_applied_input_for_jts": float(
                previous_input["max_relative_step_change"]
            ),
            "closed_map_A": closed_a,
            "closed_map_B": closed_b,
        },
        "qualification": (
            "central h/h2 agreement for both mechanical torque inputs and the "
            "closed discrete map; diagnostics present alone is not a pass"
        ),
    }


def _source_descriptor(source: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(source)
    result["raw_path"] = Path(result["raw_path"])
    return result


def run_specs(specs: Sequence[Mapping[str, Any]],
              reference_sources: Sequence[Mapping[str, Any]], output: Path,
              *, analyzer: Callable[..., dict[str, Any]] = analyze_snapshot
              ) -> dict[str, Any]:
    """Capture and analyze injected R3 specs against injected stored sources.

    Source descriptors contain ``raw_path``, ``architecture``, ``mass_kg``,
    ``com_m``, ``source_kind`` and ``requested_times_s``.  Optional keys are
    ``candidate`` (filter), ``detector_module``, ``include_actual_peaks``, and
    ``applied_torque_key``.  No R2 directory or protocol marker is implicit.
    """
    output = Path(output)
    if output.exists():
        raise RuntimeError("completed stability evidence exists; no overwrite")
    analyses: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    module_hashes: dict[str, str] = {}
    branch_qualifications: dict[str, Any] = {}
    for raw_spec in specs:
        spec = copy.deepcopy(dict(raw_spec))
        module, controller_class = _resolve_class(spec)
        label = str(spec.get("label", spec.get("id", spec["module"])))
        architecture = str(spec["architecture"])
        module_hashes[str(spec["module"])] = _sha256(Path(module.__file__).resolve())
        params = dict(spec.get("params", {}))
        if architecture == "torque" and params.get("schedule") in {"mass", "mass_phase"}:
            branch_qualifications[label] = qualify_mass_schedule(
                float(params["mass_start_kg"]), float(params["mass_full_kg"])
            )
        matched = 0
        for raw_source in reference_sources:
            source = _source_descriptor(raw_source)
            if str(source["architecture"]) != architecture:
                continue
            if source.get("candidate") not in (None, label, spec.get("id")):
                continue
            if source.get("source_kind") not in set(
                    spec.get("source_kinds", ("forced_r1", "actual_candidate"))):
                continue
            detector_class = None
            if source.get("detector_module"):
                detector_class = importlib.import_module(
                    str(source["detector_module"])
                ).Detector
            capture = capture_snapshots(
                source["raw_path"], architecture, float(source["mass_kg"]),
                source["com_m"], source["requested_times_s"], controller_class,
                source_kind=str(source["source_kind"]), candidate_label=label,
                detector_class=detector_class,
                include_actual_peaks=bool(source.get("include_actual_peaks", False)),
                applied_torque_key=source.get("applied_torque_key"),
            )
            captures.append({
                "candidate": label, "source_kind": capture["source_kind"],
                "architecture": architecture, "mass_kg": capture["mass_kg"],
                "com_m": capture["com_m"],
                "source_raw_name": capture["source_raw_name"],
                "source_raw_sha256": capture["source_raw_sha256"],
                "actuation_replay": capture["actuation_replay"],
                "sample_indices": [row["sample_index"] for row in capture["snapshots"]],
                "requested_times_s": [row["requested_times_s"] for row in capture["snapshots"]],
                "peak_tags": [row["peak_tags"] for row in capture["snapshots"]],
            })
            for snapshot in capture["snapshots"]:
                row = analyzer(
                    architecture, capture, float(snapshot["time_s"]),
                    controller_class=controller_class, candidate_label=label,
                )
                row["operating_point_qualification"] = (
                    "actual candidate controller/shared state, queue, drive, and physical trace"
                    if capture["source_kind"] == "actual_candidate" else
                    "candidate controller/shared state and actuation replayed causally on frozen R1 observations and physical trace"
                )
                row["fd_convergence"] = fd_convergence(row)
                row["combined_radius_fd_pass"] = bool(
                    row["closed_loop"][
                        "protocol_radius_pass_lt_1_plus_1e_minus_6"
                    ] and row["fd_convergence"]["pass"]
                )
                row["source_kind"] = capture["source_kind"]
                row["sample_index"] = snapshot["sample_index"]
                row["peak_tags"] = snapshot["peak_tags"]
                row["module"] = spec["module"]
                row["module_sha256"] = module_hashes[str(spec["module"])]
                row["parameters"] = params
                analyses.append(row)
            matched += 1
        if not matched:
            raise ValueError(f"no reference source matched {label}")
        if module_hashes[str(spec["module"])] != _sha256(Path(module.__file__).resolve()):
            raise RuntimeError("candidate source changed during stability analysis")

    pass_rows = [row.get("closed_loop", {}).get(
        "protocol_radius_pass_lt_1_plus_1e_minus_6") for row in analyses]
    fd_pass_count = int(sum(row["fd_convergence"]["pass"] for row in analyses))
    combined_pass_count = int(sum(row["combined_radius_fd_pass"] for row in analyses))
    result = {
        "schema": "R3_STABILITY.v1", "episodes_run": 0,
        "method": (
            "stored-observation controller/detector replay; actual traces exact torque checked; "
            "elastic local MuJoCo FD; exact 1ms ZOH; one-cycle command/JTS delays"
        ),
        "module_sha256": module_hashes,
        "specs": copy.deepcopy(list(specs)),
        "captures": captures,
        "mass_schedule_branch_qualifications": branch_qualifications,
        "analyses": analyses,
        "summary": {
            "total_cases": len(analyses),
            "protocol_radius_pass_cases": int(sum(value is True for value in pass_rows)),
            "protocol_radius_fail_cases": int(sum(value is False for value in pass_rows)),
            "fd_convergence_pass_cases": fd_pass_count,
            "fd_convergence_fail_cases": len(analyses) - fd_pass_count,
            "combined_radius_fd_pass_cases": combined_pass_count,
            "combined_radius_fd_fail_cases": len(analyses) - combined_pass_count,
            "actual_candidate_cases": int(sum(
                row["source_kind"] == "actual_candidate" for row in analyses
            )),
            "forced_r1_cases": int(sum(
                row["source_kind"] == "forced_r1" for row in analyses
            )),
        },
        "scope_warning": (
            "finite fixed-phase/fixed-contact local checks only; not a global, nonlinear, "
            "LTV, hybrid, passivity, or robust-stability proof"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True,
                        help="JSON list of candidate specs")
    parser.add_argument("--reference-sources", type=Path, required=True,
                        help="JSON list of private raw-source descriptors")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    specs = json.loads(args.specs.read_text())
    sources = json.loads(args.reference_sources.read_text())
    if not isinstance(specs, list) or not isinstance(sources, list):
        raise ValueError("specs and reference sources must be JSON lists")
    result = run_specs(specs, sources, args.output)
    print(result["summary"])


if __name__ == "__main__":
    main()
