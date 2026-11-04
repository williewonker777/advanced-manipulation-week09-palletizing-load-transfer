#!/usr/bin/env python3
"""R2 P0 physical characterization using frozen R1 controllers.

Plant/log channels are consumed only after an episode has finished.  Runtime
recording is limited to the detector's documented observed inputs and values
derived from them by the frozen R1 estimator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from sim.drive import Drive
from sim.nominal import Nominal
from sim.plant import Plant
from sim.runner import run_episode
import r1_detector
from tools import v2_resources

MASSES = (0.85, 2.6, 5.4, 7.7)
COM = (0.013, -0.009, 0.019)
ARCHITECTURES = ("servo", "torque")
GRAVITY = 9.81
PLACE_WINDOW = (3.6, 4.1)
PHASE_TIMES = {"grasp": T.T_GRASP, "transport": 2.65, "release": T.T_RELEASE}
STABILITY_SNAPSHOT_TIMES = (1.3, 2.5, 3.75, 3.87, 3.89, 4.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingDetector(r1_detector.Detector):
    """Record observed inputs and frozen-estimator derivatives only."""

    def reset(self):
        super().reset()
        self.rows: list[tuple[Any, ...]] = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        # The runner does not time detector.update.  This places each periodic
        # capacity guard immediately before the corresponding compute call.
        if int(round(float(t) / C.DT)) % 250 == 0:
            v2_resources.wait_for_capacity()
        result = super().update(t, th, thd, tau_m, tau_s, grip)
        packet = self._shared
        q = np.asarray(packet.q_prev, float).copy()
        v = np.asarray(packet.v, float).copy()
        qdd_ref = T.reference(float(t))[2]
        nom = packet.nom_box if grip else packet.nom
        model_tau = nom.inverse_dynamics(q, v, qdd_ref)
        self.rows.append((
            float(t), np.asarray(th).copy(), np.asarray(thd).copy(),
            np.asarray(tau_m).copy(), np.asarray(tau_s).copy(), bool(grip),
            q, v, np.asarray(packet.r_joint).copy(),
            np.asarray(packet.last_wrench).copy(),
            np.asarray(packet.last_table_wrench).copy(),
            np.asarray(packet.last_residual_wrench).copy(),
            np.asarray(tau_s, float) - model_tau,
            np.asarray(th, float) - T.reference(float(t))[0],
        ))
        return result

    def arrays(self) -> dict[str, np.ndarray]:
        names = (
            "obs_t", "encoder", "encoder_velocity", "motor_torque",
            "jts", "grip", "estimated_q", "estimated_qd",
            "joint_residual", "estimated_wrench", "estimated_table_wrench",
            "estimated_residual_wrench", "jts_model_gap", "encoder_ref_error",
        )
        columns = zip(*self.rows, strict=True)
        return {name: np.asarray(values) for name, values in zip(names, columns, strict=True)}


def quintic_shift(t: float, start: float, end: float, amplitude: float) -> float:
    """Quintic 0-to-amplitude shift, held after *end*."""
    if t <= start:
        return 0.0
    if t >= end:
        return float(amplitude)
    s = (t - start) / (end - start)
    return float(amplitude * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5))


def shifted_reference(controller, base_now, base_at, t: float,
                      start: float, end: float, amplitude: float):
    """Add a world-z perturbation without numerically differentiating the R1 reference."""
    q, qd, qdd = (np.asarray(value, float) for value in base_now)
    if amplitude == 0.0:
        # Exact zero-probe parity, including R1's delay compensation and 5 ms
        # endpoint shaping.  Do not replace its analytical derivatives with a
        # finite difference.
        return q, qd, qdd

    def perturbation(at: float) -> np.ndarray:
        qr = np.asarray(base_at(float(np.clip(at, 0.0, T.T_END)))[0], float)
        jacobian = controller.shared_estimator._point_jacobian(qr)[0]
        twist = np.array(
            [0.0, 0.0, quintic_shift(at, start, end, amplitude), 0.0, 0.0, 0.0]
        )
        return np.linalg.lstsq(jacobian, twist, rcond=1e-8)[0]

    h = C.DT
    dm, d0, dp = perturbation(t - h), perturbation(t), perturbation(t + h)
    return q + d0, qd + (dp - dm) / (2.0 * h), qdd + (dp - 2.0 * d0 + dm) / h**2


def probe_controller(base_cls, architecture: str, config: dict[str, float]):
    """Return an owned reference-only subclass; frozen classes remain unchanged."""
    start, end, amplitude = (float(config[k]) for k in ("start_s", "end_s", "amplitude_m"))
    if not (0.0 <= start < end <= T.T_END) or not np.isfinite(amplitude):
        raise ValueError("probe requires finite amplitude_m and 0 <= start_s < end_s <= T_END")

    if architecture == "servo":
        class Probe(base_cls):
            def _reference(self, t, q_d, qd_d, qdd_d):
                base_now = super()._reference(t, q_d, qd_d, qdd_d)

                def base_at(at):
                    return super(Probe, self)._reference(at, *T.reference(at))

                return shifted_reference(self, base_now, base_at, t, start, end, amplitude)
    else:
        class Probe(base_cls):
            def _reference(self, t):
                base_now = super()._reference(t)

                def base_at(at):
                    return super(Probe, self)._reference(at)

                return shifted_reference(self, base_now, base_at, t, start, end, amplitude)
    Probe.__name__ = f"R2P0Probe{architecture.title()}"
    return Probe


def reconstruct_geometry(qpos: np.ndarray, mass: float, com: tuple[float, ...]):
    """Reconstruct actual box bottom and TCP z from logged state, post hoc."""
    q0 = T.reference(0.0)[0]
    template = Plant(q0, mass, com)
    data = mujoco.MjData(template.m)
    box_id = template.m.body("box_free").id
    table_id = template.m.geom("table_place").id
    half = np.asarray(C.BOX_SIZE, float) / 2.0
    bottom = np.empty(len(qpos))
    tcp_z = np.empty(len(qpos))
    nominal = Nominal()
    for index, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_forward(template.m, data)
        rotation = data.xmat[box_id].reshape(3, 3)
        bottom[index] = data.xpos[box_id, 2] - np.abs(rotation[2]) @ half
        q = state[template.th_q].copy()
        q[template.E] += state[template.ph_q]
        tcp_z[index] = nominal.tcp_position(q)[2]
    mujoco.mj_forward(template.m, data)
    table_top = float(data.geom_xpos[table_id, 2] + template.m.geom_size[table_id, 2])
    return bottom, tcp_z, table_top


def first_crossing(t: np.ndarray, force: np.ndarray, threshold: float) -> dict[str, float | None]:
    indices = np.flatnonzero((t >= PLACE_WINDOW[0]) & (t < T.T_RELEASE) & (force > threshold))
    if not len(indices):
        return {"threshold_n": float(threshold), "time_s": None, "remaining_to_release_s": None}
    k = int(indices[0])
    return {"threshold_n": float(threshold), "time_s": float(t[k]),
            "remaining_to_release_s": float(T.T_RELEASE - t[k])}


def spectrum(values: np.ndarray, dt: float = C.DT) -> dict[str, Any]:
    """Compact per-axis one-sided spectral characterization."""
    x = np.asarray(values, float)
    x = x - np.mean(x, axis=0, keepdims=True)
    freq = np.fft.rfftfreq(len(x), dt)
    power = np.abs(np.fft.rfft(x, axis=0)) ** 2 / max(len(x) ** 2, 1)
    nonzero = freq > 0.0
    peak_indices = np.argmax(power[nonzero], axis=0)
    peak_freq = freq[nonzero][peak_indices]
    bands = {}
    for lo, hi in ((0.0, 5.0), (5.0, 50.0), (50.0, 200.0), (200.0, 500.1)):
        mask = (freq >= lo) & (freq < hi)
        bands[f"{lo:g}-{hi:g}_hz_rms"] = np.sqrt(np.sum(power[mask], axis=0)).tolist()
    return {"peak_frequency_hz": peak_freq.tolist(), **bands}


def force_displacement_curve(force, displacement, mask, bins: int = 12):
    values = displacement[mask]
    forces = force[mask]
    if len(values) < 2 or np.ptp(values) == 0.0:
        return []
    edges = np.linspace(float(values.min()), float(values.max()), bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (values >= lo) & (values <= hi if hi == edges[-1] else values < hi)
        if np.any(selected):
            rows.append({"displacement_m": float(np.mean(values[selected])),
                         "force_mean_n": float(np.mean(forces[selected])),
                         "force_min_n": float(np.min(forces[selected])),
                         "force_max_n": float(np.max(forces[selected])),
                         "samples": int(np.count_nonzero(selected))})
    return rows


def longest_true_run(mask: np.ndarray, t: np.ndarray) -> dict[str, float | None]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {"duration_s": 0.0, "start_s": None, "end_s": None}
    splits = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    run = max(splits, key=len)
    return {"duration_s": float(len(run) * C.DT), "start_s": float(t[run[0]]),
            "end_s": float(t[run[-1]] + C.DT)}


def late_contact_dynamics(t: np.ndarray, force: np.ndarray, bottom: np.ndarray,
                          table_top: float, mass: float) -> dict[str, Any]:
    pre_release = (t >= PLACE_WINDOW[0]) & (t < T.T_RELEASE)
    half_support = pre_release & (force >= 0.5 * mass * GRAVITY)
    sustained = longest_true_run(half_support, t)
    contact = np.flatnonzero(pre_release & (force > 1.0))
    fall = None
    if len(contact):
        after = np.flatnonzero((np.arange(len(t)) > contact[0]) & pre_release & (force <= 1e-9))
        if len(after):
            fall = {"time_s": float(t[after[0]]),
                    "elapsed_from_first_1n_contact_s": float(t[after[0]] - t[contact[0]])}
    last20 = (t >= T.T_RELEASE - 0.020) & (t < T.T_RELEASE)
    velocity = np.gradient(bottom, C.DT)
    gap = bottom - table_top
    before_release_values = half_support[np.flatnonzero(pre_release)]
    trailing = 0
    for value in before_release_values[::-1]:
        if not value:
            break
        trailing += 1
    return {
        "half_mg_longest_contiguous_before_release": sustained,
        "half_mg_contiguous_immediately_before_release_s": float(C.DT * trailing),
        "first_force_return_to_zero_after_1n_contact": fall,
        "last_20ms_box_bottom_minus_table_top_m": {
            "mean": float(np.mean(gap[last20])), "minimum": float(np.min(gap[last20])),
            "maximum": float(np.max(gap[last20]))},
        "last_20ms_box_bottom_world_z_velocity_m_s": {
            "mean": float(np.mean(velocity[last20])), "minimum": float(np.min(velocity[last20])),
            "maximum": float(np.max(velocity[last20]))},
        "interpretation": "unilateral damped contact/bounce; a transient 0.5mg crossing is not sustained G4 support or a stiffness sample",
    }


def nominal_snapshots(detector: RecordingDetector, t: np.ndarray, contact_time: float | None):
    phases = dict(PHASE_TIMES)
    phases["precontact"] = max(PLACE_WINDOW[0], (contact_time or T.T_RELEASE) - 0.05)
    arrays = detector.arrays()
    result = {}
    for name, target in phases.items():
        k = int(np.argmin(np.abs(t - target)))
        q = arrays["estimated_q"][k]
        grip = bool(arrays["grip"][k])
        packet = detector._shared
        nom = packet.nom_box if grip else packet.nom
        mass_matrix = nom.mass_matrix(q)
        jacobian = packet._point_jacobian(q)[0]
        eps = 1e-5
        gravity_derivative = np.column_stack([
            (nom.gravity(q + eps * np.eye(6)[j]) - nom.gravity(q - eps * np.eye(6)[j])) / (2 * eps)
            for j in range(6)
        ])
        result[name] = {"time_s": float(t[k]), "estimated_q_rad": q.tolist(),
                        "mass_matrix": mass_matrix.tolist(), "bottom_jacobian_6d": jacobian.tolist(),
                        "gravity_derivative": gravity_derivative.tolist(),
                        "payload_model": "known mass/COM nominal model" if grip else "empty nominal model"}
    return result


def summarize_episode(architecture: str, mass: float, result: dict[str, Any], log: dict[str, Any],
                      detector: RecordingDetector, raw_file: Path) -> dict[str, Any]:
    t, force = np.asarray(log["t"]), np.asarray(log["f_table"])
    bottom, tcp_z, table_top = reconstruct_geometry(log["qpos"], mass, COM)
    contact_1 = first_crossing(t, force, 1.0)
    contact_alt = first_crossing(t, force, 0.05 * mass * GRAVITY)
    contact_time = contact_1["time_s"]
    contact_k = int(np.argmin(np.abs(t - contact_time))) if contact_time is not None else None
    tcp_disp = tcp_z[contact_k] - tcp_z if contact_k is not None else np.zeros_like(tcp_z)
    bottom_disp = bottom[contact_k] - bottom if contact_k is not None else np.zeros_like(bottom)
    crossing_50 = first_crossing(t, force, 0.5 * mass * GRAVITY)
    if crossing_50["time_s"] is not None and contact_time is not None:
        cross_k = int(np.argmin(np.abs(t - crossing_50["time_s"])))
        crossing_50.update(
            reference_contact_available=True,
            tcp_displacement_from_1n_contact_m=float(tcp_disp[cross_k]),
            box_bottom_displacement_from_1n_contact_m=float(bottom_disp[cross_k]),
            required_average_tcp_speed_m_s=float(tcp_disp[cross_k] / max(T.T_RELEASE - contact_time, C.DT)),
            required_average_box_bottom_speed_m_s=float(bottom_disp[cross_k] / max(T.T_RELEASE - contact_time, C.DT)),
        )
    elif crossing_50["time_s"] is not None:
        crossing_50.update(reference_contact_available=False,
                           tcp_displacement_from_1n_contact_m=None,
                           box_bottom_displacement_from_1n_contact_m=None,
                           required_average_tcp_speed_m_s=None,
                           required_average_box_bottom_speed_m_s=None)
    last20 = (t >= T.T_RELEASE - 0.020) & (t < T.T_RELEASE)
    place = (t >= PLACE_WINDOW[0]) & (t <= PLACE_WINDOW[1])
    velocity = np.gradient(np.asarray(log["q"]), C.DT, axis=0)
    arrays = detector.arrays()
    approach = place & (t < T.T_RELEASE) & (force > 0.0)
    bottom_speed = np.gradient(bottom, C.DT)
    quasi = approach & (np.abs(bottom_speed) <= np.quantile(np.abs(bottom_speed[approach]), 0.25) if np.any(approach) else False)
    dynamic = approach & ~quasi
    return {
        "architecture": architecture, "mass_kg": mass, "com_m": list(COM),
        "raw_npz": raw_file.name, "raw_sha256": sha256_file(raw_file),
        "runner": {k: result[k] for k in ("rms_mm", "max_mm", "diverged", "saturation_pct", "torque_jitter", "ctrl_time_us")},
        "contact": {"force_gt_1n": contact_1, "force_gt_0p05mg": contact_alt,
                    "force_50pct_mg_crossing": crossing_50},
        "support": {"last_20ms_mean_n": float(np.mean(force[last20])),
                    "last_20ms_min_n": float(np.min(force[last20])),
                    "last_20ms_mean_over_mg": float(np.mean(force[last20]) / (mass * GRAVITY)),
                    "last_20ms_min_over_mg": float(np.min(force[last20]) / (mass * GRAVITY)),
                    "place_peak_n": float(np.max(force[place])),
                    "place_peak_over_mg": float(np.max(force[place]) / (mass * GRAVITY))},
        "geometry": {"place_table_top_z_m": table_top,
                     "box_bottom_at_1n_contact_z_m": (float(bottom[contact_k])
                                                       if contact_k is not None else None),
                     "tcp_at_1n_contact_z_m": (float(tcp_z[contact_k])
                                                if contact_k is not None else None),
                     "weld_compliance_included": True},
        "force_vs_normal_tcp_displacement": (force_displacement_curve(force, tcp_disp, approach)
                                               if contact_k is not None else []),
        "force_vs_actual_box_bottom_displacement": (force_displacement_curve(force, bottom_disp, approach)
                                                      if contact_k is not None else []),
        "late_contact_dynamics": late_contact_dynamics(t, force, bottom, table_top, mass),
        "quasi_static_dynamic": {
            "quasi_static_established": False,
            "classification": "posthoc lower-quartile absolute box-bottom speed is retained only as a descriptive low-speed subset; contact loss/bounce prevents a quasi-static stiffness inference",
            "speed_cutoff_m_s": float(np.quantile(np.abs(bottom_speed[approach]), 0.25)) if np.any(approach) else None,
            "quasi_static_force_mean_n": float(np.mean(force[quasi])) if np.any(quasi) else None,
            "dynamic_force_mean_n": float(np.mean(force[dynamic])) if np.any(dynamic) else None,
            "quasi_static_samples": int(np.count_nonzero(quasi)), "dynamic_samples": int(np.count_nonzero(dynamic))},
        "headroom": {
            "torque_peak_abs_nm": np.max(np.abs(log["tau"]), axis=0).tolist(),
            "torque_headroom_nm": (C.TAU_MAX - np.max(np.abs(log["tau"]), axis=0)).tolist(),
            "velocity_peak_abs_rad_s": np.max(np.abs(velocity), axis=0).tolist(),
            "velocity_headroom_rad_s": (C.QD_MAX - np.max(np.abs(velocity), axis=0)).tolist()},
        "observed_signal_spectra": {
            key: spectrum(arrays[key]) for key in
            ("encoder", "jts", "joint_residual", "jts_model_gap", "encoder_ref_error")},
        "nominal_snapshots": nominal_snapshots(detector, t, contact_time),
    }


def run_one(architecture: str, mass: float, raw_dir: Path, probe: dict[str, float] | None = None):
    module = importlib.import_module(f"r1_{architecture}")
    cls = module.Controller if probe is None else probe_controller(module.Controller, architecture, probe)
    before = v2_resources.wait_for_capacity()
    controller, detector = cls(), RecordingDetector()
    result, log = run_episode(controller, box_mass=mass, box_com=COM, detector=detector)
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = "baseline" if probe is None else "probe"
    raw_file = raw_dir / f"{architecture}_{mass:g}kg_{suffix}.npz"
    payload = {k: np.asarray(v) for k, v in log.items() if k != "plant"}
    payload.update(detector.arrays())
    bottom, tcp_z, table_top = reconstruct_geometry(payload["qpos"], mass, COM)
    payload.update(actual_box_bottom_z=bottom, actual_tcp_z=tcp_z,
                   place_table_top_z=np.asarray(table_top))
    np.savez_compressed(raw_file, **payload)
    summary = summarize_episode(architecture, mass, result, log, detector, raw_file)
    summary["host_before"] = before
    summary["host_after"] = v2_resources.snapshot()
    return summary


def aggregate_boundaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate directly measured boundaries without fitting a contact law."""
    by_architecture = {}

    def numeric_range(values):
        valid = [float(value) for value in values if value is not None]
        return [min(valid), max(valid)] if valid else [None, None]

    for architecture in ARCHITECTURES:
        selected = [row for row in rows if row["architecture"] == architecture]
        remaining = [row["contact"]["force_gt_1n"]["remaining_to_release_s"] for row in selected]
        crossing = [row["contact"]["force_50pct_mg_crossing"] for row in selected]
        tcp_displacement = [row.get("tcp_displacement_from_1n_contact_m") for row in crossing]
        bottom_displacement = [row.get("box_bottom_displacement_from_1n_contact_m") for row in crossing]
        tcp_speed = [row.get("required_average_tcp_speed_m_s") for row in crossing]
        bottom_speed = [row.get("required_average_box_bottom_speed_m_s") for row in crossing]
        valid_tcp_speed = np.asarray([value for value in tcp_speed if value is not None])
        valid_bottom_speed = np.asarray([value for value in bottom_speed if value is not None])
        by_architecture[architecture] = {
            "contact_remaining_to_release_s_range": numeric_range(remaining),
            "tcp_displacement_to_50pct_mg_m_range": numeric_range(tcp_displacement),
            "actual_box_bottom_displacement_to_50pct_mg_m_range": numeric_range(bottom_displacement),
            "required_average_tcp_speed_to_50pct_mg_m_s_range": numeric_range(tcp_speed),
            "required_average_box_bottom_speed_to_50pct_mg_m_s_range": numeric_range(bottom_speed),
            "episodes_above_r1_0p01m_s_tcp_rate": int(np.count_nonzero(valid_tcp_speed > 0.01)),
            "episodes_above_r1_0p01m_s_box_bottom_rate": int(np.count_nonzero(valid_bottom_speed > 0.01)),
            "episodes_without_1n_contact": int(np.count_nonzero([value is None for value in remaining])),
            "episodes_without_50pct_mg_crossing": int(np.count_nonzero(
                [row.get("time_s") is None for row in crossing])),
            "episodes_with_50pct_crossing_but_no_1n_reference": int(np.count_nonzero(
                [row.get("time_s") is not None and row.get("reference_contact_available") is False
                 for row in crossing])),
            "episodes": len(selected),
        }
    return {
        "definition": "observed displacement from first F>1 N sample to first F>0.5mg sample divided by measured time remaining from first contact to release",
        "r1_0p01m_s_comparison": "measured descriptive comparison only; not a selected bound and not a linear-contact extrapolation",
        "by_architecture": by_architecture,
    }


def numeric_state(instance) -> dict[str, Any]:
    """JSON-safe numeric/scalar state; intentionally excludes model/drive objects."""
    result = {}
    for key, value in vars(instance).items():
        if key in {"nom", "nom_box", "drive", "shared_estimator"}:
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            result[key] = value
        elif isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            result[key] = value.tolist()
        elif isinstance(value, (tuple, list)) and all(isinstance(item, (bool, int, float)) for item in value):
            result[key] = list(value)
    return result


def _flatten_numeric(prefix: str, value: Any, output: dict[str, np.ndarray]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else key, item, output)
    elif isinstance(value, (list, tuple)) and any(isinstance(item, dict) for item in value):
        for index, item in enumerate(value):
            _flatten_numeric(f"{prefix}.{index}", item, output)
    elif value is not None and not isinstance(value, str):
        array = np.asarray(value)
        if array.dtype.kind in "biuf":
            output[prefix] = array


def mirrored_servo_drive(cmd: np.ndarray, th: np.ndarray, thd: np.ndarray,
                         velocity_state: np.ndarray, integral_state: np.ndarray):
    """Public Drive equations with caller-owned state; does not inspect a Drive instance."""
    pos, _vel, vel_ff, tau_ff = np.asarray(cmd, float)
    velocity_next = velocity_state + C.DRIVE_VEL_FILTER * (thd - velocity_state)
    velocity_error = vel_ff + C.DRIVE_KP * (pos - th) - velocity_next
    integral_next = integral_state + velocity_error * C.DT
    torque = tau_ff + C.DRIVE_KV * velocity_error + C.DRIVE_KI * integral_next
    return torque, velocity_next, integral_next


def weld_relation_from_logged_qpos(qpos: np.ndarray, mass: float) -> dict[str, Any]:
    """Reconstruct the grasp-time weld relation post hoc in separate MjData."""
    template = Plant(T.reference(0.0)[0], mass, COM)
    data = mujoco.MjData(template.m)
    data.qpos[:] = qpos
    mujoco.mj_forward(template.m, data)
    flange = template.m.body("link_6").id
    box = template.m.body("box_free").id
    rotation = data.xmat[flange].reshape(3, 3)
    inverse, relative = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(inverse, data.xquat[flange])
    mujoco.mju_mulQuat(relative, inverse, data.xquat[box])
    return {
        "source": "raw qpos immediately before grasp, reconstructed post hoc; never controller input",
        "translation_in_flange_m": (rotation.T @ (data.xpos[box] - data.xpos[flange])).tolist(),
        "relative_quaternion_wxyz": relative.tolist(),
    }


def capture_stability_snapshots(raw_dir: Path) -> list[Path]:
    """Replay observed inputs only to recover frozen-controller numeric states."""
    written = []
    for architecture in ARCHITECTURES:
        module = importlib.import_module(f"r1_{architecture}")
        for mass in MASSES:
            raw_file = raw_dir / f"{architecture}_{mass:g}kg_baseline.npz"
            # Materialize once: repeated indexing on NpzFile would repeatedly
            # decompress zip members throughout the 6000-cycle replay.
            with np.load(raw_file, allow_pickle=False) as archive:
                raw = {name: archive[name] for name in archive.files}
            controller = module.Controller()
            controller.nom, controller.nom_box = Nominal(), Nominal()
            controller.nom_box.set_payload(mass, COM)
            controller.box_mass, controller.box_com = mass, COM
            if architecture == "servo":
                controller.drive = Drive()
            controller.reset()
            pending = (np.array([T.reference(0.0)[0], np.zeros(6), np.zeros(6), np.zeros(6)])
                       if architecture == "servo" else np.zeros(6))
            drive_velocity = np.zeros(6)
            drive_integral = np.zeros(6)
            maximum_torque_replay_error = 0.0
            target_indices = {int(np.argmin(np.abs(raw["obs_t"] - target))): target
                              for target in STABILITY_SNAPSHOT_TIMES}
            snapshots = []
            for index, t in enumerate(raw["obs_t"]):
                if index % 250 == 0:
                    v2_resources.wait_for_capacity()
                before = None
                if index in target_indices:
                    before = {"controller": numeric_state(controller),
                              "shared_estimator": numeric_state(controller.shared_estimator),
                              "actuation": {
                                  "pending_command": np.asarray(pending).tolist(),
                                  "previous_applied_torque": (raw["tau"][index - 1].tolist()
                                                               if index else np.zeros(6).tolist()),
                                  "mirrored_drive_velocity_state": drive_velocity.tolist(),
                                  "mirrored_drive_integral_state": drive_integral.tolist(),
                                  "encoder_previous": (raw["encoder"][index - 1].tolist()
                                                       if index else raw["encoder"][0].tolist())},
                              "physical_before_step": {
                                  "qpos": (raw["qpos"][index - 1].tolist()
                                           if index else raw["qpos"][0].tolist()),
                                  "qpos_previous": (raw["qpos"][index - 2].tolist()
                                                    if index >= 2 else raw["qpos"][0].tolist()),
                                  "qpos_finite_difference_per_s": (
                                      (raw["qpos"][index - 1] - raw["qpos"][index - 2]) / C.DT
                                      if index >= 2 else np.zeros_like(raw["qpos"][0])).tolist(),
                                  "note": "qpos[k-1] is pre-step-k; coordinate difference is provided, not claimed to be MuJoCo qvel"}}
                q_d, qd_d, qdd_d = T.reference(float(t))
                output = controller.compute(
                    float(t), raw["encoder"][index].copy(),
                    raw["encoder_velocity"][index].copy(), raw["jts"][index].copy(),
                    int(raw["grip"][index]), q_d, qd_d, qdd_d)
                output = np.asarray(output, float)
                if architecture == "servo":
                    predicted, drive_velocity_next, drive_integral_next = mirrored_servo_drive(
                        pending, raw["encoder"][index], raw["encoder_velocity"][index],
                        drive_velocity, drive_integral)
                else:
                    predicted = np.asarray(pending, float)
                    drive_velocity_next, drive_integral_next = drive_velocity, drive_integral
                predicted = np.clip(predicted, -C.TAU_MAX, C.TAU_MAX)
                replay_error = float(np.max(np.abs(predicted - raw["tau"][index])))
                maximum_torque_replay_error = max(maximum_torque_replay_error, replay_error)
                if replay_error > 1e-10:
                    raise AssertionError(
                        f"{architecture} {mass:g} kg actuation replay mismatch at {t:.3f}: {replay_error}"
                    )
                if before is not None:
                    snapshots.append({
                        "requested_time_s": target_indices[index], "sample_index": index,
                        "time_s": float(t), "before_compute": before,
                        "input": {"th": raw["encoder"][index].tolist(),
                                  "thd": raw["encoder_velocity"][index].tolist(),
                                  "tau_s": raw["jts"][index].tolist(),
                                  "grip": bool(raw["grip"][index]),
                                  "q_d": q_d.tolist(), "qd_d": qd_d.tolist(),
                                  "qdd_d": qdd_d.tolist()},
                        "output": output.tolist(),
                        "after_compute": {"controller": numeric_state(controller),
                                          "shared_estimator": numeric_state(controller.shared_estimator),
                                          "actuation": {
                                              "predicted_applied_torque": predicted.tolist(),
                                              "raw_applied_torque": raw["tau"][index].tolist(),
                                              "torque_replay_max_abs_error": replay_error,
                                              "next_pending_command": output.tolist(),
                                              "mirrored_drive_velocity_state": drive_velocity_next.tolist(),
                                              "mirrored_drive_integral_state": drive_integral_next.tolist()}},
                    })
                pending = output.copy()
                drive_velocity, drive_integral = drive_velocity_next, drive_integral_next
            grasp_index = int(np.searchsorted(raw["obs_t"], T.T_GRASP))
            payload = {
                "schema": "R2_P0_STABILITY_SNAPSHOTS.v1", "architecture": architecture,
                "mass_kg": mass, "com_m": list(COM), "source_raw_npz": raw_file.name,
                "source_raw_sha256": sha256_file(raw_file),
                "replay_statement": "frozen R1 compute replay from recorded measured th/thd/tau_s/grip only; no Plant/log/oracle channel",
                "actuation_replay": {
                    "maximum_applied_torque_abs_error_nm": maximum_torque_replay_error,
                    "assertion_tolerance_nm": 1e-10,
                    "servo_drive_state": "independently reconstructed from public equations; no Drive instance state read",
                    "runner_delay": "pending command drives current applied torque; current controller output queues next cycle"},
                "grasp_weld_relation": weld_relation_from_logged_qpos(
                    raw["qpos"][grasp_index - 1], mass),
                "snapshots": snapshots,
            }
            json_file = raw_dir / f"{architecture}_{mass:g}kg_stability_snapshots.json"
            npz_file = raw_dir / f"{architecture}_{mass:g}kg_stability_snapshots.npz"
            json_file.write_text(json.dumps(payload, indent=2) + "\n")
            flat = {}
            _flatten_numeric("", payload, flat)
            np.savez_compressed(npz_file, **flat)
            written.extend((json_file, npz_file))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--public-out", type=Path)
    parser.add_argument("--snapshots-only", action="store_true",
                        help="replay existing raw observed inputs into private numeric stability snapshots")
    parser.add_argument("--probe-config", type=Path,
                        help="explicit root-reviewed JSON; absence runs the fixed eight-episode baseline")
    args = parser.parse_args(argv)
    if args.snapshots_only:
        written = capture_stability_snapshots(args.raw_dir)
        print(f"wrote {len(written)} private snapshot files", flush=True)
        return
    if args.public_out is None:
        parser.error("--public-out is required unless --snapshots-only is used")
    probe = json.loads(args.probe_config.read_text()) if args.probe_config else None
    rows = []
    for architecture in ARCHITECTURES:
        for mass in MASSES:
            rows.append(run_one(architecture, mass, args.raw_dir, probe))
            print(architecture, mass, rows[-1]["contact"], flush=True)
    sources = {name: sha256_file(ROOT / name) for name in
               ("r1_servo.py", "r1_torque.py", "r1_detector.py", "tools/v2_resources.py", "tools/r2_p0.py")}
    freeze = ROOT / "results/v2_protocol/R1_FROZEN_FOR_R2.json"
    output = {
        "schema": "R2_P0_BASELINE.v1" if probe is None else "R2_P0_PROBE.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "analysis-only physical characterization; Plant/log values never enter controller or detector inputs",
        "official": False, "candidate_evaluation": False, "sequential_workers": 1,
        "conditions": {"masses_kg": list(MASSES), "com_m": list(COM),
                       "architectures": list(ARCHITECTURES), "episodes": len(rows)},
        "probe_config": probe,
        "r1_freeze_sha256": sha256_file(freeze), "source_sha256": sources,
        "baseline_execution_source_sha256": dict(sources),
        "postprocess_source_sha256": sources["tools/r2_p0.py"],
        "resource_policy": {"guard": "wait_for_capacity before each episode and every 250 cycles outside controller timing",
                            "min_available_ram_gib": 6, "max_total_simulations": 4,
                            "owned_simulation_workers": 1, "signals_sent_to_other_processes": 0},
        "contact_model_statement": "Force-displacement is reported as sampled nonlinear data; no linear contact assumption is made.",
        "transient_support_warning": "The first 0.5mg crossing is impact/transient evidence only; sustained pre-release support is reported separately.",
        "r1_reference_timing": {
            "runner_command_delay_s": 0.001,
            "endpoint_advance_profile_runner_time_s": [3.749, 3.899],
            "maximum_reference_advance_s": 0.005,
            "post_release_advance_return_runner_time_s": [3.899, 3.999],
            "payload_unload_profile_runner_time_s": [3.84, 3.9],
            "source": "frozen R1 Controller constants and _reference equations; descriptive, not Plant truth"},
        "derived_boundary_measurements": aggregate_boundaries(rows),
        "episodes": rows,
    }
    args.public_out.parent.mkdir(parents=True, exist_ok=True)
    args.public_out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
