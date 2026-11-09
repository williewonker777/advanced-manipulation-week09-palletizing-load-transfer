"""Portable normal-only fitting for the fixed R5 detector model grid.

Only documented signal arrays and public reference data enter model fitting.
A phase with no selected fit samples is fail-closed: its numeric location and
scale slots are placeholders and the learned branch is inactive there.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import sys
from typing import Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import r1_detector as _r1  # noqa: E402
from sim import trajectory as _trajectory  # noqa: E402


N_SAMPLES = 6000
DT = 0.001
FIT_STRIDE = 5
EMA_ALPHA = 0.1
ON_COUNT = 4
CLEAR_COUNT = 8
RIDGE = 0.01
CLIP = 8.0
SCALE_FLOOR = 0.25
PHASE_BOUNDARIES = np.array(
    [
        _trajectory.T_GRASP - 0.08,
        _trajectory.T_GRASP + 0.20,
        _trajectory.T_RELEASE - 0.15,
        _trajectory.T_RELEASE + 0.10,
    ],
    dtype=float,
)

_ARRAY6 = ("th", "thd", "tau_m", "tau_s", "r_joint", "wrench", "ref_q", "ref_qd", "ref_qdd")
_BOOL = ("grip", "active", "r1_detect")
_REQUIRED = {"t", *_ARRAY6, *_BOOL, "mass", "com", "mode", "role", "normal_only"}
_FORBIDDEN = {
    "collision", "collision_label", "contact", "external_force", "force",
    "true_q", "true_torque", "q_true", "tau_true", "offset", "drift",
    "seed", "label", "kind", "official", "perturb", "perturbation",
    "qpos", "qvel", "f_table", "f_tool", "sim_state",
}

_SPECS = (
    {"id": "scale_active_phase_mad", "family": "SCALE_ACTIVE", "gate": "r1_active", "target_dims": 2, "scale": "phase_mad", "ridge": None},
    {"id": "scale_active_state_log", "family": "SCALE_ACTIVE", "gate": "r1_active", "target_dims": 2, "scale": "state_logscale", "ridge": RIDGE},
    {"id": "rff_active_b64", "family": "RFF_ACTIVE", "gate": "r1_active", "target_dims": 2, "bands": 64, "lengthscale": 1.5, "parameter_seed": 377003, "ridge": RIDGE, "scale": "phase_mad"},
    {"id": "rff_active_b128", "family": "RFF_ACTIVE", "gate": "r1_active", "target_dims": 2, "bands": 128, "lengthscale": 1.5, "parameter_seed": 377004, "ridge": RIDGE, "scale": "phase_mad"},
    {"id": "rff_full_b64", "family": "RFF_FULL", "gate": "t>=0.06", "target_dims": 3, "bands": 64, "lengthscale": 1.5, "parameter_seed": 377005, "ridge": RIDGE, "scale": "phase_mad"},
    {"id": "rff_full_b128", "family": "RFF_FULL", "gate": "t>=0.06", "target_dims": 3, "bands": 128, "lengthscale": 1.5, "parameter_seed": 377006, "ridge": RIDGE, "scale": "phase_mad"},
    {"id": "ar_full_lag20ms", "family": "AR_FULL", "gate": "t>=0.06", "target_dims": 2, "lag_ms": 20, "ridge": RIDGE, "scale": "phase_mad"},
    {"id": "ar_full_lag50ms", "family": "AR_FULL", "gate": "t>=0.06", "target_dims": 2, "lag_ms": 50, "ridge": RIDGE, "scale": "phase_mad"},
)


def specifications():
    """Return a defensive copy of the eight pre-registered configurations."""
    return deepcopy(list(_SPECS))


def _resolve_spec(spec):
    identifier = spec if isinstance(spec, str) else spec.get("id")
    matches = [candidate for candidate in _SPECS if candidate["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"unknown specification: {identifier!r}")
    canonical = deepcopy(matches[0])
    if not isinstance(spec, str) and dict(spec) != canonical:
        raise ValueError("specification differs from the pre-registered values")
    return canonical


def _validate_episode(ep: Mapping, allowed_roles=None, require_normal=False):
    if not isinstance(ep, Mapping):
        raise TypeError("episode must be a mapping")
    forbidden = sorted(set(ep) & _FORBIDDEN)
    if forbidden:
        raise ValueError(f"forbidden fields: {forbidden}")
    missing = sorted(_REQUIRED - set(ep))
    if missing:
        raise ValueError(f"missing fields: {missing}")
    extra = sorted(set(ep) - _REQUIRED)
    if extra:
        raise ValueError(f"unapproved fields: {extra}")
    if allowed_roles is not None and ep["role"] not in set(allowed_roles):
        raise ValueError(f"role {ep['role']!r} is not allowed here")
    if not isinstance(ep["normal_only"], (bool, np.bool_)):
        raise TypeError("normal_only must be boolean")
    if require_normal and not bool(ep["normal_only"]):
        raise ValueError("fit/calibration episodes must be normal_only=True")
    if ep["mode"] not in ("servo", "torque"):
        raise ValueError("mode must be servo or torque")

    t = np.asarray(ep["t"])
    if t.shape != (N_SAMPLES,) or not np.issubdtype(t.dtype, np.number):
        raise ValueError("t must be a numeric (6000,) array")
    if not np.all(np.isfinite(t)) or not np.allclose(t, np.arange(N_SAMPLES) * DT, rtol=0.0, atol=1e-12):
        raise ValueError("t must be the fixed 0..5.999 s, 1 ms grid")
    for key in _ARRAY6:
        value = np.asarray(ep[key])
        if value.shape != (N_SAMPLES, 6) or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{key} must be a numeric (6000,6) array")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains non-finite values")
    for key in _BOOL:
        value = np.asarray(ep[key])
        if value.shape != (N_SAMPLES,) or value.dtype != np.bool_:
            raise ValueError(f"{key} must be a boolean (6000,) array")
    mass = np.asarray(ep["mass"])
    com = np.asarray(ep["com"])
    if mass.shape != () or not np.isfinite(float(mass)):
        raise ValueError("mass must be a finite scalar")
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise ValueError("com must be a finite length-three vector")


def _phase_index(t):
    return np.searchsorted(PHASE_BOUNDARIES, np.asarray(t, dtype=float), side="right")


def _normal_features(ep):
    return np.vstack([_r1.normal_features(t, ep["mass"], ep["com"]) for t in np.asarray(ep["t"])])


def _augmented_features(ep):
    base = _normal_features(ep)
    phase = np.eye(5, dtype=float)[_phase_index(ep["t"])]
    grip = np.asarray(ep["grip"], dtype=float)[:, None]
    mass_grip = grip * float(ep["mass"])
    return np.column_stack((base, phase, grip, mass_grip))


def _r1_prediction(ep):
    params = _r1.MODEL_PARAMETERS[ep["mode"]]
    mean = np.asarray(params["mean"], dtype=float)
    scale = np.asarray(params["scale"], dtype=float)
    weights = np.asarray(params["weights"], dtype=float)
    z = np.clip((_normal_features(ep) - mean) / scale, -CLIP, CLIP)
    return z @ weights


def _gate(spec, ep):
    if spec["gate"] == "r1_active":
        return np.asarray(ep["active"], dtype=bool).copy()
    return np.asarray(ep["t"], dtype=float) >= 0.06


def _standardizer(x):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _standardize(x, mean, scale):
    return np.clip((x - mean) / scale, -CLIP, CLIP)


def _ridge(x, y, ridge=RIDGE):
    n = max(len(x), 1)
    penalty = np.eye(x.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    lhs = (x.T @ x) / n + penalty
    rhs = (x.T @ y) / n
    return np.linalg.solve(lhs, rhs)


def _phase_location(residual, phase, selected):
    result = np.zeros((5, residual.shape[1]), dtype=float)
    for index in range(5):
        values = residual[selected & (phase == index)]
        if len(values):
            result[index] = np.median(values, axis=0)
    return result


def _phase_mad(residual, phase, selected, location=None):
    result = np.ones(5, dtype=float)
    if location is None:
        location = np.zeros((5, residual.shape[1]), dtype=float)
    for index in range(5):
        values = residual[selected & (phase == index)] - location[index]
        if len(values):
            norm = np.linalg.norm(values, axis=1)
            centre = np.median(norm)
            mad = np.median(np.abs(norm - centre)) * 1.4826
            result[index] = max(float(mad), SCALE_FLOOR)
    return result


def _ar_raw_features(ep, lag_samples):
    base = _augmented_features(ep)
    n = len(base)
    delayed = np.zeros((n, 16), dtype=float)
    source = np.column_stack((
        np.asarray(ep["wrench"])[:, :2],
        np.asarray(ep["r_joint"]),
        np.asarray(ep["th"]) - np.asarray(ep["ref_q"]),
    ))
    if lag_samples < n:
        delayed[lag_samples:, :14] = source[:-lag_samples]
    # Mean of the ten samples ending at the lagged instant, including it.
    for k in range(lag_samples + 9, n):
        delayed[k, 14:] = np.mean(ep["wrench"][k - lag_samples - 9:k - lag_samples + 1, :2], axis=0)
    return np.column_stack((base, delayed))


def _fit_dataset(spec, episodes):
    rows = []
    targets = []
    phases = []
    gram_masks = []
    for ep in episodes:
        gate = _gate(spec, ep)
        if spec["family"] == "AR_FULL":
            x = _ar_raw_features(ep, spec["lag_ms"])
        else:
            x = _augmented_features(ep)
        rows.append(x[gate])
        targets.append(np.asarray(ep["wrench"])[gate, :spec["target_dims"]])
        phases.append(_phase_index(ep["t"])[gate])
        gram_masks.append((np.flatnonzero(gate) % FIT_STRIDE) == 0)
    return tuple(map(np.concatenate, (rows, targets, phases, gram_masks)))


def fit(spec, normal_episodes: Iterable[Mapping]):
    """Fit normal episodes; unobserved phases are recorded and fail closed."""
    spec = _resolve_spec(spec)
    episodes = list(normal_episodes)
    if not episodes:
        raise ValueError("at least one fit episode is required")
    for ep in episodes:
        _validate_episode(ep, {"fit"}, require_normal=True)
    modes = {ep["mode"] for ep in episodes}
    if len(modes) != 1:
        raise ValueError("a model may cover only one controller mode")

    model = {"format": "r5.normal-model.v1", "spec": spec, "mode": modes.pop()}
    phase_all = []
    residual_all = []
    gate_all = []

    if spec["family"] == "SCALE_ACTIVE":
        for ep in episodes:
            prediction = _r1_prediction(ep)
            residual_all.append(np.asarray(ep["wrench"])[:, :spec["target_dims"]] - prediction)
            phase_all.append(_phase_index(ep["t"]))
            gate_all.append(_gate(spec, ep))
        residual = np.concatenate(residual_all)
        phase = np.concatenate(phase_all)
        selected = np.concatenate(gate_all)
        if not np.any(selected):
            raise ValueError("fit gate selected no samples")
        counts = np.bincount(phase[selected], minlength=5)
        model["phase_fit_counts"] = [int(value) for value in counts]
        location = _phase_location(residual, phase, selected)
        model["location"] = location
        if spec["scale"] == "phase_mad":
            model["phase_scale"] = _phase_mad(residual, phase, selected, location)
        else:
            x = np.concatenate([_augmented_features(ep) for ep in episodes])
            mean, scale = _standardizer(x[selected])
            z = np.column_stack((np.ones(np.count_nonzero(selected)), _standardize(x[selected], mean, scale)))
            centred_norm = np.linalg.norm(residual[selected] - location[phase[selected]], axis=1)
            target = np.log(np.maximum(centred_norm, SCALE_FLOOR))
            subsample = (np.flatnonzero(selected) % FIT_STRIDE) == 0
            model.update(feature_mean=mean, feature_scale=scale, logscale_weights=_ridge(z[subsample], target[subsample]))
        return model

    x, target, _, gram_mask = _fit_dataset(spec, episodes)
    if not len(x):
        raise ValueError("fit gate selected no samples")
    fit_phases = np.concatenate([
        _phase_index(ep["t"])[_gate(spec, ep)] for ep in episodes
    ])
    counts = np.bincount(fit_phases, minlength=5)
    model["phase_fit_counts"] = [int(value) for value in counts]
    mean, scale = _standardizer(x)
    z = _standardize(x, mean, scale)
    if spec["family"].startswith("RFF"):
        rng = np.random.default_rng(spec["parameter_seed"])
        omega = rng.normal(0.0, 1.0 / spec["lengthscale"], size=(z.shape[1], spec["bands"]))
        bias = rng.uniform(0.0, 2.0 * np.pi, size=spec["bands"])
        phi = np.column_stack((np.ones(len(z)), z, np.cos(z @ omega + bias) * np.sqrt(2.0 / spec["bands"])))
        weights = _ridge(phi[gram_mask], target[gram_mask])
        model.update(feature_mean=mean, feature_scale=scale, omega=omega, bias=bias, weights=weights)
    else:
        phi = np.column_stack((np.ones(len(z)), z))
        weights = _ridge(phi[gram_mask], target[gram_mask])
        model.update(feature_mean=mean, feature_scale=scale, weights=weights)

    # Scale is derived on every fit sample, not only Gram/RHS subsamples.
    predictions = []
    for ep in episodes:
        predictions.append(_predict(model, ep))
        residual_all.append(np.asarray(ep["wrench"])[:, :spec["target_dims"]] - predictions[-1])
        phase_all.append(_phase_index(ep["t"]))
        gate_all.append(_gate(spec, ep))
    model["phase_scale"] = _phase_mad(
        np.concatenate(residual_all), np.concatenate(phase_all), np.concatenate(gate_all)
    )
    return model


def _predict(model, ep):
    spec = model["spec"]
    if spec["family"] == "SCALE_ACTIVE":
        return _r1_prediction(ep) + model["location"][_phase_index(ep["t"])]
    if spec["family"] == "AR_FULL":
        raw = _ar_raw_features(ep, spec["lag_ms"])
    else:
        raw = _augmented_features(ep)
    z = _standardize(raw, model["feature_mean"], model["feature_scale"])
    if spec["family"].startswith("RFF"):
        phi = np.column_stack((np.ones(len(z)), z, np.cos(z @ model["omega"] + model["bias"]) * np.sqrt(2.0 / spec["bands"])))
    else:
        phi = np.column_stack((np.ones(len(z)), z))
    return phi @ model["weights"]


def trace(model, ep):
    """Trace a frozen model, disabling its branch in unobserved fit phases."""
    _validate_episode(ep, {"fit", "calibration", "audit", "analysis_eval"})
    if ep["mode"] != model["mode"]:
        raise ValueError("episode mode differs from model mode")
    prediction = _predict(model, ep)
    error = np.asarray(ep["wrench"])[:, :model["spec"]["target_dims"]] - prediction
    phase = _phase_index(ep["t"])
    if model["spec"]["family"] == "SCALE_ACTIVE" and model["spec"]["scale"] == "state_logscale":
        raw = _augmented_features(ep)
        z = np.column_stack((np.ones(N_SAMPLES), _standardize(raw, model["feature_mean"], model["feature_scale"])))
        scale = np.clip(np.exp(z @ model["logscale_weights"]), SCALE_FLOOR, 1000.0)
    else:
        scale = model["phase_scale"][phase]
    active = _gate(model["spec"], ep)
    supported_phase = np.asarray(model["phase_fit_counts"], dtype=int) > 0
    supported_sample = supported_phase[phase]
    scale = np.asarray(scale, dtype=float).copy()
    scale[~supported_sample] = 1.0
    score = np.linalg.norm(error, axis=1) / scale
    active &= supported_sample
    ema = np.zeros(N_SAMPLES, dtype=float)
    state = 0.0
    for index in range(N_SAMPLES):
        if active[index] and np.isfinite(score[index]):
            state += EMA_ALPHA * (score[index] - state)
        else:
            state = 0.0
        ema[index] = state
    return {
        "prediction": prediction.copy(), "error_norm": np.linalg.norm(error, axis=1),
        "scale": np.asarray(scale).copy(), "score": score.copy(), "ema": ema,
        "active": active.copy(), "r1_detect": np.asarray(ep["r1_detect"]).copy(),
    }


def _four_run_peak(ema, active):
    peak = 0.0
    for index in range(ON_COUNT - 1, len(ema)):
        window = slice(index - ON_COUNT + 1, index + 1)
        if np.all(active[window]):
            peak = max(peak, float(np.min(ema[window])))
    return peak


def calibrate(model, normal_cal_episodes):
    """Calibrate threshold only, with 5% headroom over every normal 4-run."""
    episodes = list(normal_cal_episodes)
    if not episodes:
        raise ValueError("at least one calibration episode is required")
    peak = 0.0
    for ep in episodes:
        _validate_episode(ep, {"calibration"}, require_normal=True)
        if ep["mode"] != model["mode"]:
            raise ValueError("episode mode differs from model mode")
        result = trace(model, ep)
        peak = max(peak, _four_run_peak(result["ema"], result["active"]))
    return 1.05 * peak + 1e-12


def alarms(result, threshold):
    """Return the required R1-OR-model stateful detection trace."""
    active = np.asarray(result["active"], dtype=bool)
    over = active & (np.asarray(result["ema"]) > float(threshold))
    own = np.zeros(len(over), dtype=bool)
    count = clear = 0
    detected = False
    for index, value in enumerate(over):
        if not active[index]:
            count = clear = 0
            detected = False
            own[index] = False
            continue
        count = count + 1 if value else 0
        clear = 0 if value else clear + 1
        if count >= ON_COUNT:
            detected = True
        elif clear >= CLEAR_COUNT:
            detected = False
        own[index] = detected
    return own | np.asarray(result["r1_detect"], dtype=bool)


def statistics(model, episodes):
    """Summarize model-only normal residuals; does not fit or calibrate."""
    values = []
    for ep in episodes:
        result = trace(model, ep)
        selected = result["active"]
        values.append(result["error_norm"][selected])
    joined = np.concatenate(values) if values else np.empty(0)
    return {
        "samples": int(len(joined)),
        "rmse": float(np.sqrt(np.mean(joined ** 2))) if len(joined) else float("nan"),
        "max": float(np.max(joined)) if len(joined) else float("nan"),
    }


def serialized(model):
    """Convert a fitted model to a JSON-safe, deterministic nested object."""
    if isinstance(model, np.ndarray):
        return model.tolist()
    if isinstance(model, np.generic):
        return model.item()
    if isinstance(model, dict):
        return {key: serialized(value) for key, value in model.items()}
    if isinstance(model, (list, tuple)):
        return [serialized(value) for value in model]
    return deepcopy(model)


SIGNAL_KEYS = {
    "t", "th", "thd", "tau_m", "tau_s", "grip", "r_joint", "wrench",
    "active", "r1_detect", "ref_q", "ref_qd", "ref_qdd",
}
PROTOCOL_PATH = ROOT / "results/v2_protocol/PROTOCOL_R5.json"
DESIGN_PATH = ROOT / "results/v2_protocol/R5_S2_DESIGN.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_digest(rows):
    joined = "\n".join(row["signal_sha256"] for row in rows).encode("ascii")
    return hashlib.sha256(joined).hexdigest()


def _prepare_output(dataset_root, output):
    dataset_root = Path(dataset_root).resolve()
    output = Path(output).resolve()
    if "logs" in {part.lower() for part in output.parts}:
        raise ValueError("output below a logs directory is not allowed")
    registered = (ROOT / "models/r5").resolve()
    if output == registered or output.is_relative_to(registered):
        raise ValueError("registered models are read-only")
    if output == dataset_root or output.is_relative_to(dataset_root):
        raise ValueError("output must be outside the input dataset")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("output must be a new or empty directory")
    else:
        output.mkdir(parents=True)
    return dataset_root, output


def _public_contract():
    protocol = _load_json(PROTOCOL_PATH)
    design = _load_json(DESIGN_PATH)
    plan_sha = protocol["S2_analysis"]["plan_sha256"]
    design_sha = protocol["S2_analysis"]["design_sha256"]
    if design.get("source_design_sha256") != design_sha:
        raise ValueError("public normal-data design does not match protocol")
    if design.get("not_validation_development_or_holdout") is not True:
        raise ValueError("public normal-data design lacks split isolation")
    return protocol, design, plan_sha, design_sha


def _load_index(dataset_root, role, mode, design, plan_sha, design_sha):
    folder = (dataset_root / role / mode).resolve()
    expected_folder = (dataset_root / role / mode).resolve()
    if folder != expected_folder:
        raise ValueError("normal-data folder mismatch")
    index_path = folder / "index.json"
    document = _load_json(index_path)
    if (document.get("schema") != "R5_S2_DATA_INDEX.v1"
            or document.get("role") != role or document.get("mode") != mode
            or document.get("plan_sha256") != plan_sha
            or document.get("design_sha256") != design_sha):
        raise ValueError("normal-data index provenance mismatch")
    rows = document.get("rows")
    expected_cases = design["roles"][role]
    if not isinstance(rows, list) or [row.get("case") for row in rows] != expected_cases:
        raise ValueError("normal-data cases differ from the public design")
    episodes = [_load_episode(row, role, mode, folder) for row in rows]
    return {
        "path": index_path, "sha256": _sha256(index_path), "rows": rows,
        "source_sha256": _source_digest(rows), "episodes": episodes,
    }


def _load_episode(row, role, mode, folder):
    case = row.get("case")
    if (row.get("role") != role or row.get("mode") != mode
            or row.get("normal_only") is not True or row.get("official") is not False
            or not isinstance(case, dict) or case.get("kind") != "reference"
            or case.get("collision") is not None or case.get("pick_offset") is not None):
        raise ValueError("fit and calibration require declared normal reference rows")
    status = row.get("status")
    if (not isinstance(status, dict) or status.get("diverged") is not False
            or status.get("samples") != N_SAMPLES):
        raise ValueError("incomplete or divergent normal row")
    signal_name = row.get("signal_file")
    if not isinstance(signal_name, str):
        raise ValueError("normal row has no signal file")
    signal_path = (folder / signal_name).resolve()
    if not signal_path.is_relative_to(folder) or not signal_path.is_file():
        raise ValueError("normal signal path escapes its role folder")
    if _sha256(signal_path) != row.get("signal_sha256"):
        raise ValueError("normal signal hash mismatch")
    with np.load(signal_path, allow_pickle=False) as raw:
        if set(raw.files) != SIGNAL_KEYS:
            raise ValueError("normal signal fields differ from the allowlist")
        episode = {name: raw[name] for name in raw.files}
    episode.update(
        mass=case["mass"], com=np.asarray(case["com"], dtype=float), mode=mode,
        role=role, normal_only=True,
    )
    _validate_episode(episode, {role}, require_normal=True)
    return episode


def _coverage(episode, spec):
    if spec["gate"] != "t>=0.06":
        return None
    phase = _phase_index(episode["t"])
    counts = [int(np.sum((phase == index) & (episode["t"] >= 0.06)))
              for index in range(5)]
    if min(counts) < 200:
        raise ValueError("full-gate normal data have insufficient phase coverage")
    return counts


def _write_json(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError("refuse to overwrite output")
    path.write_text(json.dumps(serialized(value), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _own_detection(result, threshold):
    own_input = dict(result)
    own_input["r1_detect"] = np.zeros_like(result["r1_detect"], dtype=bool)
    return alarms(own_input, threshold)


def reproduce(dataset_root, output):
    """Fit and calibrate all two-mode, eight-spec normal-only models."""
    dataset_root, output = _prepare_output(dataset_root, output)
    _, design, plan_sha, design_sha = _public_contract()
    sources = {}
    for role in ("fit", "calibration"):
        for mode in ("servo", "torque"):
            sources[(role, mode)] = _load_index(
                dataset_root, role, mode, design, plan_sha, design_sha)

    outputs = []
    for mode in ("servo", "torque"):
        fit_source = sources[("fit", mode)]
        calibration_source = sources[("calibration", mode)]
        fit_episodes = fit_source["episodes"]
        calibration_episodes = calibration_source["episodes"]
        for spec in specifications():
            fit_coverage = [_coverage(episode, spec) for episode in fit_episodes]
            calibration_coverage = [_coverage(episode, spec)
                                    for episode in calibration_episodes]
            model = fit(spec, fit_episodes)
            threshold = calibrate(model, calibration_episodes)
            if any(np.any(_own_detection(trace(model, episode), threshold))
                   for episode in calibration_episodes):
                raise RuntimeError("calibration produced a normal alarm")
            value = {
                "spec": spec, "mode": mode, "threshold": threshold,
                "model": serialized(model),
                "fit_statistics": statistics(model, fit_episodes),
                "calibration_statistics": statistics(model, calibration_episodes),
                "fit_rows": len(fit_episodes),
                "calibration_rows": len(calibration_episodes),
                "fit_index_sha256": fit_source["sha256"],
                "calibration_index_sha256": calibration_source["sha256"],
                "source_signal_sha256": {
                    "fit": fit_source["source_sha256"],
                    "calibration": calibration_source["source_sha256"],
                },
                "learning_code_sha256": _sha256(__file__),
                "full_gate_phase_coverage": {
                    "fit": fit_coverage, "calibration": calibration_coverage,
                },
                "float_parameters": int(sum(
                    value.size for value in model.values()
                    if isinstance(value, np.ndarray))),
                "normal_only": True,
                "audit_or_evaluation_files_read": 0,
            }
            filename = mode + "_" + spec["id"] + ".json"
            target = output / filename
            _write_json(target, value)
            outputs.append({
                "mode": mode, "spec_id": spec["id"], "file": filename,
                "sha256": _sha256(target), "threshold": threshold,
            })

    manifest = {
        "schema": "R5_NORMAL_LEARNING.v1",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "public_design_sha256": _sha256(DESIGN_PATH),
        "source_design_sha256": design_sha,
        "plan_sha256": plan_sha,
        "learning_code_sha256": _sha256(__file__),
        "normal_only": True,
        "fit_and_calibration_only": True,
        "audit_or_evaluation_files_read": 0,
        "sources": {
            mode: {
                role: {
                    "index_sha256": sources[(role, mode)]["sha256"],
                    "source_signal_sha256": sources[(role, mode)]["source_sha256"],
                    "rows": len(sources[(role, mode)]["rows"]),
                }
                for role in ("fit", "calibration")
            }
            for mode in ("servo", "torque")
        },
        "outputs": outputs,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = reproduce(args.dataset, args.output)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
