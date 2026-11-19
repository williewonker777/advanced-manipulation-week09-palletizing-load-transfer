"""Fit signed normal vertical-to-horizontal dynamics and audit without adjustment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r27_loaded_observer.lateral_witness import LoadedLateralOnset


PROTOCOL_SHA256 = "efcf058b61a1eab589a99e4039f1335db37598920a07b43a555b8b3b21f8d7af"
PLAN_SHA256 = "9a00dcf6584f51c47a7f51c1c2ac200b0b76b4ad54edd3446149b6a366672fea"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(directory):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    raw = directory / "normal_observer_v2.npz"
    if sha(raw) != manifest["npz_sha256"]:
        raise ValueError("normal capture bytes differ")
    return manifest, np.load(raw, allow_pickle=False)


def eligible_deltas(rows, lag, grip_stability=96):
    data = np.asarray(rows, float)
    tick = data[:, 0].astype(int)
    force = data[:, 5:8]
    values = []
    for index in range(lag, len(data)):
        window = slice(index - lag, index + 1)
        if (tick[index] - tick[index - lag] == lag
                and int(data[index, 2]) >= grip_stability
                and not bool(np.any(data[window, 4]))):
            values.append(force[index] - force[index - lag])
    return np.asarray(values, float)


def fit_coupling(arrays, lag):
    delta = np.concatenate([eligible_deltas(arrays[key], lag) for key in arrays.files])
    design = np.c_[np.ones(len(delta)), delta[:, 2]]
    coefficients, _, rank, _ = np.linalg.lstsq(design, delta[:, :2], rcond=None)
    if rank != 2 or not np.isfinite(coefficients).all():
        raise ValueError("normal coupling design is not finite rank two")
    residual = delta[:, :2] - design @ coefficients
    return coefficients, residual


def scores(rows, lag, coefficients):
    delta = eligible_deltas(rows, lag)
    design = np.c_[np.ones(len(delta)), delta[:, 2]]
    residual = delta[:, :2] - design @ np.asarray(coefficients)
    return np.linalg.norm(residual, axis=1)


def replay(rows, model):
    state = LoadedLateralOnset(model)
    bits = []
    for row in np.asarray(rows, float):
        if bool(row[4]):
            state.reset()
            bits.append(False)
        else:
            bits.append(state.feed(int(row[0]), row[5:8], int(row[2]), bool(row[3])))
    bits = np.asarray(bits, bool)
    rises = np.flatnonzero(bits & ~np.r_[False, bits[:-1]])
    return rises.astype(int).tolist(), int(bits.sum()), int(state.onset_count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(path.exists() or path.is_symlink() for path in (args.model, args.report)):
        parser.error("fresh v3 model and report required")
    if sha(args.plan) != PLAN_SHA256 or sha(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("locked v3 plan or protocol differs")
    fit_manifest, fit = load_split(args.fit)
    audit_manifest, audit = load_split(args.audit)
    onset_coefficients, onset_residual = fit_coupling(fit, 8)
    sustain_coefficients, sustain_residual = fit_coupling(fit, 32)
    onset_max = float(np.linalg.norm(onset_residual, axis=1).max())
    sustain_max = float(np.linalg.norm(sustain_residual, axis=1).max())
    sustain_threshold = 1.05 * sustain_max + 0.05
    model = {
        "schema": "r27-loaded-lateral-model.v1",
        "proposal": "normal_vertical_coupling_v3",
        "fit_rule": "OLS xy change from [1,z change]; threshold 1.05 * fit max + 0.05 N",
        "grip_stability_samples": 96,
        "onset_lag_samples": 8,
        "sustain_lag_samples": 32,
        "onset_persistence_samples": 3,
        "pending_samples": 25,
        "pulse_samples": 8,
        "rearm_quiet_samples": 25,
        "onset_xy_from_z": onset_coefficients.tolist(),
        "sustain_xy_from_z": sustain_coefficients.tolist(),
        "onset_fit_residual_max_N": onset_max,
        "sustain_fit_residual_max_N": sustain_max,
        "onset_threshold_N": 1.05 * onset_max + 0.05,
        "sustain_threshold_N": sustain_threshold,
        "ownership_release_N": 0.5 * sustain_threshold,
        "protocol_sha256": sha(args.protocol),
        "plan_sha256": sha(args.plan),
        "fit_npz_sha256": fit_manifest["npz_sha256"],
    }
    records, passed = [], True
    for split_name, arrays in (("fit", fit), ("audit", audit)):
        for key in arrays.files:
            onset = scores(arrays[key], 8, onset_coefficients)
            sustain = scores(arrays[key], 32, sustain_coefficients)
            rises, active, count = replay(arrays[key], model)
            row = {
                "split": split_name,
                "key": key,
                "eligible_samples": int(len(onset)),
                "onset_residual_max_N": float(onset.max()),
                "sustain_residual_max_N": float(sustain.max()),
                "candidate_rise_rows": rises,
                "candidate_active_samples": active,
                "candidate_onset_count": count,
            }
            records.append(row)
            passed = passed and not rises and active == 0 and count == 0
    report = {
        "schema": "r27-loaded-observer-normal-fit-audit.v3",
        "passed": bool(passed),
        "episodes": len(records),
        "fit_episodes": fit_manifest["episodes"],
        "audit_episodes": audit_manifest["episodes"],
        "model": model,
        "audit_npz_sha256": audit_manifest["npz_sha256"],
        "records": records,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise ValueError("locked v3 normal falsifier failed; no model emitted")
    args.model.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
