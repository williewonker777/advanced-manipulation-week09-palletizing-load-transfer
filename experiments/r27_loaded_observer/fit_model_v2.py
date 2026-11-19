"""Fit and independently audit the locked settled-unsupported v2 proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r27_loaded_observer.features_v2 import normal_scores
from experiments.r27_loaded_observer.witness import LoadedForceOnset


PROTOCOL_SHA256 = "d4d010e5c0bd58a1fdbd5d2a28c8d988629afc31cffb7d13e25e48ebd4ed419b"
PLAN_SHA256 = "9a00dcf6584f51c47a7f51c1c2ac200b0b76b4ad54edd3446149b6a366672fea"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(directory):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    raw = directory / "normal_observer_v2.npz"
    if sha(raw) != manifest["npz_sha256"]:
        raise ValueError("v2 normal capture bytes differ")
    return manifest, np.load(raw, allow_pickle=False)


def replay(rows, model):
    state = LoadedForceOnset(model)
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
        parser.error("fresh v2 model and report required")
    if sha(args.plan) != PLAN_SHA256 or sha(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("locked v2 plan or protocol differs")
    fit_manifest, fit = load_split(args.fit)
    audit_manifest, audit = load_split(args.audit)

    onset_values, sustain_values = [], []
    for key in fit.files:
        onset, sustain = normal_scores(fit[key])
        onset_values.extend(onset.tolist())
        sustain_values.extend(sustain.tolist())
    fit_onset_max = float(max(onset_values))
    fit_sustain_max = float(max(sustain_values))
    sustain_threshold = 1.05 * fit_sustain_max + 0.05
    model = {
        "schema": "r27-loaded-observer-model.v1",
        "proposal": "settled_unsupported_v2",
        "fit_rule": "threshold = 1.05 * eligible fit maximum + 0.05 N",
        "grip_stability_samples": 96,
        "onset_lag_samples": 8,
        "sustain_lag_samples": 32,
        "onset_persistence_samples": 3,
        "pending_samples": 25,
        "pulse_samples": 8,
        "rearm_quiet_samples": 25,
        "onset_threshold_N": 1.05 * fit_onset_max + 0.05,
        "sustain_threshold_N": sustain_threshold,
        "ownership_release_N": 0.5 * sustain_threshold,
        "fit_onset_max_N": fit_onset_max,
        "fit_sustain_max_N": fit_sustain_max,
        "protocol_sha256": sha(args.protocol),
        "plan_sha256": sha(args.plan),
        "fit_npz_sha256": fit_manifest["npz_sha256"],
    }
    records, passed = [], True
    for split_name, arrays in (("fit", fit), ("audit", audit)):
        for key in arrays.files:
            onset, sustain = normal_scores(arrays[key])
            rises, active, count = replay(arrays[key], model)
            row = {
                "split": split_name,
                "key": key,
                "eligible_samples": int(len(onset)),
                "onset_max_N": float(onset.max()),
                "sustain_max_N": float(sustain.max()),
                "candidate_rise_rows": rises,
                "candidate_active_samples": active,
                "candidate_onset_count": count,
            }
            records.append(row)
            passed = passed and not rises and active == 0 and count == 0
    report = {
        "schema": "r27-loaded-observer-normal-fit-audit.v2",
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
        raise ValueError("locked v2 normal falsifier failed; no model emitted")
    args.model.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
