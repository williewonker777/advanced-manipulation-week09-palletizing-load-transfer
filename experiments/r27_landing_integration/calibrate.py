"""Fit the preregistered grip-force bound on normal G4 episodes only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r26_delivery_pair import probe
from experiments.r26_delivery_pair.candidate import Detector as C1
from sim import config as C


HERE = Path(__file__).resolve().parent


class Meter(C1):
    """Read-only instrumentation; detector output remains exactly C1."""

    def reset(self):
        super().reset()
        self.grip_force = NormalForce(
            self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D,
            jts_hz=C.JTS_CUTOFF)
        self.force_trace = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.grip_force.update(t, th, tau_s, bool(grip))
        value = None if sample is None or not grip else sample.force_n.copy()
        self.force_trace.append(value)
        return residual, report


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")

    plan = json.loads((HERE / "plan.json").read_bytes())
    cases, split = probe.cases()
    if split["fit"] != plan["normal_split"]["fit"]:
        raise ValueError("fit-only normal split differs from preregistration")
    rows = []
    maximum = 0.0
    for key in split["fit"]:
        detector, _, metrics, log = probe.run(cases[key], Meter)
        force = np.asarray([
            [np.nan, np.nan, np.nan] if value is None else value
            for value in detector.force_trace], dtype=float)
        valid = np.isfinite(force).all(axis=1)
        if not valid.any() or metrics["diverged"]:
            raise ValueError("invalid normal fit episode: " + key)
        horizontal = np.linalg.norm(force[valid, :2], axis=1)
        local = float(horizontal.max())
        maximum = max(maximum, local)
        rows.append({
            "id": key,
            "valid_samples": int(valid.sum()),
            "max_horizontal_N": local,
            "max_xyz_N": force[valid].max(axis=0).tolist(),
            "min_xyz_N": force[valid].min(axis=0).tolist(),
            "baseline_rises_s": probe.replay.rise_times(log),
            "physical_sha256": probe.replay.physical_signature(log),
        })
    model = {
        "schema": "r27-gripped-world-force-bound.v1",
        "fit_episode_ids": split["fit"],
        "fit_episode_count": len(rows),
        "fit_max_horizontal_N": maximum,
        "horizontal_bound_N": 1.20 * maximum + 0.50,
        "release_fraction": 0.80,
        "onset_consecutive_samples": 2,
        "fit_rows_sha256": hashlib.sha256(json.dumps(
            rows, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest(),
        "plan_sha256": sha(HERE / "plan.json"),
    }
    args.out.mkdir()
    (args.out / "fit.json").write_text(json.dumps({
        "schema": "r27-gripped-force-normal-fit.v1", "rows": rows,
        "episodes": len(rows), "model": model,
    }, indent=2, allow_nan=False) + "\n")
    (args.out / "model.json").write_text(
        json.dumps(model, indent=2, allow_nan=False) + "\n")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
