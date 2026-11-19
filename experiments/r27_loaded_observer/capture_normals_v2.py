"""Recapture normal splits with observer-effective physical eligibility telemetry."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from experiments.r15_delayed_wrench.observer import LAG, NormalForce
from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as R26
from sim import config as C


PROTOCOL_SHA256 = "d4d010e5c0bd58a1fdbd5d2a28c8d988629afc31cffb7d13e25e48ebd4ed419b"
PLAN_SHA256 = "9a00dcf6584f51c47a7f51c1c2ac200b0b76b4ad54edd3446149b6a366672fea"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Probe(R26):
    def reset(self):
        super().reset()
        self.r27_observer = NormalForce(
            self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D,
            jts_hz=C.JTS_CUTOFF)
        self.r27_eligibility = deque(maxlen=LAG + 1)
        self.r27_rows = []
        self.r27_grip_run = 0

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        self.r27_grip_run = self.r27_grip_run + 1 if grip else 0
        self.r27_eligibility.append((
            int(self.r27_grip_run),
            bool(getattr(self._shared, "last_table_allowed", False))))
        sample = self.r27_observer.update(t, th, tau_s, grip)
        if sample is not None and len(self.r27_eligibility) == self.r27_eligibility.maxlen:
            effective_grip_run, effective_table_allowed = self.r27_eligibility[0]
            self.r27_rows.append([
                int(round(1000.0 * float(t))),
                int(round(1000.0 * float(sample.effective_time))),
                int(effective_grip_run),
                int(bool(report)),
                int(effective_table_allowed),
                *np.asarray(sample.force_n, float).tolist(),
            ])
        return residual, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--split", choices=("fit", "audit"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output directory required")
    if mujoco.__version__ != "3.17.0":
        raise RuntimeError("MuJoCo 3.17.0 required")
    if sha(args.plan) != PLAN_SHA256 or sha(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("locked v2 normal plan or protocol differs")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    baseline = json.loads(args.baseline_manifest.read_bytes())
    baseline_rows = {row["key"]: row for row in baseline["records"]}
    args.out.mkdir(parents=True)

    plan = json.loads(args.plan.read_bytes())
    cases = plan[args.split + "_cases"]
    variants = plan[args.split + "_variants"]
    arrays, records = {}, []
    for case in cases:
        for variant_name, variant in variants.items():
            holder = []

            def factory():
                item = Probe()
                holder.append(item)
                return item

            _, log = lab.run_episode(case, variant, factory, (None, None))
            key = case["id"] + "__" + variant_name
            rows = np.asarray(holder[0].r27_rows, float)
            if rows.ndim != 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
                raise ValueError("invalid v2 public observer capture")
            signature = lab.certified.trace_signature({"none": log})
            detect = np.asarray(log["detect"], bool)
            rises = np.flatnonzero(detect & ~np.r_[False, detect[:-1]]).astype(int).tolist()
            base = baseline_rows[key]
            if signature != base["physical_signature"] or rises != base["native_rise_indices"]:
                raise ValueError("passive v2 capture changed frozen baseline")
            arrays[key] = rows
            records.append({
                "key": key,
                "observer_samples": int(len(rows)),
                "native_rise_indices": rises,
                "physical_signature": signature,
                "v1_baseline_exact": True,
            })
            print(f"{args.split} v2 {len(records)}/{len(cases) * len(variants)}", flush=True)

    raw_path = args.out / "normal_observer_v2.npz"
    with raw_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = {
        "schema": "r27-loaded-observer-normal-capture.v2",
        "split": args.split,
        "episodes": len(records),
        "runtime": {"mujoco": mujoco.__version__, "cpu_workers": 1},
        "protocol_sha256": sha(args.protocol),
        "plan_sha256": sha(args.plan),
        "capture_script_sha256": sha(__file__),
        "baseline_manifest_sha256": sha(args.baseline_manifest),
        "npz_sha256": sha(raw_path),
        "records": records,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
