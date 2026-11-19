"""Fit and audit the preregistered vertical innovation bound on normal traces only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r21_full_protection import runner_v2 as raw
from experiments.r22_census import check
from experiments.r31_held_identity.candidate import Detector as Parent


class Probe(Parent):
    instance = None

    def reset(self):
        super().reset()
        self.samples = []
        Probe.instance = self

    def update(self, *args):
        result = super().update(*args)
        projection = self.r27_projection.last_force
        shared = self._shared.last_residual_wrench
        self.samples.append({
            "projection_z": None if projection is None else float(projection[2]),
            "shared_z": None if shared is None else float(shared[2]),
            "eligible": bool(args[-1] and not self._shared.last_table_allowed),
        })
        return result


def maximum(samples, width):
    values = []
    best = 0.0
    for sample in samples:
        p = sample["projection_z"]
        w = sample["shared_z"]
        if not sample["eligible"] or p is None or w is None:
            values.clear()
            continue
        values.append(np.array([p, w], dtype=float))
        if len(values) < 2 * width:
            continue
        before = np.mean(values[-2 * width:-width], axis=0)
        now = np.mean(values[-width:], axis=0)
        delta = now - before
        if delta[0] * delta[1] > 0:
            best = max(best, float(np.min(np.abs(delta))))
    return best


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    cases = check.build_cases("my_torque")
    references = {case["case_id"]: case for case in cases if case["kind"] == "reference"}
    rows = []
    for cohort in ("fit", "audit"):
        for case_id in plan[f"{cohort}_reference_ids"]:
            raw.execute(Probe, references[case_id])
            rows.append({"cohort": cohort, "case_id": case_id,
                         "maximum_N": maximum(Probe.instance.samples, plan["window_samples"])})
    fit_max = max(row["maximum_N"] for row in rows if row["cohort"] == "fit")
    threshold = 1.25 * fit_max + 0.10
    audit_max = max(row["maximum_N"] for row in rows if row["cohort"] == "audit")
    result = {
        "schema": "r33-carry-vertical-calibration.v1",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "fit_maximum_N": fit_max,
        "threshold_N": threshold,
        "audit_maximum_N": audit_max,
        "audit_pass": audit_max <= threshold,
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
