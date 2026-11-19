"""Post-failure score attribution for the two disclosed stock targets only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.r26_holdout import evaluator as lab
from experiments.r27_loaded_observer.candidate import Detector, MODEL
from experiments.r27_loaded_observer.lateral_witness import lateral_residual


CANDIDATE_SHA256 = "6c24a8d1944e3c5c23fd2c6a24f1b2afce1054c4817ec443cf5513bb21f87d4b"
PROTOCOL_SHA256 = "d489e5f9985d84e681310132692eec2a51f9ccc0c91a2652247f5c8f9025c10a"
CASES = ("s26_long_carry_signed", "s26_long_place_lateral")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Probe(Detector):
    def reset(self):
        super().reset()
        self.post_rows = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        if self.loaded_force_sample is not None:
            self.post_rows.append([
                int(round(1000.0 * float(t))),
                int(self.loaded_force_eligible),
                int(report),
                *np.asarray(self.loaded_force_sample, float).tolist(),
            ])
        return residual, report


def run(case, impact):
    holder = []

    def factory():
        item = Probe()
        holder.append(item)
        return item

    _, log = lab.run_episode(case, {"kind": "stock"}, factory, (impact, None))
    return log, np.asarray(holder[0].post_rows, float)


def score(rows, lag, coefficients, first, last):
    values = []
    for index in range(lag, len(rows)):
        tick = int(rows[index, 0])
        if not first <= tick < last or not bool(rows[index, 1]):
            continue
        if int(rows[index, 0] - rows[index - lag, 0]) != lag:
            continue
        vector = lateral_residual(rows[index, 3:6], rows[index - lag, 3:6], coefficients)
        values.append((tick, float(np.linalg.norm(vector)), vector.tolist()))
    peak = max(values, key=lambda item: item[1])
    above = [item[1] > (MODEL["onset_threshold_N"] if lag == 8
                        else MODEL["sustain_threshold_N"]) for item in values]
    longest = current = 0
    for item in above:
        current = current + 1 if item else 0
        longest = max(longest, current)
    return {"peak_tick": peak[0], "peak_N": peak[1], "vector_at_peak_N": peak[2],
            "above_samples": int(sum(above)), "longest_above_run": int(longest)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    if mujoco.__version__ != "3.17.0":
        raise RuntimeError("MuJoCo 3.17.0 required")
    if (sha(args.protocol) != PROTOCOL_SHA256
            or sha(Path(__file__).with_name("candidate.py")) != CANDIDATE_SHA256):
        raise ValueError("frozen candidate or protocol differs")
    plan = json.loads(lab.PLAN.read_bytes())
    records = []
    for case_id in CASES:
        case = next(item for item in plan["single"] if item["id"] == case_id)
        none, none_rows = run(case, None)
        forced, forced_rows = run(case, case["impact"])
        start, boundary = lab.actual_start(forced["t"], case["impact"])
        end = min(6000, start + plan["event_window_ms"])
        records.append({
            "id": case_id,
            "boundary": boundary,
            "prefix": lab.certified.prefix(none, forced, start, start),
            "force_exact": lab.exact_force(forced, (case["impact"], None)),
            "onset_threshold_N": MODEL["onset_threshold_N"],
            "sustain_threshold_N": MODEL["sustain_threshold_N"],
            "none_onset": score(none_rows, 8, MODEL["onset_xy_from_z"], start, end),
            "forced_onset": score(forced_rows, 8, MODEL["onset_xy_from_z"], start, end),
            "none_32ms": score(none_rows, 32, MODEL["sustain_xy_from_z"], start, end),
            "forced_32ms": score(forced_rows, 32, MODEL["sustain_xy_from_z"], start, end),
        })
    result = {
        "schema": "r27-loaded-observer-post-failure-attribution.v1",
        "classification": "Post-selection diagnosis only; candidate and model unchanged.",
        "runtime": {"mujoco": mujoco.__version__, "cpu_workers": 1},
        "episodes": 4,
        "candidate_sha256": CANDIDATE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "records": records,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
