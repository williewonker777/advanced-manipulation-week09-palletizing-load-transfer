"""Staged four-twin assessment for the frozen support-aware servo witness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import my_detector
from experiments.r18_paired_impulses.runner import run_episode
from experiments.r18_paired_impulses.scoring import assess
from experiments.r21_support_yaw.candidate import Detector as Candidate
from my_servo import Controller
from tools.r14_paired_events import match_false_events

ROOT = Path(__file__).parent
ARMS = {"r10": my_detector.Detector, "candidate": Candidate}
TWINS = {"none": (None, None), "first": (0, None), "second": (None, 1), "both": (0, 1)}
PHYSICAL = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos",
            "f_ext", "f_ext_xyz_committed", "f_ext_xyz_native", "f_ext_xyz_second")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(plan):
    if plan.get("status") != "FROZEN_BEFORE_ANY_FORCE_OUTCOME":
        raise ValueError("unfrozen plan")
    for name, expected in plan["pins"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"pin mismatch: {name}")


def first_delay(result, key):
    values = result["absolute_delays_ms"][key]
    return None if not values else min(values)


def run_arm(case, detector):
    impacts = case["impacts"]
    traces = {}
    for name, selection in TWINS.items():
        pair = tuple(None if i is None else impacts[i] for i in selection)
        metrics, trace = run_episode(Controller(), case["mass"], tuple(case["com"]),
                                     detector=detector(), impacts=pair)
        trace["r18_diverged"] = bool(metrics["diverged"])
        traces[name] = trace
    return traces


def evaluate_case(case, plan):
    traces = {name: run_arm(case, cls) for name, cls in ARMS.items()}
    scores = {name: assess(case, arm, plan["quiet_ms"], plan["event_window_ms"])
              for name, arm in traces.items()}
    parity = {twin: {key: bool(np.array_equal(traces["r10"][twin][key],
                                               traces["candidate"][twin][key]))
                     for key in PHYSICAL} for twin in TWINS}
    retained = {}
    for event in ("isolated_first", "isolated_second", "both_first", "both_second"):
        base = first_delay(scores["r10"], event)
        candidate = first_delay(scores["candidate"], event)
        retained[event] = {"r10_ms": base, "candidate_ms": candidate,
                           "retained_5ms": base is None or candidate is not None and candidate <= base + 5.0}
    false_match = match_false_events(scores["candidate"]["false_times_s"]["both"],
                                     scores["r10"]["false_times_s"]["both"])
    second = (first_delay(scores["candidate"], "isolated_second"),
              first_delay(scores["candidate"], "both_second"))
    second_onset = float(case["impacts"][1][0])
    first_only_near_second = [x for x in scores["candidate"]["rises_s"]["first"]
                              if second_onset - 0.02 <= x < second_onset + plan["event_window_ms"] / 1000.0]
    gates = {
        "candidate_pair_assessment": scores["candidate"]["passed"],
        "exact_physics_all_twins": all(all(fields.values()) for fields in parity.values()),
        "actual_second_isolated_and_both": all(value is not None for value in second),
        "no_first_only_delayed_rise_near_second": not first_only_near_second,
        "retain_all_r10_events_5ms": all(x["retained_5ms"] for x in retained.values()),
        "no_added_both_false": false_match["passed"],
    }
    return {"id": case["id"], "scores": scores, "physical_equal": parity,
            "retention": retained, "first_only_near_second_s": first_only_near_second,
            "false_match": false_match, "gates": gates, "passed": all(gates.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("protected", "reserved"), required=True)
    parser.add_argument("--protected-result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("fresh output required")
    raw = (ROOT / "plan.json").read_bytes()
    plan = json.loads(raw)
    verify(plan)
    if args.stage == "reserved":
        if args.protected_result is None:
            parser.error("reserved requires protected result")
        protected = json.loads(args.protected_result.read_bytes())
        if protected.get("plan_sha256") != hashlib.sha256(raw).hexdigest() or not protected.get("passed"):
            parser.error("same-plan passing protected result required")
    cases = plan[args.stage + "_cases"]
    rows = [evaluate_case(case, plan) for case in cases]
    result = {"schema": "r21-support-yaw-paired-result.v1", "stage": args.stage,
              "plan_sha256": hashlib.sha256(raw).hexdigest(), "episodes": len(rows) * 8,
              "rows": rows, "passed": all(row["passed"] for row in rows)}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"stage": args.stage, "episodes": result["episodes"],
                      "passed": result["passed"],
                      "failed": [row["id"] for row in rows if not row["passed"]]}))


if __name__ == "__main__":
    main()
