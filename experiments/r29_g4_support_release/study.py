"""Execute the frozen disclosed R29 unloading-source study without retuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.r26_delivery_pair.controller import Controller
from experiments.r29_g4_support_release.candidate import Detector as Candidate
from experiments.r29_g4_support_release.diagnose import C1Trace, MemoryTrace, RouteTrace
from experiments.r28_landing_replay import evaluator as replay


HERE = Path(__file__).resolve().parent
CANDIDATE_SHA256 = "edc89c20eeeb469bd16096ed978bbde0d17dd4a35ddd666ba0c3c298d8b7c6c9"
PLAN_SHA256 = "aea18818ca3a4d295bc9ca5700c6e5276deb3371296fce9c0f6d1699dd4d3335"
CANDIDATE_PLAN_SHA256 = "7b74c2b23affd87a4dce9b4dedb9d54303b9da9c0598a7f3c5b142fc185a1152"


class CandidateTrace(RouteTrace, Candidate):
    pass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(case, detector, impact):
    log = replay.run_episode(case, detector, (impact, None), Controller)
    return log, list(detector.route_trace)


def compact(log, impact, starts, scoring):
    return replay.trace_record(log, (impact, None), starts, scoring)


def positive_record(forced, reference, impact, scoring, trace):
    start = replay.actual_start(forced["t"], impact)
    event = replay.event_record(forced, reference, start, scoring)
    record = compact(forced, impact, [start], scoring)
    prefix = replay.prefix_exact(reference, forced, start)
    return {
        "actual_input_tick": start,
        "event": event,
        "trace": record,
        "prefix": prefix,
        "duration_delta_samples_vs_none": record["active_samples"] - int(
            np.count_nonzero(reference["detect"])),
        "online_at_input": trace[start],
        "online_at_rises": {str(edge): trace[edge] for edge in replay.edge_indices(forced)},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh --out required")
    if (digest(HERE / "candidate.py") != CANDIDATE_SHA256
            or digest(HERE / "plan.json") != PLAN_SHA256
            or digest(HERE / "candidate_plan.json") != CANDIDATE_PLAN_SHA256):
        parser.error("frozen R29 source or plan changed")
    protocol = json.loads((HERE / "plan.json").read_bytes())
    fixed = protocol["fixed_controls"]
    authored = json.loads(replay.PLAN.read_bytes())
    scoring = authored["scoring"]
    tool, quiet = authored["known_tool_pair"]
    mass0 = dict(authored["positive_controls"][0]); mass0.pop("impact")
    mass8 = dict(authored["positive_controls"][1]); mass8.pop("impact")
    bases = {row["id"]: row for row in (tool, quiet, mass0, mass8)}
    arms = {"c1": C1Trace, "memory": MemoryTrace, "candidate": CandidateTrace}
    none_logs, normal_rows = {}, []
    episodes = 0
    for case_id, case in bases.items():
        records = {}
        for arm, detector in arms.items():
            log, trace = run(case, detector, None)
            none_logs[(case_id, arm)] = log
            records[arm] = compact(log, None, [], scoring)
            records[arm]["online_at_rises"] = {
                str(edge): trace[edge] for edge in replay.edge_indices(log)}
            episodes += 1
        normal_rows.append({"id": case_id, "case": case, "records": records,
                            "same_physics": replay.same_physics(
                                {name: {"none": value} for name, value in records.items()})})

    positives = []
    groups = (
        ("near_support", tool, fixed["near_support_time_s"]),
        ("near_quiet", quiet, fixed["near_quiet_time_s"]),
        ("late_postrelease", quiet, fixed["late_postrelease_time_s"]),
    )
    for group, case, onset in groups:
        for vector in fixed["vectors_N"]:
            impact = (float(onset), vector)
            records = {}
            for arm, detector in arms.items():
                forced, trace = run(case, detector, impact)
                reference = none_logs[(case["id"], arm)]
                records[arm] = positive_record(forced, reference, impact, scoring, trace)
                episodes += 1
            positives.append({
                "id": f"{group}_{vector}", "group": group, "case_id": case["id"],
                "impact": impact, "records": records,
                "same_force_physics": all(
                    records[arm]["trace"]["physical_sha256"]
                    == records["c1"]["trace"]["physical_sha256"] for arm in arms),
            })
    for case, control in ((mass0, fixed["pickup_controls"][0]),
                          (mass8, fixed["pickup_controls"][1])):
        impact = (float(control["time_s"]), control["vector_N"])
        records = {}
        for arm, detector in arms.items():
            forced, trace = run(case, detector, impact)
            records[arm] = positive_record(
                forced, none_logs[(case["id"], arm)], impact, scoring, trace)
            episodes += 1
        positives.append({
            "id": f"pickup_y15_mass{int(case['mass'])}", "group": "pickup",
            "case_id": case["id"], "impact": impact, "records": records,
            "same_force_physics": all(
                records[arm]["trace"]["physical_sha256"]
                == records["c1"]["trace"]["physical_sha256"] for arm in arms),
        })

    tool_normal = next(row for row in normal_rows if row["id"] == tool["id"])
    quiet_normals = [row for row in normal_rows if row["id"] != tool["id"]]
    removed = [edge for edge in tool_normal["records"]["memory"]["rises"]
               if edge not in tool_normal["records"]["candidate"]["rises"]]
    positive_gates = []
    for row in positives:
        memory = row["records"]["memory"]
        candidate = row["records"]["candidate"]
        old = memory["event"]["delays_ms"]
        new = candidate["event"]["delays_ms"]
        positive_gates.append({
            "id": row["id"],
            "input_present_but_candidate_missed": bool(old and not new),
            "first_delay_delta_ms": None if not old or not new else new[0] - old[0],
            "candidate_not_later_plus5": not old or bool(new and new[0] <= old[0] + 5),
            "candidate_no_more_raw_rises": len(candidate["event"]["raw_rise_indices"])
            <= len(memory["event"]["raw_rise_indices"]),
            "candidate_duration_le_plus5": candidate["trace"]["active_samples"]
            <= memory["trace"]["active_samples"] + 5,
            "force_exact": candidate["trace"]["force_exact"]["passed"],
            "prefix_exact": candidate["prefix"]["passed"],
            "same_force_physics": row["same_force_physics"],
        })
    protection = {
        "removed_only_known_inherited_edges": removed == [3903, 3923],
        "no_new_normal_edges": all(
            row["records"]["candidate"]["rises"] == row["records"]["memory"]["rises"]
            for row in quiet_normals),
        "all_positive_controls_preserved": all(
            not row["input_present_but_candidate_missed"]
            and row["candidate_not_later_plus5"] and row["candidate_no_more_raw_rises"]
            and row["candidate_duration_le_plus5"] and row["force_exact"]
            and row["prefix_exact"] and row["same_force_physics"]
            for row in positive_gates),
        "pickup_delays_exact": {
            row["id"]: row["records"]["candidate"]["event"]["delays_ms"]
            for row in positives if row["group"] == "pickup"
        } == {"pickup_y15_mass0": [38], "pickup_y15_mass8": [43]},
    }
    result = {
        "schema": "r29-g4-empty-inverse-release-result.v1",
        "classification": protocol["classification"],
        "pins": {"candidate": CANDIDATE_SHA256, "plan": PLAN_SHA256,
                 "candidate_plan": CANDIDATE_PLAN_SHA256},
        "episodes": episodes,
        "normal_rows": normal_rows,
        "positive_rows": positives,
        "positive_gates": positive_gates,
        "removed_memory_edges": removed,
        "protection_gates": protection,
        "passed": all(value if isinstance(value, bool) else True
                      for value in protection.values()),
        "claim_status": "disclosed_postmortem_only_no_promotion",
    }
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": episodes, "removed": removed,
                      "protection_gates": protection, "passed": result["passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
