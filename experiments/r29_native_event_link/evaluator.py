"""Run the preregistered known my_torque native-link panel with one worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.r24_holdout import evaluator as lab24
from experiments.r26_holdout import evaluator as lab26
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r27_load_change.manifold import Detector as R27
from experiments.r29_native_event_link.candidate import Detector as R29


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
ARMS = {"c3": C3, "r27": R27, "r29": R29}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(plan):
    import mujoco

    if mujoco.__version__ != "3.17.0":
        raise ValueError("MuJoCo 3.17.0 required")
    actual = {name: sha(ROOT / name) for name in plan["source_pins"]}
    if actual != plan["source_pins"]:
        raise ValueError("preregistered dependency changed")
    if plan.get("status") != "FROZEN_BEFORE_R29_RESULTS":
        raise ValueError("R29 plan is not frozen")
    return actual


def detector_run(run, case, variant, detector_type, impacts):
    detector = detector_type()
    metrics, log = run(case, variant, lambda: detector, impacts)
    diagnostics = None
    if isinstance(detector, R29):
        diagnostics = {
            "optional_starts": detector.r29_optional_starts,
            "native_links": detector.r29_native_links,
            "consumed_samples": detector.r29_consumed_samples,
            "link_active_at_end": detector.r29_link.active,
            "routes_complete": detector.r29_routes_complete,
            "active_routes_at_end": list(detector.r29_active_routes),
        }
    return metrics, log, diagnostics


def p26_case(plan):
    source = json.loads(lab26.PLAN.read_bytes())
    case = next(row for row in source["paired"] if row["id"] == "p26_close_place_cross")
    variant = source["variants"]["stock"]
    assignments = {"none": (None, None), "long": (case["long"], None),
                   "first": (case["impacts"][0], None), "second": (None, case["impacts"][1]),
                   "both": tuple(case["impacts"])}
    traces = {name: {tw: detector_run(lab26.run_episode, case, variant, detector, impacts)
                     for tw, impacts in assignments.items()} for name, detector in ARMS.items()}
    t = traces["c3"]["both"][1]["t"]
    long_start, long_boundary = lab26.actual_start(t, case["long"])
    starts = [lab26.actual_start(t, impact) for impact in case["impacts"]]
    first, second = starts[0][0], starts[1][0]
    scoring = plan["scoring"]["p26"]
    slots = {"long_single": ("long", long_start, long_start + scoring["event_window_ms"], ["none"]),
             "isolated_first": ("first", first, second, ["none"]),
             "isolated_second": ("second", second, second + scoring["event_window_ms"], ["none"]),
             "both_first": ("both", first, second, ["none", "second"]),
             "both_second": ("both", second, second + scoring["event_window_ms"], ["first"])}
    records = {}
    base = {tw: lab26.certified.physical_hash(row[1]) for tw, row in traces["c3"].items()}
    for name, twins in traces.items():
        events = {slot: lab26.event(twins[tw][1], [twins[ref][1] for ref in refs], begin,
                                    min(6000, end), scoring)
                  for slot, (tw, begin, end, refs) in slots.items()}
        common = {tw: lab26.common(row[1], assignments[tw],
                                  [first if assignments[tw][0] else None,
                                   second if assignments[tw][1] else None], scoring)
                  for tw, row in twins.items()}
        prefixes = {
            "long": lab26.certified.prefix(twins["none"][1], twins["long"][1], long_start, long_start),
            "first": lab26.certified.prefix(twins["none"][1], twins["first"][1], first, first),
            "second": lab26.certified.prefix(twins["none"][1], twins["second"][1], second, second),
            "both_first": lab26.certified.prefix(twins["none"][1], twins["both"][1], first, first),
            "both_second": lab26.certified.prefix(twins["first"][1], twins["both"][1], second, second),
        }
        records[name] = {"events": events, "twins": common, "prefixes": prefixes,
                         "physical_exact_vs_c3": all(lab26.certified.physical_hash(row[1]) == base[tw]
                                                     for tw, row in twins.items()),
                         "diagnostics": {tw: row[2] for tw, row in twins.items() if row[2] is not None}}
    return {"id": case["id"], "family": "p26", "case": case, "variant": variant,
            "boundaries": {"long": long_boundary, "first": starts[0][1], "second": starts[1][1]},
            "records": records, "episodes": len(ARMS) * len(assignments)}


def stress_cases(plan):
    original = json.loads(lab24.PLAN.read_bytes())
    policy = json.loads((ROOT / "experiments/r25_fragment/selection_policy.json").read_bytes())["paired_stress"]
    base = next(row for row in original["paired"] if row["id"] == "p24_transfer_near_cross")
    variant = original["variants"]["stock"]
    scoring = plan["scoring"]["stress"]
    rows = []
    for index in range(3):
        gap, vector = policy["conditions"][index]
        case = dict(base, id=f"cross_stress_{index}", impacts=[
            [policy["first_time"], policy["first_vector"]],
            [policy["first_time"] + gap, vector]])
        assignments = {"none": (None, None), "first": (case["impacts"][0], None),
                       "second": (None, case["impacts"][1]), "both": tuple(case["impacts"])}
        traces = {name: {tw: detector_run(lab24.run_episode, case, variant, detector, impacts)
                         for tw, impacts in assignments.items()} for name, detector in ARMS.items()}
        t = traces["c3"]["both"][1]["t"]
        starts = [lab24.certified.actual_start(t, impact) for impact in case["impacts"]]
        first, second = starts[0][0], starts[1][0]
        slots = {"isolated_first": ("first", first, second, ["none"]),
                 "isolated_second": ("second", second, second + scoring["event_window_ms"], ["none"]),
                 "both_first": ("both", first, second, ["none", "second"]),
                 "both_second": ("both", second, second + scoring["event_window_ms"], ["first"])}
        base_hash = {tw: lab24.certified.physical_hash(row[1]) for tw, row in traces["c3"].items()}
        records = {}
        for name, twins in traces.items():
            events = {slot: lab24.event_record(twins[tw][1], [twins[ref][1] for ref in refs], begin,
                                               min(6000, end), scoring["quiet_ms"],
                                               scoring["reference_twin_ms"])
                      for slot, (tw, begin, end, refs) in slots.items()}
            common = {tw: lab24.record_common(row[1], assignments[tw],
                                              [first if assignments[tw][0] else None,
                                               second if assignments[tw][1] else None], scoring)
                      for tw, row in twins.items()}
            prefixes = {
                "first": lab24.certified.prefix(twins["none"][1], twins["first"][1], first, first),
                "second": lab24.certified.prefix(twins["none"][1], twins["second"][1], second, second),
                "both_first": lab24.certified.prefix(twins["none"][1], twins["both"][1], first, first),
                "both_second": lab24.certified.prefix(twins["first"][1], twins["both"][1], second, second),
            }
            records[name] = {"events": events, "twins": common, "prefixes": prefixes,
                             "physical_exact_vs_c3": all(lab24.certified.physical_hash(row[1]) == base_hash[tw]
                                                         for tw, row in twins.items()),
                             "diagnostics": {tw: row[2] for tw, row in twins.items() if row[2] is not None}}
        rows.append({"id": case["id"], "family": "stress", "case": case, "variant": variant,
                     "boundaries": [item[1] for item in starts], "records": records,
                     "episodes": len(ARMS) * len(assignments)})
    return rows


def event_floor(reference, candidate, slack, cost_slack):
    if not reference["strict"]:
        return {"latency": True, "duration": True}
    return {"latency": bool(candidate["strict"] and candidate["delays_ms"][0] <=
                            reference["delays_ms"][0] + slack),
            "duration": bool(candidate["strict"] and candidate["active_samples_in_slot"] <=
                             reference["active_samples_in_slot"] + cost_slack)}


def row_gates(row, plan):
    scoring = plan["scoring"]
    slack = scoring["latency_slack_ms"]
    cost = scoring["duration_slack_samples"]
    candidate = row["records"]["r29"]
    references = [row["records"]["c3"], row["records"]["r27"]]
    floors = {name: {arm: event_floor(ref["events"][name], candidate["events"][name], slack, cost)
                     for arm, ref in zip(("c3", "r27"), references)}
              for name in candidate["events"]}
    candidate_false = [value for twin in candidate["twins"].values() for value in twin["false_times_s"]]
    reference_false = {arm: [value for twin in ref["twins"].values() for value in twin["false_times_s"]]
                       for arm, ref in zip(("c3", "r27"), references)}
    gates = {
        "all_strict_reference_latencies_preserved": all(item["latency"] for slot in floors.values()
                                                          for item in slot.values()),
        "all_strict_reference_durations_preserved": all(item["duration"] for slot in floors.values()
                                                          for item in slot.values()),
        "no_new_duplicates_vs_c3": all(candidate["events"][name]["duplicate_rises"] <=
                                         references[0]["events"][name]["duplicate_rises"]
                                         for name in candidate["events"]),
        "no_new_false_vs_both": all(len(candidate_false) <= len(values) for values in reference_false.values()),
        "no_new_tool_prealarm_vs_both": all(
            not candidate["twins"][tw]["tool"]["prealarm"] or
            all(ref["twins"][tw]["tool"]["prealarm"] for ref in references)
            for tw in candidate["twins"]),
        "no_new_stuck_tail_vs_both": all(
            not candidate["twins"][tw]["stuck_tail"] or
            all(ref["twins"][tw]["stuck_tail"] for ref in references)
            for tw in candidate["twins"]),
        "all_prefixes_exact": all(value["passed"] for record in row["records"].values()
                                    for value in record["prefixes"].values()),
        "all_force_exact": all(twin["force_exact"]["passed"] for record in row["records"].values()
                               for twin in record["twins"].values()),
        "all_physics_exact": all(record["physical_exact_vs_c3"] for record in row["records"].values()),
    }
    target_slot = "long_single" if row["id"] == "p26_close_place_cross" else "both_second"
    if row["id"] in ("p26_close_place_cross", "cross_stress_0"):
        event = candidate["events"][target_slot]
        gates["target_exactly_one"] = event["exactly_one"] and event["duplicate_rises"] == 0
    if row["id"] in ("p26_close_place_cross", "cross_stress_1", "cross_stress_2"):
        event = candidate["events"]["both_second"]
        strict_reference = any(ref["events"]["both_second"]["strict"] for ref in references)
        gates["distinct_second_not_consumed"] = not strict_reference or event["strict"]
    return gates, floors, {"candidate": candidate_false, **reference_false}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh output required")
    plan_raw = PLAN.read_bytes()
    plan = json.loads(plan_raw)
    before = verify(plan)
    rows = [p26_case(plan), *stress_cases(plan)]
    if sum(row["episodes"] for row in rows) != plan["episode_accounting"]["total"]:
        raise ValueError("episode accounting differs")
    for row in rows:
        gates, floors, false = row_gates(row, plan)
        row["gates"] = gates
        row["event_floors"] = floors
        row["false_times_s"] = false
        row["passed"] = all(gates.values())
    after = {name: sha(ROOT / name) for name in plan["source_pins"]}
    result = {
        "schema": "r29-native-event-link-known-result.v1",
        "classification": plan["classification"],
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "source_pins": before,
        "sources_unchanged": before == after,
        "episodes": sum(row["episodes"] for row in rows),
        "workers": 1,
        "rows": rows,
        "failed_rows": [{"id": row["id"], "gates": [name for name, passed in row["gates"].items()
                                                        if not passed]} for row in rows if not row["passed"]],
        "protection_passed": all(row["passed"] for row in rows) and before == after,
        "claim_status": "known_development_only_no_default_promotion",
    }
    args.out.mkdir()
    (args.out / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2,
                                                       allow_nan=False) + "\n")
    print(json.dumps({"episodes": result["episodes"], "protection_passed": result["protection_passed"],
                      "failed_rows": result["failed_rows"],
                      "targets": {row["id"]: row["records"]["r29"]["events"][
                          "long_single" if row["id"] == "p26_close_place_cross" else "both_second"]
                          for row in rows[:2]}}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
