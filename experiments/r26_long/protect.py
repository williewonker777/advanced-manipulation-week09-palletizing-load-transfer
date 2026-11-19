"""Protect disclosed close forces, short seconds, and the corrected R25 pickup."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r24_holdout import evaluator as r24_eval
from experiments.r25_holdout import evaluator
from experiments.r26_long.candidate import Detector


R25_SHA = "83987dba8ce14fb6cc0283021d6bd062ac2fedb317ceb8da67c3d4b2b8c4f4fe"
PICKUP_SHA = "bd99e89f35df7afadb51bf59734135508643b3bfcb69455dbee1bc17ea30b856"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(case, variant, plan, old):
    assignment = {"none": (None, None), "long": (case["long"], None),
                  "first": (case["impacts"][0], None),
                  "second": (None, case["impacts"][1]),
                  "both": tuple(case["impacts"])}
    traces = {name: evaluator.run_episode(case, variant, Detector, impacts)[1]
              for name, impacts in assignment.items()}
    t = traces["both"]["t"]
    long_start = evaluator.actual_start(t, case["long"])[0]
    first = evaluator.actual_start(t, case["impacts"][0])[0]
    second = evaluator.actual_start(t, case["impacts"][1])[0]
    window = plan["event_window_ms"]
    slots = {"long_single": ("long", long_start, min(6000, long_start + window), ["none"]),
             "isolated_first": ("first", first, min(second, first + window), ["none"]),
             "isolated_second": ("second", second, min(6000, second + window), ["none"]),
             "both_first": ("both", first, min(second, first + window), ["none", "second"]),
             "both_second": ("both", second, min(6000, second + window), ["first"])}
    events = {slot: evaluator.event(traces[twin], [traces[ref] for ref in refs], begin, end, plan)
              for slot, (twin, begin, end, refs) in slots.items()}
    common = {twin: evaluator.common(log, impacts,
                                      [first if impacts[0] else None,
                                       second if impacts[1] else None], plan)
              for (twin, impacts), log in zip(assignment.items(), traces.values(), strict=True)}
    costs = {}
    for slot, previous in old["records"]["r25"]["events"].items():
        current = events[slot]
        costs[slot] = {
            "frozen": previous,
            "candidate": current,
            "strict_floor": not previous["strict"] or current["strict"],
            "latency_plus_5_ms": (not previous["strict"] or
                                  current["delays_ms"][0] <= previous["delays_ms"][0] + 5),
            "duration_plus_5_samples": (not previous["strict"] or
                                        current["active_samples_in_slot"] <=
                                        previous["active_samples_in_slot"] + 5),
            "zero_duplicate": current["duplicate_rises"] == 0,
        }
    gates = {
        "all_frozen_strict_cost_floors": all(all(value for key, value in row.items()
                                                    if key not in ("frozen", "candidate"))
                                                 for row in costs.values()),
        "zero_false": all(not row["false_times_s"] for row in common.values()),
        "no_stuck_tail": all(not row["stuck_tail"] for row in common.values()),
        "force_exact": all(row["force_exact"]["passed"] for row in common.values()),
        "physical_exact": evaluator.certified.trace_signature(traces) == old["r24_physics_sha256"],
    }
    return {"id": case["id"], "variant": variant["kind"], "events": events,
            "costs": costs, "gates": gates, "passed": all(gates.values())}


def pickup(plan, archive):
    case = next(row for row in plan["single"] if row["id"] == "s24_transfer_offaxis_early")
    variant = plan["variants"]["stock"]
    _, none = r24_eval.run_episode(case, variant, Detector, (None, None))
    _, force = r24_eval.run_episode(case, variant, Detector, (case["impact"], None))
    start, boundary = r24_eval.certified.actual_start(force["t"], case["impact"])
    event = r24_eval.event_record(force, [none], start,
                                  min(6000, start + plan["event_window_ms"]),
                                  plan["quiet_ms"], plan["reference_twin_ms"])
    frozen = archive["duration"]["records"]["r25"]
    old = frozen["event"]
    gates = {
        "strict": event["strict"],
        "latency_plus_5_ms": event["delays_ms"][0] <= old["delays_ms"][0] + 5,
        "duration_plus_5_samples": event["active_samples_in_slot"] <= old["active_samples_in_slot"] + 5,
        "zero_duplicate": event["duplicate_rises"] == 0,
        "none_quiet": not np.asarray(none["detect"], bool).any(),
        "force_exact": (r24_eval.certified.exact_force(none, (None, None))["passed"] and
                        r24_eval.certified.exact_force(force, (case["impact"], None))["passed"]),
        "none_physics_exact": r24_eval.certified.physical_hash(none) == frozen["none"]["physical_sha256"],
        "force_physics_exact": r24_eval.certified.physical_hash(force) == frozen["force"]["physical_sha256"],
        "boundary_exact": boundary == archive["duration"]["boundary"],
    }
    return {"id": case["id"], "variant": variant["kind"], "frozen": old,
            "candidate": event, "gates": gates, "passed": all(gates.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r25", required=True, type=Path)
    parser.add_argument("--pickup", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    if sha(args.r25) != R25_SHA or sha(args.pickup) != PICKUP_SHA:
        raise ValueError("unexpected disclosed archive")
    old_r25 = json.loads(args.r25.read_bytes())
    old_pickup = json.loads(args.pickup.read_bytes())
    plan = json.loads(evaluator.PLAN.read_bytes())
    r24_plan = json.loads(r24_eval.PLAN.read_bytes())
    old_rows = {(row["id"], row["variant"]): row for row in old_r25["rows"]}
    rows = [paired(case, variant, plan, old_rows[(case["id"], name)])
            for case in plan["paired"] for name, variant in plan["variants"].items()]
    pickup_row = pickup(r24_plan, old_pickup)
    short = [row for row in rows if row["id"] == "p25_close_detach_boundary"]
    opposed = [row for row in rows if row["id"] == "p25_close_detached_opposed"]
    claims = {
        "all_paired_protection": all(row["passed"] for row in rows),
        "corrected_short_second_preserved": all(
            row["costs"]["isolated_second"][key]
            for row in short for key in ("strict_floor", "latency_plus_5_ms",
                                         "duration_plus_5_samples", "zero_duplicate")),
        "opposed_all_five_strict": all(all(event["strict"] for event in row["events"].values())
                                           for row in opposed),
        "pickup_preserved": pickup_row["passed"],
    }
    document = {"schema": "r26-long-disclosed-protection.v1",
                "r25_archive_sha256": sha(args.r25),
                "pickup_archive_sha256": sha(args.pickup),
                "episodes": len(rows) * 5 + 2,
                "rows": rows, "pickup": pickup_row, "claims": claims,
                "passed": all(claims.values())}
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": document["episodes"], "claims": claims,
                      "failed": [row["variant"] + ":" + row["id"]
                                 for row in rows if not row["passed"]],
                      "pickup": pickup_row["candidate"]}, indent=2))
    if not document["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
