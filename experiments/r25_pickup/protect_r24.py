"""Compare R25 with frozen R24 joint on every disclosed R24 row."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r23_detached.free import Detector as R23
from experiments.r24_composed.detector import Detector as R24
from experiments.r24_holdout import evaluator
from experiments.r25_pickup.candidate import Detector as R25


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "experiments/r24_holdout/plan.json"
ARMS = {"r23_free": R23, "r24": R25, "r24_joint": R24}


def events(row, arm):
    record = row["records"][arm]
    return ({"event": record["event"]} if row["kind"] == "single"
            else record.get("events", {}))


def rises(event):
    return event["raw_rise_indices"]


def compare(row):
    if row["kind"] == "negative":
        old, new = row["records"]["r24_joint"], row["records"]["r24"]
        gates = {
            "no_new_false": len(new["false_times_s"]) <= len(old["false_times_s"]),
            "no_new_false_active": new["false_active_samples"] <= old["false_active_samples"],
            "no_new_tool_prealarm": not new["tool"]["prealarm"] or old["tool"]["prealarm"],
            "preserve_tool": evaluator.prior.tool_floor(
                old["tool"], new["tool"], 5),
            "no_stuck_tail": not new["stuck_tail"],
        }
        changes = old != new
        slots = {}
    else:
        old_events, new_events = events(row, "r24_joint"), events(row, "r24")
        base_events = events(row, "r23_free")
        slots = {}
        for slot, old in old_events.items():
            new, base = new_events[slot], base_events[slot]
            slots[slot] = {
                "r23": base, "r24": old, "r25": new,
                "joint_strict_preserved": (not old["strict"] or
                                            (new["strict"] and
                                             new["delays_ms"][0] <= old["delays_ms"][0] + 5)),
                "joint_duplicate_not_increased": (new["duplicate_rises"] <=
                                                    old["duplicate_rises"]),
                "r23_cost_floor": (not base["strict"] or
                                    (new["strict"] and
                                     new["active_samples_in_slot"] <=
                                     base["active_samples_in_slot"] + 5)),
            }
        old_twins, new_twins = (row["records"]["r24_joint"]["twins"],
                                row["records"]["r24"]["twins"])
        gates = {
            "joint_strict_preserved": all(value["joint_strict_preserved"]
                                          for value in slots.values()),
            "joint_duplicate_not_increased": all(value["joint_duplicate_not_increased"]
                                                  for value in slots.values()),
            "r23_cost_floor": all(value["r23_cost_floor"] for value in slots.values()),
            "no_new_false": all(len(new_twins[twin]["false_times_s"]) <=
                                len(old_twins[twin]["false_times_s"])
                                for twin in new_twins),
            "no_new_false_active": all(new_twins[twin]["false_active_samples"] <=
                                       old_twins[twin]["false_active_samples"]
                                       for twin in new_twins),
            "no_new_tool_prealarm": all(not new_twins[twin]["tool"]["prealarm"] or
                                        old_twins[twin]["tool"]["prealarm"]
                                        for twin in new_twins),
            "preserve_tool": all(evaluator.prior.tool_floor(
                old_twins[twin]["tool"], new_twins[twin]["tool"], 5)
                                 for twin in new_twins),
            "no_stuck_tail": all(not record["stuck_tail"]
                                 for record in new_twins.values()),
        }
        changes = any(rises(value["r24"]) != rises(value["r25"]) or
                      value["r24"]["active_samples_in_slot"] !=
                      value["r25"]["active_samples_in_slot"]
                      for value in slots.values())
    return {
        "variant": row["variant"], "id": row["id"], "kind": row["kind"],
        "changed": changes, "slots": slots, "gates": gates,
        "passed": all(gates.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    plan = json.loads(PLAN.read_bytes())
    tasks = [(kind, case, variant, ARMS, plan)
             for kind in ("negative", "single", "paired")
             for case in plan[kind] for variant in plan["variants"].values()]
    with ProcessPoolExecutor(args.jobs) as pool:
        raw = list(pool.map(evaluator.one, tasks))
    rows = [compare(row) for row in raw]
    invalid = [row["variant"] + ":" + row["id"] for row in raw
               if not row["protection"]["physical_and_force_exact"]]
    failures = [{"variant": row["variant"], "id": row["id"],
                 "gates": [name for name, passed in row["gates"].items() if not passed]}
                for row in rows if not row["passed"]]
    doc = {
        "schema": "r25-disclosed-r24-protection.v1",
        "classification": "All disclosed R24 rows; not independent R25 evidence",
        "episodes": sum(row["episodes"] for row in raw),
        "rows": rows,
        "summary": {
            "rows": len(rows), "changed": sum(row["changed"] for row in rows),
            "failures": failures, "invalid": invalid,
            "passed": not failures and not invalid,
        },
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps(doc["summary"], indent=2))


if __name__ == "__main__":
    main()
