"""Protect frozen R24 joint behavior on the entire disclosed R23 suite."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r23_holdout import evaluator
from experiments.r24_composed.detector import Detector as R24
from experiments.r25_pickup.candidate import Detector as R25


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "experiments/r23_holdout/plan.json"
ARMS = {"r22_light": R24, "r23": R25}


def event_map(row, arm):
    record = row["records"][arm]
    return ({"event": record["event"]} if row["kind"] == "single"
            else record.get("events", {}))


def comparison(row):
    old, new = row["records"]["r22_light"], row["records"]["r23"]
    if row["kind"] == "negative":
        gates = {
            "no_new_false": len(new["false_times_s"]) <= len(old["false_times_s"]),
            "no_new_tool_prealarm": not new["tool"]["prealarm"] or old["tool"]["prealarm"],
            "preserve_tool": evaluator.tool_floor(old["tool"], new["tool"], 5),
            "no_stuck_tail": not new["stuck_tail"],
        }
        slots = {}
    else:
        old_events, new_events = event_map(row, "r22_light"), event_map(row, "r23")
        slots = {slot: {"r24": event, "r25": new_events[slot]}
                 for slot, event in old_events.items()}
        gates = {
            "strict_preserved": all(
                not value["r24"]["strict"] or
                (value["r25"]["strict"] and
                 value["r25"]["delays_ms"][0] <= value["r24"]["delays_ms"][0] + 5)
                for value in slots.values()),
            "duplicate_not_increased": all(
                value["r25"]["duplicate_rises"] <= value["r24"]["duplicate_rises"]
                for value in slots.values()),
            "no_new_false": all(len(new["twins"][twin]["false_times_s"]) <=
                                len(old["twins"][twin]["false_times_s"])
                                for twin in new["twins"]),
            "no_new_tool_prealarm": all(
                not new["twins"][twin]["tool"]["prealarm"] or
                old["twins"][twin]["tool"]["prealarm"] for twin in new["twins"]),
            "preserve_tool": all(evaluator.tool_floor(
                old["twins"][twin]["tool"], new["twins"][twin]["tool"], 5)
                                 for twin in new["twins"]),
            "no_stuck_tail": all(not item["stuck_tail"] for item in new["twins"].values()),
        }
    return {"variant": row["variant"], "id": row["id"], "kind": row["kind"],
            "changed": old != new, "slots": slots, "gates": gates,
            "passed": all(gates.values())}


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
    rows = [comparison(row) for row in raw]
    invalid = [row["variant"] + ":" + row["id"] for row in raw
               if not row["safety_floor"]["physical_and_force_exact"]]
    failures = [{"variant": row["variant"], "id": row["id"],
                 "gates": [name for name, passed in row["gates"].items() if not passed]}
                for row in rows if not row["passed"]]
    doc = {
        "schema": "r25-disclosed-r23-protection.v1",
        "classification": "Entire disclosed R23 suite; not independent R25 evidence",
        "episodes": sum(row["episodes"] for row in raw), "rows": rows,
        "summary": {"rows": len(rows),
                    "changed": sum(row["changed"] for row in rows),
                    "failures": failures, "invalid": invalid,
                    "passed": not failures and not invalid},
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps(doc["summary"], indent=2))


if __name__ == "__main__":
    main()
