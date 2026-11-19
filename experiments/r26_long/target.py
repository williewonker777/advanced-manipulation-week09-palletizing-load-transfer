"""Evaluate the R26 ownership candidate on the disclosed long signed target."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.r25_holdout import evaluator
from experiments.r26_long.candidate import Detector


ARCHIVE_SHA = "83987dba8ce14fb6cc0283021d6bd062ac2fedb317ceb8da67c3d4b2b8c4f4fe"
CASE_ID = "s25_long_transfer_signed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    if sha(args.archive) != ARCHIVE_SHA:
        raise ValueError("unexpected disclosed R25 archive")
    archive = json.loads(args.archive.read_bytes())
    plan = json.loads(evaluator.PLAN.read_bytes())
    case = next(row for row in plan["single"] if row["id"] == CASE_ID)
    frozen = {row["variant"]: row for row in archive["rows"] if row["id"] == CASE_ID}
    rows = []
    for name, variant in plan["variants"].items():
        _, none = evaluator.run_episode(case, variant, Detector, (None, None))
        _, forced = evaluator.run_episode(case, variant, Detector, (case["impact"], None))
        start, boundary = evaluator.actual_start(forced["t"], case["impact"])
        event = evaluator.event(forced, [none], start,
                                min(6000, start + plan["event_window_ms"]), plan)
        common = {"none": evaluator.common(none, (None, None), [], plan),
                  "single": evaluator.common(forced, (case["impact"], None), [start], plan)}
        old = frozen[name]
        physical = (evaluator.certified.trace_signature({"none": none, "single": forced}) ==
                    old["r24_physics_sha256"])
        old_event = old["records"]["r25"]["event"]
        gates = {
            "exactly_one": event["strict"] and event["duplicate_rises"] == 0,
            "latency_plus_5_ms": (bool(event["delays_ms"]) and
                                  event["delays_ms"][0] <= old_event["delays_ms"][0] + 5),
            "zero_false": all(not item["false_times_s"] for item in common.values()),
            "no_stuck_tail": all(not item["stuck_tail"] for item in common.values()),
            "force_exact": all(item["force_exact"]["passed"] for item in common.values()),
            "physical_exact": physical,
            "boundary_exact": boundary == old["records"]["r25"]["boundary"],
        }
        rows.append({"variant": name, "frozen": old_event, "candidate": event,
                     "common": common, "gates": gates, "passed": all(gates.values())})
    document = {"schema": "r26-long-target-result.v1", "archive_sha256": sha(args.archive),
                "candidate": "experiments.r26_long.candidate:Detector", "rows": rows,
                "passed": all(row["passed"] for row in rows)}
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({row["variant"]: {"old": row["frozen"]["delays_ms"],
                                               "new": row["candidate"]["delays_ms"],
                                               "active": row["candidate"]["active_samples_in_slot"],
                                               "gates": row["gates"]}
                      for row in rows}, indent=2))
    if not document["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
