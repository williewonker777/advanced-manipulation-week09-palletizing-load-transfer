"""Run the disclosed R25 close-pair and pickup guards for the frozen candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.r25_holdout import evaluator
from experiments.r24_holdout import evaluator as r24_eval
from experiments.r26_long import protect as guard
from experiments.r26_yaw_acquisition.candidate import Detector


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r25", required=True, type=Path)
    parser.add_argument("--pickup", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    if guard.sha(args.r25) != guard.R25_SHA or guard.sha(args.pickup) != guard.PICKUP_SHA:
        raise ValueError("unexpected disclosed archive")
    guard.Detector = Detector
    old_r25 = json.loads(args.r25.read_bytes())
    old_pickup = json.loads(args.pickup.read_bytes())
    plan = json.loads(evaluator.PLAN.read_bytes())
    r24_plan = json.loads(r24_eval.PLAN.read_bytes())
    old_rows = {(row["id"], row["variant"]): row for row in old_r25["rows"]}
    rows = [guard.paired(case, variant, plan, old_rows[(case["id"], name)])
            for case in plan["paired"] for name, variant in plan["variants"].items()]
    pickup = guard.pickup(r24_plan, old_pickup)
    claims = {
        "all_close_pairs_preserved": all(row["passed"] for row in rows),
        "short_second_preserved": all(
            row["costs"]["isolated_second"][key]
            for row in rows if row["id"] == "p25_close_detach_boundary"
            for key in ("strict_floor", "latency_plus_5_ms",
                        "duration_plus_5_samples", "zero_duplicate")),
        "opposed_all_slots_strict": all(
            event["strict"] for row in rows if row["id"] == "p25_close_detached_opposed"
            for event in row["events"].values()),
        "pickup_preserved": pickup["passed"],
    }
    document = {
        "schema": "r26-yaw-acquisition-protection.v1",
        "candidate": "experiments.r26_yaw_acquisition.candidate:Detector",
        "episodes": len(rows) * 5 + 2,
        "rows": rows,
        "pickup": pickup,
        "claims": claims,
        "passed": all(claims.values()),
    }
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": document["episodes"], "claims": claims,
                      "failed": [row["variant"] + ":" + row["id"]
                                 for row in rows if not row["passed"]]}, indent=2))
    if not document["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
