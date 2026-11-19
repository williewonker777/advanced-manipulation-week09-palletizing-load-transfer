"""Evaluate the frozen yaw-acquisition candidate on the disclosed long target."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.r25_holdout import evaluator
from experiments.r26_yaw_acquisition.candidate import Detector


ARCHIVE_SHA = "83987dba8ce14fb6cc0283021d6bd062ac2fedb317ceb8da67c3d4b2b8c4f4fe"
CASE_ID = "s25_long_transfer_signed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Observed(Detector):
    def reset(self):
        super().reset()
        self.acquisition = []

    def update(self, *args):
        residual, report = super().update(*args)
        self.acquisition.append({
            "report": bool(report),
            "yaw": bool(self.yaw_active),
            "episode": bool(self.yaw_acquisition_episode_active),
            "consumed": bool(self.yaw_reacquisition_consumed),
            "count": int(self.yaw_reacquisition_count),
            "provenance_complete": bool(self.yaw_reacquisition_provenance_complete),
        })
        return residual, report


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
        none_detector, force_detector = Observed(), Observed()
        _, none = evaluator.run_episode(case, variant, lambda: none_detector, (None, None))
        _, forced = evaluator.run_episode(case, variant, lambda: force_detector, (case["impact"], None))
        start, boundary = evaluator.actual_start(forced["t"], case["impact"])
        event = evaluator.event(forced, [none], start,
                                min(6000, start + plan["event_window_ms"]), plan)
        old = frozen[name]
        old_event = old["records"]["r25"]["event"]
        common = {
            "none": evaluator.common(none, (None, None), [], plan),
            "single": evaluator.common(forced, (case["impact"], None), [start], plan),
        }
        gates = {
            "one_rise": bool(event["strict"] and event["duplicate_rises"] == 0),
            "first_rise_exact": event["delays_ms"] == old_event["delays_ms"][:1],
            "zero_false": all(not row["false_times_s"] for row in common.values()),
            "no_stuck_tail": all(not row["stuck_tail"] for row in common.values()),
            "force_exact": all(row["force_exact"]["passed"] for row in common.values()),
            "physical_exact": (evaluator.certified.trace_signature({"none": none, "single": forced}) ==
                               old["r24_physics_sha256"]),
            "boundary_exact": boundary == old["records"]["r25"]["boundary"],
            "provenance_complete": all(row["provenance_complete"]
                                       for row in force_detector.acquisition),
            "consumed_reacquisition": force_detector.yaw_reacquisition_count > 0,
        }
        rows.append({"variant": name, "frozen": old_event, "candidate": event,
                     "reacquisition_count": force_detector.yaw_reacquisition_count,
                     "gates": gates, "passed": all(gates.values())})
    document = {
        "schema": "r26-yaw-acquisition-target.v1",
        "candidate": "experiments.r26_yaw_acquisition.candidate:Detector",
        "candidate_sha256": sha(Path(__file__).with_name("candidate.py")),
        "archive_sha256": sha(args.archive),
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({row["variant"]: {"event": row["candidate"],
                                      "count": row["reacquisition_count"],
                                      "gates": row["gates"]}
                      for row in rows}, indent=2))
    if not document["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
