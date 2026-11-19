"""Create a compact, source-pinned scientific summary from a verified R22 result."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from experiments.r22_challenge import verify


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_single(row, arm):
    event = row["records"][arm]["event"]
    return event["exactly_one"] and event["quiet_before"]


def strict_pair_slots(row, arm):
    return sum(event["exactly_one"] and event["quiet_before"]
               for event in row["records"][arm]["events"].values())


def strict_pair_row(row, arm):
    return strict_pair_slots(row, arm) == 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh output required")
    document = json.loads(args.result.read_bytes())
    verified = verify.validate(document, sha(args.lock))
    rows = document["rows"]
    by_arm = {}
    for arm in document["arms"]:
        negative = [row for row in rows if row["kind"] == "negative"]
        single = [row for row in rows if row["kind"] == "single"]
        paired = [row for row in rows if row["kind"] == "paired"]
        by_arm[arm] = {
            "negative_false_rises": sum(len(row["records"][arm]["false_times_s"]) for row in negative),
            "tool_present_rows": sum(row["records"][arm]["tool"]["present"] for row in negative),
            "tool_detected_rows": sum(row["records"][arm]["tool"]["detected"] for row in negative),
            "single_one_to_one": sum(strict_single(row, arm) for row in single),
            "single_total": len(single),
            "paired_all_four_one_to_one": sum(strict_pair_row(row, arm) for row in paired),
            "paired_rows": len(paired),
            "paired_one_to_one_slots": sum(strict_pair_slots(row, arm) for row in paired),
            "paired_slots": 4 * len(paired),
        }
    failed_gates = Counter((row["kind"], gate) for row in rows
                           for gate, passed in row["comparison"].items() if not passed)
    compact_failures = []
    for row in rows:
        if row["passed"]:
            continue
        candidate = row["records"]["r22"]
        detail = {"id": row["id"], "kind": row["kind"], "variant": row["variant"],
                  "failed_gates": [key for key, passed in row["comparison"].items() if not passed]}
        if row["kind"] == "single":
            event = candidate["event"]
            detail["candidate_event"] = {key: event[key] for key in
                                         ("raw_rise_indices", "attributable_rise_indices",
                                          "delays_ms", "quiet_before")}
        elif row["kind"] == "paired":
            detail["candidate_failed_slots"] = {
                slot: {key: event[key] for key in ("raw_rise_indices", "attributable_rise_indices",
                                                   "delays_ms", "quiet_before")}
                for slot, event in candidate["events"].items()
                if not (event["exactly_one"] and event["quiet_before"])}
        compact_failures.append(detail)
    result = {"schema": "r22-independent-protection-summary.v1",
              "classification": document["classification"], "verdict": "PASS" if document["passed"] else "FAIL",
              "plan_sha256": document["plan_sha256"], "evaluator_sha256": sha(HERE / "evaluator.py"),
              "lock_sha256": document["lock_sha256"], "result_sha256": sha(args.result),
              "episodes": document["episodes"], "rows": len(rows),
              "mechanical_effect_observed": document["mechanical_effect_observed"],
              "verified_accounting": verified, "arm_totals": by_arm,
              "failed_gate_counts": {kind + ":" + gate: count for (kind, gate), count in sorted(failed_gates.items())},
              "failed_rows": compact_failures,
              "tool_test_gap": "Two negative rows produced real tool contact, but no arm detected either; preservation is therefore not demonstrated by this cohort.",
              "interpretation": "Every candidate failure remains a release failure; this finite challenge was not used for tuning."}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"verdict": result["verdict"], "episodes": result["episodes"],
                      "failed_rows": len(result["failed_rows"])}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
