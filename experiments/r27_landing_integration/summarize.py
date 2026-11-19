"""Build the immutable compact verdict from completed R27 landing evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    files = {
        "fit": args.evidence / "fit/fit.json",
        "audit": args.evidence / "audit.json",
        "official": args.evidence / "official/result.json",
        "mechanism": args.evidence / "mechanism.json",
        "tool": args.evidence / "tool_guard.json",
    }
    docs = {name: read(path) for name, path in files.items()}
    official = docs["official"]["records"]
    if official["c1"]["counts"]["impact_outcomes"]["detected"] != 15:
        raise ValueError("public C1 baseline did not reproduce")
    tool_failures = [{"id": row["id"], "failed": [name for name, passed
                      in row["gates"].items() if not passed]}
                     for row in docs["tool"]["rows"] if not row["passed"]]
    doc = {
        "schema": "r27-landing-integration-result.v1",
        "classification": "Post-disclosure bounded development; not blind, broad qualification, hardware, deployment, or safety evidence.",
        "frozen": docs["official"]["sources"],
        "normal_calibration": {
            "fit_episodes": docs["fit"]["episodes"],
            "fit_max_horizontal_N": docs["fit"]["model"]["fit_max_horizontal_N"],
            "frozen_bound_N": docs["fit"]["model"]["horizontal_bound_N"],
            "separated_audit_episodes": docs["audit"]["episodes"],
            "audit_passed": docs["audit"]["passed"],
        },
        "authentic_official": {
            "episodes": docs["official"]["episodes"],
            "c1": official["c1"]["counts"],
            "candidate": official["r27"]["counts"],
            "detected_gain": (official["r27"]["counts"]["impact_outcomes"]["detected"] -
                              official["c1"]["counts"]["impact_outcomes"]["detected"]),
            "same_control_cells": docs["official"]["same_control_cells"],
            "gained_strata": ["mass 0 kg, 1.42 s, 15 N horizontal",
                               "mass 8 kg, 1.42 s, 15 N horizontal"],
        },
        "bounded_mechanism": {
            "episodes": docs["mechanism"]["episodes"],
            "integrity": docs["mechanism"]["integrity"],
            "pickup_y15_strict": {name: arm["pickup_y15"]["event"]["strict"]
                                  for name, arm in docs["mechanism"]["records"].items()},
            "pickup_x15_strict": {name: arm["pickup_x15"]["event"]["strict"]
                                  for name, arm in docs["mechanism"]["records"].items()},
            "pickup_z15_strict": {name: arm["pickup_z15"]["event"]["strict"]
                                  for name, arm in docs["mechanism"]["records"].items()},
            "pair_strict": {name: {key: event["strict"] for key, event
                            in arm["events"].items()} for name, arm
                            in docs["mechanism"]["pair_records"].items()},
            "worst_update_p99_us": {name: max(record["update_p99_us"]
                                      for key, record in arm.items()
                                      if key != "events") for name, arm
                                      in docs["mechanism"]["records"].items()},
        },
        "counterexample": {
            "tool_guard_episodes": docs["tool"]["episodes"],
            "genuine_contact_rows": docs["tool"]["genuine_contact_rows"],
            "failures": tool_failures,
            "tool_high_true_contact_preserved": {
                name: next(row for row in docs["tool"]["rows"]
                           if row["id"] == "tool_high")["records"][name]["tool"]
                for name in ("c1", "r27")},
            "tool_high_rises": {
                name: next(row for row in docs["tool"]["rows"]
                           if row["id"] == "tool_high")["records"][name]["rises"]
                for name in ("c1", "r27")},
            "diagnosis": "The gripped inverse-force source also crosses its frozen normal bound during the high-offset placement/load-transfer episode, creating a new 3.857 s edge before the inherited 3.903 s edge. This violates the preregistered no-new-rise guard even though the genuine tool contact remains detected at the same delay.",
        },
        "verdict": {
            "candidate_eligible": False,
            "reason": "New late rise in a disclosed genuine-tool counterexample; no retuning permitted after frozen positive evaluation.",
            "broad_guard_authorized": False,
        },
        "evidence_sha256": {name: sha(path) for name, path in files.items()},
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"detected_gain": doc["authentic_official"]["detected_gain"],
                      "candidate_eligible": False,
                      "counterexample": tool_failures}, indent=2))


if __name__ == "__main__":
    main()
