"""Fail-closed validation and exact-repeat comparison for R25 holdout results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / "plan.json"
PROTOCOL = HERE / "protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(document, lock_sha):
    plan = json.loads(PLAN.read_bytes())
    protocol = json.loads(PROTOCOL.read_bytes())
    source_pins = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    expected = {(kind, case["id"], variant)
                for kind in ("negative", "single", "paired")
                for case in plan[kind] for variant in plan["variants"]}
    rows = document.get("rows", [])
    observed = [(row.get("kind"), row.get("id"), row.get("variant")) for row in rows]
    effects = document.get("mechanical_effect_observed", {})
    if (document.get("schema") != "r25-independent-variable-pulse-result.v1"
            or document.get("plan_sha256") != sha(PLAN)
            or document.get("lock_sha256") != lock_sha
            or document.get("arms") != ["r24_joint", "r25"]
            or document.get("episodes") != plan["required_episodes"]["both_arms"]
            or source_pins != protocol["source_pins"]
            or document.get("source_pins") != source_pins
            or set(observed) != expected or len(observed) != len(set(observed))
            or set(effects) != set(plan["variants"]) or not all(effects.values())):
        raise ValueError("incomplete or mismatched R25 holdout result")
    gains, misses = [], []
    for row in rows:
        expected_episodes = {"negative": 2, "single": 4, "paired": 10}[row["kind"]]
        protection = row.get("incumbent_safety", {})
        detection = row.get("positive_qualification", {})
        if (row.get("episodes") != expected_episodes
                or set(row.get("records", {})) != {"r24_joint", "r25"}
                or not protection or set(detection) != {"required", "passed"}
                or row.get("passed") != (all(protection.values()) and detection["passed"])):
            raise ValueError("R25 row accounting or gates differ: " + str(row.get("id")))
        if row.get("positive_gains"):
            gains.append({"variant": row["variant"], "id": row["id"],
                          "kind": row["kind"], "slots": row["positive_gains"]})
        if row.get("absolute_misses"):
            misses.append({"variant": row["variant"], "id": row["id"],
                           "kind": row["kind"], "slots": row["absolute_misses"]})
    safety_pass = all(all(row["incumbent_safety"].values()) for row in rows)
    positive_pass = all(row["positive_qualification"]["passed"] for row in rows
                         if row["positive_qualification"]["required"])
    contact_rows = [row["variant"] + ":" + row["id"] for row in rows
                    if row["kind"] == "negative" and row.get("contact_probe") and
                    row["records"]["r24_joint"]["tool"]["present"]]
    required = plan["contact_evidence"]["minimum_probe_variant_rows_with_real_contact"]
    contact = {"rows": contact_rows, "required": required,
               "adequate": len(contact_rows) >= required}
    claims = {"safety_qualification": False,
              "incumbent_safety_floor_pass": safety_pass,
              "positive_qualification_pass": positive_pass,
              "positive_gain_count": sum(len(row["slots"]) for row in gains),
              "absolute_miss_count": sum(len(row["slots"]) for row in misses)}
    failures = [row["variant"] + ":" + row["id"] for row in rows if not row["passed"]]
    passed = safety_pass and positive_pass and contact["adequate"]
    if (document.get("real_contact_evidence") != contact
            or document.get("claims") != claims
            or document.get("positive_gains") != gains
            or document.get("absolute_misses") != misses
            or document.get("failed") != failures
            or document.get("passed") != passed):
        raise ValueError("top-level R25 verdict differs from row evidence")
    return {"episodes": document["episodes"], "rows": len(rows),
            "failed": failures, "passed": passed, "claims": claims,
            "real_contact_evidence": contact}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh output required")
    lock_sha = sha(args.lock)
    first = json.loads(args.first.read_bytes())
    summary = validate(first, lock_sha)
    exact_repeat = None
    repeat_sha = None
    if args.repeat:
        repeat = json.loads(args.repeat.read_bytes())
        validate(repeat, lock_sha)
        exact_repeat = first == repeat
        repeat_sha = sha(args.repeat)
        if not exact_repeat:
            raise ValueError("repeated R25 result differs")
    receipt = {
        "schema": "r25-independent-variable-pulse-verification.v1",
        "plan_sha256": sha(PLAN), "lock_sha256": lock_sha,
        "first_result_sha256": sha(args.first),
        "repeat_result_sha256": repeat_sha, "exact_repeat": exact_repeat,
        "summary": summary,
    }
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
