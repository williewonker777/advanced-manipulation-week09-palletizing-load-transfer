"""Fail-closed validation and exact repeat comparison for R22 challenge results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
PROTOCOL = HERE / "protocol.json"
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(document, lock_sha):
    plan = json.loads(PLAN.read_bytes())
    protocol = json.loads(PROTOCOL.read_bytes())
    source_pins = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    expected = {(kind, case["id"], variant) for kind in ("negative", "single", "paired")
                for case in plan[kind] for variant in plan["variants"]}
    rows = document.get("rows", [])
    observed = [(row.get("kind"), row.get("id"), row.get("variant")) for row in rows]
    if (document.get("schema") != "r22-independent-protection-result.v1"
            or document.get("plan_sha256") != sha(PLAN)
            or document.get("lock_sha256") != lock_sha
            or document.get("arms") != ["r10", "r21_yaw", "r22"]
            or document.get("episodes") != plan["required_episodes"]["all_three_arms"]
            or source_pins != protocol["source_pins"]
            or document.get("source_pins") != source_pins
            or set(observed) != expected or len(observed) != len(set(observed))
            or set(document.get("mechanical_effect_observed", {})) != set(plan["variants"])
            or not all(document["mechanical_effect_observed"].values())):
        raise ValueError("incomplete or mismatched R22 challenge result")
    for row in rows:
        expected_episodes = {"negative": 3, "single": 6, "paired": 12}[row["kind"]]
        if (row.get("episodes") != expected_episodes or set(row.get("records", {})) !=
                {"r10", "r21_yaw", "r22"} or row.get("passed") != all(row["comparison"].values())):
            raise ValueError("row accounting or gates differ: " + str(observed))
    failures = [row["variant"] + ":" + row["id"] for row in rows if not row["passed"]]
    if document.get("failed") != failures or document.get("passed") != (not failures):
        raise ValueError("top-level verdict differs from row gates")
    return {"episodes": document["episodes"], "rows": len(rows), "failed": failures,
            "passed": document["passed"]}


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
            raise ValueError("repeated R22 result differs")
    receipt = {"schema": "r22-independent-protection-verification.v1",
               "plan_sha256": sha(PLAN), "lock_sha256": lock_sha,
               "first_result_sha256": sha(args.first), "repeat_result_sha256": repeat_sha,
               "exact_repeat": exact_repeat, "summary": summary}
    with args.out.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt["summary"], sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
