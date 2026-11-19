"""Reusable targeted D4/R23 guard for a formally frozen detector entrypoint."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from experiments.r22_census import check
from experiments.r23_holdout import evaluator as r23
from experiments.r24_loaded.d4 import load_cases
from experiments.r25_composed.detector import Detector as R25
from experiments.r26_history.d4 import occupied


def resolve(spec):
    module, symbol = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def d4_case(candidate, case_id):
    cases = load_cases()
    case = cases[case_id]
    reference = next(row for row in cases if row["kind"] == "reference" and
                     check.old.reference_key(row) == check.old.reference_key(case))
    old, new = check.old.episode_record(R25, case), check.old.episode_record(candidate, case)
    old_ref = check.old.episode_record(R25, reference)
    new_ref = check.old.episode_record(candidate, reference)
    old_score, new_score = check.score(case, old, old_ref), check.score(case, new, new_ref)
    gates = check.compare(new_score, old_score)
    gates.update({
        "latency_plus_5_ms": (not old_score["strict"] or
                              (new_score["strict"] and
                               new_score["delay_ms"] <= old_score["delay_ms"] + 5)),
        "duration_plus_5_samples": (not old_score["strict"] or
                                    occupied(new) <= occupied(old) + 5),
        "physical_exact": (old["physical_sha256"] == new["physical_sha256"] and
                           old_ref["physical_sha256"] == new_ref["physical_sha256"]),
    })
    return {"case_id": case_id, "identity": case, "r25": old_score,
            "candidate": new_score, "r25_active": occupied(old),
            "candidate_active": occupied(new), "r25_intervals": old["detect_intervals"],
            "candidate_intervals": new["detect_intervals"], "gates": gates,
            "passed": all(gates.values())}


def r23_case(candidate, case_id):
    plan = json.loads(r23.PLAN.read_bytes())
    case = next(row for row in plan["paired"] if row["id"] == case_id)
    rows = [r23.paired_case(case, variant, {"r22_light": R25, "r23": candidate}, plan)
            for variant in plan["variants"].values()]
    exact = all(row["records"]["r22_light"]["events"] == row["records"]["r23"]["events"]
                and all(row["safety_floor"].values()) for row in rows)
    return {"case_id": case_id, "rows": rows,
            "exact_r25_event_and_safety_preservation": exact, "passed": exact}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--d4-case", type=int, default=299)
    parser.add_argument("--r23-case", default="p23_transfer_to_post_detach")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    candidate = resolve(args.candidate)
    d4 = d4_case(candidate, args.d4_case)
    transfer = r23_case(candidate, args.r23_case)
    document = {"schema": "r26-targeted-history-adapter.v1",
                "candidate": args.candidate, "d4": d4, "r23": transfer,
                "passed": d4["passed"] and transfer["passed"]}
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"d4_passed": d4["passed"], "r23_passed": transfer["passed"],
                      "passed": document["passed"]}, indent=2))
    if not document["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
