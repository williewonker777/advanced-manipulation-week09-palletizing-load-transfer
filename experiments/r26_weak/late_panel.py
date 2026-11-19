"""Recheck the disclosed late +/- weak panel with true active-sample occupancy."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json

from experiments.r24_holdout import evaluator as lab
from experiments.r25_composed.detector import Detector as R25
from experiments.r26_weak.candidate import Detector as Candidate


def main(args):
    base = json.loads(args.r24_plan.read_text())
    registered = json.loads(args.weak_plan.read_text())["heldback_development"]
    time_value = max(registered["times_s"])
    cases = []
    for vector in registered["force_vectors_N"]:
        for mass, com in registered["payloads"]:
            cases.append({"id": f"late_{len(cases):02d}", "phase": "measured_loaded",
                          "mass": mass, "com": com, "offset": None,
                          "impact": [time_value, vector]})
    plan = dict(base)
    plan["negative"], plan["paired"], plan["single"] = [], [], cases
    plan["variants"] = {key: base["variants"][key] for key in registered["mechanics"]}
    arms = {"r23_free": R25, "r24": Candidate}
    work = [("single", case, variant, arms, plan)
            for case in cases for variant in plan["variants"].values()]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lab.one, work))
    result = {"schema": "r26-weak-late-occupancy.v1", "rows_count": len(rows),
              "episodes": sum(row["episodes"] for row in rows),
              "passed_rows": sum(row["passed"] for row in rows),
              "strict_rows": sum(row["records"]["r24"]["event"]["strict"] for row in rows),
              "false_free_rows": sum(row["protection"]["candidate_zero_false_rises"]
                                      for row in rows), "rows": rows}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r24-plan", type=Path, required=True)
    parser.add_argument("--weak-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    main(parser.parse_args())
