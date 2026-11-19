"""Evaluate the frozen world-horizontal candidate on preregistered held-back development."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import importlib
import json

from experiments.r24_composed.detector import Detector as R24
from experiments.r24_holdout import evaluator as lab
from experiments.r25_weak.nearby_plan import heldback_plan, tasks


def constructor(key):
    module, name = key.split(":", 1)
    return getattr(importlib.import_module(module), name)


def main(args):
    registered = json.loads(args.registered.read_text())
    candidate = constructor(args.candidate)
    plan = heldback_plan(json.loads(args.r24_plan.read_text()), registered)
    work = tasks(lab, plan, {"r23_free": R24, "r24": candidate})
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lab.one, work))
    original = registered["heldback_development"]["force_vectors_N"][0]
    summary = {"rows": len(rows), "episodes": sum(row["episodes"] for row in rows),
               "passed_rows": sum(row["passed"] for row in rows),
               "gained_rows": sum(bool(row["positive_gains"]) for row in rows),
               "strict_rows": sum(row["scoped_detection"]["passed"] for row in rows),
               "false_free_rows": sum(row["protection"]["candidate_zero_false_rises"]
                                      for row in rows),
               "false_active_free_rows": sum(row["protection"][
                   "candidate_zero_false_active_samples"] for row in rows)}
    directions = {"original": [], "opposite": []}
    variants = len(plan["variants"])
    for offset, case in enumerate(plan["single"]):
        sign = ("original" if sum(a * b for a, b in zip(case["impact"][1], original)) > 0
                else "opposite")
        directions[sign].extend(rows[offset * variants + j]["scoped_detection"]["passed"]
                                for j in range(variants))
    result = {"schema": "r25-weak-world-horizontal-heldback.v1", "summary": summary,
              "direction_strict": {key: sum(values) for key, values in directions.items()},
              "direction_total": {key: len(values) for key, values in directions.items()},
              "rows": rows}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered", type=Path, required=True)
    parser.add_argument("--r24-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", default="experiments.r25_weak.normal_force_detector:Detector")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    main(parser.parse_args())
