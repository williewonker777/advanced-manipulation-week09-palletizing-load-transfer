"""Run the frozen R26 lifecycle on the two disclosed mechanism targets."""
from pathlib import Path
import argparse
import json

from experiments.r25_composed.detector import Detector as R25
from experiments.r25_holdout import evaluator as lab
from experiments.r26_weak.candidate import Detector as Candidate


def main(args):
    plan = json.loads(args.plan.read_text())
    variant = plan["variants"]["stock"]
    single = next(row for row in plan["single"] if row["id"] == "s25_weak_signed_carry")
    paired = next(row for row in plan["paired"] if row["id"] == "p25_close_transfer_cross")
    arms = {"r24_joint": R25, "r25": Candidate}
    rows = [lab.single_case(single, variant, arms, plan),
            lab.paired_case(paired, variant, arms, plan)]
    result = {"schema": "r26-weak-targeted.v1", "episodes": sum(r["episodes"] for r in rows),
              "passed_rows": sum(r["passed"] for r in rows), "rows": rows}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": result["episodes"], "passed_rows": result["passed_rows"],
                      "rows": [{"id": row["id"], "passed": row["passed"],
                                "gains": row["positive_gains"],
                                "misses": row["absolute_misses"],
                                "failed_protection": [k for k, v in row[
                                    "incumbent_safety"].items() if not v]}
                               for row in rows]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
