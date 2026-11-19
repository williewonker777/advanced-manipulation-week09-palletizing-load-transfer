"""Targeted evaluation of the frozen support-certificate treatment."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json

from experiments.r23_holdout import evaluator as r23
from experiments.r24_holdout import evaluator as r24
from experiments.r25_holdout import evaluator as r25
from experiments.r25_composed.detector import Detector as R25
from experiments.r26_support_certificate.candidate import Detector as Candidate


def main(args):
    p23 = json.loads(args.r23_plan.read_text())
    p24 = json.loads(args.r24_plan.read_text())
    p25 = json.loads(args.r25_plan.read_text())
    arms24 = {"r23_free": R25, "r24": Candidate}
    arms25 = {"r24_joint": R25, "r25": Candidate}
    mid = next(row for row in p24["single"] if row["id"] == "s24_transfer_offaxis_mid")
    weak = next(row for row in p25["single"] if row["id"] == "s25_weak_signed_carry")
    cross = next(row for row in p25["paired"] if row["id"] == "p25_close_transfer_cross")
    rows = {
        "s24_mid": r24.single_case(mid, p24["variants"]["stock"], arms24, p24),
        "r25_weak": r25.single_case(weak, p25["variants"]["stock"], arms25, p25),
        "r25_cross": r25.paired_case(cross, p25["variants"]["stock"], arms25, p25),
    }
    disclosed = json.loads(args.disclosed.read_text())
    selected = {(row["case"], row["mechanics"])
                for row in disclosed["arms"]["r23_free"]["failed_rows"] if row["false_s"]}
    cases = {row["id"]: (kind, row) for kind in ("negative", "single", "paired")
             for row in p23[kind]}
    variants = {row["kind"]: row for row in p23["variants"].values()}
    work = [(cases[case][0], cases[case][1], variants[variant],
             {"r22_light": R25, "r23": Candidate}, p23)
            for case, variant in sorted(selected)]
    with ProcessPoolExecutor(args.workers) as pool:
        controls = list(pool.map(r23.one, work))
    result = {"schema": "r26-support-certificate-targeted.v1", "targets": rows,
              "controls": controls,
              "control_false_free": sum(row["safety_floor"].get(
                  "candidate_zero_false_rises", True) for row in controls),
              "control_rows": len(controls)}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"targets": {key: {"passed": row["passed"],
                                         "gains": row["positive_gains"],
                                         "misses": row.get("absolute_misses", [])}
                                        for key, row in rows.items()},
                      "control_false_free": result["control_false_free"],
                      "control_rows": result["control_rows"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r23-plan", type=Path, required=True)
    parser.add_argument("--r24-plan", type=Path, required=True)
    parser.add_argument("--r25-plan", type=Path, required=True)
    parser.add_argument("--disclosed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    main(parser.parse_args())
