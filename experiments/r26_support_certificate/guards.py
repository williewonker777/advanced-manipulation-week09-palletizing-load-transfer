"""Bounded old-late, long-input, and signed support-contact guards."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json

import numpy as np

from experiments.r24_holdout import evaluator as r24
from experiments.r25_holdout import evaluator as r25
from experiments.r25_composed.detector import Detector as R25
from experiments.r26_support_certificate.candidate import Detector as Candidate


class Observed(Candidate):
    def reset(self):
        super().reset()
        self.support_trace = []

    def update(self, *inputs):
        residual, report = super().update(*inputs)
        self.support_trace.append((float(self._shared.last_table_wrench[2]),
                                   bool(self._shared.last_table_allowed),
                                   self.r26_certificate.last_lower_bound,
                                   self.r26_certificate.last_maximum_normal))
        return residual, report


def late_plan(base, weak):
    registered = weak["heldback_development"]
    cases = []
    for vector in registered["force_vectors_N"]:
        for mass, com in registered["payloads"]:
            cases.append({"id": f"late_{len(cases):02d}", "phase": "measured_loaded",
                          "mass": mass, "com": com, "offset": None,
                          "impact": [max(registered["times_s"]), vector]})
    plan = dict(base)
    plan["negative"], plan["paired"], plan["single"] = [], [], cases
    plan["variants"] = {key: base["variants"][key] for key in registered["mechanics"]}
    return plan


def main(args):
    p24 = json.loads(args.r24_plan.read_text())
    p25 = json.loads(args.r25_plan.read_text())
    weak = json.loads(args.weak_plan.read_text())
    plan = late_plan(p24, weak)
    arms24 = {"r23_free": R25, "r24": Candidate}
    work = [("single", case, variant, arms24, plan)
            for case in plan["single"] for variant in plan["variants"].values()]
    with ProcessPoolExecutor(args.workers) as pool:
        late = list(pool.map(r24.one, work))

    long_case = next(row for row in p25["single"] if row["id"] == "s25_long_transfer_diag")
    long_row = r25.single_case(long_case, p25["variants"]["stock"],
                              {"r24_joint": R25, "r25": Candidate}, p25)

    source = next(row for row in p24["single"] if row["id"] == "s24_detached_offaxis_b")
    contact_spec = json.loads(args.contact_panel.read_text())
    contact = []
    for index, impact in enumerate(contact_spec["impacts"]):
        case = dict(source, id=f"support_contact_{index}", impact=impact)
        row = r24.single_case(case, p24["variants"]["stock"], arms24, p24)
        detector = Observed()
        _, log = r24.run_episode(case, p24["variants"]["stock"], lambda: detector,
                                 (impact, None))
        start = r24.certified.actual_start(log["t"], impact)[0]
        values = np.asarray(detector.support_trace)[start:start + p24["event_window_ms"]]
        row["measured_support"] = {
            "maximum_table_normal_n": float(np.max(values[:, 0])),
            "table_allowed_samples": int(np.count_nonzero(values[:, 1])),
            "maximum_robust_lower_bound_n": float(np.max(values[:, 2])),
            "maximum_normal_used_by_certificate_n": float(np.max(values[:, 3])),
        }
        contact.append(row)
    result = {"schema": "r26-support-certificate-guards.v1", "late": late,
              "long_transfer": long_row, "support_contact": contact,
              "summary": {"late_rows": len(late),
                          "late_passed": sum(row["passed"] for row in late),
                          "late_false_free": sum(row["protection"]["candidate_zero_false_rises"]
                                                 for row in late),
                          "long_passed": long_row["passed"],
                          "contact_passed": sum(row["passed"] for row in contact)}}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r24-plan", type=Path, required=True)
    parser.add_argument("--r25-plan", type=Path, required=True)
    parser.add_argument("--weak-plan", type=Path, required=True)
    parser.add_argument("--contact-panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    main(parser.parse_args())
