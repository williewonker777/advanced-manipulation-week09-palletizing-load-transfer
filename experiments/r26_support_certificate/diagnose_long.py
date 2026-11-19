"""Post-rejection lifecycle segmentation of the disclosed long transfer input."""
from pathlib import Path
import argparse
import json

import numpy as np

from experiments.r25_holdout import evaluator as lab
from experiments.r26_support_certificate.candidate import Detector


class Observed(Detector):
    def reset(self):
        super().reset()
        self.trace = []

    def update(self, *inputs):
        before = self.r26_cert_lifecycle.phase
        residual, report = super().update(*inputs)
        life = self.r26_cert_lifecycle
        self.trace.append({"phase_before": before, "phase": life.phase,
                           "score_n": self.r26_cert_force.last_score,
                           "raw_n": self.r26_cert_force.last_raw.tolist(),
                           "corrected_n": self.r26_cert_force.last_vector.tolist(),
                           "lower_bound_n": self.r26_certificate.last_lower_bound,
                           "maximum_normal_n": self.r26_certificate.last_maximum_normal,
                           "direction": None if life.direction is None else life.direction.tolist(),
                           "owned": life.owned, "report": bool(report)})
        return residual, report


def main(args):
    plan = json.loads(args.plan.read_text())
    case = next(row for row in plan["single"] if row["id"] == "s25_long_transfer_diag")
    detector = Observed()
    _, log = lab.run_episode(case, plan["variants"]["stock"], lambda: detector,
                             (case["impact"], None))
    starts, ends, quiet = [], [], []
    for tick, row in enumerate(detector.trace):
        if row["phase_before"] == "quiet" and row["phase"] == "active":
            starts.append({"tick": tick, **row})
        if row["phase_before"] == "active" and row["phase"] == "rearm":
            ends.append({"tick": tick, **row})
        if row["phase_before"] == "rearm" and row["phase"] == "quiet":
            quiet.append({"tick": tick, **row})
    bits = np.asarray(log["detect"], bool)
    rises = np.flatnonzero(bits & ~np.r_[False, bits[:-1]]).astype(int).tolist()
    result = {"schema": "r26-support-long-segmentation.v1",
              "classification": "post-rejection diagnostic; frozen candidate unchanged",
              "case": case, "starts": starts, "ends": ends, "rearmed": quiet,
              "detect_rises": rises,
              "start_to_start_gaps": [b["tick"] - a["tick"] for a, b in zip(starts, starts[1:])],
              "end_to_next_start_gaps": [b["tick"] - a["tick"]
                                         for a, b in zip(ends, starts[1:])]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
