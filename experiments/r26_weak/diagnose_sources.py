"""Post-rejection diagnostic of real weak pulses and the 3.855 s nuisance twins."""
from pathlib import Path
import argparse
import json

import numpy as np

from experiments.r23_holdout import evaluator as r23
from experiments.r24_holdout import evaluator as r24
from experiments.r25_holdout import evaluator as r25
from experiments.r26_weak.candidate import Detector


class Observed(Detector):
    def reset(self):
        super().reset()
        self.r26_trace = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        before = self.r26_lifecycle.phase
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        packet = self._shared
        ctrl = self.ctrl
        self.r26_trace.append({
            "t": float(t), "phase_before": before, "phase": self.r26_lifecycle.phase,
            "score_n": float(self.r26_force.last_score),
            "vector_n": self.r26_force.last_vector.tolist(),
            "grip": bool(grip), "report": bool(report),
            "owned": bool(self.r26_lifecycle.owned),
            "load_fraction": float(packet.load_fraction),
            "lambda_support": float(packet.lambda_support),
            "lambda_support_valid": bool(packet.lambda_support_valid),
            "table_normal_n": float(packet.last_table_wrench[2]),
            "table_allowed": bool(packet.last_table_allowed),
            "hybrid_alpha": float(getattr(ctrl, "hybrid_alpha", 0.0)),
            "commanded_load": float(getattr(ctrl, "commanded_load", 0.0)),
            "command_norm": float(np.linalg.norm(tau_m)),
            "jts_norm": float(np.linalg.norm(tau_s)),
        })
        return residual, report


def episodes(trace):
    answer = []
    start = None
    for i, row in enumerate(trace):
        if row["phase_before"] == "quiet" and row["phase"] == "active":
            start = i
        if start is not None and row["phase_before"] == "active" and row["phase"] == "rearm":
            block = trace[start:i + 1]
            peak = max(range(len(block)), key=lambda j: block[j]["score_n"])
            fields = ("score_n", "load_fraction", "lambda_support", "table_normal_n",
                      "hybrid_alpha", "commanded_load", "command_norm", "jts_norm")
            answer.append({
                "start_tick": start, "end_tick": i,
                "start_s": block[0]["t"], "end_s": block[-1]["t"],
                "duration_samples": len(block), "owned": bool(block[-1]["owned"]),
                "release_reported": any(row["report"] for row in trace[i:i + 8]),
                "grip_true_samples": sum(row["grip"] for row in block),
                "table_allowed_samples": sum(row["table_allowed"] for row in block),
                "support_valid_samples": sum(row["lambda_support_valid"] for row in block),
                "peak_offset": peak,
                "start": {key: block[0][key] for key in fields},
                "peak": {key: block[peak][key] for key in fields},
                "end": {key: block[-1][key] for key in fields},
                "maximum": {key: max(row[key] for row in block) for key in fields},
            })
            start = None
    return answer


def run(factory, case, variant, impacts):
    detector = Observed()
    _, log = factory(case, variant, lambda: detector, impacts)
    return {"episodes": episodes(detector.r26_trace),
            "detect_rises": np.flatnonzero(np.asarray(log["detect"], bool) &
                ~np.r_[False, np.asarray(log["detect"], bool)[:-1]]).astype(int).tolist()}


def main(args):
    p23 = json.loads(args.r23_plan.read_text())
    p24 = json.loads(args.r24_plan.read_text())
    p25 = json.loads(args.r25_plan.read_text())
    v23 = next(row for row in p23["variants"].values() if row["kind"] == "actuator_skew")
    negative = {}
    for case_id in ("p23_post_detach_diagonal_pair", "s23_post_detach_radial"):
        case = next(row for kind in ("single", "paired") for row in p23[kind]
                    if row["id"] == case_id)
        negative[case_id] = run(r23.run_episode, case, v23, (None, None))

    cross = next(row for row in p25["paired"] if row["id"] == "p25_close_transfer_cross")
    first, second = cross["impacts"]
    real_cross = {
        "isolated_second": run(r25.run_episode, cross, p25["variants"]["stock"],
                               (None, second)),
        "both_second": run(r25.run_episode, cross, p25["variants"]["stock"],
                           (first, second)),
    }
    mid = next(row for row in p24["single"] if row["id"] == "s24_transfer_offaxis_mid")
    real_mid = run(r24.run_episode, mid, p24["variants"]["stock"], (mid["impact"], None))
    result = {"schema": "r26-weak-source-diagnostic.v1",
              "classification": "post-v1-rejection diagnostic only; candidate unchanged",
              "negative_3855": negative, "real_cross": real_cross, "real_s24_mid": real_mid}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r23-plan", type=Path, required=True)
    parser.add_argument("--r24-plan", type=Path, required=True)
    parser.add_argument("--r25-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
