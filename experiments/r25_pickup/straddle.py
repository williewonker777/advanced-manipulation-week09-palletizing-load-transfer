"""Diagnose frozen H1 on preregistered pickup-straddling second inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r24_composed.detector import Detector as R24
from experiments.r24_holdout import evaluator
from experiments.r25_pickup.candidate import Detector as R25


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "experiments/r24_holdout/plan.json"
ARMS = {"r24": R24, "r25": R25}
SECOND = [
    ("same_sign_immediate", (1.401, [5.4, -6.0, 2.2])),
    ("opposite_sign_immediate", (1.401, [-5.4, 6.0, -2.2])),
    ("same_sign_late_consumption", (1.405, [5.4, -6.0, 2.2])),
    ("opposite_sign_late_consumption", (1.405, [-5.4, 6.0, -2.2])),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(bits):
    values = np.asarray(bits, bool)
    rises = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    falls = np.flatnonzero(~values & np.r_[False, values[:-1]])
    if len(values) and values[-1]:
        falls = np.r_[falls, len(values)]
    return [[int(a), int(b)] for a, b in zip(rises, falls, strict=True)]


def probe(base):
    class Probe(base):
        def reset(self):
            super().reset()
            self.straddle_trace = []

        def update(self, t, th, thd, tau_m, tau_s, grip):
            residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
            index = int(round(1000 * float(t)))
            if 1385 <= index <= 1435:
                packet = self._shared
                normal = getattr(self, "normal_model", None)
                self.straddle_trace.append({
                    "index": index, "output": bool(report), "grip": bool(grip),
                    "r10": bool(packet.detected or
                                (normal is not None and normal.detected)),
                    "horizontal": bool(self.horizontal_detected),
                    "yaw": bool(self.yaw_active),
                    "yaw_value_Nm": (None if self.yaw_value is None
                                     else float(self.yaw_value)),
                    "yaw_bound_Nm": float(self.bounds[int(bool(grip))]),
                    "r24_latch": bool(getattr(self, "pick_event_active", False)),
                    "completion_active": bool(
                        getattr(self, "pick_completion_active", False)),
                    "completion_selected": bool(
                        getattr(self, "pick_completion_selected", False)),
                    "completion_reason": getattr(self, "pick_completion_reason", None),
                })
            return residual, report

    return Probe


def run(case, variant, base, impacts):
    detector = probe(base)()
    _, log = evaluator.run_episode(case, variant, lambda: detector, impacts)
    return detector, log


def event(log, references, start, plan):
    return evaluator.event_record(log, references, start,
                                  min(6000, start + plan["event_window_ms"]),
                                  plan["quiet_ms"], plan["reference_twin_ms"])


def row(case, variant, second_id, second_impact, plan):
    assignment = {"none": (None, None), "first": (case["impact"], None),
                  "second": (None, second_impact), "both": (case["impact"], second_impact)}
    traces = {name: {twin: run(case, variant, base, impacts)
                     for twin, impacts in assignment.items()}
              for name, base in ARMS.items()}
    t = traces["r24"]["both"][1]["t"]
    first = evaluator.certified.actual_start(t, case["impact"])[0]
    second = evaluator.certified.actual_start(t, second_impact)[0]
    records = {}
    for name, twins in traces.items():
        detect_first = np.asarray(twins["first"][1]["detect"], bool)
        detect_both = np.asarray(twins["both"][1]["detect"], bool)
        stop = min(6000, second + plan["event_window_ms"])
        attributable_active = np.flatnonzero(
            detect_both[second:stop] & ~detect_first[second:stop]) + second
        records[name] = {
            "isolated_second": event(twins["second"][1], [twins["none"][1]],
                                     second, plan),
            "both_second": event(twins["both"][1], [twins["first"][1]],
                                 second, plan),
            "attributable_active_indices_both_vs_first": attributable_active.tolist(),
            "twins": {twin: {
                "output_intervals": intervals(pair[1]["detect"]),
                "force_intervals": intervals(np.any(
                    np.asarray(pair[1]["f_ext_xyz_committed"]) != 0.0, axis=1)),
                "force_exact": evaluator.certified.exact_force(pair[1], assignment[twin]),
                "physical_sha256": evaluator.certified.physical_hash(pair[1]),
                "trace_1385_1435": pair[0].straddle_trace,
            } for twin, pair in twins.items()},
        }
    for twin in assignment:
        if records["r24"]["twins"][twin]["physical_sha256"] != \
                records["r25"]["twins"][twin]["physical_sha256"]:
            raise ValueError("physical trace differs by detector")
    exact = all(item["force_exact"]["passed"] for arm in records.values()
                for item in arm["twins"].values())
    r24_attr = bool(records["r24"]["attributable_active_indices_both_vs_first"])
    r25_attr = bool(records["r25"]["attributable_active_indices_both_vs_first"])
    return {
        "id": second_id, "first_actual": first, "second_actual": second,
        "second_impact": second_impact, "records": records,
        "diagnosis": {
            "exact_force_and_physics": exact,
            "r24_has_attributable_output_for_second": r24_attr,
            "r25_has_attributable_output_for_second": r25_attr,
            "h1_lost_attributable_r24_output": r24_attr and not r25_attr,
            "both_detectors_lack_attributable_output": not r24_attr and not r25_attr,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    plan = json.loads(PLAN.read_bytes())
    case = next(item for item in plan["single"]
                if item["id"] == "s24_transfer_offaxis_early")
    rows = [row(case, plan["variants"]["stock"], name, impact, plan)
            for name, impact in SECOND]
    doc = {
        "schema": "r25-pickup-straddling-second-diagnosis.v1",
        "classification": "Preregistered disclosed-case mechanism test; not independent R25 evidence",
        "candidate_sha256": sha(ROOT / "experiments/r25_pickup/candidate.py"),
        "episodes": len(rows) * len(ARMS) * 4,
        "rows": rows,
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({row["id"]: row["diagnosis"] for row in rows}, indent=2))


if __name__ == "__main__":
    main()
