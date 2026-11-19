"""Neutral public R10-versus-early reproduction on stock simulator cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


import my_detector
import r1_torque
from experiments.r14_stock_handoff.detector import Detector
from experiments.r14_stock_handoff.qualification import compact
from experiments.r15_early_com.controller import Controller
from sim.runner import run_episode
from tools.r11_placement import score
from tools.r14_paired_events import match_false_events


GROUPS = (
    ("ec1", 3.45, (0.019, -0.014, 0.021), (0.0058, -0.0065, 0.0044)),
    ("ec2", 4.90, (-0.022, 0.018, -0.013), (-0.0062, -0.0048, 0.0052)),
    ("ec3", 6.55, (0.016, 0.023, -0.020), (0.0072, 0.0054, 0.0042)),
    ("ec4", 7.75, (-0.015, -0.020, 0.017), (-0.0054, 0.0076, 0.0055)),
)
VARIANTS = ("none", "explicit_zero", "xy", "up", "down")
EXPOSED = (
    {"id": "sh3", "mass": 7.10, "com": [-0.016, -0.019, 0.027],
     "offset": [0.005, 0.007, 0.0]},
    {"id": "sh5", "mass": 6.65, "com": [-0.025, 0.018, -0.020],
     "offset": [0.004, -0.006, 0.0048]},
)


def cases():
    rows = []
    for identifier, mass, com, shift in GROUPS:
        x, y, z = shift
        offsets = {
            "none": None,
            "explicit_zero": [0.0, 0.0, 0.0],
            "xy": [x, y, 0.0],
            "up": [x, y, z],
            "down": [x, y, -z],
        }
        rows.extend({"id": f"{identifier}:{name}", "mass": mass,
                     "com": list(com), "offset": offsets[name]}
                    for name in VARIANTS)
    return rows + [dict(row) for row in EXPOSED]


def ratios(candidate, baseline):
    return {key: float(value / max(baseline["cost"][key], 1e-12))
            for key, value in candidate["cost"].items()}


def episode(kind, case, height, early):
    controller = (r1_torque.Controller() if kind == "r10"
                  else Controller(height, early))
    detector = my_detector.Detector() if kind == "r10" else Detector()
    metrics, log = run_episode(controller, case["mass"], tuple(case["com"]),
                               detector, pick_offset=case["offset"])
    record = compact(metrics, log, controller, case["mass"])
    if kind == "early":
        record["early"] = controller.early_result
        record["early_selected_at_s"] = controller.early_decided_at_s
        record["early_updates"] = controller.early_updates
    return record


def compare(case, height, early):
    reference = episode("r10", case, height, early)
    candidate = episode("early", case, height, early)
    cost = ratios(candidate, reference)
    g4 = score(reference, candidate)
    false_events = match_false_events(candidate["false_alarm_times"],
                                      reference["false_alarm_times"])
    tool = (reference["tool_delay_ms"] is None or
            candidate["tool_delay_ms"] is not None and
            candidate["tool_delay_ms"] <= reference["tool_delay_ms"] + 5.0 + 1e-9)
    gates = {
        "status": reference["status"] == candidate["status"] == "ok",
        "G4": bool(g4["all_four_g4"]),
        "cost_r10": max(cost.values()) <= 1.05,
        "saturation": candidate["saturation_pct"] <= reference["saturation_pct"] + 1e-12,
        "false_event_matching": bool(false_events["passed"]),
        "tool": bool(tool),
        "controller_p99_below_800us": candidate["ctrl_p99_us"] <= 800.0,
    }
    return {
        "id": case["id"], "case": case,
        "arms": {"r10": reference, "early": candidate},
        "cost_ratios": cost, "g4": g4, "false_event_match": false_events,
        "gates": gates, "passed": all(gates.values()),
    }


def summarize(rows):
    matrix = [row for row in rows if row["id"].startswith("ec")]
    failures = [{"id": row["id"],
                 "gates": [key for key, value in row["gates"].items() if not value]}
                for row in rows if not row["passed"]]
    return {
        "cases": len(rows), "matrix_cases": len(matrix),
        "g4_cases": sum(row["gates"]["G4"] for row in rows),
        "matrix_g4_cases": sum(row["gates"]["G4"] for row in matrix),
        "matrix_all_gates": all(row["passed"] for row in matrix),
        "qualified": all(row["passed"] for row in rows),
        "failures": failures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", required=True, type=Path)
    parser.add_argument("--early", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    height = json.loads(args.height.read_text())
    early = json.loads(args.early.read_text())
    rows = [compare(case, height, early) for case in cases()]
    result = {
        "schema": "r15-portable-early-stock.v1", "rows": rows,
        "summary": summarize(rows), "external_pulses": 0,
        "threshold_or_delay_changes": False,
    }
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
