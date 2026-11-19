"""Deterministic protection screen for the optional single-file R20 detector."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

import my_detector
import my_torque
from experiments.r20_horizontal.candidate_detector import Detector as Modular
from experiments.r21_pick_observer.candidate import Detector as ModularR21
from experiments.r21_yaw_impulse.coarse import Detector as ModularYaw
from sim.runner import run_episode
from tools.d4_robustness import build_cases
from tools.r5_metrics import event_result


ROOT = Path(__file__).resolve().parents[2]
PHYSICAL = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "f_ext")


def load(path):
    spec = importlib.util.spec_from_file_location(
        "_r21_optional_" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Detector


def selected_cases():
    rows = build_cases("my_torque")
    references = [r for r in rows if r["kind"] == "reference" and r["com"] == (0., 0., .015)]
    impacts = [r for r in rows if r["kind"] == "collision" and r["phase"] == "pick"
               and r["force_n"] in (10., 15., 40.)
               and r["direction"] in ("+x", "-x", "+y", "-y")]
    offsets = [r for r in rows if r["kind"] == "offset"]
    if [len(references), len(impacts), len(offsets)] != [7, 12, 16]:
        raise ValueError("frozen D4 screen selection differs")
    return references, impacts, offsets


def episode(detector_type, case, *, collision_marker=True):
    collision = case.get("collision") if collision_marker else None
    result, log = run_episode(
        my_torque.Controller(), box_mass=case["mass"], box_com=case["com"],
        detector=detector_type(), collision=collision, pick_offset=case.get("pick_offset"))
    return result, log


def exact_physics(left, right):
    return all(np.array_equal(left[key], right[key]) for key in PHYSICAL)


def paired_record(detector_type, case):
    force_result, force_log = episode(detector_type, case)
    reference_result, reference_log = episode(detector_type, case, collision_marker=False)
    return {
        "result": force_result,
        "event": event_result(force_log["t"], force_log["detect"], case,
                              force_log["f_tool"], reference_log["detect"]),
        "force_log": force_log,
        "reference_result": reference_result,
        "reference_event": event_result(reference_log["t"], reference_log["detect"],
                                        {**case, "collision": None}, reference_log["f_tool"]),
        "reference_log": reference_log,
    }


def compare_record(case, baseline, candidate):
    gates = {
        "force_physics_exact": exact_physics(baseline["force_log"], candidate["force_log"]),
        "reference_physics_exact": exact_physics(
            baseline["reference_log"], candidate["reference_log"]),
        "baseline_alarm_samples_preserved": bool(np.all(
            ~baseline["force_log"]["detect"] | candidate["force_log"]["detect"])),
        "no_reference_false_alarm_increase": (
            candidate["reference_event"]["false_alarms"]
            <= baseline["reference_event"]["false_alarms"]),
        "no_collision_trial_false_alarm_increase": (
            candidate["event"]["false_alarms"] <= baseline["event"]["false_alarms"]),
        "baseline_collision_detection_preserved": (
            baseline["event"]["outcome"] != "detected"
            or candidate["event"]["outcome"] == "detected"),
        "baseline_tool_detection_preserved": (
            not baseline["event"]["tool_detected"] or candidate["event"]["tool_detected"]),
        "no_new_tool_prealarm": (
            not candidate["event"]["tool_prealarm"] or baseline["event"]["tool_prealarm"]),
    }
    return {
        "case_id": case["case_id"], "kind": case["kind"], "mass": case["mass"],
        "com": case["com"], "collision": case.get("collision"),
        "pick_offset": case.get("pick_offset"),
        "baseline": {k: baseline["event"][k] for k in (
            "outcome", "delay_ms", "prealarm", "false_alarms",
            "tool_present", "tool_detected", "tool_prealarm", "tool_delay_ms")},
        "candidate": {k: candidate["event"][k] for k in (
            "outcome", "delay_ms", "prealarm", "false_alarms",
            "tool_present", "tool_detected", "tool_prealarm", "tool_delay_ms")},
        "gates": gates, "passed": all(gates.values()),
    }


def direct_record(case, baseline_type, candidate_type):
    left_result, left = episode(baseline_type, case)
    right_result, right = episode(candidate_type, case)
    return {
        "id": case["id"],
        "samples": len(left["t"]),
        "detect_exact": bool(np.array_equal(left["detect"], right["detect"])),
        "residual_exact": bool(np.array_equal(left["r"], right["r"])),
        "physical_exact": exact_physics(left, right),
        "delay_exact": left_result["detect_delay_ms"] == right_result["detect_delay_ms"],
        "false_exact": left_result["false_alarm_times"] == right_result["false_alarm_times"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", type=Path, required=True)
    parser.add_argument("--baseline-standalone", type=Path)
    parser.add_argument("--modular", choices=("r20", "r21", "yaw"), default="r20")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("new output required")
    candidate_type = load(args.standalone)
    baseline_type = load(args.baseline_standalone) if args.baseline_standalone else my_detector.Detector
    modular_type = {"r20": Modular, "r21": ModularR21, "yaw": ModularYaw}[args.modular]
    references, impacts, offsets = selected_cases()
    rows = []
    for case in (*references, *impacts, *offsets):
        if case["kind"] == "collision":
            baseline, candidate = paired_record(baseline_type, case), paired_record(candidate_type, case)
        else:
            b_result, b_log = episode(baseline_type, case)
            c_result, c_log = episode(candidate_type, case)
            b_event = event_result(b_log["t"], b_log["detect"], case, b_log["f_tool"])
            c_event = event_result(c_log["t"], c_log["detect"], case, c_log["f_tool"])
            baseline = {"force_log": b_log, "reference_log": b_log, "event": b_event,
                        "reference_event": b_event}
            candidate = {"force_log": c_log, "reference_log": c_log, "event": c_event,
                         "reference_event": c_event}
        row = compare_record(case, baseline, candidate)
        rows.append(row)
        print(case["case_id"], row["passed"], flush=True)

    parity_cases = []
    for mass in (0., 8.):
        parity_cases.append({"id": f"normal-{mass:g}", "mass": mass,
                             "com": (.02, -.01, .03), "collision": None,
                             "pick_offset": None})
        for force in (10., 15.):
            parity_cases.append({"id": f"early-{mass:g}-{force:g}", "mass": mass,
                                 "com": (.02, -.01, .03),
                                 "collision": (1.42, (0., force, 0.)), "pick_offset": None})
    parity = [direct_record(case, modular_type, candidate_type) for case in parity_cases]
    gains = [row["case_id"] for row in rows
             if row["baseline"]["outcome"] != "detected"
             and row["candidate"]["outcome"] == "detected"]
    failures = [{"case_id": row["case_id"],
                 "gates": [name for name, ok in row["gates"].items() if not ok]}
                for row in rows if not row["passed"]]
    result = {
        "schema": "r21-r20-release-protection.v1",
        "classification": "reused exposed D4 subset, not independent holdout",
        "standalone_sha256": hashlib.sha256(args.standalone.read_bytes()).hexdigest(),
        "baseline_standalone_sha256": (hashlib.sha256(args.baseline_standalone.read_bytes()).hexdigest()
                                         if args.baseline_standalone else None),
        "modular_reference": args.modular,
        "episodes": 2 * (len(references) + len(offsets)) + 4 * len(impacts)
                    + 2 * len(parity),
        "selection": {"references": len(references), "impacts": len(impacts),
                      "offsets": len(offsets), "parity": len(parity)},
        "rows": rows, "parity": parity, "candidate_only_detections": gains,
        "failed": failures,
        "gates": {
            "all_D4_subset_protection": not failures,
            "all_modular_standalone_parity": all(
                all(value for key, value in row.items() if key != "id") for row in parity),
        },
    }
    result["passed"] = all(result["gates"].values())
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": result["episodes"], "failed": len(failures),
                      "gains": gains, "passed": result["passed"]}))


if __name__ == "__main__":
    main()
