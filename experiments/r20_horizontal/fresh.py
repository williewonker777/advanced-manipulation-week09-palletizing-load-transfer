"""Post-freeze four-trial assay for the R20 horizontal candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import my_detector
from experiments.r20_horizontal import candidate_detector
from my_torque import Controller
from sim.runner import COLLISION_HOLD, COLLISION_RAMP, run_episode


PHYSICAL_KEYS = ("t", "q", "q_d", "tau", "e_mm", "f_table",
                 "f_tool", "f_ext", "qpos")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rises(log):
    detected = np.asarray(log["detect"], bool)
    index = np.flatnonzero(detected & ~np.r_[False, detected[:-1]])
    return [float(log["t"][k]) for k in index]


def same_logs(left, right):
    return {key: bool(np.array_equal(left[key], right[key]))
            for key in PHYSICAL_KEYS}


def unmatched(candidate_times, baseline_times, tolerance=0.0011):
    return [float(t) for t in candidate_times
            if not any(abs(float(t) - float(ref)) <= tolerance
                       for ref in baseline_times)]


def tool_outcome(log):
    force = np.asarray(log["f_tool"]) > 1.0
    detect = np.asarray(log["detect"], bool)
    if not force.any():
        return {"present": False, "detected": False, "prealarm": False}
    first = int(np.argmax(force)); last = int(np.flatnonzero(force)[-1])
    stop = min(len(force), last + 300)
    return {"present": True, "detected": bool(detect[first:stop].any()),
            "prealarm": bool(detect[:first].any())}


def twin_at(reference, hit_time, radius_ms):
    if hit_time is None:
        return False
    t = np.asarray(reference["t"])
    detect = np.asarray(reference["detect"], bool)
    k = int(np.searchsorted(t, hit_time))
    radius = int(radius_ms)
    return bool(detect[max(0, k - radius):min(len(detect), k + radius + 1)].any())


def classify(result, force_log, reference_log, collision_time, radius_ms):
    delay = result["detect_delay_ms"]
    hit_time = None if delay is None else collision_time + delay / 1000.0
    twin = twin_at(reference_log, hit_time, radius_ms)
    if result["alarm_before_collision"]:
        official = "alarm_before"
    elif delay is None:
        official = "missed"
    elif twin:
        official = "indistinguishable"
    else:
        official = "detected"
    end = collision_time + 2 * COLLISION_RAMP + COLLISION_HOLD + 0.2
    strict = bool(hit_time is not None and not result["alarm_before_collision"]
                  and not twin and collision_time <= hit_time < end)
    return {"official": official, "strict_true_positive": strict,
            "delay_ms": delay, "alarm_before": result["alarm_before_collision"],
            "reference_twin": twin, "false_rise_times": result["false_alarm_times"],
            "all_rise_times": rises(force_log)}


def run(detector_type, case, collision):
    result, log = run_episode(
        Controller(), box_mass=case["mass"], box_com=case["com"],
        detector=detector_type(), collision=collision)
    return result, log


def main(plan_path, out_path):
    raw = Path(plan_path).read_bytes(); plan = json.loads(raw)
    source = Path(candidate_detector.__file__)
    actual_candidate = sha256(source)
    if actual_candidate != plan["candidate_sha256"]:
        raise RuntimeError("candidate source differs from frozen plan")
    radius = int(plan["reference_twin_ms"])
    output = {"schema": "R20_HORIZONTAL_FRESH_RESULT.v1",
              "plan_sha256": hashlib.sha256(raw).hexdigest(),
              "candidate_sha256": actual_candidate, "cases": []}
    for case in plan["cases"]:
        collision = (case["time"], case["force"])
        base_none, base_none_log = run(my_detector.Detector, case, None)
        cand_none, cand_none_log = run(candidate_detector.Detector, case, None)
        base_force, base_force_log = run(my_detector.Detector, case, collision)
        cand_force, cand_force_log = run(candidate_detector.Detector, case, collision)
        base_class = classify(base_force, base_force_log, base_none_log,
                              case["time"], radius)
        cand_class = classify(cand_force, cand_force_log, cand_none_log,
                              case["time"], radius)
        none_equal = same_logs(base_none_log, cand_none_log)
        force_equal = same_logs(base_force_log, cand_force_log)
        output["cases"].append({
            "case": case,
            "no_force": {
                "r10_false_rise_times": base_none["false_alarm_times"],
                "candidate_false_rise_times": cand_none["false_alarm_times"],
                "r10_all_rise_times": rises(base_none_log),
                "candidate_all_rise_times": rises(cand_none_log),
                "candidate_unmatched_false_rises": unmatched(
                    cand_none["false_alarm_times"], base_none["false_alarm_times"]),
                "r10_tool": tool_outcome(base_none_log),
                "candidate_tool": tool_outcome(cand_none_log),
            },
            "force": {"r10": base_class, "candidate": cand_class,
                      "candidate_unmatched_false_rises": unmatched(
                          cand_force["false_alarm_times"], base_force["false_alarm_times"]),
                      "r10_tool": tool_outcome(base_force_log),
                      "candidate_tool": tool_outcome(cand_force_log)},
            "none_log_equal": none_equal, "force_log_equal": force_equal,
            "none_all_physical_equal": all(v for k, v in none_equal.items() if k != "f_ext"),
            "force_all_physical_equal": all(v for k, v in force_equal.items() if k != "f_ext"),
            "none_applied_force_equal": none_equal["f_ext"],
            "force_applied_force_equal": force_equal["f_ext"],
        })
    rows = output["cases"]
    output["gates"] = {
        "source_pin": actual_candidate == plan["candidate_sha256"],
        "physical_and_force_equal": all(
            r["none_all_physical_equal"] and r["force_all_physical_equal"]
            and r["none_applied_force_equal"] and r["force_applied_force_equal"]
            for r in rows),
        "no_extra_no_force_false": all(
            not r["no_force"]["candidate_unmatched_false_rises"] for r in rows),
        "no_extra_force_false": all(
            not r["force"]["candidate_unmatched_false_rises"] for r in rows),
        "no_strict_baseline_loss": all(
            not r["force"]["r10"]["strict_true_positive"]
            or r["force"]["candidate"]["strict_true_positive"] for r in rows),
    }
    output["passed"] = all(output["gates"].values())
    Path(out_path).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(); main(args.plan, args.out)
