"""Fresh original 29-episode torque evaluator records for R25 versus R26."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import importlib
import json

import numpy as np

from my_torque import Controller
from sim.runner import run_episode
from experiments.r24_holdout import evaluator as certified_lab


R25 = "experiments.r25_composed.detector:Detector"
R26 = "experiments.r26_long.candidate:Detector"
EXPECTED_R26 = "a0570079ad5ce8425e9fbdc291b20401a9056b3bca35824237d157ec37a73057"


def constructor(key):
    module, name = key.split(":", 1)
    return getattr(importlib.import_module(module), name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(bits):
    bits = np.asarray(bits, bool)
    starts = np.flatnonzero(bits & ~np.r_[False, bits[:-1]])
    ends = np.flatnonzero(bits & ~np.r_[bits[1:], False]) + 1
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def physical_hash(log):
    keys = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "f_ext")
    return {key: hashlib.sha256(np.ascontiguousarray(log[key]).tobytes()).hexdigest()
            for key in keys}


def capture(job):
    arm, role, mass, collision = job
    metrics, log = run_episode(Controller(), box_mass=mass,
                               detector=constructor(arm)(), collision=collision)
    bits = np.asarray(log["detect"], bool)
    rises = np.flatnonzero(bits & ~np.r_[False, bits[:-1]])
    row = {"role": role, "mass": mass, "collision": collision,
           "metrics": {key: metrics[key] for key in
                       ("rms_mm", "max_mm", "saturation_pct", "detect_delay_ms",
                        "alarm_before_collision", "false_alarms", "false_alarm_times")},
           "detect_intervals": intervals(bits), "rise_indices": rises.astype(int).tolist(),
           "tail": bool(bits[-1]), "tool": certified_lab.certified.tool_record(log),
           "physical_sha256": physical_hash(log)}
    if collision is not None:
        onset = int(round(collision[0] / .001))
        active_force = np.flatnonzero(np.asarray(log["f_ext"]) > 0)
        if len(active_force):
            onset = int(active_force[0])
        end = min(len(bits), onset + 320)
        attributable = rises[(rises >= onset) & (rises < end)]
        row["event"] = {"onset": onset, "end": end,
                        "quiet_before": not bits[max(0, onset - 25):onset].any(),
                        "attributable_rises": attributable.astype(int).tolist(),
                        "duplicate_rises": max(0, int(len(attributable)) - 1),
                        "active_samples": int(bits[onset:end].sum())}
    return row


def active_at(interval_rows, begin, end):
    return any(a < end and b > begin for a, b in interval_rows)


def strict(row, reference):
    delay = row["metrics"]["detect_delay_ms"]
    if row["metrics"]["alarm_before_collision"] or delay is None:
        return False
    tick = int(round((row["collision"][0] + delay / 1000.0) / .001))
    return not active_at(reference["detect_intervals"], tick - 20, tick + 21)


def main(args):
    if sha("experiments/r26_long/candidate.py") != EXPECTED_R26:
        raise ValueError("frozen R26 candidate changed")
    tasks = []
    for mass in (0.0, 4.0, 8.0):
        tasks.append(("control", mass, None))
    for mass in (0.0, 8.0):
        tasks.append(("reference", mass, None))
        for time_value in (1.42, 3.0, 3.86):
            for force in (10.0, 15.0, 40.0, 100.0):
                tasks.append(("collision", mass, [time_value, [0.0, force, 0.0]]))
    arms = {}
    for arm in (R25, R26):
        with ProcessPoolExecutor(args.workers) as pool:
            arms[arm] = list(pool.map(capture, [(arm, *task) for task in tasks]))
    references = {arm: {row["mass"]: row for row in rows if row["role"] == "reference"}
                  for arm, rows in arms.items()}
    comparisons, regressions, gains = [], [], []
    for old, new in zip(arms[R25], arms[R26], strict=True):
        if (old["role"], old["mass"], old["collision"]) != (
                new["role"], new["mass"], new["collision"]):
            raise ValueError("mixed official row order")
        gates = {"physical_exact": old["physical_sha256"] == new["physical_sha256"],
                 "no_new_false": new["metrics"]["false_alarms"] <= old["metrics"]["false_alarms"],
                 "no_stuck_tail": not new["tail"]}
        old_strict = new_strict = False
        if old["role"] == "control":
            gates["tool_floor"] = certified_lab.prior.tool_floor(old["tool"], new["tool"], 5)
        elif old["role"] == "collision":
            old_strict = strict(old, references[R25][old["mass"]])
            new_strict = strict(new, references[R26][new["mass"]])
            gates.update({
                "strict_floor": not old_strict or new_strict,
                "latency_plus_5_ms": (not old_strict or
                    new["metrics"]["detect_delay_ms"] <= old["metrics"]["detect_delay_ms"] + 5),
                "duration_plus_5_samples": (not old_strict or
                    new["event"]["active_samples"] <= old["event"]["active_samples"] + 5),
                "no_new_duplicate": new["event"]["duplicate_rises"] <= old["event"]["duplicate_rises"],
                "quiet_preserved": not old["event"]["quiet_before"] or new["event"]["quiet_before"],
            })
        item = {"role": old["role"], "mass": old["mass"], "collision": old["collision"],
                "r25": old, "r26": new, "r25_strict": old_strict,
                "r26_strict": new_strict, "gates": gates}
        comparisons.append(item)
        failed = [key for key, value in gates.items() if not value]
        if failed:
            regressions.append({"role": old["role"], "mass": old["mass"],
                                "collision": old["collision"], "failures": failed})
        if new_strict and not old_strict:
            gains.append({"mass": old["mass"], "collision": old["collision"]})
    document = {"schema": "r26-history-original-29.v1", "episodes": 58,
                "per_arm_episodes": 29, "candidate_sha256": EXPECTED_R26,
                "evaluate_sha256": sha("evaluate.py"), "arms": arms,
                "comparisons": comparisons, "regressions": regressions,
                "strict_gains": gains, "passed": not regressions}
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": document["episodes"], "regressions": regressions,
                      "strict_gains": gains, "passed": document["passed"]}, indent=2))
    if regressions:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    main(parser.parse_args())
