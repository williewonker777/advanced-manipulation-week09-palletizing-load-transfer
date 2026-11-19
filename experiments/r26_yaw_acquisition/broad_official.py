"""Fresh original 29-episode torque comparison for the admitted candidate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r24_holdout import evaluator as certified_lab
from experiments.r26_history import official as base


R25 = base.R25
CANDIDATE = "experiments.r26_yaw_acquisition.candidate:Detector"
CANDIDATE_SHA = "212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784"
SOURCE = Path(__file__).with_name("candidate.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", choices=(1, 2), type=int, default=2)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    if base.sha(SOURCE) != CANDIDATE_SHA:
        raise ValueError("admitted acquisition candidate changed")
    tasks = []
    for mass in (0.0, 4.0, 8.0):
        tasks.append(("control", mass, None))
    for mass in (0.0, 8.0):
        tasks.append(("reference", mass, None))
        for time_value in (1.42, 3.0, 3.86):
            for force in (10.0, 15.0, 40.0, 100.0):
                tasks.append(("collision", mass, [time_value, [0.0, force, 0.0]]))
    arms = {}
    for arm in (R25, CANDIDATE):
        with ProcessPoolExecutor(args.workers) as pool:
            arms[arm] = list(pool.map(base.capture, [(arm, *task) for task in tasks]))
    references = {arm: {row["mass"]: row for row in rows if row["role"] == "reference"}
                  for arm, rows in arms.items()}
    rows, regressions, gains = [], [], []
    for old, new in zip(arms[R25], arms[CANDIDATE], strict=True):
        if (old["role"], old["mass"], old["collision"]) != (new["role"], new["mass"], new["collision"]):
            raise ValueError("mixed official row order")
        gates = {"physical_exact": old["physical_sha256"] == new["physical_sha256"],
                 "no_new_false": new["metrics"]["false_alarms"] <= old["metrics"]["false_alarms"],
                 "no_stuck_tail": not new["tail"]}
        old_strict = new_strict = False
        if old["role"] == "control":
            gates["tool_floor"] = certified_lab.prior.tool_floor(old["tool"], new["tool"], 5)
        elif old["role"] == "collision":
            old_strict = base.strict(old, references[R25][old["mass"]])
            new_strict = base.strict(new, references[CANDIDATE][new["mass"]])
            gates.update({
                "strict_floor": not old_strict or new_strict,
                "latency_plus_5_ms": (not old_strict or
                                      new["metrics"]["detect_delay_ms"] <= old["metrics"]["detect_delay_ms"] + 5),
                "duration_plus_5_samples": (not old_strict or
                                            new["event"]["active_samples"] <= old["event"]["active_samples"] + 5),
                "no_new_duplicate": new["event"]["duplicate_rises"] <= old["event"]["duplicate_rises"],
                "quiet_preserved": not old["event"]["quiet_before"] or new["event"]["quiet_before"],
            })
        failed = [key for key, value in gates.items() if not value]
        rows.append({"role": old["role"], "mass": old["mass"], "collision": old["collision"],
                     "r25": old, "candidate": new, "r25_strict": old_strict,
                     "candidate_strict": new_strict, "gates": gates, "failures": failed})
        if failed:
            regressions.append({"role": old["role"], "mass": old["mass"],
                                "collision": old["collision"], "failures": failed})
        if new_strict and not old_strict:
            gains.append({"mass": old["mass"], "collision": old["collision"]})
    document = {"schema": "r26-acquisition-original-29.v1", "episodes": 58,
                "per_arm_episodes": 29, "candidate": CANDIDATE,
                "candidate_sha256": CANDIDATE_SHA, "evaluate_sha256": base.sha("evaluate.py"),
                "arms": arms, "comparisons": rows, "regressions": regressions,
                "strict_gains": gains, "passed": not regressions}
    args.out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": 58, "regressions": regressions,
                      "strict_gains": gains, "passed": document["passed"]}, indent=2))
    if regressions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
