"""Run named R26 landing arms against disclosed, fixed stock-simulator scenarios."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.r24_transfer_replay import evaluator as replay


HERE = Path(__file__).resolve().parent
R24 = HERE.parent / "r24_transfer_replay"
SCENARIOS = {
    "normal-sh5": ("known_normal", "sh5"),
    "signed-x-positive": ("signed_late_force", "late_x_pos"),
    "signed-x-negative": ("signed_late_force", "late_x_neg"),
    "signed-z-positive": ("signed_late_force", "late_z_pos"),
    "signed-z-negative": ("signed_late_force", "late_z_neg"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def resolve(entrypoint):
    module, symbol = entrypoint.split(":", 1)
    return getattr(importlib.import_module(module), symbol)


def registrations():
    data = read(HERE / "registration.json")
    if data.get("schema") != "r26-transfer-arm-registration.v1":
        raise ValueError("arm registration schema differs")
    arms = data.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"baseline", "r24", "registered-c1"}:
        raise ValueError("complete named arm registration required")
    for name, item in arms.items():
        if item.get("status") == "not_frozen":
            if item.get("detector") is not None:
                raise ValueError("unfrozen C1 must not name a detector")
        elif not all(isinstance(item.get(key), str) and item[key]
                     for key in ("controller", "detector")):
            raise ValueError("complete frozen arm entrypoints required: " + name)
        if name == "registered-c1" and item.get("status") != "not_frozen":
            source = HERE.parent / "r26_delivery_pair" / "candidate.py"
            if digest(source) != item.get("detector_sha256"):
                raise ValueError("registered-c1 detector changed after freeze")
    return data


def scenario(name):
    kind, identifier = SCENARIOS[name]
    plan = read(R24 / "plan.json")
    if kind == "known_normal":
        case = next(item for item in plan["known_replay"] if item["id"] == identifier)
        return kind, case, None
    impact = next(item for item in plan["late_impulses"] if item["id"] == identifier)
    collision = [float(impact["time_s"]), list(impact["force_n"])]
    return kind, plan["impulse_base"], collision


def run_episode(case, arm, collision):
    controller = resolve(arm["controller"])
    detector = resolve(arm["detector"])
    runner = replay.run_function(case["mechanics"]["table_solref_time_scale"])
    metrics, log = runner(controller(), float(case["mass"]), tuple(case["com"]),
                          detector(), collision=collision, pick_offset=case["offset"])
    requested = np.asarray(log["plant"].r24_requested_force)
    committed = np.asarray(log["plant"].r24_committed_force)
    expected = replay.expected_force(log["t"], collision)
    if (metrics["diverged"] or len(log["t"]) != 6000
            or not np.array_equal(requested, committed)
            or not np.array_equal(committed, expected)):
        raise ValueError("invalid trajectory, sample count, or signed input")
    log["signed_force"] = committed
    return metrics, log


def assess(name, arm_name, arm):
    kind, case, collision = scenario(name)
    twin_metrics, twin = run_episode(case, arm, None)
    if collision is None:
        result = {
            "kind": kind,
            "official_false_alarm_times_s": twin_metrics["false_alarm_times"],
            "reported_rises_s": replay.rise_times(twin),
            "physical_sha256": replay.physical_signature(twin),
        }
        if arm_name == "r24":
            expected = case["expected"]
            if (result["official_false_alarm_times_s"] != expected["r24_false"]
                    or result["physical_sha256"] != expected["r24_physical_sha256"]):
                raise ValueError("R24 known-normal reproduction differs")
        return result
    forced_metrics, forced = run_episode(case, arm, collision)
    plan = read(R24 / "plan.json")
    result = replay.force_assessment(forced, twin, collision, plan)
    result.update({
        "kind": kind,
        "matched_no_force_official_false_alarm_times_s": twin_metrics["false_alarm_times"],
        "forced_official_false_alarm_times_s": forced_metrics["false_alarm_times"],
        "matched_no_force_physical_sha256": replay.physical_signature(twin),
        "forced_physical_sha256": replay.physical_signature(forced),
    })
    if not result["signed_force_exact"] or not result["force_free_prefix_exact"]:
        raise ValueError("signed-force or matched-prefix integrity differs")
    return result


def evaluate(scenario_name, arm_names):
    if mujoco.__version__ != "3.17.0":
        raise ValueError("course MuJoCo 3.17.0 required")
    registration = registrations()
    output = {}
    for arm_name in arm_names:
        arm = registration["arms"][arm_name]
        if arm.get("status") == "not_frozen":
            raise ValueError("registered-c1 is unavailable until source and entrypoint are frozen")
        output[arm_name] = assess(scenario_name, arm_name, arm)
    return {
        "schema": "r26-transfer-disclosed-replay.v1",
        "classification": "finite local simulation research; not ORIGINAL, hidden evaluation, hardware, deployment, or safety proof",
        "scenario": scenario_name,
        "scenario_definition": {
            "source": "frozen experiments/r24_transfer_replay/plan.json",
            "kind": SCENARIOS[scenario_name][0],
            "id": SCENARIOS[scenario_name][1],
            "matched_no_force_twin": SCENARIOS[scenario_name][0] == "signed_late_force",
        },
        "arms": output,
        "runtime": {"mujoco": mujoco.__version__},
        "original_demo": {"included": False, "entrypoint": ["run.py", "--box", "8"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--arm", action="append", choices=("registered-c1", "r24", "baseline"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error("fresh regular output path required")
    try:
        result = evaluate(args.scenario, tuple(dict.fromkeys(args.arm)))
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n")
    except (ImportError, AttributeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
