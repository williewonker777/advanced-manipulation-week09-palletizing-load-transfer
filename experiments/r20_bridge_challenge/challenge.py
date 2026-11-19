"""Run a frozen R19-vs-R20 four-twin challenge on the original course engine."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.r16_payload_contract.controller import Controller
from experiments.r18_paired_impulses import runner, scoring
from experiments.r18_single_pass import paired_replay
from experiments.r19_close_pairs import assay
from experiments.r19_impulse_end.detector import Detector as R19
from experiments.r20_event_bridge.detector import Detector as R20
from tools.r14_paired_events import match_false_events, preserve_tool
from tools.r5_metrics import event_result


ARMS = {"r19": R19, "r20": R20}
SLOW_PINS = {
    "experiments/r20_slow_handoff/state.py":
        "ddaecbfb5ba49b05a592251a275b43133af781ecc0d2c8696edbbc3eda0452ab",
    "experiments/r20_slow_handoff/detector.py":
        "d03c76fb199797ee012d473034c391a1167705969e908e03b57940dccb3ca7c6",
    "experiments/r20_slow_handoff/plan.json":
        "ef498711c85d8c7e82e5ee23882782c023535a58190199f45e86230a9fc67f0b",
}
REQUIRED_PLAN_KEYS = {
    "schema", "status", "candidate", "comparator", "policy_pins",
    "quiet_ms", "window_ms", "pulse_duration_ms", "fresh",
    "exposed_controls", "fresh_acceptance", "restrictions",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def assignments(case):
    first, second = case["impacts"]
    return {"none": (None, None), "first": (first, None),
            "second": (None, second), "both": (first, second)}


def validate_plan(plan):
    if (set(plan) != REQUIRED_PLAN_KEYS
            or plan["schema"] != "r20-bridge-challenge-plan.v1"
            or plan["status"] != "FROZEN_BEFORE_ANY_CHALLENGE_OUTCOME"
            or plan["quiet_ms"] != 20 or plan["window_ms"] != 320
            or len(plan["fresh"]) != 6 or len(plan["exposed_controls"]) != 3):
        raise ValueError("invalid frozen challenge plan")
    rows = plan["fresh"] + plan["exposed_controls"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate case id")
    for row in rows:
        if set(row) != {"id", "mass", "com", "offset", "gap_ms", "impacts"}:
            raise ValueError("unexpected case fields")
        first, second = row["impacts"]
        actual_gap = round(1000 * (float(second[0]) - float(first[0]))) - plan["pulse_duration_ms"]
        if actual_gap != row["gap_ms"] or len(row["com"]) != 3:
            raise ValueError("inconsistent predeclared gap or COM")
        if np.asarray(first[1]).shape != (3,) or np.asarray(second[1]).shape != (3,):
            raise ValueError("three-axis impacts required")
    for rel, expected in plan["policy_pins"].items():
        source = Path(inspect.getsourcefile(R20)).resolve().parents[2] / rel
        if sha(source) != expected:
            raise ValueError("frozen policy pin differs: " + rel)


def resolve_arms(names):
    selected = {}
    for name in names:
        if name in ARMS:
            selected[name] = ARMS[name]
        elif name == "slow":
            from experiments.r20_slow_handoff.detector import Detector as Slow
            root = Path(inspect.getsourcefile(Slow)).resolve().parents[2]
            for rel, expected in SLOW_PINS.items():
                if sha(root / rel) != expected:
                    raise ValueError("frozen slow policy pin differs: " + rel)
            selected[name] = Slow
        else:
            raise ValueError("unknown arm: " + name)
    if "r19" not in selected or len(selected) < 2:
        raise ValueError("r19 and at least one frozen candidate arm required")
    return selected


def episode(detector_type, case, height, early, model, impacts):
    detector = detector_type()
    detector.model_path = Path(model)
    metrics, log = runner.run_episode(
        Controller(height, early), case["mass"], tuple(case["com"]), detector,
        pick_offset=case.get("offset"), impacts=impacts,
    )
    log["r18_diverged"] = bool(metrics["diverged"])
    if not scoring.finite_grid(log):
        raise ValueError("incomplete or nonfinite episode: " + case["id"])
    return log


def run_arm(detector_type, case, height, early, model, quiet_ms, window_ms):
    traces = {name: episode(detector_type, case, height, early, model, impacts)
              for name, impacts in assignments(case).items()}
    record, raw = assay.characterize(case, traces, quiet_ms, window_ms)
    return traces, record, raw


def slot_status(prior, candidate):
    rows = {}
    for slot in assay.SLOTS:
        old, new = prior["slots"][slot], candidate["slots"][slot]
        old_strict = len(old["raw_delays_ms"]) == len(old["credited_delays_ms"]) == 1
        new_strict = len(new["raw_delays_ms"]) == len(new["credited_delays_ms"]) == 1
        raw_uncredited = bool(old["raw_delays_ms"] and not old["credited_delays_ms"])
        rows[slot] = {
            "r19_raw_ms": old["raw_delays_ms"],
            "r19_credited_ms": old["credited_delays_ms"],
            "r20_raw_ms": new["raw_delays_ms"],
            "r20_credited_ms": new["credited_delays_ms"],
            "r19_strict": old_strict,
            "r20_strict": new_strict,
            "gain": bool(new_strict and not old_strict and not raw_uncredited),
            "loss": bool(old_strict and not new_strict),
            "disallowed_reference_raw_reclassification": bool(
                new_strict and not old_strict and raw_uncredited),
        }
    return rows


def valid_force(record):
    keys = ("complete_finite_strict_grid", "signed_profiles_exact",
            "force_free_prefixes", "all_required_prefixes_exact",
            "distinct_non_noop_impacts", "no_sign_inversion")
    return bool(record["signed_and_scalar_exact"]
                and all(record["r18_gates"][key] for key in keys))


def evaluate_case(case, role, height, early, model, arm_types=None,
                  quiet_ms=20, window_ms=320):
    arm_types = ARMS if arm_types is None else arm_types
    traces, records, raw = {}, {}, {}
    for arm, detector_type in arm_types.items():
        traces[arm], records[arm], raw[arm] = run_arm(
            detector_type, case, height, early, model, quiet_ms, window_ms)
    original = {
        arm: {
            "force_and_prefix_valid": valid_force(records[arm]),
            "all_r18_gates": bool(all(records[arm]["r18_gates"].values())),
            "exactly_one_per_slot": assay.one_per_pulse(records[arm]),
        } for arm in arm_types
    }
    candidates = {}
    for arm in arm_types:
        if arm == "r19":
            continue
        comparison = assay.compare(records[arm], records["r19"])
        physics = {
            twin: {key: bool(np.array_equal(traces["r19"][twin][key],
                                            traces[arm][twin][key]))
                   for key in paired_replay.PHYSICAL}
            for twin in scoring.TWINS
        }
        slots = slot_status(records["r19"], records[arm])
        gates = {
            "force_prefix_and_original_gates": bool(
                original[arm]["force_and_prefix_valid"]
                and original[arm]["all_r18_gates"]),
            "exactly_one_per_slot": original[arm]["exactly_one_per_slot"],
            "retain_latency_false_tool_tail": assay.all_comparisons([comparison]),
            "physical_trajectories_equal": all(
                value for twin in physics.values() for value in twin.values()),
            "no_disallowed_reference_raw_reclassification": not any(
                row["disallowed_reference_raw_reclassification"] for row in slots.values()),
        }
        candidates[arm] = {
            "slot_status": slots, "comparison_to_r19": comparison,
            "physical_equality": physics, "gates": gates,
            "passed": all(gates.values()),
        }
    return {
        "id": case["id"], "role": role, "case": case,
        "episodes": 4 * len(arm_types),
        "original_protocol": original, "candidates": candidates,
        "assessment_sha256": {
            arm: paired_replay.signature(raw[arm]) for arm in arm_types},
        "arms": records, "passed": all(row["passed"] for row in candidates.values()),
    }


def evaluate_normal(case, role, height, early, model, arm_types):
    traces = {arm: episode(detector, case, height, early, model, (None, None))
              for arm, detector in arm_types.items()}
    reference = traces["r19"]
    t = np.asarray(reference["t"])
    result = {}
    ref_tool = event_result(t, reference["detect"], {}, reference["f_tool"])
    ref_rises = t[scoring.rise_indices(reference)].tolist()
    for arm, trace in traces.items():
        if arm == "r19":
            continue
        rises = t[scoring.rise_indices(trace)].tolist()
        physical = {key: bool(np.array_equal(reference[key], trace[key]))
                    for key in paired_replay.PHYSICAL}
        gates = {
            "complete_finite": scoring.finite_grid(trace),
            "physical_trajectory_equal": all(physical.values()),
            "no_added_false_rise": match_false_events(rises, ref_rises)["passed"],
            "preserve_tool": preserve_tool(
                event_result(t, trace["detect"], {}, trace["f_tool"]), ref_tool),
            "no_new_stuck_tail": not bool(trace["detect"][-1]) or bool(reference["detect"][-1]),
        }
        result[arm] = {"rise_times_s": rises, "active_samples": int(np.count_nonzero(trace["detect"])),
                       "physical_equality": physical, "gates": gates,
                       "passed": all(gates.values())}
    return {"id": case["id"], "role": role, "episodes": len(arm_types),
            "r19_rise_times_s": ref_rises, "candidates": result,
            "passed": all(row["passed"] for row in result.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=Path, required=True)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="r19,r20")
    parser.add_argument("--roles", default="fresh,exposed_controls")
    parser.add_argument("--normal-only", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("new output path required")
    try:
        plan = read(args.plan)
        validate_plan(plan)
        height, early = read(args.height), read(args.early)
        arm_types = resolve_arms([name.strip() for name in args.arms.split(",") if name.strip()])
        roles = [name.strip() for name in args.roles.split(",") if name.strip()]
        if not roles or any(role not in ("fresh", "exposed_controls") for role in roles):
            raise ValueError("roles must select fresh and/or exposed_controls")
        rows = []
        for role in roles:
            for case in plan[role]:
                if args.normal_only:
                    row = evaluate_normal(case, role, height, early, args.model, arm_types)
                else:
                    row = evaluate_case(case, role, height, early, args.model, arm_types,
                                        plan["quiet_ms"], plan["window_ms"])
                rows.append(row)
                print(case["id"], row["passed"], flush=True)
        fresh = [row for row in rows if row["role"] == "fresh"]
        candidate_names = [name for name in arm_types if name != "r19"]
        result = {
            "schema": ("r20-bridge-normal-result.v1" if args.normal_only
                       else "r20-bridge-challenge-result.v1"),
            "research_only": True,
            "normal_only": args.normal_only,
            "arms": list(arm_types),
            "input_sha256": {
                "plan": sha(args.plan), "height": sha(args.height),
                "early": sha(args.early), "model": sha(args.model),
                "course_runner": runner.source_sha256(),
                **plan["policy_pins"], **({} if "slow" not in arm_types else SLOW_PINS),
            },
            "episodes": sum(row["episodes"] for row in rows),
            "fresh_cases": len(fresh), "exposed_controls": len(rows) - len(fresh),
            "fresh_strict_r19": (None if args.normal_only else sum(
                row["original_protocol"]["r19"]["exactly_one_per_slot"] for row in fresh)),
            "fresh_candidate_summary": (None if args.normal_only else {
                arm: {
                    "strict": sum(row["original_protocol"][arm]["exactly_one_per_slot"] for row in fresh),
                    "slot_gains": sum(slot["gain"] for row in fresh
                                      for slot in row["candidates"][arm]["slot_status"].values()),
                    "slot_losses": sum(slot["loss"] for row in fresh
                                       for slot in row["candidates"][arm]["slot_status"].values()),
                    "disallowed_reclassifications": sum(
                        slot["disallowed_reference_raw_reclassification"] for row in fresh
                        for slot in row["candidates"][arm]["slot_status"].values()),
                } for arm in candidate_names}),
            "fresh_all_protected_gates": all(row["passed"] for row in fresh),
            "verdict": "PASS" if all(row["passed"] for row in fresh) else "FAIL_DIAGNOSTIC_ONLY",
            "rows": rows,
        }
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in (
        "episodes", "fresh_strict_r19", "fresh_candidate_summary", "verdict")}))


if __name__ == "__main__":
    main()
