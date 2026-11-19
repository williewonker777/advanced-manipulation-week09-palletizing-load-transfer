"""Authored staged R16 stock qualification; invoked only inside a public overlay."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

import my_detector
import r1_torque
from experiments.r14_stock_handoff.detector import Detector
from experiments.r14_stock_handoff.qualification import KEYS, compact
from experiments.r15_early_com.controller import Controller as R15
from sim.runner import run_episode
from tools.r11_placement import score
from tools.r14_paired_events import (equal_prefix, force_realization,
                                     match_false_events, preserve_tool)
from tools.r5_metrics import event_result

HERE = Path(__file__).resolve().parent
ARMS = ("r10", "r15", "r16")
CANDIDATE = None
LAW = None
CONTRACT_BASELINE = None
CONTRACT_DETECTOR = None
RAW_C = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controller(kind, height, early):
    if CANDIDATE is None:
        raise ValueError("candidate controller not authenticated")
    return {"r10": lambda: r1_torque.Controller(),
            "r15": lambda: R15(height, early),
            "raw_c": lambda: RAW_C(height, early),
            "contract_baseline": lambda: CONTRACT_BASELINE(height, early),
            "r16": lambda: CANDIDATE(height, early)}[kind]()


def episode(kind, case, height, early, collision=None):
    ctrl = controller(kind, height, early)
    if kind == "r10":
        detector = my_detector.Detector()
    elif LAW == "payload-contract" and kind in ("contract_baseline", "r16"):
        detector = CONTRACT_DETECTOR()
    else:
        detector = Detector()
    metrics, log = run_episode(ctrl, float(case["mass"]), tuple(case["com"]), detector,
                               collision=collision, pick_offset=case.get("offset"))
    record = compact(metrics, log, ctrl, float(case["mass"]))
    return record, log


def ratio(candidate, baseline):
    return {key: float(value / max(baseline["cost"][key], 1e-12))
            for key, value in candidate["cost"].items()}


def tool_gate(candidate, reference):
    return (reference["tool_delay_ms"] is None or
            candidate["tool_delay_ms"] is not None and
            candidate["tool_delay_ms"] <= reference["tool_delay_ms"] + 5.0 + 1e-9)


def saturation_both(candidate, first, second):
    return candidate["saturation_pct"] <= min(
        first["saturation_pct"], second["saturation_pct"]) + 1e-12


def normal_case(case, height, early):
    records = {kind: episode(kind, case, height, early)[0] for kind in ARMS}
    r10, r15, candidate = (records[kind] for kind in ARMS)
    costs = {name: ratio(candidate, records[name]) for name in ("r10", "r15")}
    false_events = {name: match_false_events(candidate["false_alarm_times"],
                                              records[name]["false_alarm_times"])
                    for name in ("r10", "r15")}
    g4 = score(r10, candidate)
    gates = {
        "status": all(record["status"] == "ok" for record in records.values()),
        "G4": bool(g4["all_four_g4"]),
        "cost_r10": max(costs["r10"].values()) <= 1.05,
        "cost_r15": max(costs["r15"].values()) <= 1.05,
        "false_events_r10": false_events["r10"]["passed"],
        "false_events_r15": false_events["r15"]["passed"],
        "tool_r10": tool_gate(candidate, r10),
        "tool_r15": tool_gate(candidate, r15),
        "saturation": candidate["saturation_pct"] <= max(r10["saturation_pct"], r15["saturation_pct"]) + 1e-12,
        "runtime": candidate["ctrl_p99_us"] <= 800.0,
    }
    return {"id": case["id"], "case": case, "arms": records, "cost_ratios": costs,
            "g4": g4, "false_event_match": false_events, "gates": gates,
            "passed": all(gates.values())}


def exact_logs(left, right):
    return {key: bool(np.array_equal(left[key], right[key])) for key in KEYS}


def nonclock_record(record):
    return {key: value for key, value in record.items() if key not in ("ctrl_p99_us", "registration")}


def exact_pair(left_record, left_log, right_record, right_log):
    fields = exact_logs(left_log, right_log)
    metrics = nonclock_record(left_record) == nonclock_record(right_record)
    return {"fields": fields, "metrics": metrics,
            "passed": bool(metrics and all(fields.values()))}


def contract_normal_case(case, height, early):
    kinds = ("r10", "r15", "raw_c", "contract_baseline", "r16")
    records, logs = {}, {}
    for kind in kinds:
        records[kind], logs[kind] = episode(kind, case, height, early)
    r10, raw_r15, candidate = records["r10"], records["r15"], records["r16"]
    costs = {name: ratio(candidate, records[name]) for name in ("r10", "r15")}
    false_events = {name: match_false_events(candidate["false_alarm_times"],
                                              records[name]["false_alarm_times"])
                    for name in ("r10", "r15")}
    identities = {
        "loaded_D_equals_raw_C": exact_pair(records["r16"], logs["r16"],
                                              records["raw_c"], logs["raw_c"]),
        "loaded_contract_baseline_equals_raw_R15": exact_pair(
            records["contract_baseline"], logs["contract_baseline"],
            records["r15"], logs["r15"]),
    }
    g4 = score(r10, candidate)
    gates = {
        "status": all(record["status"] == "ok" for record in records.values()),
        "G4": bool(g4["all_four_g4"]),
        "cost_r10": max(costs["r10"].values()) <= 1.05,
        "cost_raw_r15": max(costs["r15"].values()) <= 1.05,
        "false_events_r10": false_events["r10"]["passed"],
        "false_events_raw_r15": false_events["r15"]["passed"],
        "tool_r10": tool_gate(candidate, r10),
        "tool_raw_r15": tool_gate(candidate, raw_r15),
        # Contract law is bounded by BOTH comparators, not their maximum.
        "saturation_both": saturation_both(candidate, r10, raw_r15),
        "runtime": candidate["ctrl_p99_us"] <= 800.0,
        "loaded_D_equals_raw_C": identities["loaded_D_equals_raw_C"]["passed"],
        "loaded_baseline_equals_raw_R15": identities[
            "loaded_contract_baseline_equals_raw_R15"]["passed"],
    }
    return {"id": case["id"], "case": case, "arms": records,
            "arm_roles": {"r15": "frozen_raw_comparator", "raw_c": "frozen_balanced_law",
                          "contract_baseline": "domain-dispatched_baseline",
                          "r16": "payload_contract_candidate"},
            "cost_ratios": costs, "g4": g4, "false_event_match": false_events,
            "identity": identities, "gates": gates, "passed": all(gates.values())}


def unloaded(height, early):
    if LAW == "payload-contract":
        return contract_unloaded(height, early)
    case = {"id": "unloaded", "mass": 0.0, "com": [0.0, 0.0, 0.0], "offset": None}
    records, logs = {}, {}
    for kind in ARMS:
        records[kind], logs[kind] = episode(kind, case, height, early)
    parity = {name: {key: bool(np.array_equal(logs[name][key], logs["r10"][key]))
                     for key in KEYS} for name in ("r15", "r16")}
    return {"id": "unloaded", "arms": records, "parity_vs_r10": parity,
            "passed": all(all(fields.values()) for fields in parity.values())}


def rises(log):
    detect = np.asarray(log["detect"], dtype=bool)
    return np.flatnonzero(detect & ~np.r_[False, detect[:-1]]).tolist()


def preserves_rises(candidate, reference):
    return all(index in set(rises(candidate)) for index in rises(reference))


def contract_unloaded(height, early):
    case = {"id": "unloaded", "mass": 0.0, "com": [0.0, 0.0, 0.0], "offset": None}
    kinds = ("r10", "r15", "contract_baseline", "r16")
    records, logs = {}, {}
    for kind in kinds:
        records[kind], logs[kind] = episode(kind, case, height, early)
    baseline = exact_pair(records["contract_baseline"], logs["contract_baseline"],
                          records["r10"], logs["r10"])
    candidate = exact_pair(records["r16"], logs["r16"], records["r10"], logs["r10"])
    raw_gap = exact_pair(records["r15"], logs["r15"], records["r10"], logs["r10"])
    gates = {
        "status_all": all(record["status"] == "ok" for record in records.values()),
        "contract_baseline_exact_r10": baseline["passed"],
        "payload_candidate_exact_r10": candidate["passed"],
        "contract_baseline_no_extra_events": records["contract_baseline"]["false_alarm_times"]
            == records["r10"]["false_alarm_times"],
        "payload_candidate_no_extra_events": records["r16"]["false_alarm_times"]
            == records["r10"]["false_alarm_times"],
        "raw_r15_preserves_r10_rises": preserves_rises(logs["r15"], logs["r10"]),
        "contract_baseline_preserves_r10_rises": preserves_rises(
            logs["contract_baseline"], logs["r10"]),
        "payload_candidate_preserves_r10_rises": preserves_rises(logs["r16"], logs["r10"]),
    }
    return {"id": "unloaded", "arms": records,
            "arm_roles": {"r15": "frozen_raw_R15_known_gap_not_a_passing_candidate",
                          "contract_baseline": "domain-dispatched_baseline",
                          "r16": "payload_contract_candidate"},
            "exact": {"raw_r15_known_gap": raw_gap, "contract_baseline": baseline,
                      "payload_candidate": candidate},
            "gates": gates, "passed": all(gates.values())}


def normal(stage, cases, height, early, auth, case_id=None):
    selected = cases if case_id is None else [case for case in cases if case["id"] == case_id]
    if not selected:
        raise ValueError("unknown case id for stage")
    scorer = contract_normal_case if LAW == "payload-contract" else normal_case
    rows = [scorer(case, height, early) for case in selected]
    zero = unloaded(height, early) if stage == "fresh" and case_id is None else None
    complete = case_id is None and len(rows) == len(cases)
    passed = bool(complete and all(row["passed"] for row in rows)
                  and (zero is None or zero["passed"]))
    failed = [{"id": row["id"], "gates": [k for k, v in row["gates"].items() if not v]}
              for row in rows if not row["passed"]]
    if zero is not None and not zero["passed"]:
        failed.append({"id": "unloaded", "gates": [k for k, v in zero.get(
            "gates", {"exact_parity": False}).items() if not v]})
    return {"schema": "r16-progressive-stage.v1", "stage": stage, "auth": auth,
            "complete": complete, "diagnostic_case": case_id, "unloaded": zero,
            "rows": rows, "passed": passed, "failed": failed}


def impact_case(case, height, early):
    collision = case["collision"]
    events, prefixes, forces, statuses = {}, {}, {}, {}
    for kind in ARMS:
        normal_record, normal_log = episode(kind, case, height, early)
        forced_record, forced_log = episode(kind, case, height, early, collision)
        statuses[kind] = normal_record["status"] == forced_record["status"] == "ok"
        prefixes[kind] = equal_prefix(normal_log, forced_log, collision[0])
        forces[kind] = force_realization(forced_log["t"], forced_log["f_ext"], collision)
        events[kind] = event_result(forced_log["t"], forced_log["detect"],
                                    {"collision": collision}, forced_log["f_tool"],
                                    normal_log["detect"])
    r10, r15, candidate = (events[kind] for kind in ARMS)
    fa = {name: match_false_events(candidate["false_alarm_times"],
                                   events[name]["false_alarm_times"])
          for name in ("r10", "r15")}
    retained = (r10["outcome"] != "detected" or
                candidate["outcome"] == "detected" and
                candidate["delay_ms"] <= r10["delay_ms"] + 5.0 + 1e-9)
    gates = {
        "status": all(statuses.values()),
        "force_realized": all(force["passed"] for force in forces.values()),
        "matched_prefix": all(prefix["passed"] for prefix in prefixes.values()),
        "retain_r10_true_positive": retained,
        "no_candidate_prealarm": not candidate["prealarm"],
        "false_events_r10": fa["r10"]["passed"],
        "false_events_r15": fa["r15"]["passed"],
        "tool_r10": preserve_tool(candidate, r10),
        "tool_r15": preserve_tool(candidate, r15),
    }
    return {"id": case["id"], "claim": case["claim"], "case": case,
            "arms": events, "prefix": prefixes, "force": forces,
            "false_event_match": fa, "gates": gates, "passed": all(gates.values())}


def contract_impact_case(case, height, early):
    collision = case["collision"]
    kinds = ("r10", "r15", "raw_c", "contract_baseline", "r16")
    events, prefixes, forces, statuses = {}, {}, {}, {}
    normal_records, normal_logs, forced_records, forced_logs = {}, {}, {}, {}
    for kind in kinds:
        normal_records[kind], normal_logs[kind] = episode(kind, case, height, early)
        forced_records[kind], forced_logs[kind] = episode(kind, case, height, early, collision)
        statuses[kind] = normal_records[kind]["status"] == forced_records[kind]["status"] == "ok"
        prefixes[kind] = equal_prefix(normal_logs[kind], forced_logs[kind], collision[0])
        forces[kind] = force_realization(forced_logs[kind]["t"], forced_logs[kind]["f_ext"], collision)
        events[kind] = event_result(forced_logs[kind]["t"], forced_logs[kind]["detect"],
                                    {"collision": collision}, forced_logs[kind]["f_tool"],
                                    normal_logs[kind]["detect"])
    r10, raw_r15, candidate = events["r10"], events["r15"], events["r16"]
    fa = {name: match_false_events(candidate["false_alarm_times"],
                                   events[name]["false_alarm_times"])
          for name in ("r10", "r15")}
    retained = (r10["outcome"] != "detected" or
                candidate["outcome"] == "detected" and
                candidate["delay_ms"] <= r10["delay_ms"] + 5.0 + 1e-9)
    identities = {
        "normal_D_equals_raw_C": exact_pair(normal_records["r16"], normal_logs["r16"],
                                              normal_records["raw_c"], normal_logs["raw_c"]),
        "forced_D_equals_raw_C": exact_pair(forced_records["r16"], forced_logs["r16"],
                                              forced_records["raw_c"], forced_logs["raw_c"]),
        "normal_baseline_equals_raw_R15": exact_pair(
            normal_records["contract_baseline"], normal_logs["contract_baseline"],
            normal_records["r15"], normal_logs["r15"]),
        "forced_baseline_equals_raw_R15": exact_pair(
            forced_records["contract_baseline"], forced_logs["contract_baseline"],
            forced_records["r15"], forced_logs["r15"]),
    }
    gates = {
        "status": all(statuses.values()),
        "force_realized": all(force["passed"] for force in forces.values()),
        "matched_prefix": all(prefix["passed"] for prefix in prefixes.values()),
        "retain_r10_true_positive": retained,
        "no_candidate_prealarm": not candidate["prealarm"],
        "false_events_r10": fa["r10"]["passed"],
        "false_events_raw_r15": fa["r15"]["passed"],
        "tool_r10": preserve_tool(candidate, r10),
        "tool_raw_r15": preserve_tool(candidate, raw_r15),
        "loaded_D_equals_raw_C": (identities["normal_D_equals_raw_C"]["passed"]
                                   and identities["forced_D_equals_raw_C"]["passed"]),
        "loaded_baseline_equals_raw_R15": (
            identities["normal_baseline_equals_raw_R15"]["passed"]
            and identities["forced_baseline_equals_raw_R15"]["passed"]),
    }
    return {"id": case["id"], "claim": case["claim"], "case": case,
            "arms": events, "arm_roles": {"r15": "frozen_raw_comparator",
                "raw_c": "frozen_balanced_law", "contract_baseline": "domain_baseline",
                "r16": "payload_contract_candidate"},
            "prefix": prefixes, "force": forces, "identity": identities,
            "false_event_match": fa, "gates": gates, "passed": all(gates.values())}


def impact(cases, height, early, auth, case_id=None):
    selected = cases if case_id is None else [case for case in cases if case["id"] == case_id]
    if not selected:
        raise ValueError("unknown impact case")
    scorer = contract_impact_case if LAW == "payload-contract" else impact_case
    rows = [scorer(case, height, early) for case in selected]
    complete = case_id is None and len(rows) == len(cases)
    return {"schema": "r16-progressive-stage.v1", "stage": "impact", "auth": auth,
            "complete": complete, "diagnostic_case": case_id, "rows": rows,
            "passed": bool(complete and all(row["passed"] for row in rows)),
            "failed": [{"id": row["id"], "gates": [k for k, v in row["gates"].items() if not v]}
                       for row in rows if not row["passed"]]}


def validate_upstream(path, expected_stage, auth):
    document = json.loads(Path(path).read_text())
    if (document.get("schema") != "r16-progressive-stage.v1"
            or document.get("stage") != expected_stage
            or document.get("complete") is not True or document.get("passed") is not True
            or document.get("auth") != auth):
        raise ValueError("complete passing authenticated upstream stage required")
    return document


def main(argv=None):
    global CANDIDATE, LAW, CONTRACT_BASELINE, CONTRACT_DETECTOR, RAW_C
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("exposed", "fresh", "impact"), required=True)
    parser.add_argument("--law", choices=("release", "normal-phase", "balanced",
                                          "payload-contract"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--height", type=Path, required=True)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--auth", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    height = json.loads(args.height.read_text())
    early = json.loads(args.early.read_text())
    auth = json.loads(args.auth.read_text())
    try:
        plan = json.loads(args.plan.read_text())
        expected_module = {"release": "experiments.r16_release.controller",
                           "normal-phase": "experiments.r16_normal_phase.controller",
                           "balanced": "experiments.r16_balanced.controller",
                           "payload-contract": "experiments.r16_payload_contract.controller"}[args.law]
        if (plan.get("schema") != "r16-progressive-handoff-plan.v1"
                or plan.get("law") != args.law
                or plan.get("candidate_module") != expected_module
                or auth.get("law") != args.law
                or auth.get("plan_sha256") != sha(args.plan)):
            raise ValueError("law-specific plan/authentication mismatch")
        module = importlib.import_module(expected_module)
        CANDIDATE = module.Controller
        LAW = args.law
        if LAW == "payload-contract":
            CONTRACT_BASELINE, CONTRACT_DETECTOR = module.Baseline, module.Detector
            RAW_C = importlib.import_module("experiments.r16_balanced.controller").Controller
        if args.stage == "exposed":
            if args.upstream is not None:
                raise ValueError("exposed stage accepts no upstream")
            result = normal("exposed", plan["exposed_cases"], height, early, auth, args.case)
        elif args.stage == "fresh":
            if args.upstream is None:
                raise ValueError("fresh stage requires exposed upstream")
            validate_upstream(args.upstream, "exposed", auth)
            result = normal("fresh", plan["fresh_cases"], height, early, auth, args.case)
            result["upstream_sha256"] = sha(args.upstream)
        else:
            if args.upstream is None:
                raise ValueError("impact stage requires fresh upstream")
            upstream = validate_upstream(args.upstream, "fresh", auth)
            if not isinstance(upstream.get("upstream_sha256"), str):
                raise ValueError("fresh stage lacks exposed chain")
            result = impact(plan["impact_cases"], height, early, auth, args.case)
            result["upstream_sha256"] = sha(args.upstream)
            result["exposed_sha256"] = upstream["upstream_sha256"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"stage": args.stage, "complete": result["complete"],
                      "passed": result["passed"], "failed": result["failed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
