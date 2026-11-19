"""Execute the sealed R24 actuator/transfer holdout after candidate lock."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import types

import numpy as np

from experiments.r23_holdout import evaluator as prior
from experiments.r23_detached.free import Detector as R23
from my_torque import Controller


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
PROTOCOL = HERE / "protocol.json"
ARMS = ("r23_free", "r24")
certified = prior.certified


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_plant(second, variant):
    parent = certified.paired_runner.instrumented(second)
    kind = variant["kind"]

    class HoldoutPlant(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r24_kind = kind
            self.r24_gain = np.ones(6)
            self.r24_delay_steps = 0
            self.r24_commands = []
            if kind == "actuator_delay_gain":
                self.r24_gain = np.asarray(variant["joint_gain"], dtype=float)
                self.r24_delay_steps = int(variant["delay_steps"])
            elif kind == "damping_coupled":
                scale = np.asarray(variant["joint_damping_scale"], dtype=float)
                self.m.dof_damping[:len(scale)] *= scale
            elif kind == "contact_armature":
                tables = [self.m.geom(name).id for name in ("table_pick", "table_place")]
                if self.has_box:
                    box = self.m.body("box_free").id
                    tables.extend(np.flatnonzero(self.m.geom_bodyid == box).tolist())
                self.m.geom_friction[tables, 0] *= variant["table_friction_scale"]
                self.m.geom_solref[tables, 0] *= variant["table_solref_time_scale"]
                if self.m.neq:
                    self.m.eq_solref[:, 0] *= variant["suction_solref_time_scale"]
                scale = np.asarray(variant["joint_armature_scale"], dtype=float)
                self.m.dof_armature[:len(scale)] *= scale

        def step(self, tau):
            command = np.asarray(tau, dtype=float) * self.r24_gain
            if self.r24_delay_steps:
                self.r24_commands.append(command.copy())
                index = len(self.r24_commands) - 1 - self.r24_delay_steps
                command = (np.zeros_like(command) if index < 0
                           else self.r24_commands[index])
            return super().step(command)

    return HoldoutPlant


def run_episode(case, variant, detector, impacts):
    plant = local_plant(impacts[1], variant)
    function = types.FunctionType(
        certified.stock.run_episode.__code__,
        dict(certified.stock.run_episode.__globals__, Plant=plant),
        certified.stock.run_episode.__name__,
        certified.stock.run_episode.__defaults__,
        certified.stock.run_episode.__closure__,
    )
    function.__kwdefaults__ = certified.stock.run_episode.__kwdefaults__
    metrics, log = function(
        Controller(), box_mass=case["mass"], box_com=tuple(case["com"]),
        pick_offset=case.get("offset"), detector=detector(), collision=impacts[0])
    owner = log["plant"]
    log["f_ext_xyz_native"] = np.asarray(owner.r18_native_force)
    log["f_ext_xyz_second"] = np.asarray(owner.r18_second_force)
    log["f_ext_xyz_committed"] = np.asarray(owner.r18_committed_force)
    if any(np.asarray(log[key]).shape[0] != 6000
           for key in certified.PHYSICAL + ("r", "detect")):
        raise ValueError("incomplete 6000-sample trace")
    if any(not np.isfinite(np.asarray(log[key])).all()
           for key in certified.PHYSICAL + ("r", "detect")):
        raise ValueError("nonfinite holdout trace")
    if metrics["diverged"]:
        raise ValueError("holdout episode diverged")
    return metrics, log


def event_record(log, references, start, end, quiet, radius):
    row = prior.event_record(log, references, start, end, quiet, radius)
    bits = np.asarray(log["detect"], bool)
    row["active_samples_in_slot"] = int(bits[start:end].sum())
    return row


def false_active_samples(log, starts, window):
    bits = np.asarray(log["detect"], bool)
    valid = certified.intervals(log, starts, window)
    return int((bits & ~valid).sum())


def loaded_transfer_quiet(log, plan):
    first, last = plan["loaded_transfer_window_ms"]
    bits = np.asarray(log["detect"], bool)
    return {"active_samples": int(bits[first:last].sum()),
            "rises": int(((bits & ~np.r_[False, bits[:-1]])[first:last]).sum()),
            "passed": not bits[first:last].any()}


def record_common(log, impacts, starts, plan):
    window = plan["event_window_ms"]
    return {
        "false_times_s": certified.false_record(log, starts, window),
        "false_active_samples": false_active_samples(log, starts, window),
        "force_exact": certified.exact_force(log, impacts),
        "tool": certified.tool_record(log),
        "loaded_transfer_normal": loaded_transfer_quiet(log, plan),
        "stuck_tail": bool(log["detect"][-1]),
    }


def preservation(reference, candidate, slack, cost_slack):
    if not reference["strict"]:
        return {"event": True, "cost": True}
    event = (candidate["strict"] and candidate["delays_ms"][0] <=
             reference["delays_ms"][0] + slack)
    cost = (candidate["strict"] and candidate["active_samples_in_slot"] <=
            reference["active_samples_in_slot"] + cost_slack)
    return {"event": event, "cost": cost}


def negative_case(case, variant, arms, plan):
    traces = {name: run_episode(case, variant, detector, (None, None))
              for name, detector in arms.items()}
    base_hash = certified.physical_hash(traces["r23_free"][1])
    records = {}
    for name, (_, log) in traces.items():
        records[name] = record_common(log, (None, None), [], plan)
        records[name]["physical_exact_vs_r23"] = certified.physical_hash(log) == base_hash
    old, new = records["r23_free"], records["r24"]
    tools = prior.tool_gates(old["tool"], new["tool"], plan["preservation_slack_ms"])
    safety = {
        "candidate_zero_false_rises": not new["false_times_s"],
        "candidate_zero_false_active_samples": new["false_active_samples"] == 0,
        "candidate_loaded_transfer_normal_quiet": new["loaded_transfer_normal"]["passed"],
        **tools,
        "candidate_no_stuck_tail": not new["stuck_tail"],
        "physical_and_force_exact": all(
            row["physical_exact_vs_r23"] and row["force_exact"]["passed"]
            for row in records.values()),
    }
    return {
        "id": case["id"], "kind": "negative", "variant": variant["kind"],
        "contact_probe": case["contact_probe"], "episodes": len(arms),
        "r23_physics_sha256": certified.trace_signature({"none": traces["r23_free"][1]}),
        "records": records, "protection": safety,
        "scoped_detection": {"required": False, "passed": True},
        "positive_gains": [], "passed": all(safety.values()),
    }


def single_case(case, variant, arms, plan):
    traces = {
        name: {"none": run_episode(case, variant, detector, (None, None)),
               "force": run_episode(case, variant, detector, (case["impact"], None))}
        for name, detector in arms.items()
    }
    t = traces["r23_free"]["force"][1]["t"]
    start, boundary = certified.actual_start(t, case["impact"])
    base_hash = {tw: certified.physical_hash(pair[1])
                 for tw, pair in traces["r23_free"].items()}
    records = {}
    for name, twins in traces.items():
        none, force = twins["none"][1], twins["force"][1]
        records[name] = {
            "event": event_record(force, [none], start,
                                  min(6000, start + plan["event_window_ms"]),
                                  plan["quiet_ms"], plan["reference_twin_ms"]),
            "twins": {
                "none": record_common(none, (None, None), [], plan),
                "force": record_common(force, (case["impact"], None), [start], plan),
            },
            "prefix": certified.prefix(none, force, start, start),
            "boundary": boundary,
            "physical_exact_vs_r23": all(
                certified.physical_hash(twins[tw][1]) == base_hash[tw] for tw in twins),
        }
    old, new = records["r23_free"], records["r24"]
    floor = preservation(old["event"], new["event"], plan["preservation_slack_ms"],
                         plan["alarm_cost_slack_samples"])
    safety = {
        "preserve_r23_strict_event": floor["event"],
        "preserve_r23_alarm_sample_cost": floor["cost"],
        "candidate_zero_duplicate_rises": new["event"]["duplicate_rises"] == 0,
        "candidate_zero_false_rises": all(not row["false_times_s"]
                                           for row in new["twins"].values()),
        "candidate_zero_false_active_samples": all(row["false_active_samples"] == 0
                                                     for row in new["twins"].values()),
        "candidate_loaded_transfer_normal_quiet": new["twins"]["none"][
            "loaded_transfer_normal"]["passed"],
        "no_new_tool_prealarm": all(
            not new["twins"][tw]["tool"]["prealarm"] or
            old["twins"][tw]["tool"]["prealarm"] for tw in new["twins"]),
        "preserve_r23_tool_detection": all(
            prior.tool_floor(old["twins"][tw]["tool"], new["twins"][tw]["tool"],
                             plan["preservation_slack_ms"]) for tw in new["twins"]),
        "candidate_no_stuck_tail": all(not row["stuck_tail"]
                                        for row in new["twins"].values()),
        "prefix_exact": all(row["prefix"]["passed"] for row in records.values()),
        "physical_and_force_exact": all(
            row["physical_exact_vs_r23"] and
            all(twin["force_exact"]["passed"] for twin in row["twins"].values())
            for row in records.values()),
    }
    detection = {"required": True, "passed": new["event"]["strict"]}
    gains = ["event"] if new["event"]["strict"] and not old["event"]["strict"] else []
    return {
        "id": case["id"], "phase": case["phase"], "kind": "single",
        "variant": variant["kind"], "episodes": 2 * len(arms),
        "r23_physics_sha256": certified.trace_signature(
            {tw: pair[1] for tw, pair in traces["r23_free"].items()}),
        "records": records, "protection": safety, "scoped_detection": detection,
        "positive_gains": gains, "passed": all(safety.values()) and detection["passed"],
    }


def paired_case(case, variant, arms, plan):
    assignment = {"none": (None, None), "first": (case["impacts"][0], None),
                  "second": (None, case["impacts"][1]), "both": tuple(case["impacts"])}
    traces = {name: {tw: run_episode(case, variant, detector, impacts)
                     for tw, impacts in assignment.items()}
              for name, detector in arms.items()}
    t = traces["r23_free"]["both"][1]["t"]
    starts = [certified.actual_start(t, impact) for impact in case["impacts"]]
    first, second = starts[0][0], starts[1][0]
    window = plan["event_window_ms"]
    windows = {
        "isolated_first": ("first", first, min(second, first + window), ["none"]),
        "isolated_second": ("second", second, min(6000, second + window), ["none"]),
        "both_first": ("both", first, min(second, first + window), ["none", "second"]),
        "both_second": ("both", second, min(6000, second + window), ["first"]),
    }
    base_hash = {tw: certified.physical_hash(pair[1])
                 for tw, pair in traces["r23_free"].items()}
    records = {}
    for name, twins in traces.items():
        events = {slot: event_record(twins[tw][1], [twins[ref][1] for ref in refs],
                                     begin, end, plan["quiet_ms"], plan["reference_twin_ms"])
                  for slot, (tw, begin, end, refs) in windows.items()}
        common = {tw: record_common(pair[1], impacts,
                                    [first if impacts[0] else None,
                                     second if impacts[1] else None], plan)
                  for (tw, impacts), pair in zip(assignment.items(), twins.values())}
        records[name] = {
            "events": events, "twins": common,
            "prefix_first": certified.prefix(twins["none"][1], twins["first"][1], first, first),
            "prefix_second": certified.prefix(twins["none"][1], twins["second"][1], second, second),
            "prefix_both_first": certified.prefix(twins["none"][1], twins["both"][1], first, first),
            "prefix_both_second": certified.prefix(twins["first"][1], twins["both"][1], second, second),
            "boundary": [item[1] for item in starts],
            "physical_exact_vs_r23": all(
                certified.physical_hash(twins[tw][1]) == base_hash[tw] for tw in twins),
        }
    old, new = records["r23_free"], records["r24"]
    floors = {slot: preservation(event, new["events"][slot],
                                 plan["preservation_slack_ms"],
                                 plan["alarm_cost_slack_samples"])
              for slot, event in old["events"].items()}
    safety = {
        "preserve_r23_strict_slots": all(row["event"] for row in floors.values()),
        "preserve_r23_alarm_sample_cost": all(row["cost"] for row in floors.values()),
        "candidate_zero_duplicate_rises": all(
            event["duplicate_rises"] == 0 for event in new["events"].values()),
        "candidate_zero_false_rises": all(not row["false_times_s"]
                                           for row in new["twins"].values()),
        "candidate_zero_false_active_samples": all(row["false_active_samples"] == 0
                                                     for row in new["twins"].values()),
        "candidate_loaded_transfer_normal_quiet": new["twins"]["none"][
            "loaded_transfer_normal"]["passed"],
        "no_new_tool_prealarm": all(
            not new["twins"][tw]["tool"]["prealarm"] or
            old["twins"][tw]["tool"]["prealarm"] for tw in new["twins"]),
        "preserve_r23_tool_detection": all(
            prior.tool_floor(old["twins"][tw]["tool"], new["twins"][tw]["tool"],
                             plan["preservation_slack_ms"]) for tw in new["twins"]),
        "candidate_no_stuck_tail": all(not row["stuck_tail"]
                                        for row in new["twins"].values()),
        "all_prefixes_exact": all(
            row[key]["passed"] for row in records.values()
            for key in ("prefix_first", "prefix_second", "prefix_both_first", "prefix_both_second")),
        "physical_and_force_exact": all(
            row["physical_exact_vs_r23"] and
            all(twin["force_exact"]["passed"] for twin in row["twins"].values())
            for row in records.values()),
    }
    detection = {"required": True,
                 "passed": all(event["strict"] for event in new["events"].values())}
    gains = [slot for slot, event in new["events"].items()
             if event["strict"] and not old["events"][slot]["strict"]]
    return {
        "id": case["id"], "phase": case["phase"], "kind": "paired",
        "variant": variant["kind"], "episodes": 4 * len(arms),
        "r23_physics_sha256": certified.trace_signature(
            {tw: pair[1] for tw, pair in traces["r23_free"].items()}),
        "records": records, "protection": safety, "scoped_detection": detection,
        "positive_gains": gains, "passed": all(safety.values()) and detection["passed"],
    }


def dry_validate():
    plan = json.loads(PLAN.read_bytes())
    protocol = json.loads(PROTOCOL.read_bytes())
    cases = [(kind, row["id"]) for kind in ("negative", "single", "paired")
             for row in plan[kind]]
    if len(cases) != len(set(cases)) or len(cases) != 13:
        raise ValueError("R24 case accounting differs")
    per_arm = len(plan["variants"]) * (len(plan["negative"]) +
                                       2 * len(plan["single"]) +
                                       4 * len(plan["paired"]))
    if (per_arm != plan["required_episodes"]["per_arm"] or
            2 * per_arm != plan["required_episodes"]["both_arms"] or
            not 100 <= 2 * per_arm <= 300):
        raise ValueError("R24 episode accounting differs")
    t = .001 * np.arange(6000)
    for case in plan["single"]:
        certified.actual_start(t, case["impact"])
    for case in plan["paired"]:
        active = []
        for impact in case["impacts"]:
            force = np.stack([certified.pulse(value, impact) for value in t])
            active.append(np.flatnonzero(np.any(force != 0., axis=1)))
        if not all(len(row) for row in active) or active[0][-1] >= active[1][0]:
            raise ValueError("R24 paired impulses overlap or have no support")
    actual = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if actual != protocol["source_pins"]:
        raise ValueError("R24 protocol source differs")
    return {"schema": plan["schema"], "cases": len(cases),
            "episodes": 2 * per_arm, "variants": len(plan["variants"]),
            "worker_cap": 2}


def resolve(lock):
    if (lock.get("schema") != "r24-candidate-lock.v1" or
            lock.get("status") != "LOCKED_BEFORE_HOLDOUT_OUTCOMES" or
            lock.get("plan_sha256") != sha(PLAN) or
            lock.get("protocol_sha256") != sha(PROTOCOL) or
            lock.get("evaluator_sha256") != sha(Path(__file__))):
        raise ValueError("candidate lock does not match sealed R24 evaluator")
    for name, expected in lock.get("transitive_sources", {}).items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("candidate transitive source pin differs: " + name)
    module, symbol = lock["entrypoint"].rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def one(item):
    kind, case, variant, arms, plan = item
    return {"negative": negative_case, "single": single_case,
            "paired": paired_case}[kind](case, variant, arms, plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.lock or args.out:
            parser.error("dry-run does not accept lock or output")
        print(json.dumps(dry_validate(), sort_keys=True))
        return 0
    if args.lock is None or args.out is None:
        parser.error("--lock and --out are required for execution")
    if os.path.lexists(args.out):
        parser.error("fresh output required")
    plan_raw, lock_raw = PLAN.read_bytes(), args.lock.read_bytes()
    plan, lock = json.loads(plan_raw), json.loads(lock_raw)
    protocol = json.loads(PROTOCOL.read_bytes())
    pinned_before = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if pinned_before != protocol["source_pins"]:
        raise ValueError("sealed R24 dependency differs")
    candidate = resolve(lock)
    arms = {"r23_free": R23, "r24": candidate}
    tasks = [(kind, case, variant, arms, plan)
             for kind in ("negative", "single", "paired")
             for case in plan[kind] for variant in plan["variants"].values()]
    before = {name: sha(ROOT / name) for name in lock["transitive_sources"]}
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(one, tasks))
    after = {name: sha(ROOT / name) for name in lock["transitive_sources"]}
    pinned_after = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if pinned_before != pinned_after or before != after or before != lock["transitive_sources"]:
        raise ValueError("sealed source changed during R24 holdout")
    effects = {"stock": True}
    stock = {(row["kind"], row["id"]): row["r23_physics_sha256"]
             for row in rows if row["variant"] == "stock"}
    for variant in plan["variants"]:
        if variant != "stock":
            effects[variant] = any(
                row["r23_physics_sha256"] != stock[(row["kind"], row["id"])]
                for row in rows if row["variant"] == variant)
    if not all(effects.values()):
        raise ValueError("declared mechanical intervention had no observed effect")
    episodes = sum(row["episodes"] for row in rows)
    if episodes != plan["required_episodes"]["both_arms"]:
        raise ValueError("incomplete R24 episode accounting")
    contact_rows = [row["variant"] + ":" + row["id"] for row in rows
                    if row["kind"] == "negative" and row["contact_probe"] and
                    row["records"]["r23_free"]["tool"]["present"]]
    required = plan["contact_evidence"]["minimum_probe_variant_rows_with_real_r23_contact"]
    contact = {"rows": contact_rows, "required": required,
               "adequate": len(contact_rows) >= required}
    protection = all(all(row["protection"].values()) for row in rows)
    detection = all(row["scoped_detection"]["passed"] for row in rows
                    if row["scoped_detection"]["required"])
    gains = [{"variant": row["variant"], "id": row["id"], "kind": row["kind"],
              "slots": row["positive_gains"]} for row in rows if row["positive_gains"]]
    failures = [row["variant"] + ":" + row["id"] for row in rows if not row["passed"]]
    passed = protection and detection and contact["adequate"]
    document = {
        "schema": "r24-independent-actuator-transfer-result.v1",
        "plan_sha256": sha(PLAN), "lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "classification": plan["classification"], "arms": list(arms),
        "episodes": episodes, "source_pins": pinned_before,
        "mechanical_effect_observed": effects, "real_contact_evidence": contact,
        "claims": {"safety_qualification": False,
                   "absolute_protection_pass": protection,
                   "scoped_detection_pass": detection,
                   "positive_gain_count": sum(len(row["slots"]) for row in gains)},
        "positive_gains": gains, "rows": rows, "failed": failures, "passed": passed,
    }
    with args.out.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"episodes": episodes, "passed": passed,
                      "failed": len(failures),
                      "positive_gains": document["claims"]["positive_gain_count"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
