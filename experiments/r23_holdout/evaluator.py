"""Run the sealed R23 transfer/post-detachment holdout after candidate lock."""
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

from experiments.r22_challenge import evaluator as certified
from experiments.r22_light.candidate import Detector as R22
from my_torque import Controller


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
PROTOCOL = HERE / "protocol.json"
ARMS = ("r22_light", "r23")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_plant(second, variant):
    """Create a local research copy; never mutate the authenticated kit model."""
    parent = certified.paired_runner.instrumented(second)
    kind = variant["kind"]

    class HoldoutPlant(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r23_kind = kind
            if kind == "contact_surface":
                tables = [self.m.geom(name).id for name in ("table_pick", "table_place")]
                if self.has_box:
                    box = self.m.body("box_free").id
                    tables.extend(np.flatnonzero(self.m.geom_bodyid == box).tolist())
                self.m.geom_friction[tables, 0] *= variant["table_friction_scale"]
                self.m.geom_solref[tables, 0] *= variant["table_solref_time_scale"]
                if self.m.neq:
                    self.m.eq_solref[:, 0] *= variant["suction_solref_time_scale"]
            self.r23_gain = np.asarray(variant.get("joint_gain", np.ones(6)), dtype=float)

        def step(self, tau):
            return super().step(np.asarray(tau, dtype=float) * self.r23_gain)

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
        Controller(),
        box_mass=case["mass"],
        box_com=tuple(case["com"]),
        pick_offset=case.get("offset"),
        detector=detector(),
        collision=impacts[0],
    )
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


def strict(event):
    return bool(event["exactly_one"] and event["quiet_before"])


def event_record(log, references, start, end, quiet, radius):
    row = certified.event_record(log, references, start, end, quiet, radius)
    row["duplicate_rises"] = max(0, len(row["raw_rise_indices"]) - 1)
    row["strict"] = strict(row)
    return row


def tool_floor(reference, candidate, slack):
    return (not reference["detected"] or
            (candidate["detected"] and
             candidate["delay_ms"] <= reference["delay_ms"] + slack))


def tool_gates(reference, candidate, slack):
    return {
        "no_new_tool_prealarm": not candidate["prealarm"] or reference["prealarm"],
        "preserve_r22_tool_detection": tool_floor(reference, candidate, slack),
    }


def record_common(log, impacts, starts, window):
    return {
        "false_times_s": certified.false_record(log, starts, window),
        "force_exact": certified.exact_force(log, impacts),
        "tool": certified.tool_record(log),
        "stuck_tail": bool(log["detect"][-1]),
    }


def negative_case(case, variant, arms, plan):
    traces = {name: run_episode(case, variant, detector, (None, None))
              for name, detector in arms.items()}
    base_hash = certified.physical_hash(traces["r22_light"][1])
    records = {}
    for name, (_, log) in traces.items():
        records[name] = record_common(log, (None, None), [], plan["event_window_ms"])
        records[name]["physical_exact_vs_r22"] = certified.physical_hash(log) == base_hash
    old, new = records["r22_light"], records["r23"]
    safety = {
        "candidate_zero_false_rises": not new["false_times_s"],
        **tool_gates(old["tool"], new["tool"], plan["preservation_slack_ms"]),
        "candidate_no_stuck_tail": not new["stuck_tail"],
        "physical_and_force_exact": all(
            row["physical_exact_vs_r22"] and row["force_exact"]["passed"]
            for row in records.values()),
    }
    return {
        "id": case["id"], "kind": "negative", "variant": variant["kind"],
        "contact_probe": case["contact_probe"], "episodes": len(arms),
        "r22_physics_sha256": certified.trace_signature({"none": traces["r22_light"][1]}),
        "records": records, "safety_floor": safety,
        "scoped_detection": {"required": False, "passed": True},
        "positive_gains": [], "passed": all(safety.values()),
    }


def single_case(case, variant, arms, plan):
    radius = plan["reference_twin_ms"]
    window = plan["event_window_ms"]
    quiet = plan["quiet_ms"]
    traces = {
        name: {
            "none": run_episode(case, variant, detector, (None, None)),
            "force": run_episode(case, variant, detector, (case["impact"], None)),
        }
        for name, detector in arms.items()
    }
    t = traces["r22_light"]["force"][1]["t"]
    start, boundary = certified.actual_start(t, case["impact"])
    base_hash = {tw: certified.physical_hash(pair[1])
                 for tw, pair in traces["r22_light"].items()}
    records = {}
    for name, twins in traces.items():
        none, force = twins["none"][1], twins["force"][1]
        common = {tw: record_common(pair[1],
                                    (case["impact"], None) if tw == "force" else (None, None),
                                    [start] if tw == "force" else [], window)
                  for tw, pair in twins.items()}
        records[name] = {
            "event": event_record(force, [none], start, min(6000, start + window),
                                  quiet, radius),
            "twins": common,
            "prefix": certified.prefix(none, force, start, start),
            "boundary": boundary,
            "physical_exact_vs_r22": all(
                certified.physical_hash(twins[tw][1]) == base_hash[tw] for tw in twins),
        }
    old, new = records["r22_light"], records["r23"]
    old_event, new_event = old["event"], new["event"]
    preserve = (not old_event["strict"] or
                (new_event["strict"] and
                 new_event["delays_ms"][0] <=
                 old_event["delays_ms"][0] + plan["preservation_slack_ms"]))
    safety = {
        "preserve_r22_strict_event": preserve,
        "candidate_zero_duplicate_rises": new_event["duplicate_rises"] == 0,
        "candidate_zero_false_rises": all(not row["false_times_s"]
                                           for row in new["twins"].values()),
        "no_new_tool_prealarm": all(
            not new["twins"][tw]["tool"]["prealarm"] or
            old["twins"][tw]["tool"]["prealarm"] for tw in new["twins"]),
        "preserve_r22_tool_detection": all(
            tool_floor(old["twins"][tw]["tool"], new["twins"][tw]["tool"],
                       plan["preservation_slack_ms"]) for tw in new["twins"]),
        "candidate_no_stuck_tail": all(not row["stuck_tail"]
                                        for row in new["twins"].values()),
        "prefix_exact": all(row["prefix"]["passed"] for row in records.values()),
        "physical_and_force_exact": all(
            row["physical_exact_vs_r22"] and
            all(twin["force_exact"]["passed"] for twin in row["twins"].values())
            for row in records.values()),
    }
    detection = {"required": True, "passed": new_event["strict"]}
    gains = (["event"] if new_event["strict"] and not old_event["strict"] else [])
    return {
        "id": case["id"], "phase": case["phase"], "kind": "single",
        "variant": variant["kind"], "episodes": 2 * len(arms),
        "r22_physics_sha256": certified.trace_signature(
            {tw: pair[1] for tw, pair in traces["r22_light"].items()}),
        "records": records, "safety_floor": safety,
        "scoped_detection": detection, "positive_gains": gains,
        "passed": all(safety.values()) and detection["passed"],
    }


def paired_case(case, variant, arms, plan):
    radius = plan["reference_twin_ms"]
    window = plan["event_window_ms"]
    quiet = plan["quiet_ms"]
    assignment = {
        "none": (None, None),
        "first": (case["impacts"][0], None),
        "second": (None, case["impacts"][1]),
        "both": tuple(case["impacts"]),
    }
    traces = {name: {tw: run_episode(case, variant, detector, impacts)
                     for tw, impacts in assignment.items()}
              for name, detector in arms.items()}
    t = traces["r22_light"]["both"][1]["t"]
    starts = [certified.actual_start(t, impact) for impact in case["impacts"]]
    first, second = starts[0][0], starts[1][0]
    windows = {
        "isolated_first": ("first", first, min(second, first + window), ["none"]),
        "isolated_second": ("second", second, min(6000, second + window), ["none"]),
        "both_first": ("both", first, min(second, first + window), ["none", "second"]),
        "both_second": ("both", second, min(6000, second + window), ["first"]),
    }
    base_hash = {tw: certified.physical_hash(pair[1])
                 for tw, pair in traces["r22_light"].items()}
    records = {}
    for name, twins in traces.items():
        events = {
            slot: event_record(twins[tw][1], [twins[ref][1] for ref in refs],
                               begin, end, quiet, radius)
            for slot, (tw, begin, end, refs) in windows.items()
        }
        common = {
            tw: record_common(pair[1], impacts,
                              [first if impacts[0] else None,
                               second if impacts[1] else None], window)
            for (tw, impacts), pair in zip(assignment.items(), twins.values())
        }
        records[name] = {
            "events": events,
            "twins": common,
            "prefix_first": certified.prefix(twins["none"][1], twins["first"][1],
                                               first, first),
            "prefix_second": certified.prefix(twins["none"][1], twins["second"][1],
                                                second, second),
            "prefix_both_first": certified.prefix(twins["none"][1], twins["both"][1],
                                                    first, first),
            "prefix_both_second": certified.prefix(twins["first"][1], twins["both"][1],
                                                     second, second),
            "boundary": [item[1] for item in starts],
            "physical_exact_vs_r22": all(
                certified.physical_hash(twins[tw][1]) == base_hash[tw] for tw in twins),
        }
    old, new = records["r22_light"], records["r23"]
    preservation = all(
        not event["strict"] or
        (new["events"][slot]["strict"] and
         new["events"][slot]["delays_ms"][0] <=
         event["delays_ms"][0] + plan["preservation_slack_ms"])
        for slot, event in old["events"].items())
    safety = {
        "preserve_r22_strict_slots": preservation,
        "candidate_zero_duplicate_rises": all(
            event["duplicate_rises"] == 0 for event in new["events"].values()),
        "candidate_zero_false_rises": all(not row["false_times_s"]
                                           for row in new["twins"].values()),
        "no_new_tool_prealarm": all(
            not new["twins"][tw]["tool"]["prealarm"] or
            old["twins"][tw]["tool"]["prealarm"] for tw in new["twins"]),
        "preserve_r22_tool_detection": all(
            tool_floor(old["twins"][tw]["tool"], new["twins"][tw]["tool"],
                       plan["preservation_slack_ms"]) for tw in new["twins"]),
        "candidate_no_stuck_tail": all(not row["stuck_tail"]
                                        for row in new["twins"].values()),
        "all_prefixes_exact": all(
            row[key]["passed"] for row in records.values()
            for key in ("prefix_first", "prefix_second",
                        "prefix_both_first", "prefix_both_second")),
        "physical_and_force_exact": all(
            row["physical_exact_vs_r22"] and
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
        "r22_physics_sha256": certified.trace_signature(
            {tw: pair[1] for tw, pair in traces["r22_light"].items()}),
        "records": records, "safety_floor": safety,
        "scoped_detection": detection, "positive_gains": gains,
        "passed": all(safety.values()) and detection["passed"],
    }


def resolve(lock):
    if (lock.get("schema") != "r23-candidate-lock.v1" or
            lock.get("status") != "LOCKED_BEFORE_HOLDOUT_OUTCOMES" or
            lock.get("plan_sha256") != sha(PLAN) or
            lock.get("evaluator_sha256") != sha(Path(__file__))):
        raise ValueError("candidate lock does not match sealed R23 evaluator")
    sources = lock.get("transitive_sources", {})
    for name, expected in sources.items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("candidate transitive source pin differs: " + name)
    module, symbol = lock["entrypoint"].rsplit(":", 1)
    detector = getattr(importlib.import_module(module), symbol)
    return detector


def one(item):
    kind, case, variant, arms, plan = item
    return {"negative": negative_case, "single": single_case,
            "paired": paired_case}[kind](case, variant, arms, plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh output required")
    plan_raw, lock_raw = PLAN.read_bytes(), args.lock.read_bytes()
    plan, lock = json.loads(plan_raw), json.loads(lock_raw)
    protocol = json.loads(PROTOCOL.read_bytes())
    pinned_before = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if pinned_before != protocol["source_pins"]:
        raise ValueError("sealed R23 dependency differs")
    candidate = resolve(lock)
    arms = {"r22_light": R22, "r23": candidate}
    tasks = [(kind, case, variant, arms, plan)
             for kind in ("negative", "single", "paired")
             for case in plan[kind] for variant in plan["variants"].values()]
    before = {name: sha(ROOT / name) for name in lock["transitive_sources"]}
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(one, tasks))
    after = {name: sha(ROOT / name) for name in lock["transitive_sources"]}
    pinned_after = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if pinned_before != pinned_after:
        raise ValueError("sealed dependency changed during holdout")
    if before != after or before != lock["transitive_sources"]:
        raise ValueError("candidate changed during holdout")
    effects = {"stock": True}
    stock_signatures = {(row["kind"], row["id"]): row["r22_physics_sha256"]
                        for row in rows if row["variant"] == "stock"}
    for variant in plan["variants"]:
        if variant == "stock":
            continue
        selected = [row for row in rows if row["variant"] == variant]
        effects[variant] = any(
            row["r22_physics_sha256"] != stock_signatures[(row["kind"], row["id"])]
            for row in selected)
    if not all(effects.values()):
        raise ValueError("declared mechanical intervention had no observed physical effect")
    episodes = sum(row["episodes"] for row in rows)
    if episodes != plan["required_episodes"]["both_arms"]:
        raise ValueError("incomplete episode accounting")
    contact_rows = [row["variant"] + ":" + row["id"] for row in rows
                    if row["kind"] == "negative" and row["contact_probe"] and
                    row["records"]["r22_light"]["tool"]["present"]]
    required_contacts = plan["contact_evidence"][
        "minimum_probe_variant_rows_with_real_r22_contact"]
    contact_adequate = len(contact_rows) >= required_contacts
    safety_pass = all(all(row["safety_floor"].values()) for row in rows)
    detection_pass = all(row["scoped_detection"]["passed"] for row in rows
                         if row["scoped_detection"]["required"])
    gains = [{"variant": row["variant"], "id": row["id"], "kind": row["kind"],
              "slots": row["positive_gains"]} for row in rows if row["positive_gains"]]
    failures = [row["variant"] + ":" + row["id"] for row in rows if not row["passed"]]
    passed = safety_pass and detection_pass and contact_adequate
    document = {
        "schema": "r23-independent-transfer-detachment-result.v1",
        "plan_sha256": sha(PLAN),
        "lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "classification": plan["classification"],
        "arms": list(arms), "episodes": episodes, "source_pins": pinned_before,
        "mechanical_effect_observed": effects,
        "real_contact_evidence": {"rows": contact_rows, "required": required_contacts,
                                  "adequate": contact_adequate},
        "claims": {"safety_qualification": False,
                   "scoped_regression_floor_pass": safety_pass,
                   "scoped_detection_pass": detection_pass,
                   "positive_gain_count": sum(len(row["slots"]) for row in gains)},
        "positive_gains": gains, "rows": rows, "failed": failures, "passed": passed,
    }
    with args.out.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"episodes": episodes, "passed": passed,
                      "failed": len(failures), "positive_gains":
                      document["claims"]["positive_gain_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
