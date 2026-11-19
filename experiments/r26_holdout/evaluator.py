"""Execute the sealed R26 long-vs-close holdout after candidate lock."""
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
import mujoco

from experiments.r25_holdout import evaluator as prior
from experiments.r25_composed.detector import Detector as R25
from experiments.r26_holdout import runner
from my_torque import Controller


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
PROTOCOL = HERE / "protocol.json"
certified = prior.certified
tool_floor = prior.prior.prior.tool_floor
tool_gates = prior.prior.prior.tool_gates


def require_runtime():
    if mujoco.__version__ != "3.17.0":
        raise RuntimeError("R26 requires authenticated MuJoCo 3.17.0")
    return mujoco.__version__


require_runtime()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_episode(case, variant, detector, impacts):
    plant = runner.instrumented(impacts, variant)
    function = types.FunctionType(
        certified.stock.run_episode.__code__,
        dict(certified.stock.run_episode.__globals__, Plant=plant),
        certified.stock.run_episode.__name__, certified.stock.run_episode.__defaults__,
        certified.stock.run_episode.__closure__)
    function.__kwdefaults__ = certified.stock.run_episode.__kwdefaults__
    metrics, log = function(
        Controller(), box_mass=case["mass"], box_com=tuple(case["com"]),
        pick_offset=case.get("offset"), detector=detector(), collision=None)
    owner = log["plant"]
    log["f_ext_xyz_native"] = np.asarray(owner.r26_first)
    log["f_ext_xyz_second"] = np.asarray(owner.r26_second)
    log["f_ext_xyz_committed"] = np.asarray(owner.r26_committed)
    log["f_ext"] = np.linalg.norm(log["f_ext_xyz_native"], axis=1)
    if any(np.asarray(log[key]).shape[0] != 6000
           for key in certified.PHYSICAL + ("r", "detect")):
        raise ValueError("incomplete 6000-sample trace")
    if any(not np.isfinite(np.asarray(log[key])).all()
           for key in certified.PHYSICAL + ("r", "detect")):
        raise ValueError("nonfinite R26 trace")
    if metrics["diverged"]:
        raise ValueError("R26 episode diverged")
    return metrics, log


def exact_force(log, impacts):
    target = runner.expected(log["t"], impacts)
    gates = {
        "first": bool(np.array_equal(log["f_ext_xyz_native"], target["first"])),
        "second": bool(np.array_equal(log["f_ext_xyz_second"], target["second"])),
        "committed": bool(np.array_equal(log["f_ext_xyz_committed"], target["committed"])),
        "scalar": bool(np.array_equal(log["f_ext"], target["scalar"])),
    }
    return {"passed": all(gates.values()), "gates": gates}


def actual_start(t, impact):
    values = np.stack([runner.pulse(value, impact) for value in t])
    active = np.flatnonzero(np.any(values != 0.0, axis=1))
    if not len(active):
        raise ValueError("nonzero registered pulse required")
    nominal = int(np.searchsorted(t, impact[0]))
    return int(active[0]), {"nominal_index": nominal,
                            "first_nonzero_index": int(active[0]),
                            "last_nonzero_index": int(active[-1]),
                            "duration_samples": int(len(active)),
                            "nominal_sample_zero": bool(not np.any(values[nominal]))}


def event(log, references, start, end, plan):
    return prior.event(log, references, start, end, plan)


def common(log, impacts, starts, plan):
    return {"false_times_s": certified.false_record(log, starts, plan["event_window_ms"]),
            "force_exact": exact_force(log, impacts),
            "tool": certified.tool_record(log),
            "stuck_tail": bool(log["detect"][-1])}


def preserved(reference, candidate, plan):
    return prior.preserved(reference, candidate, plan)


def negative_case(case, variant, arms, plan):
    traces = {name: run_episode(case, variant, detector, (None, None))
              for name, detector in arms.items()}
    base = certified.physical_hash(traces["r25_joint"][1])
    records = {}
    for name, (_, log) in traces.items():
        records[name] = common(log, (None, None), [], plan)
        records[name]["physical_exact_vs_r25"] = certified.physical_hash(log) == base
    old, new = records["r25_joint"], records["r26"]
    protection = {
        "candidate_zero_false_rises": not new["false_times_s"],
        **tool_gates(old["tool"], new["tool"], plan["preservation_slack_ms"]),
        "candidate_no_stuck_tail": not new["stuck_tail"],
        "physical_and_force_exact": all(
            row["physical_exact_vs_r25"] and row["force_exact"]["passed"]
            for row in records.values()),
    }
    return {"id": case["id"], "role": case["role"], "kind": "negative",
            "variant": variant["kind"], "contact_probe": case["contact_probe"],
            "episodes": len(arms),
            "r25_physics_sha256": certified.trace_signature({"none": traces["r25_joint"][1]}),
            "records": records, "incumbent_safety": protection,
            "positive_qualification": {"required": False, "passed": True},
            "positive_gains": [], "absolute_misses": [],
            "passed": all(protection.values())}


def single_case(case, variant, arms, plan):
    assignment = {"none": (None, None), "single": (case["impact"], None)}
    traces = {name: {tw: run_episode(case, variant, detector, impacts)
                     for tw, impacts in assignment.items()} for name, detector in arms.items()}
    start, boundary = actual_start(traces["r25_joint"]["single"][1]["t"], case["impact"])
    base = {tw: certified.physical_hash(pair[1])
            for tw, pair in traces["r25_joint"].items()}
    records = {}
    for name, twins in traces.items():
        none, forced = twins["none"][1], twins["single"][1]
        records[name] = {
            "event": event(forced, [none], start,
                           min(6000, start + plan["event_window_ms"]), plan),
            "twins": {tw: common(pair[1], assignment[tw],
                                  [start] if tw == "single" else [], plan)
                      for tw, pair in twins.items()},
            "prefix": certified.prefix(none, forced, start, start),
            "boundary": boundary,
            "physical_exact_vs_r25": all(
                certified.physical_hash(twins[tw][1]) == base[tw] for tw in twins),
        }
    old, new = records["r25_joint"], records["r26"]
    floor = preserved(old["event"], new["event"], plan)
    protection = {
        "preserve_r25_strict": floor["event"],
        "preserve_r25_strict_duration": floor["duration"],
        "candidate_zero_duplicate": new["event"]["duplicate_rises"] == 0,
        "candidate_zero_false": all(not row["false_times_s"] for row in new["twins"].values()),
        "preserve_real_tool": all(tool_floor(old["twins"][tw]["tool"],
                                             new["twins"][tw]["tool"],
                                             plan["preservation_slack_ms"])
                                  for tw in new["twins"]),
        "candidate_no_stuck_tail": all(not row["stuck_tail"] for row in new["twins"].values()),
        "prefix_exact": all(row["prefix"]["passed"] for row in records.values()),
        "physical_and_force_exact": all(
            row["physical_exact_vs_r25"] and
            all(twin["force_exact"]["passed"] for twin in row["twins"].values())
            for row in records.values()),
    }
    qualification = {"required": True, "passed": new["event"]["strict"]}
    gains = ["single"] if new["event"]["strict"] and not old["event"]["strict"] else []
    misses = [] if new["event"]["strict"] else ["single"]
    return {"id": case["id"], "phase": case["phase"], "kind": "single",
            "variant": variant["kind"], "episodes": 2 * len(arms),
            "r25_physics_sha256": certified.trace_signature(
                {tw: pair[1] for tw, pair in traces["r25_joint"].items()}),
            "records": records, "incumbent_safety": protection,
            "positive_qualification": qualification, "positive_gains": gains,
            "absolute_misses": misses,
            "passed": all(protection.values()) and qualification["passed"]}


def paired_case(case, variant, arms, plan):
    assignment = {"none": (None, None), "long": (case["long"], None),
                  "first": (case["impacts"][0], None),
                  "second": (None, case["impacts"][1]),
                  "both": tuple(case["impacts"])}
    traces = {name: {tw: run_episode(case, variant, detector, impacts)
                     for tw, impacts in assignment.items()} for name, detector in arms.items()}
    t = traces["r25_joint"]["both"][1]["t"]
    long_start, long_boundary = actual_start(t, case["long"])
    starts = [actual_start(t, impact) for impact in case["impacts"]]
    first, second = starts[0][0], starts[1][0]
    window = plan["event_window_ms"]
    slots = {"long_single": ("long", long_start, min(6000, long_start + window), ["none"]),
             "isolated_first": ("first", first, min(second, first + window), ["none"]),
             "isolated_second": ("second", second, min(6000, second + window), ["none"]),
             "both_first": ("both", first, min(second, first + window), ["none", "second"]),
             "both_second": ("both", second, min(6000, second + window), ["first"])}
    base = {tw: certified.physical_hash(pair[1])
            for tw, pair in traces["r25_joint"].items()}
    records = {}
    for name, twins in traces.items():
        events = {slot: event(twins[tw][1], [twins[ref][1] for ref in refs],
                              begin, end, plan)
                  for slot, (tw, begin, end, refs) in slots.items()}
        records[name] = {
            "events": events,
            "twins": {tw: common(pair[1], impacts,
                                  [first if impacts[0] else None,
                                   second if impacts[1] else None], plan)
                      for (tw, impacts), pair in zip(assignment.items(), twins.values())},
            "prefix_long": certified.prefix(twins["none"][1], twins["long"][1],
                                             long_start, long_start),
            "prefix_first": certified.prefix(twins["none"][1], twins["first"][1], first, first),
            "prefix_second": certified.prefix(twins["none"][1], twins["second"][1], second, second),
            "prefix_both_first": certified.prefix(twins["none"][1], twins["both"][1], first, first),
            "prefix_both_second": certified.prefix(twins["first"][1], twins["both"][1], second, second),
            "boundary": {"long": long_boundary, "first": starts[0][1], "second": starts[1][1]},
            "physical_exact_vs_r25": all(
                certified.physical_hash(twins[tw][1]) == base[tw] for tw in twins),
        }
    old, new = records["r25_joint"], records["r26"]
    floors = {slot: preserved(old_event, new["events"][slot], plan)
              for slot, old_event in old["events"].items()}
    protection = {
        "preserve_r25_strict": all(row["event"] for row in floors.values()),
        "preserve_r25_strict_duration": all(row["duration"] for row in floors.values()),
        "candidate_zero_duplicate": all(e["duplicate_rises"] == 0 for e in new["events"].values()),
        "candidate_zero_false": all(not row["false_times_s"] for row in new["twins"].values()),
        "preserve_real_tool": all(tool_floor(old["twins"][tw]["tool"],
                                             new["twins"][tw]["tool"],
                                             plan["preservation_slack_ms"])
                                  for tw in new["twins"]),
        "candidate_no_stuck_tail": all(not row["stuck_tail"] for row in new["twins"].values()),
        "all_prefixes_exact": all(row[key]["passed"] for row in records.values()
                                      for key in ("prefix_long", "prefix_first", "prefix_second",
                                                  "prefix_both_first", "prefix_both_second")),
        "physical_and_force_exact": all(
            row["physical_exact_vs_r25"] and
            all(twin["force_exact"]["passed"] for twin in row["twins"].values())
            for row in records.values()),
    }
    qualification = {"required": True,
                     "passed": all(e["strict"] for e in new["events"].values())}
    gains = [slot for slot, e in new["events"].items()
             if e["strict"] and not old["events"][slot]["strict"]]
    misses = [slot for slot, e in new["events"].items() if not e["strict"]]
    return {"id": case["id"], "phase": case["phase"], "kind": "paired",
            "variant": variant["kind"], "episodes": 5 * len(arms),
            "r25_physics_sha256": certified.trace_signature(
                {tw: pair[1] for tw, pair in traces["r25_joint"].items()}),
            "records": records, "incumbent_safety": protection,
            "positive_qualification": qualification, "positive_gains": gains,
            "absolute_misses": misses,
            "passed": all(protection.values()) and qualification["passed"]}


def dry_validate():
    plan = json.loads(PLAN.read_bytes())
    protocol = json.loads(PROTOCOL.read_bytes())
    old_plan = json.loads(prior.PLAN.read_bytes())
    cases = [(kind, row["id"]) for kind in ("negative", "single", "paired")
             for row in plan[kind]]
    per_arm = len(plan["variants"]) * (len(plan["negative"]) +
                                       2 * len(plan["single"]) +
                                       5 * len(plan["paired"]))
    if len(cases) != len(set(cases)) or len(cases) != 10 or per_arm != 78 \
            or plan["required_episodes"] != {"per_arm": 78, "both_arms": 156}:
        raise ValueError("R26 accounting differs")
    old_rows = [row for kind in ("negative", "single", "paired")
                for row in old_plan[kind]]
    new_rows = [row for kind in ("negative", "single", "paired")
                for row in plan[kind]]
    old_boxes = {(row["mass"], tuple(row["com"]), tuple(row.get("offset") or ()))
                 for row in old_rows}
    new_boxes = {(row["mass"], tuple(row["com"]), tuple(row.get("offset") or ()))
                 for row in new_rows}
    old_impacts = [impact for row in old_plan["single"] for impact in [row["impact"]]]
    old_impacts += [impact for row in old_plan["paired"]
                    for impact in [row["long"], *row["impacts"]]]
    new_impacts = [impact for row in plan["single"] for impact in [row["impact"]]]
    new_impacts += [impact for row in plan["paired"]
                    for impact in [row["long"], *row["impacts"]]]
    if (old_boxes & new_boxes
            or {row["id"] for row in old_rows} & {row["id"] for row in new_rows}
            or {impact[0] for impact in old_impacts} & {impact[0] for impact in new_impacts}
            or {impact[1] for impact in old_impacts} & {impact[1] for impact in new_impacts}
            or {tuple(impact[2]) for impact in old_impacts}
               & {tuple(impact[2]) for impact in new_impacts}
            or set(map(json.dumps, [value for key, value in old_plan["variants"].items()
                                    if key != "stock"]))
               & set(map(json.dumps, [value for key, value in plan["variants"].items()
                                      if key != "stock"]))):
        raise ValueError("R26 conditions are not independent of R25")
    t = .001 * np.arange(6000)
    for row in plan["single"]:
        actual_start(t, row["impact"])
    for row in plan["paired"]:
        actual_start(t, row["long"])
        support = [actual_start(t, impact)[1] for impact in row["impacts"]]
        if support[0]["last_nonzero_index"] >= support[1]["first_nonzero_index"]:
            raise ValueError("R26 close pulses overlap")
    actual = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if actual != protocol["source_pins"]:
        raise ValueError("R26 protocol source differs")
    return {"schema": plan["schema"], "cases": 10, "episodes": 156,
            "variants": 3, "worker_cap": 2, "mujoco": require_runtime()}


def resolve(lock):
    if (lock.get("schema") != "r26-candidate-lock.v1" or
            lock.get("status") != "LOCKED_BEFORE_HOLDOUT_OUTCOMES" or
            lock.get("plan_sha256") != sha(PLAN) or
            lock.get("protocol_sha256") != sha(PROTOCOL) or
            lock.get("evaluator_sha256") != sha(Path(__file__))):
        raise ValueError("candidate lock does not match sealed R26 evaluator")
    for name, expected in lock.get("transitive_sources", {}).items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("candidate source pin differs: " + name)
    module, symbol = lock["entrypoint"].rsplit(":", 1)
    if (not module.startswith("experiments.r26_")
            or module.startswith("experiments.r26_holdout")
            or symbol != "Detector"):
        raise ValueError("new non-holdout R26 Detector required")
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
        print(json.dumps(dry_validate(), sort_keys=True))
        return 0
    if args.lock is None or args.out is None:
        parser.error("--lock and --out required")
    if os.path.lexists(args.out):
        parser.error("fresh output required")
    plan, lock = json.loads(PLAN.read_bytes()), json.loads(args.lock.read_bytes())
    protocol = json.loads(PROTOCOL.read_bytes())
    pins = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if pins != protocol["source_pins"]:
        raise ValueError("sealed R26 dependency differs")
    candidate = resolve(lock)
    arms = {"r25_joint": R25, "r26": candidate}
    tasks = [(kind, case, variant, arms, plan)
             for kind in ("negative", "single", "paired")
             for case in plan[kind] for variant in plan["variants"].values()]
    before = {name: sha(ROOT / name) for name in lock["transitive_sources"]}
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(one, tasks))
    if before != {name: sha(ROOT / name) for name in lock["transitive_sources"]}:
        raise ValueError("candidate changed during R26 holdout")
    effects = {"stock": True}
    stock = {(r["kind"], r["id"]): r["r25_physics_sha256"]
             for r in rows if r["variant"] == "stock"}
    for variant in plan["variants"]:
        if variant != "stock":
            effects[variant] = any(r["r25_physics_sha256"] != stock[(r["kind"], r["id"])]
                                   for r in rows if r["variant"] == variant)
    episodes = sum(r["episodes"] for r in rows)
    if episodes != 156 or not all(effects.values()):
        raise ValueError("R26 accounting or mechanics differs")
    contact_rows = [r["variant"] + ":" + r["id"] for r in rows
                    if r["kind"] == "negative" and r["contact_probe"] and
                    r["records"]["r25_joint"]["tool"]["present"]]
    required = plan["contact_evidence"]["minimum_probe_variant_rows_with_real_contact"]
    contact = {"rows": contact_rows, "required": required,
               "adequate": len(contact_rows) >= required}
    safety = all(all(r["incumbent_safety"].values()) for r in rows)
    positive = all(r["positive_qualification"]["passed"] for r in rows
                   if r["positive_qualification"]["required"])
    gains = [{"variant": r["variant"], "id": r["id"], "kind": r["kind"],
              "slots": r["positive_gains"] } for r in rows if r["positive_gains"]]
    misses = [{"variant": r["variant"], "id": r["id"], "kind": r["kind"],
               "slots": r["absolute_misses"]} for r in rows if r["absolute_misses"]]
    failed = [r["variant"] + ":" + r["id"] for r in rows if not r["passed"]]
    document = {"schema": "r26-independent-long-close-result.v1",
                "plan_sha256": sha(PLAN), "lock_sha256": sha(args.lock),
                "classification": plan["classification"], "arms": list(arms),
                "episodes": episodes, "source_pins": pins,
                "mechanical_effect_observed": effects, "real_contact_evidence": contact,
                "claims": {"safety_qualification": False,
                           "incumbent_safety_floor_pass": safety,
                           "positive_qualification_pass": positive,
                           "positive_gain_count": sum(len(g["slots"]) for g in gains),
                           "absolute_miss_count": sum(len(m["slots"]) for m in misses)},
                "positive_gains": gains, "absolute_misses": misses,
                "rows": rows, "failed": failed,
                "passed": safety and positive and contact["adequate"]}
    with args.out.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"episodes": episodes, "failed": len(failed),
                      "passed": document["passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
