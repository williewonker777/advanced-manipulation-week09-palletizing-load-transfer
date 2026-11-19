"""Execute the sealed R22 three-arm challenge after a candidate source lock exists."""
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

import my_detector
from my_torque import Controller
from experiments.r18_paired_impulses import runner as paired_runner
from experiments.r21_yaw_impulse.coarse import Detector as R21
import sim.runner as stock
from tools.r14_paired_events import match_false_events


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
PROTOCOL = HERE / "protocol.json"
PHYSICAL = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "f_ext",
            "f_ext_xyz_native", "f_ext_xyz_second", "f_ext_xyz_committed")
SENSED = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "r", "detect")
CURRENT_TICK_CAUSAL = ("t", "q_d", "tau", "r", "detect")
FORCE = {"native": "f_ext_xyz_native", "extra": "f_ext_xyz_second",
         "combined": "f_ext_xyz_committed", "scalar": "f_ext"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signal_sha(value):
    data = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(data.dtype.str.encode("ascii"))
    digest.update(json.dumps(data.shape, separators=(",", ":")).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def pulse(t, impact):
    return paired_runner.pulse(float(t), impact)


def expected(t, impacts):
    native = np.stack([pulse(value, impacts[0]) for value in t])
    extra = np.stack([pulse(value, impacts[1]) for value in t])
    combined = native + extra
    scalar = np.fromiter((np.linalg.norm(row) for row in native), float, count=len(native))
    return {"native": native, "extra": extra, "combined": combined, "scalar": scalar}


def exact_force(log, impacts):
    target = expected(log["t"], impacts)
    gates = {key: bool(np.array_equal(np.asarray(log[name]), target[key]))
             for key, name in FORCE.items()}
    return {"passed": all(gates.values()), "gates": gates}


def local_plant(second, variant):
    parent = paired_runner.instrumented(second)
    kind = variant["kind"]

    class ChallengePlant(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r22_kind = kind
            if kind == "contact_suction":
                tables = [self.m.geom(name).id for name in ("table_pick", "table_place")]
                if self.has_box:
                    box = self.m.body("box_free").id
                    tables.extend(np.flatnonzero(self.m.geom_bodyid == box).tolist())
                self.m.geom_friction[tables, 0] *= variant["table_friction_scale"]
                self.m.geom_solref[tables, 0] *= variant["table_solref_time_scale"]
                if self.m.neq:
                    self.m.eq_solref[:, 0] *= variant["suction_solref_time_scale"]
            self.r22_gain = np.asarray(variant.get("joint_gain", np.ones(6)), dtype=float)

        def step(self, tau):
            return super().step(np.asarray(tau, dtype=float) * self.r22_gain)

    return ChallengePlant


def run_episode(case, variant, detector, impacts):
    plant = local_plant(impacts[1], variant)
    function = types.FunctionType(stock.run_episode.__code__,
                                  dict(stock.run_episode.__globals__, Plant=plant),
                                  stock.run_episode.__name__, stock.run_episode.__defaults__,
                                  stock.run_episode.__closure__)
    function.__kwdefaults__ = stock.run_episode.__kwdefaults__
    metrics, log = function(Controller(), box_mass=case["mass"], box_com=tuple(case["com"]),
                            pick_offset=case.get("offset"), detector=detector(),
                            collision=impacts[0])
    owner = log["plant"]
    log["f_ext_xyz_native"] = np.asarray(owner.r18_native_force)
    log["f_ext_xyz_second"] = np.asarray(owner.r18_second_force)
    log["f_ext_xyz_committed"] = np.asarray(owner.r18_committed_force)
    log["r22_diverged"] = metrics["diverged"]
    if any(np.asarray(log[key]).shape[0] != 6000 for key in PHYSICAL + ("r", "detect")):
        raise ValueError("incomplete 6000-sample trace")
    if any(not np.isfinite(np.asarray(log[key])).all() for key in PHYSICAL + ("r", "detect")):
        raise ValueError("nonfinite challenge trace")
    if metrics["diverged"]:
        raise ValueError("challenge episode diverged")
    return metrics, log


def rises(log):
    bits = np.asarray(log["detect"], bool)
    return np.flatnonzero(bits & ~np.r_[False, bits[:-1]])


def intervals(log, starts, window):
    true = np.zeros(6000, bool)
    for start in starts:
        if start is not None:
            true[start:min(6000, start + window)] = True
    tool = np.asarray(log["f_tool"]) > 1.
    if tool.any():
        first, last = np.flatnonzero(tool)[[0, -1]]
        true[first:min(6000, last + 301)] = True
    return true


def tool_record(log):
    contact = np.asarray(log["f_tool"]) > 1.
    bits = np.asarray(log["detect"], bool)
    if not contact.any():
        return {"present": False, "detected": False, "prealarm": False, "delay_ms": None,
                "peak_N": float(np.max(log["f_tool"]))}
    first, last = np.flatnonzero(contact)[[0, -1]]
    after = np.flatnonzero(bits[first:min(6000, last + 301)])
    return {"present": True, "detected": bool(len(after)), "prealarm": bool(bits[:first].any()),
            "delay_ms": None if not len(after) else int(after[0]), "peak_N": float(np.max(log["f_tool"]))}


def actual_start(t, impact):
    values = np.stack([pulse(value, impact) for value in t])
    active = np.flatnonzero(np.any(values != 0., axis=1))
    if not len(active):
        raise ValueError("nonzero programmed impulse required")
    nominal = int(np.searchsorted(t, impact[0]))
    return int(active[0]), {"nominal_index": nominal, "first_nonzero_index": int(active[0]),
                            "nominal_sample_zero": bool(not np.any(values[nominal]))}


def prefix(reference, forced, stop, through):
    before = {key: bool(np.array_equal(np.asarray(reference[key])[:stop],
                                       np.asarray(forced[key])[:stop])) for key in SENSED}
    # The runner logs q/qpos/contact after plant.step, so only detector output and
    # same-tick command/reference fields can be required through the first nonzero
    # force tick. Every retained field is still exact strictly before that input.
    causal = {key: bool(np.array_equal(np.asarray(reference[key])[:through + 1],
                                       np.asarray(forced[key])[:through + 1]))
              for key in CURRENT_TICK_CAUSAL}
    return {"force_free_before_actual_input": before, "detector_through_first_nonzero": causal,
            "passed": all(before.values()) and all(causal.values())}


def match_reference(hit, references, radius):
    return any(np.any(np.abs(rises(log) - hit) <= radius) for log in references)


def event_record(log, references, start, end, quiet, radius):
    edges = rises(log)
    raw = edges[(edges >= start) & (edges < end)]
    attributable = [int(hit) for hit in raw if not match_reference(int(hit), references, radius)]
    bits = np.asarray(log["detect"], bool)
    return {"raw_rise_indices": raw.astype(int).tolist(), "attributable_rise_indices": attributable,
            "delays_ms": [hit - start for hit in attributable],
            "quiet_before": bool(not bits[max(0, start - quiet):start].any()),
            "high_before": bool(start > 0 and bits[start - 1]),
            "exactly_one": len(raw) == len(attributable) == 1}


def false_record(log, starts, window):
    edge = rises(log)
    valid = intervals(log, starts, window)
    return np.asarray(log["t"])[edge[~valid[edge]]].tolist()


def physical_hash(log):
    return {key: signal_sha(log[key]) for key in PHYSICAL}


def trace_signature(logs):
    digest = hashlib.sha256()
    for name, log in sorted(logs.items()):
        digest.update(name.encode("ascii"))
        for key, value in sorted(physical_hash(log).items()):
            digest.update(key.encode("ascii"))
            digest.update(value.encode("ascii"))
    return digest.hexdigest()


def negative_case(case, variant, arms, plan):
    traces = {name: run_episode(case, variant, detector, (None, None))
              for name, detector in arms.items()}
    base_hash = physical_hash(traces["r10"][1])
    records = {}
    for name, (metrics, log) in traces.items():
        records[name] = {"false_times_s": false_record(log, [], plan["event_window_ms"]),
                         "tool": tool_record(log), "physical_exact_vs_r10": physical_hash(log) == base_hash,
                         "force_exact": exact_force(log, (None, None)),
                         "stuck_tail": bool(log["detect"][-1])}
    base, old, new = records["r10"], records["r21_yaw"], records["r22"]
    comparisons = {
        "no_added_false_vs_r10": match_false_events(new["false_times_s"], base["false_times_s"])["passed"],
        "no_added_false_vs_r21": match_false_events(new["false_times_s"], old["false_times_s"])["passed"],
        "no_new_tool_prealarm": not new["tool"]["prealarm"] or base["tool"]["prealarm"] or old["tool"]["prealarm"],
        "preserve_tool_r10": not base["tool"]["detected"] or (new["tool"]["detected"] and new["tool"]["delay_ms"] <= base["tool"]["delay_ms"] + 5),
        "preserve_tool_r21": not old["tool"]["detected"] or (new["tool"]["detected"] and new["tool"]["delay_ms"] <= old["tool"]["delay_ms"] + 5),
        "no_new_stuck_tail": not new["stuck_tail"] or base["stuck_tail"] or old["stuck_tail"],
        "physical_and_force_exact": all(row["physical_exact_vs_r10"] and row["force_exact"]["passed"] for row in records.values()),
    }
    return {"id": case["id"], "kind": "negative", "variant": variant["kind"], "episodes": len(arms),
            "r10_physics_sha256": trace_signature({"none": traces["r10"][1]}),
            "records": records, "comparison": comparisons, "passed": all(comparisons.values())}


def single_case(case, variant, arms, plan):
    radius, window, quiet = plan["reference_twin_ms"], plan["event_window_ms"], plan["quiet_ms"]
    traces, records = {}, {}
    for name, detector in arms.items():
        traces[name] = {"none": run_episode(case, variant, detector, (None, None)),
                        "force": run_episode(case, variant, detector, (case["impact"], None))}
    t = traces["r10"]["force"][1]["t"]
    start, boundary = actual_start(t, case["impact"])
    base_hash = {tw: physical_hash(traces["r10"][tw][1]) for tw in ("none", "force")}
    for name, twins in traces.items():
        none, force = twins["none"][1], twins["force"][1]
        records[name] = {
            "event": event_record(force, [none], start, min(6000, start + window), quiet, radius),
            "none_false_s": false_record(none, [], window),
            "force_false_s": false_record(force, [start], window),
            "prefix": prefix(none, force, start, start), "boundary": boundary,
            "force_exact": exact_force(force, (case["impact"], None)),
            "physical_exact_vs_r10": all(physical_hash(twins[tw][1]) == base_hash[tw] for tw in twins),
            "tool": tool_record(force),
            "stuck_tail": {tw: bool(pair[1]["detect"][-1]) for tw, pair in twins.items()},
        }
    base, old, new = records["r10"], records["r21_yaw"], records["r22"]
    comparisons = {
        "r22_one_to_one": new["event"]["exactly_one"] and new["event"]["quiet_before"],
        "preserve_r10": not base["event"]["exactly_one"] or (new["event"]["exactly_one"] and new["event"]["delays_ms"][0] <= base["event"]["delays_ms"][0] + 5),
        "preserve_r21": not old["event"]["exactly_one"] or (new["event"]["exactly_one"] and new["event"]["delays_ms"][0] <= old["event"]["delays_ms"][0] + 5),
        "no_added_none_false": all(match_false_events(new["none_false_s"], ref["none_false_s"])["passed"] for ref in (base, old)),
        "no_added_force_false": all(match_false_events(new["force_false_s"], ref["force_false_s"])["passed"] for ref in (base, old)),
        "no_new_stuck_tail": all(not new["stuck_tail"][tw] or base["stuck_tail"][tw] or old["stuck_tail"][tw]
                                  for tw in new["stuck_tail"]),
        "prefix_exact": all(row["prefix"]["passed"] for row in records.values()),
        "physical_and_force_exact": all(row["physical_exact_vs_r10"] and row["force_exact"]["passed"] for row in records.values()),
    }
    return {"id": case["id"], "kind": "single", "variant": variant["kind"], "episodes": 2 * len(arms),
            "r10_physics_sha256": trace_signature({tw: pair[1] for tw, pair in traces["r10"].items()}),
            "records": records, "comparison": comparisons, "passed": all(comparisons.values())}


def paired_case(case, variant, arms, plan):
    radius, window, quiet = plan["reference_twin_ms"], plan["event_window_ms"], plan["quiet_ms"]
    assignment = {"none": (None, None), "first": (case["impacts"][0], None),
                  "second": (None, case["impacts"][1]), "both": tuple(case["impacts"])}
    traces = {name: {tw: run_episode(case, variant, detector, impacts)
                     for tw, impacts in assignment.items()} for name, detector in arms.items()}
    t = traces["r10"]["both"][1]["t"]
    starts = [actual_start(t, impact) for impact in case["impacts"]]
    first, second = starts[0][0], starts[1][0]
    windows = {"isolated_first": ("first", first, min(second, first + window), ["none"]),
               "isolated_second": ("second", second, min(6000, second + window), ["none"]),
               "both_first": ("both", first, min(second, first + window), ["none", "second"]),
               "both_second": ("both", second, min(6000, second + window), ["first"])}
    base_hash = {tw: physical_hash(traces["r10"][tw][1]) for tw in assignment}
    records = {}
    for name, twins in traces.items():
        events = {slot: event_record(twins[tw][1], [twins[ref][1] for ref in refs], begin, end,
                                     quiet, radius) for slot, (tw, begin, end, refs) in windows.items()}
        records[name] = {
            "events": events,
            "false_s": {tw: false_record(pair[1], [first if impacts[0] else None,
                                                     second if impacts[1] else None], window)
                        for (tw, impacts), pair in zip(assignment.items(), twins.values())},
            "prefix_first": prefix(twins["none"][1], twins["first"][1], first, first),
            "prefix_second": prefix(twins["none"][1], twins["second"][1], second, second),
            "prefix_both_first": prefix(twins["none"][1], twins["both"][1], first, first),
            "prefix_both_second": prefix(twins["first"][1], twins["both"][1], second, second),
            "boundary": [item[1] for item in starts],
            "force_exact": {tw: exact_force(pair[1], assignment[tw]) for tw, pair in twins.items()},
            "physical_exact_vs_r10": all(physical_hash(twins[tw][1]) == base_hash[tw] for tw in twins),
            "stuck_tail": {tw: bool(pair[1]["detect"][-1]) for tw, pair in twins.items()},
        }
    base, old, new = records["r10"], records["r21_yaw"], records["r22"]
    comparisons = {"r22_one_to_one_4_slots": all(event["exactly_one"] and event["quiet_before"] for event in new["events"].values())}
    for comparator, record in (("r10", base), ("r21", old)):
        comparisons["preserve_" + comparator] = all(
            not event["exactly_one"] or (new["events"][slot]["exactly_one"] and
                                          new["events"][slot]["delays_ms"][0] <= event["delays_ms"][0] + 5)
            for slot, event in record["events"].items())
        comparisons["no_added_false_vs_" + comparator] = all(
            match_false_events(new["false_s"][tw], record["false_s"][tw])["passed"] for tw in assignment)
    comparisons["all_prefixes_exact"] = all(record[key]["passed"] for record in records.values()
                                               for key in ("prefix_first", "prefix_second", "prefix_both_first", "prefix_both_second"))
    comparisons["no_new_stuck_tail"] = all(not new["stuck_tail"][tw] or base["stuck_tail"][tw] or old["stuck_tail"][tw]
                                              for tw in assignment)
    comparisons["physical_and_force_exact"] = all(
        record["physical_exact_vs_r10"] and all(item["passed"] for item in record["force_exact"].values())
        for record in records.values())
    return {"id": case["id"], "kind": "paired", "variant": variant["kind"], "episodes": 4 * len(arms),
            "r10_physics_sha256": trace_signature({tw: pair[1] for tw, pair in traces["r10"].items()}),
            "records": records, "comparison": comparisons, "passed": all(comparisons.values())}


def resolve(lock):
    if lock["status"] != "LOCKED_BEFORE_CHALLENGE_OUTCOMES" or lock["plan_sha256"] != sha(PLAN) \
            or lock["evaluator_sha256"] != sha(Path(__file__)):
        raise ValueError("candidate lock does not match this sealed evaluator")
    for name, expected in lock["candidate_sources"].items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("candidate source pin differs: " + name)
    module, symbol = lock["entrypoint"].rsplit(":", 1)
    detector = getattr(importlib.import_module(module), symbol)
    return detector


def one(item):
    kind, case, variant, arms, plan = item
    return {"negative": negative_case, "single": single_case, "paired": paired_case}[kind](case, variant, arms, plan)


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
        raise ValueError("sealed challenge dependency differs")
    candidate = resolve(lock)
    arms = {"r10": my_detector.Detector, "r21_yaw": R21, "r22": candidate}
    tasks = [(kind, case, variant, arms, plan) for kind in ("negative", "single", "paired")
             for case in plan[kind] for variant in plan["variants"].values()]
    before = {name: sha(ROOT / name) for name in lock["candidate_sources"]}
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(one, tasks))
    after = {name: sha(ROOT / name) for name in lock["candidate_sources"]}
    pinned_after = {name: sha(ROOT / name) for name in protocol["source_pins"]}
    if pinned_before != pinned_after:
        raise ValueError("sealed dependency changed during challenge")
    if before != after or before != lock["candidate_sources"]:
        raise ValueError("candidate changed during challenge")
    effects = {"stock": True}
    stock_signatures = {(row["kind"], row["id"]): row["r10_physics_sha256"]
                        for row in rows if row["variant"] == "stock"}
    for variant in plan["variants"]:
        if variant == "stock":
            continue
        rows_v = [row for row in rows if row["variant"] == variant]
        effects[variant] = any(row["r10_physics_sha256"] != stock_signatures[(row["kind"], row["id"])]
                               for row in rows_v)
    # Physical equality is required across arms, not between deliberately different mechanics.
    if not all(effects.values()):
        raise ValueError("declared mechanical intervention had no observed physical effect")
    episodes = sum(row["episodes"] for row in rows)
    if episodes != plan["required_episodes"]["all_three_arms"]:
        raise ValueError("incomplete episode accounting")
    failures = [row["variant"] + ":" + row["id"] for row in rows if not row["passed"]]
    document = {"schema": "r22-independent-protection-result.v1", "plan_sha256": sha(PLAN),
                "lock_sha256": hashlib.sha256(lock_raw).hexdigest(), "classification": plan["classification"],
                "arms": list(arms), "episodes": episodes, "source_pins": pinned_before, "mechanical_effect_observed": effects,
                "rows": rows, "failed": failures, "passed": not failures}
    with args.out.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"episodes": episodes, "passed": document["passed"], "failed": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
