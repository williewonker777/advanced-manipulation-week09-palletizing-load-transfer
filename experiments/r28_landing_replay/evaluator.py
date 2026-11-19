"""Run the fixed known R28 landing comparison with passive detector arms."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import types

import numpy as np

from sim import config as C
from sim import runner as stock
from sim.plant import Plant as StockPlant


HERE = Path(__file__).resolve().parent
PLAN = HERE / "plan.json"
PHYSICAL = (
    "t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "f_ext",
    "f_ext_xyz_native", "f_ext_xyz_second", "f_ext_xyz_committed",
)


def resolve(entrypoint):
    module, symbol = entrypoint.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def signal_sha(value):
    data = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(data.dtype.str.encode("ascii"))
    digest.update(json.dumps(data.shape, separators=(",", ":")).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def physical_hash(log):
    return {key: signal_sha(log[key]) for key in PHYSICAL}


def pulse(t, impact):
    if impact is None:
        return np.zeros(3)
    onset, vector = impact
    age = float(t) - float(onset)
    weight = min(age / stock.COLLISION_RAMP, 1.0,
                 (2 * stock.COLLISION_RAMP + stock.COLLISION_HOLD - age) / stock.COLLISION_RAMP)
    return max(weight, 0.0) * np.asarray(vector, float)


def instrumented(second, mechanics):
    friction = float(mechanics["table_friction_scale"])
    solref = float(mechanics["table_solref_time_scale"])
    suction = float(mechanics["suction_solref_time_scale"])
    damping = float(mechanics["dof_damping_scale"])
    if any(not 0.90 <= value <= 1.10 for value in (friction, solref, suction, damping)):
        raise ValueError("mechanics outside fixed mild range")

    class ReplayPlant(StockPlant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.r28_push_count = 0
            self.r28_native_force = []
            self.r28_second_force = []
            self.r28_committed_force = []
            tables = [self.m.geom(name).id for name in ("table_pick", "table_place")]
            if self.has_box:
                box = self.m.body("box_free").id
                tables.extend(np.flatnonzero(self.m.geom_bodyid == box).tolist())
            self.m.geom_friction[tables, 0] *= friction
            self.m.geom_solref[tables, 0] *= solref
            if self.m.neq:
                self.m.eq_solref[:, 0] *= suction
            self.m.dof_damping[:] *= damping

        def push(self, native_force):
            t = self.r28_push_count * C.DT
            extra = pulse(t, second)
            combined = np.asarray(native_force, float) + extra
            super().push(combined)
            body = self.m.body("link_6").id
            self.r28_native_force.append(np.asarray(native_force, float).copy())
            self.r28_second_force.append(extra.copy())
            self.r28_committed_force.append(self.d.xfrc_applied[body, :3].copy())
            self.r28_push_count += 1

    return ReplayPlant


def run_episode(case, detector_type, impacts, controller_type):
    plant = instrumented(impacts[1], case["mechanics"])
    function = types.FunctionType(
        stock.run_episode.__code__, dict(stock.run_episode.__globals__, Plant=plant),
        stock.run_episode.__name__, stock.run_episode.__defaults__, stock.run_episode.__closure__)
    function.__kwdefaults__ = stock.run_episode.__kwdefaults__
    offset = case.get("offset")
    metrics, log = function(
        controller_type(), box_mass=float(case["mass"]), box_com=tuple(case["com"]),
        pick_offset=None if offset is None else tuple(offset), detector=detector_type(),
        collision=impacts[0])
    owner = log["plant"]
    log["f_ext_xyz_native"] = np.asarray(owner.r28_native_force)
    log["f_ext_xyz_second"] = np.asarray(owner.r28_second_force)
    log["f_ext_xyz_committed"] = np.asarray(owner.r28_committed_force)
    fields = PHYSICAL + ("r", "detect")
    if any(np.asarray(log[key]).shape[0] != 6000 for key in fields):
        raise ValueError("incomplete 6000-sample trace")
    if any(not np.isfinite(np.asarray(log[key])).all() for key in fields):
        raise ValueError("nonfinite trace")
    if metrics["diverged"]:
        raise ValueError("episode diverged")
    return log


def edge_indices(log):
    bits = np.asarray(log["detect"], bool)
    return np.flatnonzero(bits & ~np.r_[False, bits[:-1]])


def expected_force(t, impacts):
    native = np.stack([pulse(value, impacts[0]) for value in t])
    extra = np.stack([pulse(value, impacts[1]) for value in t])
    return {
        "native": native,
        "extra": extra,
        "combined": native + extra,
        "scalar": np.linalg.norm(native, axis=1),
    }


def force_exact(log, impacts):
    expected = expected_force(log["t"], impacts)
    names = {"native": "f_ext_xyz_native", "extra": "f_ext_xyz_second",
             "combined": "f_ext_xyz_committed", "scalar": "f_ext"}
    gates = {key: bool(np.array_equal(np.asarray(log[name]), expected[key]))
             for key, name in names.items()}
    return {"passed": all(gates.values()), "gates": gates}


def tool_record(log, threshold, tail):
    contact = np.asarray(log["f_tool"]) > threshold
    bits = np.asarray(log["detect"], bool)
    if not contact.any():
        return {"present": False, "first_contact_tick": None, "last_contact_tick": None,
                "detected": False, "prealarm": False, "first_latency_ms": None,
                "peak_N": float(np.max(log["f_tool"]))}
    first, last = np.flatnonzero(contact)[[0, -1]]
    after = np.flatnonzero(bits[first:min(6000, last + tail + 1)])
    return {"present": True, "first_contact_tick": int(first), "last_contact_tick": int(last),
            "detected": bool(len(after)), "prealarm": bool(bits[:first].any()),
            "first_latency_ms": None if not len(after) else int(after[0]),
            "peak_N": float(np.max(log["f_tool"]))}


def event_intervals(log, starts, window, threshold, tool_tail):
    valid = np.zeros(6000, bool)
    for start in starts:
        if start is not None:
            valid[start:min(6000, start + window)] = True
    tool = np.asarray(log["f_tool"]) > threshold
    if tool.any():
        first, last = np.flatnonzero(tool)[[0, -1]]
        valid[first:min(6000, last + tool_tail + 1)] = True
    return valid


def trace_record(log, impacts, starts, scoring):
    edges = edge_indices(log)
    valid = event_intervals(log, starts, scoring["event_window_ms"],
                            scoring["tool_force_threshold_N"], scoring["tool_tail_ms"])
    return {
        "rises": edges.astype(int).tolist(),
        "active_samples": int(np.asarray(log["detect"], bool).sum()),
        "late_false_rises_s": np.asarray(log["t"])[edges[~valid[edges]]].tolist(),
        "tool": tool_record(log, scoring["tool_force_threshold_N"], scoring["tool_tail_ms"]),
        "force_exact": force_exact(log, impacts),
        "physical_sha256": physical_hash(log),
        "stuck_tail": bool(log["detect"][-1]),
    }


def actual_start(t, impact):
    values = np.stack([pulse(value, impact) for value in t])
    active = np.flatnonzero(np.any(values != 0.0, axis=1))
    if not len(active):
        raise ValueError("positive control needs nonzero input")
    return int(active[0])


def event_record(forced, reference, start, scoring):
    end = min(6000, start + scoring["event_window_ms"])
    raw = edge_indices(forced)
    raw = raw[(raw >= start) & (raw < end)]
    reference_edges = edge_indices(reference)
    attributable = [int(hit) for hit in raw
                    if not np.any(np.abs(reference_edges - hit) <= scoring["reference_twin_ms"])]
    bits = np.asarray(forced["detect"], bool)
    return {"raw_rise_indices": raw.astype(int).tolist(),
            "attributable_rise_indices": attributable,
            "delays_ms": [hit - start for hit in attributable],
            "quiet_before": bool(not bits[max(0, start - scoring["quiet_ms"]):start].any()),
            "exactly_one": len(raw) == len(attributable) == 1,
            "strict": len(raw) == len(attributable) == 1 and
                      not bits[max(0, start - scoring["quiet_ms"]):start].any()}


def prefix_exact(reference, forced, stop):
    fields = ("t", "q_d", "tau", "r", "detect")
    gates = {key: bool(np.array_equal(np.asarray(reference[key])[:stop],
                                      np.asarray(forced[key])[:stop])) for key in fields}
    return {"passed": all(gates.values()), "gates": gates}


def same_physics(records, twins=("none",)):
    base = records["c1"]
    return {name: all(records[name][tw]["physical_sha256"] == base[tw]["physical_sha256"]
                      for tw in twins) for name in records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("fresh output required")
    plan_raw = PLAN.read_bytes()
    plan = json.loads(plan_raw)
    if plan.get("schema") != "r28-landing-replay-plan.v1" or plan.get("status") != "FROZEN":
        parser.error("R28 plan is not frozen")
    arms = {name: resolve(plan["arms"][name])
            for name in ("c1", "r27_v3", "r28", "pulse_diagnostic")}
    controller = resolve(plan["arms"]["controller"])
    scoring = plan["scoring"]
    tool_rows = []
    positive_rows = []
    episodes = 0

    for case in plan["known_tool_pair"]:
        logs = {name: run_episode(case, detector, (None, None), controller)
                for name, detector in arms.items()}
        episodes += len(arms)
        records = {name: {"none": trace_record(log, (None, None), [], scoring)}
                   for name, log in logs.items()}
        physics = same_physics(records)
        gates = {"all_physics_match_c1": all(physics.values()),
                 "all_forces_exact": all(row["none"]["force_exact"]["passed"] for row in records.values())}
        tool_rows.append({"id": case["id"], "case": case, "records": records,
                          "physics_match_c1": physics, "integrity_gates": gates})

    for case in plan["positive_controls"]:
        logs = {name: {"none": run_episode(case, detector, (None, None), controller),
                       "force": run_episode(case, detector, (tuple(case["impact"]), None), controller)}
                for name, detector in arms.items()}
        episodes += 2 * len(arms)
        start = actual_start(logs["c1"]["force"]["t"], case["impact"])
        records = {}
        for name, twins in logs.items():
            records[name] = {
                "none": trace_record(twins["none"], (None, None), [], scoring),
                "force": trace_record(twins["force"], (tuple(case["impact"]), None), [start], scoring),
                "event": event_record(twins["force"], twins["none"], start, scoring),
                "prefix": prefix_exact(twins["none"], twins["force"], start),
            }
        physics = same_physics(records, ("none", "force"))
        gates = {"all_physics_match_c1": all(physics.values()),
                 "all_forces_exact": all(row["force"]["force_exact"]["passed"] for row in records.values()),
                 "all_prefixes_exact": all(row["prefix"]["passed"] for row in records.values())}
        first_rise = {name: (row["event"]["delays_ms"][0]
                             if row["event"]["delays_ms"] else None)
                      for name, row in records.items()}
        positive_rows.append({"id": case["id"], "case": case, "actual_input_tick": start,
                              "records": records, "physics_match_c1": physics,
                              "first_rise_delay_ms": first_rise,
                              "r28_minus_r27_first_rise_ms": (
                                  None if first_rise["r28"] is None or first_rise["r27_v3"] is None
                                  else first_rise["r28"] - first_rise["r27_v3"]),
                              "integrity_gates": gates})

    if episodes != plan["runtime"]["episodes"]:
        raise ValueError("episode accounting differs")
    real_tool = next(row for row in tool_rows if row["id"] == "known_5p25kg_29mm_tool")
    no_offset = next(row for row in tool_rows if row["id"].endswith("zero_offset"))
    protection = {
        "real_tool_present_all": all(real_tool["records"][name]["none"]["tool"]["present"] for name in arms),
        "real_tool_detected_all": all(real_tool["records"][name]["none"]["tool"]["detected"] for name in arms),
        "r28_real_tool_not_later_than_c1": (
            real_tool["records"]["r28"]["none"]["tool"]["first_latency_ms"] is not None and
            real_tool["records"]["c1"]["none"]["tool"]["first_latency_ms"] is not None and
            real_tool["records"]["r28"]["none"]["tool"]["first_latency_ms"] <=
            real_tool["records"]["c1"]["none"]["tool"]["first_latency_ms"] + 5),
        "r28_real_tool_not_later_than_r27_v3": (
            real_tool["records"]["r28"]["none"]["tool"]["first_latency_ms"] is not None and
            real_tool["records"]["r27_v3"]["none"]["tool"]["first_latency_ms"] is not None and
            real_tool["records"]["r28"]["none"]["tool"]["first_latency_ms"] <=
            real_tool["records"]["r27_v3"]["none"]["tool"]["first_latency_ms"] + 5),
        "no_offset_has_no_tool_contact": all(not no_offset["records"][name]["none"]["tool"]["present"] for name in arms),
        "r28_no_more_no_offset_false_than_c1": (
            len(no_offset["records"]["r28"]["none"]["late_false_rises_s"]) <=
            len(no_offset["records"]["c1"]["none"]["late_false_rises_s"])),
        "r28_no_more_no_offset_false_than_r27_v3": (
            len(no_offset["records"]["r28"]["none"]["late_false_rises_s"]) <=
            len(no_offset["records"]["r27_v3"]["none"]["late_false_rises_s"])),
        "r28_positive_controls_strict": all(row["records"]["r28"]["event"]["strict"] for row in positive_rows),
        "r28_preserves_r27_positive_first_rise_plus5ms": all(
            row["first_rise_delay_ms"]["r27_v3"] is None or
            (row["first_rise_delay_ms"]["r28"] is not None and
             row["first_rise_delay_ms"]["r28"] <= row["first_rise_delay_ms"]["r27_v3"] + 5)
            for row in positive_rows),
        "all_integrity_gates": all(all(row["integrity_gates"].values())
                                   for row in tool_rows + positive_rows),
    }
    diagnostic = {
        "real_tool_detected": real_tool["records"]["pulse_diagnostic"]["none"]["tool"]["detected"],
        "real_tool_not_later_than_c1_plus5ms": (
            real_tool["records"]["pulse_diagnostic"]["none"]["tool"]["first_latency_ms"] is not None and
            real_tool["records"]["c1"]["none"]["tool"]["first_latency_ms"] is not None and
            real_tool["records"]["pulse_diagnostic"]["none"]["tool"]["first_latency_ms"] <=
            real_tool["records"]["c1"]["none"]["tool"]["first_latency_ms"] + 5),
        "no_more_real_tool_late_false_than_r27_v3": (
            len(real_tool["records"]["pulse_diagnostic"]["none"]["late_false_rises_s"]) <=
            len(real_tool["records"]["r27_v3"]["none"]["late_false_rises_s"])),
        "no_offset_has_no_tool_contact": not no_offset["records"]["pulse_diagnostic"]["none"]["tool"]["present"],
        "no_more_no_offset_false_than_c1": (
            len(no_offset["records"]["pulse_diagnostic"]["none"]["late_false_rises_s"]) <=
            len(no_offset["records"]["c1"]["none"]["late_false_rises_s"])),
        "positive_controls_strict": all(
            row["records"]["pulse_diagnostic"]["event"]["strict"] for row in positive_rows),
        "preserves_r27_positive_first_rise_plus5ms": all(
            row["first_rise_delay_ms"]["r27_v3"] is None or
            (row["first_rise_delay_ms"]["pulse_diagnostic"] is not None and
             row["first_rise_delay_ms"]["pulse_diagnostic"] <=
             row["first_rise_delay_ms"]["r27_v3"] + 5) for row in positive_rows),
        "all_integrity_gates": protection["all_integrity_gates"],
    }
    result = {
        "schema": "r28-landing-known-replay-result.v1",
        "classification": plan["classification"],
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "episodes": episodes,
        "arms": plan["arms"],
        "tool_rows": tool_rows,
        "positive_control_rows": positive_rows,
        "protection_gates": protection,
        "protection_passed": all(protection.values()),
        "pulse_diagnostic_gates": diagnostic,
        "pulse_diagnostic_passed": all(diagnostic.values()),
        "claim_status": "known_replay_only_no_promotion",
    }
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": episodes, "protection_gates": protection,
                      "protection_passed": result["protection_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
