"""Neutral closed-loop R15 comparison extracted from the staged control study."""
import numpy as np
import my_detector
import r1_torque
from sim.runner import run_episode
from tools.r11_placement import score
from tools.r14_paired_events import match_false_events
from experiments.r14_stock_handoff.controller import Controller as Base
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r14_stock_handoff.qualification import compact, KEYS
from experiments.r15_com_model.controller import Controller


def ratios(a, b):
    return {k: float(v / max(b["cost"][k], 1e-10)) for k, v in a["cost"].items()}


def compare(case, height, xy):
    records = {}
    logs = {}
    for arm in ("r10", "r14", "xy"):
        c = {
            "r10": lambda: r1_torque.Controller(),
            "r14": lambda: Base(height),
            "xy": lambda: Controller(height, xy),
        }[arm]()
        d = my_detector.Detector() if arm == "r10" else Coupled()
        m, log = run_episode(
            c,
            case["mass"],
            tuple(case["com"]),
            d,
            pick_offset=case.get("offset"),
        )
        rec = compact(m, log, c, case["mass"])
        if arm == "xy":
            rec["xy"] = c.xy_result
            rec["xy_updates"] = c.xy_updates
            rec["selected_at_s"] = c.xy_decided_at_s
        records[arm] = rec
        logs[arm] = log
    ref, old, new = (records[k] for k in ("r10", "r14", "xy"))
    if any(r.get("status") != "ok" for r in records.values()):
        return {"id": case["id"], "status": "incomplete", "arms": records, "passed": False}
    complete = {k: bool(np.array_equal(logs["r14"][k], logs["xy"][k])) for k in KEYS}
    prefix = {
        k: bool(np.array_equal(logs["r14"][k][:3550], logs["xy"][k][:3550]))
        for k in KEYS
    }
    h_base = score(ref, old)
    h = score(ref, new)
    costs = {k: ratios(new, records[k]) for k in ("r10", "r14")}
    has_extra = lambda e, r: not match_false_events(
        e["false_alarm_times"], r["false_alarm_times"]
    )["passed"]
    gates = {
        "g4": bool(h["all_four_g4"]),
        "cost": all(max(v.values()) <= 1.05 for v in costs.values()),
        "sat": all(
            new["saturation_pct"] <= b["saturation_pct"] + 1e-12 for b in (ref, old)
        ),
        "FA": all(not has_extra(new, b) for b in (ref, old)),
        "tool": all(
            b["tool_delay_ms"] is None
            or (
                new["tool_delay_ms"] is not None
                and new["tool_delay_ms"] <= b["tool_delay_ms"] + 5.0 + 1e-9
            )
            for b in (ref, old)
        ),
        "prefix": all(prefix.values()),
        "fallback": new["xy"]["accept"] or all(complete.values()),
        "controller_p99_below_800us": new["ctrl_p99_us"] <= 800.0,
    }
    return {
        "id": case["id"],
        "status": "ok",
        "arms": records,
        "cost_ratios": costs,
        "g4": h,
        "r14_g4": h_base,
        "official_reference_clock": True,
        "prefix": prefix,
        "full_parity": complete,
        "gates": gates,
        "passed": all(gates.values()),
    }
