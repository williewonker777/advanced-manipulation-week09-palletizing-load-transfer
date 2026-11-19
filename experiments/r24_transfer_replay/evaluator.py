"""Actual-simulation replay and preregistered challenge for frozen R24 control."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import types

import numpy as np

import my_detector
import r1_torque
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r14_stock_handoff.qualification import compact
from experiments.r19_load_transfer.controller import Controller as R19
from experiments.r24_transfer.deadband import Controller as R24
from sim import config as C
import sim.runner as stock
from sim.plant import Plant as StockPlant
from tools.r11_placement import score


HERE = Path(__file__).resolve().parent
PHYSICAL = ("t", "e_mm", "q", "q_d", "tau", "f_table", "f_tool", "qpos")
ARMS = ("r10", "r19", "r24")


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def physical_signature(log):
    output = hashlib.sha256()
    for key in PHYSICAL:
        value = np.ascontiguousarray(log[key])
        output.update(key.encode())
        output.update(str(value.shape).encode())
        output.update(value.dtype.str.encode())
        output.update(value.tobytes())
    return output.hexdigest()


def expected_force(times, collision):
    if collision is None:
        return np.zeros((len(times), 3))
    onset, vector = collision
    output = []
    for t in times:
        age = float(t) - float(onset)
        weight = min(age / stock.COLLISION_RAMP, 1.0,
                     (2 * stock.COLLISION_RAMP + stock.COLLISION_HOLD - age)
                     / stock.COLLISION_RAMP)
        output.append(max(weight, 0.0) * np.asarray(vector, dtype=float))
    return np.asarray(output)


def run_function(mechanics_scale):
    scale = float(mechanics_scale)
    if not np.isfinite(scale) or not 0.90 <= scale <= 1.10:
        raise ValueError("table solref scale outside preregistered modest range")

    class InstrumentedPlant(StockPlant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if scale != 1.0:
                for name in ("table_pick", "table_place"):
                    self.m.geom_solref[self.m.geom(name).id, 0] *= scale
            self.r24_requested_force = []
            self.r24_committed_force = []

        def push(self, force):
            requested = np.asarray(force, dtype=float).copy()
            super().push(requested)
            body = self.m.body("link_6").id
            self.r24_requested_force.append(requested)
            self.r24_committed_force.append(self.d.xfrc_applied[body, :3].copy())

    namespace = dict(stock.run_episode.__globals__, Plant=InstrumentedPlant)
    function = types.FunctionType(stock.run_episode.__code__, namespace,
                                  stock.run_episode.__name__,
                                  stock.run_episode.__defaults__,
                                  stock.run_episode.__closure__)
    function.__kwdefaults__ = stock.run_episode.__kwdefaults__
    return function


def controller(name, calibration):
    if name == "r10":
        return r1_torque.Controller()
    cls = R19 if name == "r19" else R24
    return cls(calibration["height"], calibration["information"])


def episode(name, case, calibration, collision=None):
    mechanics = case.get("mechanics", {})
    run = run_function(mechanics.get("table_solref_time_scale", 1.0))
    ctrl = controller(name, calibration)
    metrics, log = run(ctrl, float(case["mass"]), tuple(case["com"]), Coupled(),
                       collision=collision, pick_offset=case.get("offset"))
    plant = log["plant"]
    requested = np.asarray(plant.r24_requested_force)
    committed = np.asarray(plant.r24_committed_force)
    if requested.shape != (len(log["t"]), 3) or not np.array_equal(requested, committed):
        raise ValueError("incomplete or changed signed flange force")
    log["signed_force"] = committed
    return compact(metrics, log, ctrl, float(case["mass"])), log


def max_ratios(candidate, reference):
    return {key: candidate["cost"][key] / max(value, 1e-12)
            for key, value in reference["cost"].items()}


def normal_row(payload):
    case, calibration, known = payload
    records, logs = {}, {}
    for name in ARMS:
        records[name], logs[name] = episode(name, case, calibration)
    g4 = score(records["r10"], records["r24"])
    ratios = {name: max_ratios(records["r24"], records[name]) for name in ("r10", "r19")}
    row = {
        "id": case["id"], "kind": "known_replay" if known else "new_normal",
        "case": case, "arms": records, "g4": g4, "ratios": ratios,
        "physical_sha256": {name: physical_signature(logs[name]) for name in ARMS},
        "disclosure": {
            "r24_false_not_above_r19": (len(records["r24"]["false_alarm_times"])
                                          <= len(records["r19"]["false_alarm_times"])),
            "r24_cost_le_1_05_r10": max(ratios["r10"].values()) <= 1.05,
            "r24_cost_le_1_05_r19": max(ratios["r19"].values()) <= 1.05,
            "r24_saturation_not_above": (records["r24"]["saturation_pct"]
                                            <= max(records["r10"]["saturation_pct"],
                                                   records["r19"]["saturation_pct"])),
        },
    }
    if known:
        expected = case["expected"]
        support = records["r24"]["support"]
        observed_support = [support["mean_mg"], support["min_mg"], support["peak_mg"]]
        gates = {
            "r19_false_exact": records["r19"]["false_alarm_times"] == expected["r19_false"],
            "r24_false_exact": records["r24"]["false_alarm_times"] == expected["r24_false"],
            "r24_support_exact": observed_support == expected["r24_support"],
            "r24_physical_exact": row["physical_sha256"]["r24"] == expected["r24_physical_sha256"],
            "g4_all_four": g4["all_four_g4"] is True,
        }
        row["known_reproduction"] = {"gates": gates, "passed": all(gates.values())}
    return row


def rise_times(log):
    detected = np.asarray(log["detect"], dtype=bool)
    indices = np.flatnonzero(detected & ~np.r_[False, detected[:-1]])
    return np.asarray(log["t"])[indices].tolist()


def force_assessment(force_log, twin_log, collision, plan):
    onset = float(collision[0])
    window = float(plan["event_window_s"])
    guard = float(plan["twin_guard_s"])
    times = np.asarray(force_log["t"])
    force_rises = rise_times(force_log)
    twin_rises = rise_times(twin_log)
    genuine = [value for value in force_rises
               if onset < value < onset + window
               and not np.asarray(twin_log["detect"], dtype=bool)[
                   max(0, int(round((value - guard) / C.DT))):
                   min(len(times), int(round((value + guard) / C.DT)) + 1)].any()]
    forced_false = [value for value in force_rises
                    if not onset <= value < onset + window]
    prefix = times <= onset
    prefix_fields = {key: bool(np.array_equal(np.asarray(force_log[key])[prefix],
                                              np.asarray(twin_log[key])[prefix]))
                     for key in PHYSICAL}
    expected = expected_force(times, collision)
    signed_exact = bool(np.array_equal(np.asarray(force_log["signed_force"]), expected))
    return {
        "all_rises_s": force_rises,
        "twin_all_rises_s": twin_rises,
        "genuine_rises_s": genuine,
        "first_genuine_delay_ms": None if not genuine else 1000.0 * (genuine[0] - onset),
        "forced_false_rises_s": forced_false,
        "twin_false_rises_s": twin_rises,
        "signed_force_exact": signed_exact,
        "peak_signed_force_n": np.asarray(force_log["signed_force"])[
            np.argmax(np.linalg.norm(force_log["signed_force"], axis=1))].tolist(),
        "force_free_prefix_exact": all(prefix_fields.values()),
        "force_free_prefix_fields": prefix_fields,
    }


def all_matched(candidate, reference, tolerance_ms):
    tolerance = float(tolerance_ms) / 1000.0
    return all(any(abs(value - inherited) <= tolerance for inherited in reference)
               for value in candidate)


def compare(candidate, reference, plan):
    candidate_delay = candidate["first_genuine_delay_ms"]
    reference_delay = reference["first_genuine_delay_ms"]
    required = reference_delay is not None
    gates = {
        "true_nonloss": not required or candidate_delay is not None,
        "delay_le_reference_plus_5ms": (not required or candidate_delay is not None
                                         and candidate_delay <= reference_delay
                                         + float(plan["delay_margin_ms"])),
        "forced_false_no_added": all_matched(
            candidate["forced_false_rises_s"], reference["forced_false_rises_s"],
            plan["false_match_tolerance_ms"]),
        "twin_false_no_added": all_matched(
            candidate["twin_false_rises_s"], reference["twin_false_rises_s"],
            plan["false_match_tolerance_ms"]),
    }
    return {"reference_had_genuine_event": required, "gates": gates,
            "passed": all(gates.values())}


def impact_row(payload):
    impulse, base, calibration, plan = payload
    case = dict(base, id=impulse["id"])
    collision = [float(impulse["time_s"]), list(impulse["force_n"])]
    assessments = {}
    for name in ARMS:
        _, twin_log = episode(name, case, calibration)
        _, force_log = episode(name, case, calibration, collision)
        assessments[name] = force_assessment(force_log, twin_log, collision, plan)
    comparisons = {name: compare(assessments["r24"], assessments[name], plan)
                   for name in ("r10", "r19")}
    integrity = all(value["signed_force_exact"] and value["force_free_prefix_exact"]
                    for value in assessments.values())
    return {"id": impulse["id"], "kind": "predeclared_late_impulse",
            "case": case, "collision": collision, "arms": assessments,
            "candidate_vs": comparisons, "input_integrity": integrity,
            "strict_pass": integrity and all(value["passed"] for value in comparisons.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    plan = read(HERE / "plan.json")
    calibration_path = HERE / "calibration.json"
    calibration = read(calibration_path)
    if (digest(HERE.parent / "r24_transfer/deadband.py") != plan["candidate_sha256"]
            or digest(calibration_path) != plan["calibration_sha256"]):
        raise ValueError("frozen candidate or replay calibration changed")
    normal_payloads = ([(case, calibration, True) for case in plan["known_replay"]]
                       + [(case, calibration, False) for case in plan["new_normal"]])
    impact_payloads = [(impulse, plan["impulse_base"], calibration, plan)
                       for impulse in plan["late_impulses"]]
    if args.workers == 1:
        normal = [normal_row(value) for value in normal_payloads]
        impacts = [impact_row(value) for value in impact_payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            normal = list(pool.map(normal_row, normal_payloads))
            impacts = list(pool.map(impact_row, impact_payloads))
    result = {
        "schema": "r24-transfer-external-replay-result.v1",
        "classification": plan["classification"],
        "plan_sha256": digest(HERE / "plan.json"),
        "candidate_sha256": plan["candidate_sha256"],
        "calibration_sha256": plan["calibration_sha256"],
        "known_replay": normal[:len(plan["known_replay"])],
        "new_normal": normal[len(plan["known_replay"]):],
        "late_impulses": impacts,
        "known_reproduction_passed": all(row["known_reproduction"]["passed"]
                                         for row in normal[:len(plan["known_replay"])]),
        "new_challenge_strict_passed": (all(all(row["disclosure"].values()) for row in
                                                normal[len(plan["known_replay"]):])
                                         and all(row["strict_pass"] for row in impacts)),
        "promotion_eligible": False,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"known_reproduction_passed": result["known_reproduction_passed"],
                      "new_challenge_strict_passed": result["new_challenge_strict_passed"],
                      "normal_rows": len(normal), "impulse_rows": len(impacts)}))


if __name__ == "__main__":
    main()
