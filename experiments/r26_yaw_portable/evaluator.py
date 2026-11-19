"""Run the one disclosed R25 long signed-yaw case on authenticated course physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import FunctionType

import mujoco
import numpy as np

from experiments.r24_holdout import evaluator as lab
from experiments.r25_composed.detector import Detector as R25
from experiments.r25_holdout import runner
from experiments.r26_long.candidate import Detector as R26
from my_torque import Controller


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REGISTRATION = HERE / "registration.json"
ARMS = {"r25": R25, "r26_long": R26}
ROUTES = ("r10", "normal", "horizontal", "yaw", "light", "detached",
          "pickup", "continuation", "yaw_identity")


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def registration():
    data = json.loads(REGISTRATION.read_bytes())
    case = data.get("case", {})
    expected = data.get("expected", {})
    if (data.get("schema") != "r26-known-signed-yaw-registration.v1"
            or case.get("id") != "s25_long_transfer_signed"
            or set(data.get("variants", {})) != {"stock", "actuator_shift", "contact_shift"}
            or set(expected) != set(data["variants"])
            or data.get("arms") != {
                "r25": "experiments.r25_composed.detector:Detector",
                "r26_long": "experiments.r26_long.candidate:Detector",
            }
            or data.get("candidate_sha256") != digest(
                ROOT / "experiments/r26_long/candidate.py")):
        raise ValueError("disclosed signed-yaw registration or candidate differs")
    for name, row in expected.items():
        if (len(row.get("physical_signature", "")) != 64
                or row.get("force_interval") != [1624, 1803]
                or not row.get("r25_rises") or len(row.get("r26_rises", [])) != 1):
            raise ValueError("incomplete registered expectation: " + name)
    return data


def bool_intervals(values):
    values = np.asarray(values, dtype=bool)
    rises = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    falls = np.flatnonzero(~values & np.r_[False, values[:-1]])
    if len(values) and values[-1]:
        falls = np.r_[falls, len(values)]
    return [[int(first), int(last - 1)] for first, last in zip(rises, falls, strict=True)]


def rise_indices(values):
    values = np.asarray(values, dtype=bool)
    return np.flatnonzero(values & ~np.r_[False, values[:-1]]).astype(int).tolist()


def observed(base):
    class Observed(base):
        def reset(self):
            super().reset()
            self.portable_routes = []

        def update(self, t, th, thd, tau_m, tau_s, grip):
            residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
            tick = int(round(1000 * float(t)))
            if 1540 <= tick <= 1940:
                normal = getattr(self, "normal_model", None)
                light = getattr(getattr(self, "light_witness", None), "state", None)
                detached = getattr(getattr(self, "detached", None), "evidence", None)
                continuation = getattr(self, "continuation", None)
                self.portable_routes.append({
                    "tick": tick,
                    "report": bool(report),
                    "r10": bool(self._shared.detected),
                    "normal": bool(normal is not None and normal.detected),
                    "horizontal": bool(self.horizontal_detected),
                    "yaw": bool(self.yaw_active),
                    "light": bool(light is not None and light.active),
                    "detached": bool(detached is not None and detached.active),
                    "pickup": bool(getattr(self, "pick_event_active", False)
                                   or getattr(self, "pick_completion_active", False)),
                    "continuation": bool(continuation is not None
                                         and (continuation.observed or continuation.bridge)),
                    "yaw_identity": bool(getattr(self, "yaw_identity_bridge", False)),
                    "yaw_value_Nm": (None if self.yaw_value is None
                                     else float(self.yaw_value)),
                })
            return residual, report

    return Observed


def run_episode(case, variant, base, impact):
    detector = observed(base)()
    impacts = (impact, None)
    plant = runner.instrumented(impacts, variant)
    function = FunctionType(
        lab.certified.stock.run_episode.__code__,
        dict(lab.certified.stock.run_episode.__globals__, Plant=plant),
        lab.certified.stock.run_episode.__name__,
        lab.certified.stock.run_episode.__defaults__,
        lab.certified.stock.run_episode.__closure__)
    function.__kwdefaults__ = lab.certified.stock.run_episode.__kwdefaults__
    metrics, log = function(
        Controller(), box_mass=case["mass"], box_com=tuple(case["com"]),
        pick_offset=case.get("offset"), detector=detector, collision=None)
    owner = log["plant"]
    log["f_ext_xyz_native"] = np.asarray(owner.r25_first)
    log["f_ext_xyz_second"] = np.asarray(owner.r25_second)
    log["f_ext_xyz_committed"] = np.asarray(owner.r25_committed)
    log["f_ext"] = np.linalg.norm(log["f_ext_xyz_native"], axis=1)
    if metrics["diverged"] or any(np.asarray(log[key]).shape[0] != 6000
                                   for key in lab.certified.PHYSICAL + ("r", "detect")):
        raise ValueError("incomplete or diverged course episode")
    if any(not np.isfinite(np.asarray(log[key])).all()
           for key in lab.certified.PHYSICAL + ("r", "detect")):
        raise ValueError("nonfinite course episode")
    return detector, log


def exact_force(log, impacts):
    expected = runner.expected(log["t"], impacts)
    gates = {
        "native": np.array_equal(log["f_ext_xyz_native"], expected["first"]),
        "second": np.array_equal(log["f_ext_xyz_second"], expected["second"]),
        "committed": np.array_equal(log["f_ext_xyz_committed"], expected["committed"]),
        "scalar": np.array_equal(log["f_ext"], expected["scalar"]),
    }
    return {"passed": bool(all(gates.values())),
            "gates": {name: bool(value) for name, value in gates.items()}}


def force_interval(log):
    active = np.flatnonzero(np.any(np.asarray(log["f_ext_xyz_committed"]) != 0.0, axis=1))
    if not len(active):
        return None
    contiguous = np.array_equal(active, np.arange(active[0], active[-1] + 1))
    return {"first": int(active[0]), "last": int(active[-1]),
            "samples": int(len(active)), "continuous": bool(contiguous)}


def route_summary(detector):
    rows = detector.portable_routes
    result = {}
    for route in ROUTES:
        result[route] = bool_intervals([row[route] for row in rows])
    return result


def event(log, none, start, end, scoring):
    row = lab.event_record(log, [none], start, end,
                           scoring["quiet_ms"], scoring["reference_twin_ms"])
    row["duplicate_rises"] = max(0, len(row["attributable_rise_indices"]) - 1)
    row["strict"] = bool(row["quiet_before"] and not row["high_before"]
                         and row["exactly_one"])
    return row


def evaluate():
    if mujoco.__version__ != "3.17.0":
        raise ValueError("authenticated course MuJoCo 3.17.0 required")
    data = registration()
    case, scoring = data["case"], data["scoring"]
    rows = []
    for variant_name, variant in data["variants"].items():
        expected = data["expected"][variant_name]
        traces, detectors = {}, {}
        for arm_name, base in ARMS.items():
            detectors[arm_name] = {}
            traces[arm_name] = {}
            for twin, impact in (("none", None), ("force", case["impact"])):
                detector, log = run_episode(case, variant, base, impact)
                detectors[arm_name][twin], traces[arm_name][twin] = detector, log
        interval = force_interval(traces["r25"]["force"])
        if interval is None:
            raise ValueError("registered force absent")
        start = interval["first"]
        arms = {}
        for arm_name in ARMS:
            none, forced = traces[arm_name]["none"], traces[arm_name]["force"]
            rises = rise_indices(forced["detect"])
            edge_key = {"r25": "r25_rises", "r26_long": "r26_rises"}[arm_name]
            expected_rises = expected[edge_key]
            record = event(forced, none, start,
                           min(6000, start + scoring["event_window_ms"]), scoring)
            samples = {row["tick"]: row for row in detectors[arm_name]["force"].portable_routes}
            physical = lab.certified.trace_signature({"none": none, "single": forced})
            routes = route_summary(detectors[arm_name]["force"])
            allowed = {"yaw"} if arm_name == "r25" else {"yaw", "yaw_identity"}
            inactive = [name for name in ROUTES if name not in allowed and routes[name]]
            gates = {
                "registered_edges_exact": rises == expected_rises,
                "strict_twin_attribution_exact": (record["raw_rise_indices"] == rises
                                                   and record["attributable_rise_indices"] == rises),
                "none_has_no_edges": not rise_indices(none["detect"]),
                "force_exact": exact_force(forced, (case["impact"], None))["passed"],
                "none_force_exact": exact_force(none, (None, None))["passed"],
                "physical_signature_exact": physical == expected["physical_signature"],
                "prefix_exact": lab.certified.prefix(none, forced, start, start)["passed"],
                "source_ids_exact": not inactive,
                "no_stuck_tail": not bool(forced["detect"][-1]),
            }
            arms[arm_name] = {
                "alarm_rise_ticks": rises,
                "alarm_intervals": bool_intervals(forced["detect"]),
                "none_alarm_rise_ticks": rise_indices(none["detect"]),
                "event": record,
                "source_intervals_in_capture": routes,
                "sources_at_alarm_rises": [samples[tick] for tick in rises],
                "physical_signature": physical,
                "gates": gates,
                "passed": all(gates.values()),
            }
        interval_gate = ([interval["first"], interval["last"]] == expected["force_interval"]
                         and interval["samples"] == 180 and interval["continuous"])
        row = {
            "variant": variant_name,
            "force_interval": interval,
            "force_interval_exact": interval_gate,
            "arms": arms,
            "passed": bool(interval_gate and all(item["passed"] for item in arms.values())),
        }
        rows.append(row)
    document = {
        "schema": "r26-known-signed-yaw-diagnostic.v1",
        "classification": data["classification"],
        "independent": False,
        "unopened_r26_conditions_used": False,
        "original_course_physics": True,
        "case": case,
        "episodes": 12,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "limitations": ["finite disclosed simulator diagnostic", "not safety or deployment proof"],
    }
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh output required")
    result = evaluate()
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for row in result["rows"]:
        force = row["force_interval"]
        print(f"{row['variant']}: force={force['first']}..{force['last']} "
              f"continuous={force['continuous']} "
              f"R25={row['arms']['r25']['alarm_rise_ticks']} "
              f"R26={row['arms']['r26_long']['alarm_rise_ticks']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
