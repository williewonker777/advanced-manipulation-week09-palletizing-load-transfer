"""Bounded signed-force, pair, tool, and timing checks for the frozen pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np

from experiments.r23_holdout import evaluator as paired
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r26_delivery_pair.controller import Controller
from experiments.r27_landing_integration.candidate import Detector as Candidate


HERE = Path(__file__).resolve().parent
FROZEN = {
    "plan.json": "4d37047e107c9881f09ba90ecca55fc263ad0bc5338485852b8c10c4fed79c8b",
    "model.json": "447660558c5ba67ea65e53be568e493bdd5f42532b558f440884b3fb610b5939",
    "candidate.py": "ffde18b42b8ccd8250eae184147f7e498003356cfdb1a7cd46ccc5edfce0fb95",
}
CASE = {"mass": 4.0, "com": [0.02, -0.01, 0.03], "offset": None}
SINGLES = {
    "pickup_y15": (1.42, [0.0, 15.0, 0.0]),
    "pickup_x15": (1.42, [15.0, 0.0, 0.0]),
    "pickup_z15": (1.42, [0.0, 0.0, 15.0]),
    "transport_y15": (3.00, [0.0, 15.0, 0.0]),
}
PAIR = ((1.42, [0.0, 15.0, 0.0]), (1.70, [0.0, -15.0, 0.0]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(base):
    class Timed(base):
        latest = None

        def reset(self):
            super().reset()
            self.update_ns = []
            self.force_values = []
            Timed.latest = self

        def update(self, *args):
            start = time.perf_counter_ns()
            result = super().update(*args)
            self.update_ns.append(time.perf_counter_ns() - start)
            value = getattr(self, "grip_force_value", None)
            self.force_values.append(None if value is None else value.copy())
            return result

    return Timed


def run(base, impacts, offset=None):
    detector = timed(base)
    plant = paired.local_plant(impacts[1], {"kind": "stock"})
    function = types.FunctionType(
        paired.certified.stock.run_episode.__code__,
        dict(paired.certified.stock.run_episode.__globals__, Plant=plant),
        paired.certified.stock.run_episode.__name__,
        paired.certified.stock.run_episode.__defaults__,
        paired.certified.stock.run_episode.__closure__)
    function.__kwdefaults__ = paired.certified.stock.run_episode.__kwdefaults__
    metrics, log = function(
        Controller(), box_mass=CASE["mass"], box_com=tuple(CASE["com"]),
        pick_offset=offset, detector=detector(), collision=impacts[0])
    owner = log["plant"]
    log["f_ext_xyz_native"] = np.asarray(owner.r18_native_force)
    log["f_ext_xyz_second"] = np.asarray(owner.r18_second_force)
    log["f_ext_xyz_committed"] = np.asarray(owner.r18_committed_force)
    instance = detector.latest
    force = np.asarray([
        [np.nan, np.nan, np.nan] if value is None else value
        for value in instance.force_values], dtype=float)
    valid = np.isfinite(force).all(axis=1)
    bits = np.asarray(log["detect"], bool)
    rises = np.flatnonzero(bits & ~np.r_[False, bits[:-1]])
    return metrics, log, {
        "rises": rises.astype(int).tolist(),
        "active_samples": int(bits.sum()),
        "tool": paired.certified.tool_record(log),
        "force_exact": paired.certified.exact_force(log, impacts),
        "physical_sha256": paired.certified.physical_hash(log),
        "update_p99_us": float(np.percentile(instance.update_ns, 99) / 1000.0),
        "observed_peak_xyz_N": (None if not valid.any() else
                                  force[valid][np.argmax(np.linalg.norm(
                                      force[valid], axis=1))].tolist()),
    }


def event(log, refs, impact):
    start = paired.certified.actual_start(log["t"], impact)[0]
    return paired.event_record(log, refs, start, min(6000, start + 320), 20, 20)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    import mujoco
    if mujoco.__version__ != "3.17.0" or any(
            sha(HERE / name) != expected for name, expected in FROZEN.items()):
        raise ValueError("runtime or frozen source/model pin differs")
    arms = {"c1": C1, "r27": Candidate}
    records = {name: {} for name in arms}
    for name, base in arms.items():
        _, none_log, none = run(base, (None, None))
        records[name]["none"] = none
        for key, impact in SINGLES.items():
            _, log, record = run(base, (impact, None))
            record["event"] = event(log, [none_log], impact)
            records[name][key] = record
        _, _, tool = run(base, (None, None), offset=(0.015, 0.0, 0.0))
        records[name]["tool_lateral"] = tool

    pair_records = {name: {} for name in arms}
    assignment = {"none": (None, None), "first": (PAIR[0], None),
                  "second": (None, PAIR[1]), "both": PAIR}
    for name, base in arms.items():
        traces = {key: run(base, impacts) for key, impacts in assignment.items()}
        for key, (_, _, record) in traces.items():
            pair_records[name][key] = record
        pair_records[name]["events"] = {
            "isolated_first": event(traces["first"][1], [traces["none"][1]], PAIR[0]),
            "isolated_second": event(traces["second"][1], [traces["none"][1]], PAIR[1]),
            "both_first": event(traces["both"][1], [traces["none"][1], traces["second"][1]], PAIR[0]),
            "both_second": event(traces["both"][1], [traces["first"][1]], PAIR[1]),
        }
    exact = all(record["force_exact"]["passed"]
                for arm in records.values() for record in arm.values()) and all(
                    record["force_exact"]["passed"]
                    for arm in pair_records.values() for key, record in arm.items()
                    if key != "events")
    physical = all(records["c1"][key]["physical_sha256"] ==
                   records["r27"][key]["physical_sha256"] for key in records["c1"])
    physical &= all(pair_records["c1"][key]["physical_sha256"] ==
                    pair_records["r27"][key]["physical_sha256"]
                    for key in assignment)
    doc = {
        "schema": "r27-landing-bounded-mechanism-check.v1",
        "classification": "Disclosed, prespecified signed/pair/tool checks after freeze; not blind.",
        "runtime": {"mujoco": mujoco.__version__}, "sources": dict(FROZEN),
        "case": CASE, "signed_singles": SINGLES, "pair": PAIR,
        "episodes": 2 * (1 + len(SINGLES) + 1 + len(assignment)),
        "records": records, "pair_records": pair_records,
        "integrity": {"all_forces_exact": exact, "all_physics_matched": physical},
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "episodes": doc["episodes"], "integrity": doc["integrity"],
        "strict": {name: {key: value["event"]["strict"]
                            for key, value in arm.items() if "event" in value}
                   for name, arm in records.items()},
        "pair_strict": {name: {key: event["strict"]
                                for key, event in arm["events"].items()}
                        for name, arm in pair_records.items()},
        "tool": {name: arm["tool_lateral"]["tool"] for name, arm in records.items()},
        "p99_us": {name: max(item["update_p99_us"] for key, item in arm.items()
                              if key != "events") for name, arm in records.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
