"""Replay only the two disclosed R24 guard failures with frozen detector arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r23_detached.free import Detector as R23
from experiments.r24_composed.detector import Detector as Joint
from experiments.r24_holdout import evaluator
from experiments.r24_loaded.dc import Detector as DC
from experiments.r24_repeat.trial import Detector as Pickup


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "experiments/r24_holdout/plan.json"
ARMS = {"r23": R23, "pickup_only": Pickup, "dc_only": DC, "joint": Joint}
CHANNELS = ("output", "r10", "horizontal", "yaw", "light", "detached",
            "pickup_event", "changed_drive", "dc_admitted", "dc_reaction_active")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def probe_class(base):
    class Probe(base):
        def reset(self):
            super().reset()
            self.post_trace = []

        def update(self, t, th, thd, tau_m, tau_s, grip):
            residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
            packet = self._shared
            normal = bool(self.normal_model is not None and self.normal_model.detected)
            yaw = self.yaw
            reaction = finite(getattr(yaw, "reaction", 0.0)) or 0.0
            self.post_trace.append({
                "index": len(self.post_trace),
                "t_s": float(t),
                "grip": bool(grip),
                "output": bool(report),
                "r10": bool(packet.detected or normal),
                "r10_score": float(packet.last_score),
                "r10_fast_score": float(packet.last_fast_score),
                "horizontal": bool(self.horizontal_detected),
                "horizontal_score": float(self.horizontal_score),
                "yaw": bool(self.yaw_active),
                "yaw_value_Nm": finite(self.yaw_value),
                "yaw_raw_Nm": finite(getattr(yaw, "raw_value", None)),
                "yaw_reaction_Nm": reaction,
                "yaw_count": int(self.yaw_count),
                "yaw_quiet": int(self.yaw_quiet),
                "light": bool(self.light_witness.state.active),
                "detached": bool(self.detached.evidence.active),
                "pickup_event": bool(getattr(self, "pick_event_active", False)),
                "pickup_event_quiet": int(getattr(self, "pick_event_quiet", 0)),
                "pickup_selected": getattr(yaw, "pick_continuity_selected", None),
                "changed_drive": bool(getattr(self, "changed_drive", False)),
                "drive_closed": bool(getattr(getattr(self, "drive", None), "closed", False)),
                "dc_admitted": bool(getattr(yaw, "admitted", False)),
                "dc_reaction_active": bool(abs(reaction) > 0.0),
            })
            return residual, report

    return Probe


def intervals(bits):
    values = np.asarray(list(bits), bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    stops = np.flatnonzero(~values & np.r_[False, values[:-1]])
    if len(values) and values[-1]:
        stops = np.r_[stops, len(values)]
    return [[int(start), int(stop)] for start, stop in zip(starts, stops, strict=True)]


def summarize_trace(trace, log):
    force = np.asarray(log["f_ext_xyz_committed"], float)
    force_active = np.any(force != 0.0, axis=1)
    changed = []
    previous = {name: False for name in CHANNELS}
    for row, applied in zip(trace, force, strict=True):
        names = [name for name in CHANNELS if row[name] != previous[name]]
        if names:
            item = dict(row)
            item["changed"] = names
            item["applied_force_N"] = applied.tolist()
            changed.append(item)
        previous = {name: row[name] for name in CHANNELS}
    return {
        "force_intervals": intervals(force_active),
        "output_intervals": intervals(row["output"] for row in trace),
        "channel_intervals": {name: intervals(row[name] for row in trace)
                              for name in CHANNELS if name != "output"},
        "transitions": changed,
    }


def run_twins(case, variant, assignment):
    result = {}
    for arm, base in ARMS.items():
        twins = {}
        Probe = probe_class(base)
        for twin, impacts in assignment.items():
            detector = Probe()
            _, log = evaluator.run_episode(case, variant, lambda: detector, impacts)
            twins[twin] = {"log": log, "trace": detector.post_trace,
                           "summary": summarize_trace(detector.post_trace, log)}
        result[arm] = twins
    for twin in assignment:
        hashes = {arm: evaluator.certified.physical_hash(twins[twin]["log"])
                  for arm, twins in result.items()}
        if any(value != hashes["r23"] for value in hashes.values()):
            raise ValueError("detector changed physical trace: " + case["id"] + ":" + twin)
        for arm, twins in result.items():
            if not evaluator.certified.exact_force(twins[twin]["log"], assignment[twin])["passed"]:
                raise ValueError("force mismatch: " + arm + ":" + twin)
    return result


def single(case, variant, frozen):
    assignment = {"none": (None, None), "force": (case["impact"], None)}
    raw = run_twins(case, variant, assignment)
    start, boundary = evaluator.certified.actual_start(raw["r23"]["force"]["log"]["t"],
                                                       case["impact"])
    records = {}
    twins = {}
    for arm, arm_twins in raw.items():
        records[arm] = evaluator.event_record(
            arm_twins["force"]["log"], [arm_twins["none"]["log"]], start,
            min(6000, start + frozen["event_window_ms"]), frozen["quiet_ms"],
            frozen["reference_twin_ms"])
        twins[arm] = {name: value["summary"] for name, value in arm_twins.items()}
    return {"kind": "single", "id": case["id"], "mechanics": variant["kind"],
            "boundary": boundary, "records": records, "twins": twins}


def paired(case, variant, frozen):
    assignment = {"none": (None, None), "first": (case["impacts"][0], None),
                  "second": (None, case["impacts"][1]), "both": tuple(case["impacts"])}
    raw = run_twins(case, variant, assignment)
    starts = [evaluator.certified.actual_start(raw["r23"]["both"]["log"]["t"], impact)
              for impact in case["impacts"]]
    first, second = starts[0][0], starts[1][0]
    window = frozen["event_window_ms"]
    windows = {
        "isolated_first": ("first", first, min(second, first + window), ["none"]),
        "isolated_second": ("second", second, min(6000, second + window), ["none"]),
        "both_first": ("both", first, min(second, first + window), ["none", "second"]),
        "both_second": ("both", second, min(6000, second + window), ["first"]),
    }
    records = {}
    twins = {}
    for arm, arm_twins in raw.items():
        records[arm] = {
            slot: evaluator.event_record(
                arm_twins[twin]["log"], [arm_twins[ref]["log"] for ref in refs],
                begin, end, frozen["quiet_ms"], frozen["reference_twin_ms"])
            for slot, (twin, begin, end, refs) in windows.items()
        }
        twins[arm] = {name: value["summary"] for name, value in arm_twins.items()}
    return {"kind": "paired", "id": case["id"], "mechanics": variant["kind"],
            "boundary": [value[1] for value in starts], "records": records, "twins": twins}


def classify_misses(first):
    groups = {"no_real_rise": [], "duplicate": [], "quiet_25ms": [], "other": []}
    for row in first["rows"]:
        record = row["records"]["r24"]
        events = {"event": record["event"]} if row["kind"] == "single" else record.get("events", {})
        for slot, event in events.items():
            if event["strict"]:
                continue
            raw, attributable = event["raw_rise_indices"], event["attributable_rise_indices"]
            if len(raw) > 1:
                name = "duplicate"
            elif not raw or not attributable:
                name = "no_real_rise"
            elif not event["quiet_before"] or event["high_before"]:
                name = "quiet_25ms"
            else:
                name = "other"
            groups[name].append({"mechanics": row["variant"], "id": row["id"],
                                 "slot": slot, "rises": raw,
                                 "quiet_before": event["quiet_before"],
                                 "high_before": event["high_before"]})
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    first_raw = args.first.read_bytes()
    first = json.loads(first_raw)
    if hashlib.sha256(first_raw).hexdigest() != "dfce0da5e2ded2dd0dfb3e02a168aca48f661f715df0647fb1b46c1c87872c08":
        raise ValueError("unexpected immutable R24 result")
    plan = json.loads(PLAN.read_bytes())
    early = next(case for case in plan["single"] if case["id"] == "s24_transfer_offaxis_early")
    cross = next(case for case in plan["paired"] if case["id"] == "p24_transfer_near_cross")
    rows = [
        single(early, plan["variants"]["stock"], plan),
        paired(cross, plan["variants"]["actuator_delay_gain"], plan),
    ]
    frozen_rows = {(row["variant"], row["id"]): row for row in first["rows"]}
    for row in rows:
        frozen = frozen_rows[(row["mechanics"], row["id"])]
        if row["records"]["r23"] != (frozen["records"]["r23_free"]["event"]
                                           if row["kind"] == "single"
                                           else frozen["records"]["r23_free"]["events"]):
            raise ValueError("R23 event record did not reproduce")
        if row["records"]["joint"] != (frozen["records"]["r24"]["event"]
                                             if row["kind"] == "single"
                                             else frozen["records"]["r24"]["events"]):
            raise ValueError("joint event record did not reproduce")
    doc = {
        "schema": "r24-post-disclosure-causal-diagnosis.v1",
        "classification": "Two affected frozen R24 rows only; no retuning or independent claim",
        "immutable_result_sha256": hashlib.sha256(first_raw).hexdigest(),
        "source_sha256": {
            "r23": sha(ROOT / "experiments/r23_detached/free.py"),
            "pickup": sha(ROOT / "experiments/r24_repeat/trial.py"),
            "dc": sha(ROOT / "experiments/r24_loaded/dc.py"),
            "joint": sha(ROOT / "experiments/r24_composed/detector.py"),
            "evaluator": sha(ROOT / "experiments/r24_holdout/evaluator.py"),
            "plan": sha(PLAN),
        },
        "episodes": 24,
        "rows": rows,
        "non_strict_classification": classify_misses(first),
    }
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(row["mechanics"], row["id"])
        print(json.dumps(row["records"], indent=2))
        for arm in ARMS:
            print(arm, {twin: value["output_intervals"]
                        for twin, value in row["twins"][arm].items()})
    print("non_strict", {name: len(value) for name, value in
                         doc["non_strict_classification"].items()})


if __name__ == "__main__":
    main()
