"""Probe the disclosed 3.922 s post-release false against signed matched forces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r23_detached.free import Detector as R23
from experiments.r24_composed.detector import Detector as R24
from experiments.r24_holdout import evaluator


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "s24_detached_offaxis_b"
MECHANICS = "actuator_delay_gain"
FIRST_SHA = "dfce0da5e2ded2dd0dfb3e02a168aca48f661f715df0647fb1b46c1c87872c08"
QA_SHA = "f07008b56aa6ae17f83e3db8fee0e6b15d52cf01332dad3c0c5e268340486aa4"
SLOW_NAMES = ("abs_fx", "abs_fy", "negative_fz", "abs_mx", "abs_my",
              "abs_mz", "positive_out_of_height_fz")
FAST_NAMES = ("abs_dfx", "abs_dfy", "negative_dfz", "abs_dmx", "abs_dmy",
              "abs_dmz", "positive_dfz")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(packet):
    raw = np.asarray(packet.last_wrench, float)
    residual = np.asarray(packet.last_residual_wrench, float)
    innovation = np.asarray(packet.last_fast_innovation, float)
    slow = np.r_[
        np.abs(residual[:2]) / packet.force_threshold[:2],
        max(-float(raw[2]), 0.0) / packet.force_threshold[2],
        np.abs(residual[3:]) / packet.moment_threshold,
        (0.0 if packet.last_table_allowed else
         max(float(raw[2]), 0.0) / packet.out_of_height_fz_threshold),
    ]
    fast = np.r_[
        np.abs(innovation[:2]) / packet.fast_force_threshold[:2],
        max(-float(innovation[2]), 0.0) / packet.fast_force_threshold[2],
        np.abs(innovation[3:]) / packet.fast_moment_threshold,
        max(float(innovation[2]), 0.0) / packet.fast_positive_fz_threshold,
    ]
    return raw, residual, innovation, slow, fast


def observed(base):
    class Observed(base):
        def reset(self):
            super().reset()
            self.samples = []

        def update(self, t, th, thd, tau_m, tau_s, grip):
            residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
            tick = int(round(1000 * float(t)))
            if 3860 <= tick <= 4000 or 4320 <= tick <= 4520:
                packet = self._shared
                raw, standard, innovation, slow, fast = normalized(packet)
                normal = getattr(self, "normal_model", None)
                self.samples.append({
                    "tick": tick,
                    "reported": bool(report),
                    "r10": bool(packet.detected),
                    "normal_model": bool(normal is not None and normal.detected),
                    "horizontal": bool(self.horizontal_detected),
                    "yaw": bool(self.yaw_active),
                    "after_release": bool(self.detached.evidence.active),
                    "grip": bool(grip),
                    "raw_wrench_world": raw.tolist(),
                    "standard_residual_world": standard.tolist(),
                    "fast_innovation_world": innovation.tolist(),
                    "slow_standardized": dict(zip(SLOW_NAMES, slow.tolist(), strict=True)),
                    "fast_standardized": dict(zip(FAST_NAMES, fast.tolist(), strict=True)),
                    "slow_score": float(packet.last_score),
                    "fast_score": float(packet.last_fast_score),
                    "slow_persistence_count": int(packet.alarm_count),
                    "fast_persistence_count": int(packet.fast_alarm_count),
                    "clear_count": int(packet.clear_count),
                    "table_allowed": bool(packet.last_table_allowed),
                    "estimated_table_wrench_world": np.asarray(
                        packet.last_table_wrench, float).tolist(),
                    "packet_q": np.asarray(packet.q_prev, float).tolist(),
                    "packet_v": np.asarray(packet.v, float).tolist(),
                })
            return residual, report

    return Observed


def rises(samples, key):
    result, previous = [], False
    for sample in samples:
        value = bool(sample[key])
        if value and not previous:
            result.append(sample["tick"])
        previous = value
    return result


def first_cross(samples, branch, begin, end):
    for sample in samples:
        if not begin <= sample["tick"] < end:
            continue
        ratios = sample[branch]
        crossed = [name for name, value in ratios.items() if value > 1.0]
        if crossed:
            top = max(ratios, key=ratios.get)
            return {"tick": sample["tick"], "crossed": crossed,
                    "top": top, "ratios": ratios,
                    "slow_count": sample["slow_persistence_count"],
                    "fast_count": sample["fast_persistence_count"]}
    return None


def unit(vector):
    value = np.asarray(vector, float)
    norm = float(np.linalg.norm(value))
    return None if norm == 0.0 else (value / norm).tolist()


def enrich(detector, log):
    for sample in detector.samples:
        tick = sample["tick"]
        q = np.asarray(sample.pop("packet_q"), float)
        v = np.asarray(sample.pop("packet_v"), float)
        jacobian = detector._shared._point_jacobian(q)[0]
        linear = jacobian[:3] @ v
        applied = np.asarray(log["f_ext_xyz_committed"][tick], float)
        sample.update({
            "packet_linear_velocity_world_m_s": linear.tolist(),
            "packet_linear_velocity_direction": unit(linear),
            "actual_applied_force_world_N": applied.tolist(),
            "actual_applied_force_direction": unit(applied),
            "actual_table_normal_N": float(log["f_table"][tick]),
            "actual_tool_N": float(log["f_tool"][tick]),
        })


def episode(base, case, variant, impacts, plan):
    detector = observed(base)()
    _, log = evaluator.run_episode(case, variant, lambda: detector, impacts)
    enrich(detector, log)
    starts = [evaluator.certified.actual_start(log["t"], impact)[0]
              for impact in impacts if impact is not None]
    common = evaluator.record_common(log, impacts, starts, plan)
    begin = starts[0] if starts else None
    output_rises = rises(detector.samples, "reported")
    r10_rises = rises(detector.samples, "r10")
    return detector, log, {
        "actual_force_start": begin,
        "common": common,
        "output_rises_in_capture": output_rises,
        "r10_rises_in_capture": r10_rises,
        "false_window": {
            "slow_first_cross": first_cross(detector.samples, "slow_standardized", 3860, 4000),
            "fast_first_cross": first_cross(detector.samples, "fast_standardized", 3860, 4000),
        },
        "force_window": None if begin is None else {
            "slow_first_cross": first_cross(detector.samples, "slow_standardized", begin, begin + 172),
            "fast_first_cross": first_cross(detector.samples, "fast_standardized", begin, begin + 172),
        },
        "samples": detector.samples,
        "physical_sha256": evaluator.certified.physical_hash(log),
    }


def sample_at(record, tick):
    return next(sample for sample in record["samples"] if sample["tick"] == tick)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh --out required")
    if sha(args.prior) != FIRST_SHA or sha(args.archive) != QA_SHA:
        raise ValueError("unexpected disclosed R24 archive")
    prior = json.loads(args.prior.read_bytes())
    archive = json.loads(args.archive.read_bytes())
    plan = json.loads(evaluator.PLAN.read_bytes())
    case = next(item for item in plan["single"] if item["id"] == CASE_ID)
    variant = plan["variants"][MECHANICS]
    row = next(item for item in prior["rows"]
               if item["id"] == CASE_ID and item["variant"] == MECHANICS)
    registered = tuple(case["impact"])
    negated = (registered[0], (-np.asarray(registered[1], float)).tolist())
    requests = {
        "r23_none": (R23, (None, None), "r23_free", "none"),
        "r23_registered": (R23, (registered, None), "r23_free", "force"),
        "r24_none": (R24, (None, None), "r24", "none"),
        "r24_registered": (R24, (registered, None), "r24", "force"),
        "r24_negated": (R24, (negated, None), None, None),
    }
    records, logs = {}, {}
    for name, (base, impacts, archive_arm, archive_twin) in requests.items():
        detector, log, record = episode(base, case, variant, impacts, plan)
        records[name], logs[name] = record, log
        if archive_arm is not None:
            expected = row["records"][archive_arm]["twins"][archive_twin]
            if record["common"] != expected:
                raise ValueError("archived common record mismatch: " + name)
            old = archive["records"][archive_arm][archive_twin]
            rise = sample_at(record, 3922)
            if (record["common"]["false_times_s"] != [3.922] or
                    record["common"]["false_active_samples"] != 8 or
                    rise["reported"] is not True or
                    rise["fast_score"] != old["at_rise"]["r10_fast_suspicion"] or
                    rise["actual_table_normal_N"] != old["at_rise"]["actual_table_N"]):
                raise ValueError("archived false alarm mismatch: " + name)
    if evaluator.certified.trace_signature({
            "none": logs["r23_none"], "force": logs["r23_registered"]}) != row[
                "r23_physics_sha256"]:
        raise ValueError("R23 archive physics mismatch")
    if evaluator.certified.trace_signature({
            "none": logs["r24_none"], "force": logs["r24_registered"]}) != row[
                "r23_physics_sha256"]:
        raise ValueError("R24 archive physics mismatch")

    summary = {}
    for name, record in records.items():
        force_start = record["actual_force_start"]
        force_rise = None
        if force_start is not None:
            later = [tick for tick in record["output_rises_in_capture"] if tick >= force_start]
            force_rise = later[0] if later else None
        summary[name] = {
            "false_output_rise": next((tick for tick in record["output_rises_in_capture"]
                                       if 3860 <= tick < 4000), None),
            "false_rise_sample": sample_at(record, 3922),
            "false_window": record["false_window"],
            "force_start": force_start,
            "force_output_rise": force_rise,
            "force_rise_sample": (None if force_rise is None
                                  else sample_at(record, force_rise)),
            "force_window": record["force_window"],
        }
    output = {
        "schema": "r25-postrelease-direction-probe.v1",
        "classification": "Five disclosed-case physical probes; diagnosis only",
        "case": case, "mechanics": MECHANICS,
        "registered_force": registered,
        "negated_force": negated,
        "episodes": len(records),
        "archival_parity": {
            "prior_sha256": sha(args.prior), "archive_sha256": sha(args.archive),
            "four_common_records_exact": True,
            "four_false_rises_exact": True,
            "r23_and_r24_trace_signatures_exact": True,
        },
        "summary": summary,
    }
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {
        "false_fast": value["false_window"]["fast_first_cross"],
        "force_start": value["force_start"],
        "force_rise": value["force_output_rise"],
        "force_fast": (None if value["force_window"] is None else
                       value["force_window"]["fast_first_cross"]),
    } for name, value in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
