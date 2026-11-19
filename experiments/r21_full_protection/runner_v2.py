"""Run the fixed 385-case D4 census against R10 and the frozen yaw detector.

This evaluator does not tune either arm.  It instruments the unmodified stock
``sim.runner.run_episode`` only by subclassing ``Plant`` long enough to record
the signed vector passed to its public ``push`` method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

import my_detector
import my_torque
from experiments.r21_yaw_impulse.coarse import Detector as YawDetector
from sim import runner as stock_runner
from tools.check_integrity import check_integrity
from tools.d4_robustness import build_cases


ROOT = Path(__file__).resolve().parents[2]
ARMS = {"r10": my_detector.Detector, "yaw": YawDetector}
PHYSICAL = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "f_ext")
PREFIX = PHYSICAL + ("r", "detect")
ONSET_INDICES = tuple(
    sorted({int(round(1000 * value)) for values in (
        (1.38, 1.41, 1.44, 1.47, 1.50),
        (1.9, 2.3, 2.7, 3.1, 3.4),
        (3.80, 3.83, 3.86, 3.89, 3.92),
    ) for value in values})
)
TWIN_RADIUS = 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value: Any, stop: int | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(value) if stop is None else np.asarray(value)[:stop])
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_cases(cases: list[dict[str, Any]]) -> bytes:
    return (json.dumps(cases, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def intervals(bits: Any) -> list[list[int]]:
    values = np.asarray(bits, dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [[int(a), int(b)] for a, b in zip(starts, stops, strict=True)]


def interval_hits(items: list[list[int]], start: int, stop: int) -> bool:
    return any(a < stop and b > start for a, b in items)


def rises_from_intervals(items: list[list[int]]) -> list[int]:
    return [item[0] for item in items]


def expected_signed(t: Any, collision: Any) -> np.ndarray:
    values = np.zeros((len(t), 3), dtype=float)
    if collision is None:
        return values
    onset, force = collision
    force = np.asarray(force, dtype=float)
    for index, instant in enumerate(np.asarray(t)):
        age = instant - onset
        weight = min(
            age / stock_runner.COLLISION_RAMP,
            1.0,
            (2 * stock_runner.COLLISION_RAMP + stock_runner.COLLISION_HOLD - age)
            / stock_runner.COLLISION_RAMP,
        )
        values[index] = max(weight, 0.0) * force
    return values


class TracePlant(stock_runner.Plant):
    latest: "TracePlant | None" = None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.signed_pushes: list[np.ndarray] = []
        type(self).latest = self

    def push(self, force: Any) -> Any:
        self.signed_pushes.append(np.asarray(force, dtype=float).copy())
        return super().push(force)


def execute(detector_type: type, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    original = stock_runner.Plant
    TracePlant.latest = None
    stock_runner.Plant = TracePlant
    try:
        metrics, log = stock_runner.run_episode(
            my_torque.Controller(),
            box_mass=case["mass"],
            box_com=tuple(case["com"]),
            detector=detector_type(),
            collision=case.get("collision"),
            pick_offset=case.get("pick_offset"),
        )
    finally:
        stock_runner.Plant = original
    traced = TracePlant.latest
    if traced is None:
        raise RuntimeError("stock runner did not construct the instrumented Plant")
    return metrics, log, np.asarray(traced.signed_pushes, dtype=float)


def episode_record(detector_type: type, case: dict[str, Any]) -> dict[str, Any]:
    metrics, log, signed = execute(detector_type, case)
    sample_count = len(signed)
    t = np.asarray(log["t"])[:sample_count]
    detect = np.asarray(log["detect"], dtype=bool)[:sample_count]
    detect_intervals = intervals(detect)
    rise_indices = rises_from_intervals(detect_intervals)
    artificial = np.zeros(sample_count, dtype=bool)
    onset_index = None
    collision = case.get("collision")
    if collision is not None:
        onset = float(collision[0])
        onset_index = int(np.searchsorted(t, onset))
        artificial = (t >= onset) & (t < onset + 0.32)
    tool_hit = np.asarray(log["f_tool"])[:sample_count] > 1.0
    tool_window = np.zeros(sample_count, dtype=bool)
    tool_first = tool_last = None
    if tool_hit.any():
        tool_first = int(np.argmax(tool_hit))
        tool_last = int(np.flatnonzero(tool_hit)[-1])
        tool_window[tool_first:min(sample_count, tool_last + 300)] = True
    events = []
    for index in rise_indices:
        labels = []
        if artificial[index]:
            labels.append("artificial")
        if tool_window[index]:
            labels.append("tool")
        if not labels:
            labels.append("false")
        events.append({"index": index, "time_s": float(t[index]), "labels": labels})
    false_indices = [event["index"] for event in events if event["labels"] == ["false"]]
    art_indices = [event["index"] for event in events if "artificial" in event["labels"]]
    tool_indices = [event["index"] for event in events if "tool" in event["labels"]]
    expected = expected_signed(t, collision)
    signed_exact = signed.shape == expected.shape and np.array_equal(signed, expected)
    scalar_exact = signed.shape == expected.shape and np.array_equal(
        np.asarray(log["f_ext"])[:sample_count], np.linalg.norm(signed, axis=1)
    )
    nonzero = np.flatnonzero(np.linalg.norm(signed, axis=1) > 0) if sample_count else np.array([], int)
    prefix = None
    if collision is None:
        prefix = {
            str(index): {
                "before_onset": {key: array_sha(log[key], index) for key in PREFIX},
                "through_zero_force_onset": {key: array_sha(log[key], index + 1) for key in PREFIX},
            }
            for index in ONSET_INDICES
        }
    else:
        assert onset_index is not None
        prefix = {
            str(onset_index): {
                "before_onset": {key: array_sha(log[key], onset_index) for key in PREFIX},
                "through_zero_force_onset": {
                    key: array_sha(log[key], onset_index + 1) for key in PREFIX
                },
            }
        }
    return {
        "complete_fixed_grid": sample_count == 6000,
        "sample_count": sample_count,
        "metrics": {
            "rms_mm": float(metrics["rms_mm"]),
            "max_mm": float(metrics["max_mm"]),
            "diverged": bool(metrics["diverged"]),
            "saturation_pct": float(metrics["saturation_pct"]),
            "detect_delay_ms": metrics.get("detect_delay_ms"),
            "alarm_before_collision": bool(metrics.get("alarm_before_collision", False)),
            "false_alarms": int(metrics["false_alarms"]),
            "false_alarm_times": metrics["false_alarm_times"],
            "tool_contact_delay_ms": metrics.get("tool_contact_delay_ms"),
        },
        "detect_intervals": detect_intervals,
        "events": events,
        "artificial_rise_indices": art_indices,
        "tool_rise_indices": tool_indices,
        "false_rise_indices": false_indices,
        "delayed_false_indices": (
            [index for index in false_indices if collision is not None and t[index] >= collision[0] + 0.32]
        ),
        "masked_at_onset": bool(
            onset_index is not None and onset_index > 0 and detect[onset_index - 1]
        ),
        "onset_index": onset_index,
        "true_window_first_index": onset_index,
        "true_window_last_index": (
            int(np.flatnonzero(artificial)[-1]) if artificial.any() else None
        ),
        "force_first_nonzero_index": int(nonzero[0]) if len(nonzero) else None,
        "force_last_nonzero_index": int(nonzero[-1]) if len(nonzero) else None,
        "signed_expected_exact": bool(signed_exact),
        "stock_scalar_matches_signed": bool(scalar_exact),
        "tool": {
            "present": bool(tool_hit.any()),
            "first_index": tool_first,
            "last_index": tool_last,
            "peak_n": float(np.max(log["f_tool"])),
            "prealarm": bool(tool_first is not None and tool_first > 0 and detect[tool_first - 1]),
            "detected": bool(tool_indices and not (
                tool_first is not None and tool_first > 0 and detect[tool_first - 1]
            )),
        },
        "physical_sha256": {key: array_sha(log[key]) for key in PHYSICAL},
        "prefix_sha256": prefix,
    }


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case,
        "arms": {name: episode_record(detector, case) for name, detector in ARMS.items()},
    }


def reference_key(case: dict[str, Any]) -> tuple[float, tuple[float, ...]]:
    return float(case["mass"]), tuple(case["com"])


def has_twin(reference: dict[str, Any], index: int) -> bool:
    return interval_hits(reference["detect_intervals"], max(0, index - TWIN_RADIUS), index + TWIN_RADIUS + 1)


def unmatched(candidate: list[int], baseline: list[int], radius: int = 1) -> list[int]:
    return [value for value in candidate if not any(abs(value - other) <= radius for other in baseline)]


def enrich(rows: list[dict[str, Any]]) -> None:
    references = {
        (arm, reference_key(row["case"])): row["arms"][arm]
        for row in rows if row["case"]["kind"] == "reference" for arm in ARMS
    }
    for row in rows:
        case = row["case"]
        base = row["arms"]["r10"]
        candidate = row["arms"]["yaw"]
        row["physical_exact_between_arms"] = {
            key: base["physical_sha256"][key] == candidate["physical_sha256"][key]
            for key in PHYSICAL
        }
        arm_gates: dict[str, Any] = {}
        for arm, record in row["arms"].items():
            reference = (references[(arm, reference_key(case))]
                         if case["kind"] == "collision" else record)
            false_events = []
            for index in record["false_rise_indices"]:
                false_events.append({
                    "index": index,
                    "time_s": index / 1000.0,
                    "matched_no_force_within_20ms": has_twin(reference, index),
                })
            record["false_events_vs_matched_no_force"] = false_events
            record["new_false_not_in_matched_no_force"] = [
                event["index"] for event in false_events
                if not event["matched_no_force_within_20ms"]
            ]
            record["artificial_reference_twin"] = [
                index for index in record["artificial_rise_indices"] if has_twin(reference, index)
            ]
            attributable = [
                index for index in record["artificial_rise_indices"] if not has_twin(reference, index)
            ]
            record["artificial_attributable_rise_indices"] = attributable
            record["duplicate_in_window_rises"] = max(0, len(record["artificial_rise_indices"]) - 1)
            record["duplicate_attributable_rises"] = max(0, len(attributable) - 1)
            masked = record["masked_at_onset"]
            if case["kind"] == "collision":
                if masked:
                    strict = "masked"
                elif not record["artificial_rise_indices"]:
                    strict = "missed"
                elif record["artificial_rise_indices"][0] in record["artificial_reference_twin"]:
                    strict = "indistinguishable"
                else:
                    strict = "detected"
                delay = (
                    None if strict != "detected" else
                    record["artificial_rise_indices"][0] - int(record["onset_index"])
                )
                official_delay = record["metrics"]["detect_delay_ms"]
                official_index = (
                    None if official_delay is None else
                    int(round(1000 * float(case["collision"][0]) + official_delay))
                )
                official = (
                    "alarm_before_collision" if record["metrics"]["alarm_before_collision"] else
                    "missed" if official_index is None else
                    "indistinguishable" if has_twin(reference, official_index) else "detected"
                )
            else:
                strict, delay, official = "not_applicable", None, "not_applicable"
            record["strict_outcome"] = strict
            record["strict_delay_ms"] = delay
            record["official_d4_outcome"] = official
            prefix_checks = None
            if case["kind"] == "collision":
                key = str(record["onset_index"])
                prefix_checks = {
                    mode: {
                        field: digest == reference["prefix_sha256"][key][mode][field]
                        for field, digest in record["prefix_sha256"][key][mode].items()
                    }
                    for mode in ("before_onset", "through_zero_force_onset")
                }
            record["matched_no_force_prefix_exact"] = prefix_checks
            arm_gates[arm] = {
                "complete_fixed_grid": record["complete_fixed_grid"],
                "signed_expected_exact": record["signed_expected_exact"],
                "stock_scalar_matches_signed": record["stock_scalar_matches_signed"],
                "matched_prefix_exact": (
                    True if prefix_checks is None else
                    all(all(values.values()) for values in prefix_checks.values())
                ),
                "off_by_one_force_schedule": (
                    True if case["kind"] != "collision" else
                    record["force_first_nonzero_index"] == record["onset_index"] + 1
                    and record["force_last_nonzero_index"] == record["onset_index"] + 119
                    and record["true_window_first_index"] == record["onset_index"]
                    and record["true_window_last_index"] == record["onset_index"] + 319
                ),
            }
        comparison = {
            "physical_exact": all(row["physical_exact_between_arms"].values()),
            "no_new_false_vs_r10": not unmatched(
                candidate["false_rise_indices"], base["false_rise_indices"]
            ),
            "no_new_unmatched_no_force_false": not unmatched(
                candidate["new_false_not_in_matched_no_force"],
                base["new_false_not_in_matched_no_force"],
            ),
            "no_new_masked_collision": (
                case["kind"] != "collision" or not candidate["masked_at_onset"]
                or base["masked_at_onset"]
            ),
            "preserve_strict_collision": (
                case["kind"] != "collision" or base["strict_outcome"] != "detected"
                or candidate["strict_outcome"] == "detected"
            ),
            "no_new_duplicate_attributable_rises": (
                candidate["duplicate_attributable_rises"]
                <= base["duplicate_attributable_rises"]
            ),
            "preserve_tool_detection": (
                not base["tool"]["detected"] or candidate["tool"]["detected"]
            ),
            "no_new_tool_prealarm": (
                not candidate["tool"]["prealarm"] or base["tool"]["prealarm"]
            ),
        }
        row["arm_gates"] = arm_gates
        row["comparison"] = comparison
        row["passed"] = (
            all(all(gates.values()) for gates in arm_gates.values())
            and all(comparison.values())
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    collisions = [row for row in rows if row["case"]["kind"] == "collision"]
    summary: dict[str, Any] = {}
    for arm in ARMS:
        records = [row["arms"][arm] for row in rows]
        collision_records = [row["arms"][arm] for row in collisions]
        summary[arm] = {
            "official_outcomes": {
                outcome: sum(r["official_d4_outcome"] == outcome for r in collision_records)
                for outcome in ("detected", "missed", "alarm_before_collision", "indistinguishable")
            },
            "strict_outcomes": {
                outcome: sum(r["strict_outcome"] == outcome for r in collision_records)
                for outcome in ("detected", "missed", "masked", "indistinguishable")
            },
            "false_rises": sum(len(r["false_rise_indices"]) for r in records),
            "new_false_not_in_matched_no_force": sum(
                len(r["new_false_not_in_matched_no_force"]) for r in records
            ),
            "delayed_false_rises": sum(len(r["delayed_false_indices"]) for r in records),
            "duplicate_in_window_rises": sum(r["duplicate_in_window_rises"] for r in collision_records),
            "duplicate_attributable_rises": sum(
                r["duplicate_attributable_rises"] for r in collision_records
            ),
            "tool_present": sum(r["tool"]["present"] for r in records),
            "tool_missed": sum(r["tool"]["present"] and not r["tool"]["detected"] for r in records),
            "tool_prealarm": sum(r["tool"]["prealarm"] for r in records),
        }
    summary["strict_candidate_only_gains"] = [
        row["case"]["case_id"] for row in collisions
        if row["arms"]["r10"]["strict_outcome"] != "detected"
        and row["arms"]["yaw"]["strict_outcome"] == "detected"
    ]
    summary["strict_losses"] = [
        row["case"]["case_id"] for row in collisions
        if row["arms"]["r10"]["strict_outcome"] == "detected"
        and row["arms"]["yaw"]["strict_outcome"] != "detected"
    ]
    summary["new_false_case_ids"] = [
        row["case"]["case_id"] for row in rows
        if not row["comparison"]["no_new_false_vs_r10"]
    ]
    summary["new_duplicate_case_ids"] = [
        row["case"]["case_id"] for row in collisions
        if not row["comparison"]["no_new_duplicate_attributable_rises"]
    ]
    summary["failed_case_ids"] = [row["case"]["case_id"] for row in rows if not row["passed"]]
    return summary


def validate_plan(plan: dict[str, Any], raw: bytes) -> list[dict[str, Any]]:
    if plan.get("schema") != "r21-full-d4-protection-plan.v1" or plan.get("status") != "FROZEN_BEFORE_OUTCOMES":
        raise ValueError("plan is not the frozen full D4 plan")
    if plan.get("arms") != ["r10", "yaw"] or plan.get("case_count") != 385:
        raise ValueError("fixed arms/case count differ")
    for relative, digest in plan["source_pins"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("source pin differs: " + relative)
    cases = build_cases("my_torque")
    if hashlib.sha256(canonical_cases(cases)).hexdigest() != plan["matrix_sha256"]:
        raise ValueError("D4 matrix differs from preregistration")
    if len(cases) != 385 or len({case["case_id"] for case in cases}) != 385:
        raise ValueError("D4 matrix is not 385 unique IDs")
    if hashlib.sha256(raw).hexdigest() != plan["plan_sha256"]:
        # Self hashes are impossible inside their own bytes; the external pin is optional.
        if plan["plan_sha256"] != "EXTERNAL_SHA256_REQUIRED":
            raise ValueError("unexpected plan hash contract")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    if os.path.lexists(args.out):
        parser.error("new output path required")
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    cases = validate_plan(plan, raw)
    kit_integrity = check_integrity(ROOT)
    before = {relative: sha(ROOT / relative) for relative in plan["source_pins"]}
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for case in cases:
            rows.append(run_case(case))
            if len(rows) % 25 == 0:
                print(f"{len(rows)}/{len(cases)}", flush=True)
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for row in pool.imap(run_case, cases):
                rows.append(row)
                if len(rows) % 25 == 0:
                    print(f"{len(rows)}/{len(cases)}", flush=True)
    after = {relative: sha(ROOT / relative) for relative in plan["source_pins"]}
    if before != after:
        raise RuntimeError("protected source changed during census")
    rows.sort(key=lambda row: row["case"]["case_id"])
    enrich(rows)
    summary = summarize(rows)
    result = {
        "schema": "r21-full-d4-protection-result.v1",
        "classification": "full pre-existing exposed D4 census; not an independent hidden holdout",
        "plan_external_sha256": hashlib.sha256(raw).hexdigest(),
        "arms": list(ARMS),
        "cases_per_arm": len(cases),
        "episodes": len(cases) * len(ARMS),
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "kit_integrity": kit_integrity,
        "source_pins_before": before,
        "source_pins_after": after,
        "summary": summary,
        "rows": rows,
        "passed": not summary["failed_case_ids"],
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "episodes": result["episodes"], "passed": result["passed"],
        "failed_cases": len(summary["failed_case_ids"]),
        "strict_gains": len(summary["strict_candidate_only_gains"]),
        "strict_losses": len(summary["strict_losses"]),
        "new_false_cases": len(summary["new_false_case_ids"]),
        "new_duplicate_cases": len(summary["new_duplicate_case_ids"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
