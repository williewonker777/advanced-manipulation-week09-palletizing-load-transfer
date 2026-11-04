#!/usr/bin/env python3
"""Passive post-D4 timing diagnostics for the two observed worst conditions.

This is an observational follow-up selected after D4.  It never changes GC
settings, production code, weights, thresholds, or simulator modules, and its
measurements do not replace the frozen G5 evidence.  Natural GC callbacks and
per-thread rusage deltas are recorded around the original ``compute`` and
``update`` calls so coincident events can be reported without claiming cause.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.runner import run_episode  # noqa: E402
from tools import sweep, v2_resources, v2_study  # noqa: E402


D4_PATH = ROOT / "results/v2_protocol/S5_D4.json"
OUTLIER_NS = 1_000_000
USAGE_FIELDS = ("minor_page_faults", "major_page_faults",
                "voluntary_context_switches", "involuntary_context_switches")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_snapshot() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    return {
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
    }


def usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: int(after[name] - before[name]) for name in USAGE_FIELDS}


class NaturalGCTracker:
    """Record natural collections without forcing or disabling collection."""

    def __init__(self) -> None:
        self.origin_wall_ns = time.perf_counter_ns()
        self.callback_records: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self._open: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._callback = self._on_gc
        self._attached = False

    def _on_gc(self, phase: str, info: dict[str, Any]) -> None:
        thread_id = threading.get_ident()
        wall_ns = time.perf_counter_ns()
        cpu_ns = time.thread_time_ns()
        generation = int(info.get("generation", -1))
        record = {
            "phase": str(phase), "generation": generation,
            "wall_ns": wall_ns, "thread_cpu_ns": cpu_ns,
            "thread_id": thread_id,
        }
        for name in ("collected", "uncollectable"):
            if name in info:
                record[name] = int(info[name])
        self.callback_records.append(record)
        key = (thread_id, generation)
        if phase == "start":
            event = {
                "generation": generation, "thread_id": thread_id,
                "start_wall_ns": wall_ns, "start_thread_cpu_ns": cpu_ns,
                "complete": False,
            }
            self._open.setdefault(key, []).append(event)
            self.events.append(event)
        elif phase == "stop" and self._open.get(key):
            event = self._open[key].pop()
            event.update(
                stop_wall_ns=wall_ns, stop_thread_cpu_ns=cpu_ns, complete=True,
                collected=int(info.get("collected", 0)),
                uncollectable=int(info.get("uncollectable", 0)),
            )

    def attach(self) -> None:
        if self._attached:
            raise RuntimeError("GC tracker already attached")
        gc.callbacks.append(self._callback)
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        for index, callback in enumerate(gc.callbacks):
            if callback is self._callback:
                del gc.callbacks[index]
                self._attached = False
                return
        self._attached = False
        raise RuntimeError("owned GC callback disappeared before detach")


def overlapping_gc_indices(events: Iterable[dict[str, Any]], start_wall_ns: int,
                           end_wall_ns: int, thread_id: int) -> list[int]:
    """Return complete same-thread GC spans intersecting a measured call."""
    indices = []
    for index, event in enumerate(events):
        if not event.get("complete") or int(event.get("thread_id", -1)) != thread_id:
            continue
        if int(event["start_wall_ns"]) <= end_wall_ns and int(event["stop_wall_ns"]) >= start_wall_ns:
            indices.append(index)
    return indices


def serializable_gc_event(event: dict[str, Any], origin_wall_ns: int,
                          measurement_thread_id: int) -> dict[str, Any]:
    """Keep timing evidence while omitting machine-local thread identifiers."""
    record = {
        "generation": int(event["generation"]),
        "thread_role": ("measurement" if event["thread_id"] == measurement_thread_id else "other"),
        "start_offset_ns": int(event["start_wall_ns"] - origin_wall_ns),
        "complete": bool(event.get("complete", False)),
    }
    if event.get("complete"):
        record.update(
            stop_offset_ns=int(event["stop_wall_ns"] - origin_wall_ns),
            wall_duration_ns=int(event["stop_wall_ns"] - event["start_wall_ns"]),
            thread_cpu_duration_ns=int(
                event["stop_thread_cpu_ns"] - event["start_thread_cpu_ns"]),
            collected=int(event.get("collected", 0)),
            uncollectable=int(event.get("uncollectable", 0)),
        )
    return record


def serializable_gc_callback(record: dict[str, Any], origin_wall_ns: int,
                             measurement_thread_id: int) -> dict[str, Any]:
    result = {
        "phase": record["phase"], "generation": int(record["generation"]),
        "wall_offset_ns": int(record["wall_ns"] - origin_wall_ns),
        "thread_role": ("measurement" if record["thread_id"] == measurement_thread_id else "other"),
    }
    for name in ("collected", "uncollectable"):
        if name in record:
            result[name] = int(record[name])
    return result


def measure_call(component: str, simulated_t_s: float, function: Callable[..., Any],
                 *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Measure one original call; bookkeeping before/after is outside wall time."""
    before = usage_snapshot()
    thread_id = threading.get_ident()
    start_wall_ns = time.perf_counter_ns()
    start_cpu_ns = time.thread_time_ns()
    result = function(*args, **kwargs)
    end_cpu_ns = time.thread_time_ns()
    end_wall_ns = time.perf_counter_ns()
    after = usage_snapshot()
    sample = {
        "component": component, "simulated_t_s": float(simulated_t_s),
        "thread_id": thread_id, "start_wall_ns": start_wall_ns,
        "end_wall_ns": end_wall_ns, "wall_ns": end_wall_ns - start_wall_ns,
        "thread_cpu_ns": end_cpu_ns - start_cpu_ns,
        **usage_delta(before, after),
    }
    return result, sample


def diagnostic_classes(controller_base: type, detector_base: type) -> tuple[type, type]:
    """Wrap public calls without changing their arguments, result, or state."""
    class DiagnosticController(controller_base):  # type: ignore[misc, valid-type]
        def reset(self) -> None:
            self._diag_samples = []
            self._diag_capacity_guards = []
            self._diag_cycle = 0
            reset = getattr(super(), "reset", None)
            if reset is not None:
                reset()

        def compute(self, *args: Any, **kwargs: Any) -> Any:
            if self._diag_cycle % 250 == 0:
                self._diag_capacity_guards.append(v2_resources.wait_for_capacity())
            result, sample = measure_call("controller", float(args[0]), super().compute,
                                          *args, **kwargs)
            self._diag_samples.append(sample)
            self._diag_cycle += 1
            return result

    class DiagnosticDetector(detector_base):  # type: ignore[misc, valid-type]
        def reset(self) -> None:
            self._diag_samples = []
            reset = getattr(super(), "reset", None)
            if reset is not None:
                reset()

        def update(self, *args: Any, **kwargs: Any) -> Any:
            result, sample = measure_call("detector", float(args[0]), super().update,
                                          *args, **kwargs)
            self._diag_samples.append(sample)
            return result

    return DiagnosticController, DiagnosticDetector


def distribution(values_ns: Iterable[int]) -> dict[str, float | int]:
    values = np.asarray(tuple(values_ns), dtype=float) / 1000.0
    if not len(values):
        return {"samples": 0, "median_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
    return {
        "samples": int(len(values)), "median_us": float(np.median(values)),
        "p99_us": float(np.percentile(values, 99)), "max_us": float(np.max(values)),
    }


def component_summary(samples: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps = set()
    for sample in samples:
        overlaps.update(overlapping_gc_indices(events, sample["start_wall_ns"],
                                               sample["end_wall_ns"], sample["thread_id"]))
    return {
        "wall": distribution(sample["wall_ns"] for sample in samples),
        "thread_cpu": distribution(sample["thread_cpu_ns"] for sample in samples),
        "resource_delta_totals": {name: sum(int(sample[name]) for sample in samples)
                                  for name in USAGE_FIELDS},
        "calls_with_resource_delta": {name: sum(int(sample[name]) > 0 for sample in samples)
                                      for name in USAGE_FIELDS},
        "natural_gc_events_overlapping_calls": len(overlaps),
    }


def classify_outlier(outlier: dict[str, Any]) -> dict[str, Any]:
    """Describe observed coincidences, deliberately not a causal diagnosis."""
    combined = outlier["combined"]
    observations = []
    if outlier.get("matching_gc_events"):
        observations.append("natural_gc_span_overlapped_measured_call")
    if combined["major_page_faults"]:
        observations.append("major_page_fault_delta_observed")
    if combined["minor_page_faults"]:
        observations.append("minor_page_fault_delta_observed")
    if combined["voluntary_context_switches"]:
        observations.append("voluntary_context_switch_delta_observed")
    if combined["involuntary_context_switches"]:
        observations.append("involuntary_context_switch_delta_observed")
    if not observations:
        if combined["thread_cpu_ns"] >= OUTLIER_NS and combined["thread_cpu_ns"] >= 0.8 * combined["wall_ns"]:
            observations.append("cpu_time_dominated_without_recorded_gc_fault_or_context_delta")
        elif combined["wall_ns"] - combined["thread_cpu_ns"] >= 100_000:
            observations.append("wall_cpu_gap_without_recorded_thread_context_delta")
        else:
            observations.append("unresolved_by_passive_counters")
    return {
        "observations": observations,
        "causality": "coincident passive evidence only; no event is asserted to cause the outlier",
    }


def summarize_timing(controller_samples: list[dict[str, Any]],
                     detector_samples: list[dict[str, Any]],
                     gc_events: list[dict[str, Any]]) -> dict[str, Any]:
    count = min(len(controller_samples), len(detector_samples))
    paired: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for cycle in range(count):
        controller = controller_samples[cycle]
        detector = detector_samples[cycle]
        combined = {
            "wall_ns": int(controller["wall_ns"] + detector["wall_ns"]),
            "thread_cpu_ns": int(controller["thread_cpu_ns"] + detector["thread_cpu_ns"]),
            **{name: int(controller[name] + detector[name]) for name in USAGE_FIELDS},
        }
        paired.append(combined)
        if max(controller["wall_ns"], detector["wall_ns"], combined["wall_ns"]) < OUTLIER_NS:
            continue
        indices = sorted(set(
            overlapping_gc_indices(gc_events, controller["start_wall_ns"],
                                   controller["end_wall_ns"], controller["thread_id"])
            + overlapping_gc_indices(gc_events, detector["start_wall_ns"],
                                     detector["end_wall_ns"], detector["thread_id"])
        ))
        event = {
            "cycle": cycle,
            "simulated_t_s": float(controller["simulated_t_s"]),
            "controller": {key: value for key, value in controller.items()
                           if key not in {"component", "simulated_t_s", "thread_id",
                                          "start_wall_ns", "end_wall_ns"}},
            "detector": {key: value for key, value in detector.items()
                         if key not in {"component", "simulated_t_s", "thread_id",
                                        "start_wall_ns", "end_wall_ns"}},
            "combined": combined,
            "matching_gc_events": [gc_events[index] for index in indices],
        }
        event["passive_classification"] = classify_outlier(event)
        outliers.append(event)
    return {
        "paired_cycles": count,
        "unpaired_controller_calls": len(controller_samples) - count,
        "unpaired_detector_calls": len(detector_samples) - count,
        "simulated_time_alignment": all(
            abs(float(controller_samples[i]["simulated_t_s"])
                - float(detector_samples[i]["simulated_t_s"])) < 1e-12 for i in range(count)),
        "controller": component_summary(controller_samples, gc_events),
        "detector": component_summary(detector_samples, gc_events),
        "combined": {
            "wall": distribution(sample["wall_ns"] for sample in paired),
            "thread_cpu": distribution(sample["thread_cpu_ns"] for sample in paired),
            "resource_delta_totals": {name: sum(sample[name] for sample in paired)
                                      for name in USAGE_FIELDS},
        },
        "outlier_threshold_us": OUTLIER_NS / 1000.0,
        "outliers": outliers,
    }


def select_worst_final_cases(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = document.get("results")
    if not isinstance(rows, list):
        raise ValueError("D4 document lacks results array")
    selected = {}
    for architecture in ("servo", "torque"):
        candidates = [row for row in rows if row.get("variant") == "v2"
                      and row.get("architecture") == architecture]
        if not candidates:
            raise ValueError(f"D4 lacks final v2 rows for {architecture}")
        selected[architecture] = max(
            candidates, key=lambda row: float(row["timing"]["combined"]["wall"]["max_us"]))
    return selected


def selected_case_record(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("stage", "variant", "architecture", "case_id", "mass", "com", "collision",
            "pick_offset", "kind", "phase", "direction", "force_n")
    outliers = row["timing"].get("outliers", [])
    worst = max(outliers, key=lambda item: item["combined"]["wall_ns"], default=None)
    return {
        **{key: row.get(key) for key in keys},
        "original_combined_wall_max_us": row["timing"]["combined"]["wall"]["max_us"],
        "original_worst_outlier": worst,
    }


def physical_summary(result: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    summary = sweep.evaluate_control_metrics(result, log)
    summary.update(
        detect_delay_ms=result.get("detect_delay_ms"),
        alarm_before_collision=bool(result.get("alarm_before_collision", False)),
        false_alarms=int(result["false_alarms"]),
        runner_ctrl_median_us=float(result["ctrl_time_us"]),
        torque_jitter_nm=float(result["torque_jitter"]),
    )
    return summary


def run_replay(row: dict[str, Any], variant: str, repeat: int) -> dict[str, Any]:
    architecture = str(row["architecture"])
    modules = v2_study.VARIANT_MODULES[variant]
    controller_base = importlib.import_module(modules[architecture]).Controller
    detector_base = importlib.import_module(modules["detector"]).Detector
    controller_type, detector_type = diagnostic_classes(controller_base, detector_base)
    host_before = v2_resources.wait_for_capacity()
    callbacks_before = len(gc.callbacks)
    tracker = NaturalGCTracker()
    tracker.attach()
    try:
        controller, detector = controller_type(), detector_type()
        result, log = run_episode(
            controller, box_mass=float(row["mass"]), box_com=row["com"],
            detector=detector, collision=row.get("collision"),
            pick_offset=row.get("pick_offset"),
        )
    finally:
        tracker.detach()
    callbacks_after = len(gc.callbacks)
    if callbacks_after != callbacks_before:
        raise RuntimeError("GC callback count changed after owned callback removal")
    timing = summarize_timing(controller._diag_samples, detector._diag_samples, tracker.events)
    measurement_thread_id = (
        controller._diag_samples[0]["thread_id"] if controller._diag_samples
        else threading.get_ident()
    )
    for outlier in timing["outliers"]:
        outlier["matching_gc_events"] = [
            serializable_gc_event(event, tracker.origin_wall_ns, measurement_thread_id)
            for event in outlier["matching_gc_events"]
        ]
    return {
        "architecture": architecture, "variant": variant, "repeat": repeat,
        "selected_d4_case_id": int(row["case_id"]),
        "condition": {key: row.get(key) for key in
                      ("mass", "com", "collision", "pick_offset", "kind", "phase",
                       "direction", "force_n")},
        "controller": modules[architecture], "detector": modules["detector"],
        "host_before": host_before, "host_after": v2_resources.snapshot(),
        "capacity_guards": controller._diag_capacity_guards,
        "gc_thresholds_observed": list(gc.get_threshold()),
        "gc_enabled_observed": bool(gc.isenabled()),
        "gc_callback_records": [
            serializable_gc_callback(record, tracker.origin_wall_ns, measurement_thread_id)
            for record in tracker.callback_records
        ],
        "natural_gc_events": [
            serializable_gc_event(event, tracker.origin_wall_ns, measurement_thread_id)
            for event in tracker.events
        ],
        "timing": timing,
        "physical_summary": physical_summary(result, log),
        "collision_classification_note": (
            "raw delay/alarm fields only; no matching reference was added to this 12-episode diagnostic"
        ),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("refusing symlink diagnostic output")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    freeze = v2_study.require_holdout_freeze()
    v2_study.verify_freeze_sources(freeze)
    document = json.loads(D4_PATH.read_text(encoding="utf-8"))
    selected = select_worst_final_cases(document)
    before = {name: sha256_file(ROOT / name) for name in freeze["source_files"]}
    rows = []
    for architecture in ("servo", "torque"):
        for variant in ("v1", "v2"):
            for repeat in range(args.repeats):
                rows.append(run_replay(selected[architecture], variant, repeat))
                print(architecture, variant, repeat + 1, "complete", flush=True)
    v2_study.verify_freeze_sources(freeze)
    after = {name: sha256_file(ROOT / name) for name in freeze["source_files"]}
    if after != before:
        raise RuntimeError("frozen sources changed during diagnostic")
    output = {
        "metadata": {
            "official": False,
            "classification": "analysis-only passive post-D4 diagnostic",
            "selection": "worst final-v2 D4 row per architecture, selected after D4",
            "scope": "observational replay; not fit/threshold/holdout input and not a G5 replacement",
            "causality": "natural GC/rusage coincidences do not by themselves prove root cause",
            "instrumentation": (
                "getrusage and timer bookkeeping surrounds original calls; natural GC callback overhead "
                "is included when it occurs; GC settings are never changed and collection is never forced"
            ),
            "repeats": args.repeats, "episode_count": len(rows),
            "outlier_threshold_us": OUTLIER_NS / 1000.0,
            "d4_sha256": sha256_file(D4_PATH),
            "freeze_sha256": sha256_file(v2_study.FREEZE_PATH),
            "diagnostic_tool_sha256": sha256_file(Path(__file__)),
            "frozen_source_sha256": before,
            "max_owned_simulations": 1,
        },
        "selected_cases": {key: selected_case_record(value) for key, value in selected.items()},
        "results": rows,
    }
    atomic_json(args.out, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
