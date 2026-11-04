#!/usr/bin/env python3
"""Run the preregistered v2 control, development, robustness, and timing studies.

All episodes are sequential.  The simulator is used through its public runner;
only behavior-preserving subclasses collect timing data around controller and
detector calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim import trajectory as T  # noqa: E402
from sim.runner import run_episode  # noqa: E402
from tools import d4_robustness, sweep, v2_resources  # noqa: E402


PROTOCOL_PATH = ROOT / "results/v2_protocol/PROTOCOL.json"
FREEZE_PATH = ROOT / "results/v2_protocol/FREEZE.json"
GRAVITY = 9.81
ALL_VARIANTS = ("v1", "C", "L", "CL", "v2")
ARCHITECTURES = ("servo", "torque")
VARIANT_MODULES = {
    "v1": {"servo": "v1_servo", "torque": "v1_torque", "detector": "v1_detector"},
    "C": {
        "servo": "experiments.v2_normal_servo",
        "torque": "experiments.v2_normal_torque",
        "detector": "v1_detector",
    },
    "L": {
        "servo": "v1_servo",
        "torque": "v1_torque",
        "detector": "v2_detector_candidate",
    },
    "CL": {
        "servo": "experiments.v2_normal_servo",
        "torque": "experiments.v2_normal_torque",
        "detector": "v2_detector_candidate",
    },
    "v2": {"servo": "my_servo", "torque": "my_torque", "detector": "my_detector"},
}
DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
    "+xy": (1.0, 1.0, 0.0),
    "-xy": (-1.0, -1.0, 0.0),
    "+xyz": (1.0, 1.0, 1.0),
    "-xyz": (-1.0, -1.0, -1.0),
}
HOLDOUT_PHASE_RANGES = {
    "pick": (1.375, 1.515),
    "transport": (2.05, 3.35),
    "place": (3.775, 3.945),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_holdout_freeze(
    protocol_path: Path = PROTOCOL_PATH, freeze_path: Path = FREEZE_PATH
) -> dict[str, Any]:
    """Return the freeze only when it names the exact current protocol bytes."""
    if not freeze_path.is_file():
        raise RuntimeError(f"holdout is sealed: required freeze is missing: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = sha256_file(protocol_path)
    if freeze.get("protocol_sha256") != expected:
        raise RuntimeError("holdout freeze protocol_sha256 does not match PROTOCOL.json")
    return freeze


def verify_freeze_sources(freeze: dict[str, Any], root: Path = ROOT) -> None:
    """Execution gate: verify actual frozen files, not merely a protocol label."""
    files = freeze.get("source_files", {})
    if not files:
        raise RuntimeError("holdout execution requires frozen source/weight file hashes")
    for name, expected in files.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise RuntimeError("unsafe freeze source path")
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen source/weight mismatch: {name}")


def parse_variants(text: str) -> tuple[str, ...]:
    variants = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [variant for variant in variants if variant not in ALL_VARIANTS]
    if not variants or unknown:
        raise argparse.ArgumentTypeError(
            f"variants must be a nonempty comma list from {','.join(ALL_VARIANTS)}"
        )
    if len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError("variants must not contain duplicates")
    return variants


def _force(direction: str, magnitude: float) -> tuple[float, float, float]:
    vector = np.asarray(DIRECTIONS[direction], dtype=float)
    return tuple((magnitude * vector / np.linalg.norm(vector)).tolist())


def _base_case(
    stage: str,
    variant: str,
    architecture: str,
    case_id: int,
    mass: float,
    com: Iterable[float],
    collision: Any = None,
    pick_offset: Any = None,
    **tags: Any,
) -> dict[str, Any]:
    modules = VARIANT_MODULES[variant]
    return {
        "stage": stage,
        "variant": variant,
        "architecture": architecture,
        "architecture_label": "A" if architecture == "servo" else "B",
        "case_id": case_id,
        "controller": modules[architecture],
        "detector": modules["detector"],
        "mass": float(mass),
        "com": tuple(float(value) for value in com),
        "collision": collision,
        "pick_offset": None if pick_offset is None else tuple(float(v) for v in pick_offset),
        "official": False,
        **tags,
    }


def build_control_cases(protocol: dict[str, Any], variant: str, architecture: str) -> list[dict[str, Any]]:
    return [
        _base_case(
            "control", variant, architecture, index, item["mass"], item["com"],
            kind="reference", phase="none", direction="none", force_n=0.0,
        )
        for index, item in enumerate(protocol["validation"])
    ]


def build_development_cases(
    protocol: dict[str, Any], variant: str, architecture: str
) -> list[dict[str, Any]]:
    design = protocol["development"]
    conditions = list(zip(design["masses"], design["coms"], strict=True))
    cases = [
        _base_case(
            "development", variant, architecture, index, mass, com,
            kind="reference", phase="none", direction="none", force_n=0.0,
        )
        for index, (mass, com) in enumerate(conditions)
    ]
    combinations = itertools.product(
        design["collision_times_s"], design["forces_N"], design["directions"]
    )
    for index, (collision_time, force_n, direction) in enumerate(combinations):
        mass, com = conditions[index % len(conditions)]
        cases.append(_base_case(
            "development", variant, architecture, len(cases), mass, com,
            collision=(float(collision_time), _force(direction, float(force_n))),
            kind="collision", phase="development", direction=direction,
            force_n=float(force_n),
        ))
    return cases


def build_d4_cases(variant: str, architecture: str) -> list[dict[str, Any]]:
    modules = VARIANT_MODULES[variant]
    cases = d4_robustness.build_cases(modules[architecture], modules["detector"])
    for case in cases:
        case.update(
            stage="d4", variant=variant, architecture=architecture,
            architecture_label="A" if architecture == "servo" else "B", official=False,
        )
    return cases


def build_holdout_cases(
    protocol: dict[str, Any],
    variant: str,
    architecture: str,
    *,
    protocol_path: Path = PROTOCOL_PATH,
    freeze_path: Path = FREEZE_PATH,
) -> list[dict[str, Any]]:
    """Construct sealed 4-reference/30-collision/4-offset holdout cases after freeze validation."""
    require_holdout_freeze(protocol_path, freeze_path)
    design = protocol["holdout"]
    rng = np.random.default_rng(int(protocol["seeds"]["holdout"]))
    masses = tuple(float(value) for value in design["masses"])
    coms = tuple(
        tuple(values)
        for values in zip(
            rng.uniform(-0.028, 0.028, len(masses)),
            rng.uniform(-0.018, 0.018, len(masses)),
            rng.uniform(-0.012, 0.048, len(masses)),
            strict=True,
        )
    )
    offsets = tuple(tuple(float(value) for value in offset) for offset in design["offset_m"])
    conditions = tuple(zip(masses, coms, strict=True))
    cases = [
        _base_case(
            "holdout", variant, architecture, index, mass, com,
            kind="reference", phase="none", direction="none", force_n=0.0,
        )
        for index, (mass, com) in enumerate(conditions)
    ]

    phases = np.repeat(tuple(HOLDOUT_PHASE_RANGES), 10).astype(object)
    forces = np.tile(np.asarray(design["forces_N"], dtype=float), 6)
    directions = np.tile(np.asarray(design["directions"], dtype=object), 3)
    condition_indices = np.resize(np.arange(len(conditions)), 30)
    for values in (phases, forces, directions, condition_indices):
        rng.shuffle(values)
    for index in range(30):
        mass, com = conditions[int(condition_indices[index])]
        phase = str(phases[index])
        collision_time = float(rng.uniform(*HOLDOUT_PHASE_RANGES[phase]))
        force_n = float(forces[index])
        direction = str(directions[index])
        cases.append(_base_case(
            "holdout", variant, architecture, len(cases), mass, com,
            collision=(collision_time, _force(direction, force_n)),
            kind="collision", phase=phase, direction=direction, force_n=force_n,
        ))
    for (mass, com), offset in zip(conditions, offsets, strict=True):
        cases.append(_base_case(
            "holdout", variant, architecture, len(cases), mass, com, pick_offset=offset,
            kind="offset", phase="pick", direction="offset", force_n=0.0,
        ))
    return cases


def build_timing_cases(variant: str, architecture: str) -> list[dict[str, Any]]:
    return [
        _base_case(
            "timing", variant, architecture, index, mass, sweep.DEFAULT_COM,
            kind="reference", phase="none", direction="none", force_n=0.0,
        )
        for index, mass in enumerate((0.0, 4.0, 8.0))
    ]


def build_stage_cases(
    stage: str, protocol: dict[str, Any], variant: str, architecture: str
) -> list[dict[str, Any]]:
    builders = {
        "control": lambda: build_control_cases(protocol, variant, architecture),
        "development": lambda: build_development_cases(protocol, variant, architecture),
        "d4": lambda: build_d4_cases(variant, architecture),
        "holdout": lambda: build_holdout_cases(protocol, variant, architecture),
        "timing": lambda: build_timing_cases(variant, architecture),
    }
    return builders[stage]()


def g4_window_metrics(log: dict[str, Any], mass: float) -> dict[str, Any]:
    """Compute preregistered final-20-ms support and placement peak metrics."""
    times = np.asarray(log["t"], dtype=float)
    table = np.asarray(log["f_table"], dtype=float)
    final = (times >= T.T_RELEASE - 0.020) & (times < T.T_RELEASE)
    place = (times >= sweep.PLACE_WINDOW[0]) & (times < sweep.PLACE_WINDOW[1])
    mean_n = float(np.mean(table[final])) if final.any() else None
    minimum_n = float(np.min(table[final])) if final.any() else None
    peak_n = float(np.max(table[place])) if place.any() else None
    mg = float(mass) * GRAVITY
    return {
        "available": bool(final.any() and place.any()),
        "window_start_s": T.T_RELEASE - 0.020,
        "window_end_s": T.T_RELEASE,
        "table_mean_n": mean_n,
        "table_min_n": minimum_n,
        "table_mean_over_mg": None if mg <= 0.0 or mean_n is None else mean_n / mg,
        "table_min_over_mg": None if mg <= 0.0 or minimum_n is None else minimum_n / mg,
        "place_peak_n": peak_n,
        "place_peak_over_mg": None if mg <= 0.0 or peak_n is None else peak_n / mg,
    }


def _rusage_context() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    return int(usage.ru_nvcsw), int(usage.ru_nivcsw)


def timed_classes(controller_base: type, detector_base: type) -> tuple[type, type]:
    """Create behavior-preserving wrappers with paired wall/thread-CPU samples."""
    class TimedController(controller_base):  # type: ignore[misc, valid-type]
        def reset(self) -> None:
            self._study_timing = []
            self._study_guards = []
            self._study_cycle = 0
            reset = getattr(super(), "reset", None)
            if reset is not None:
                reset()

        def compute(self, *args: Any, **kwargs: Any) -> Any:
            if self._study_cycle % 250 == 0:
                self._study_guards.append(v2_resources.wait_for_capacity())
            before_context = _rusage_context()
            wall_start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            result = super().compute(*args, **kwargs)
            cpu_ns = time.thread_time_ns() - cpu_start
            wall_ns = time.perf_counter_ns() - wall_start
            after_context = _rusage_context()
            self._study_timing.append({
                "t": float(args[0]), "wall_ns": wall_ns, "cpu_ns": cpu_ns,
                "voluntary_context_switches": after_context[0] - before_context[0],
                "involuntary_context_switches": after_context[1] - before_context[1],
            })
            self._study_cycle += 1
            return result

    class TimedDetector(detector_base):  # type: ignore[misc, valid-type]
        def reset(self) -> None:
            self._study_timing = []
            reset = getattr(super(), "reset", None)
            if reset is not None:
                reset()

        def update(self, *args: Any, **kwargs: Any) -> Any:
            before_context = _rusage_context()
            wall_start = time.perf_counter_ns()
            cpu_start = time.thread_time_ns()
            result = super().update(*args, **kwargs)
            cpu_ns = time.thread_time_ns() - cpu_start
            wall_ns = time.perf_counter_ns() - wall_start
            after_context = _rusage_context()
            self._study_timing.append({
                "t": float(args[0]), "wall_ns": wall_ns, "cpu_ns": cpu_ns,
                "voluntary_context_switches": after_context[0] - before_context[0],
                "involuntary_context_switches": after_context[1] - before_context[1],
            })
            return result

    return TimedController, TimedDetector


def _distribution(values_ns: Iterable[int]) -> dict[str, float | int]:
    values = np.asarray(tuple(values_ns), dtype=float) / 1000.0
    if not len(values):
        return {"samples": 0, "median_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
    edges = np.array([0.,25.,50.,75.,100.,125.,150.,175.,200.,250.,300.,400.,500.,
                      750.,1000.,2000.,5000.,10000.,1000000.])
    counts, _ = np.histogram(values, edges)
    return {
        "samples": int(len(values)),
        "median_us": float(np.median(values)),
        "p99_us": float(np.percentile(values, 99)),
        "max_us": float(np.max(values)),
        "histogram_edges_us": edges.tolist(),
        "histogram_counts": counts.tolist(),
        "histogram_underflow": int(np.sum(values < edges[0])),
        "histogram_overflow": int(np.sum(values > edges[-1])),
    }


def timing_summary(controller: Any, detector: Any) -> dict[str, Any]:
    controller_samples = controller._study_timing
    detector_samples = detector._study_timing
    count = min(len(controller_samples), len(detector_samples))
    paired = []
    outliers = []
    for index in range(count):
        ctrl = controller_samples[index]
        det = detector_samples[index]
        combined = {
            "wall_ns": ctrl["wall_ns"] + det["wall_ns"],
            "cpu_ns": ctrl["cpu_ns"] + det["cpu_ns"],
            "voluntary_context_switches": (
                ctrl["voluntary_context_switches"] + det["voluntary_context_switches"]
            ),
            "involuntary_context_switches": (
                ctrl["involuntary_context_switches"] + det["involuntary_context_switches"]
            ),
        }
        paired.append(combined)
        if max(ctrl["wall_ns"], det["wall_ns"], combined["wall_ns"]) >= 1_000_000:
            outliers.append({
                "cycle": index,
                "simulated_t_s": ctrl["t"],
                "controller": {key: value for key, value in ctrl.items() if key != "t"},
                "detector": {key: value for key, value in det.items() if key != "t"},
                "combined": combined,
            })

    def component(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "wall": _distribution(sample["wall_ns"] for sample in samples),
            "thread_cpu": _distribution(sample["cpu_ns"] for sample in samples),
        }

    return {
        "controller": component(controller_samples[:count]),
        "detector": component(detector_samples[:count]),
        "combined": component(paired),
        "outlier_threshold_us": 1000.0,
        "outliers": outliers,
        "capacity_guard_count": len(controller._study_guards),
        "capacity_guard_max_foreign_heavy_count": max(
            (len(state["foreign_heavy_python_pids"]) for state in controller._study_guards), default=0),
        "capacity_guard_peak_load_average_1min": max(
            (state["load_average"][0] for state in controller._study_guards), default=None),
        "capacity_guard_min_mem_available_gib": min(
            (state["mem_available_gib"] for state in controller._study_guards), default=None
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_evidence(log: dict[str, Any], directory: Path, evidence_id: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{evidence_id}.npz"
    if destination.is_symlink():
        raise RuntimeError(f"refusing symlink evidence target: {destination}")
    arrays = {
        key: value for key, value in log.items()
        if isinstance(value, np.ndarray) and key != "qpos"
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{evidence_id}.", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination.name


def run_study_case(payload: dict[str, Any], evidence_dir: Path | None = None) -> dict[str, Any]:
    controller_module = importlib.import_module(payload["controller"])
    detector_module = importlib.import_module(payload["detector"])
    controller_type, detector_type = timed_classes(
        controller_module.Controller, detector_module.Detector
    )
    controller, detector = controller_type(), detector_type()
    host_before = v2_resources.wait_for_capacity()
    result, log = run_episode(
        controller,
        box_mass=payload["mass"],
        box_com=payload["com"],
        detector=detector,
        collision=payload["collision"],
        pick_offset=payload["pick_offset"],
    )
    host_after = v2_resources.snapshot()
    row = {key: _jsonable(value) for key, value in payload.items()}
    row.update(sweep.evaluate_control_metrics(result, log))
    row["g4"] = g4_window_metrics(log, payload["mass"])
    row["timing"] = timing_summary(controller, detector)
    row["runner_ctrl_median_us"] = float(result["ctrl_time_us"])
    row["torque_jitter_nm"] = float(result["torque_jitter"])
    row["tool_contact_peak_n"] = float(np.max(log["f_tool"]))
    contact = np.flatnonzero(np.asarray(log["f_tool"]) > 1.0)
    row["tool_contact_first_s"] = float(log["t"][contact[0]]) if len(contact) else None
    row["r_peak_nm"] = _jsonable(result["r_peak"])
    row.update({
        "detect_delay_ms": result.get("detect_delay_ms"),
        "alarm_before_collision": bool(result.get("alarm_before_collision", False)),
        "false_alarms": int(result["false_alarms"]),
        "tool_contact_delay_ms": result.get("tool_contact_delay_ms"),
        "host_before": host_before,
        "host_after": host_after,
        "_detect": np.asarray(log["detect"], dtype=bool),
    })
    if evidence_dir is not None:
        evidence_id = (
            f"{payload['stage']}-{payload['variant']}-{payload['architecture']}-"
            f"{payload['case_id']:04d}"
        )
        row["evidence_npz"] = _save_evidence(log, evidence_dir, evidence_id)
    return row


def classify_group(rows: list[dict[str, Any]]) -> None:
    """Apply sweep precedence within one variant/architecture reference set."""
    sweep.classify_rows(rows)


def module_hashes(variants: Iterable[str]) -> dict[str, str]:
    modules = set()
    for variant in variants:
        modules.update(VARIANT_MODULES[variant].values())
    hashes = {}
    for module in sorted(modules):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot locate source module: {module}")
        hashes[module] = sha256_file(Path(spec.origin))
    return hashes


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink output: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            compact = value.get("metadata", {}).get("stage") == "d4"
            json.dump(value, stream, ensure_ascii=False, indent=None if compact else 2,
                      separators=(",", ":") if compact else None)
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
    parser.add_argument(
        "--stage", required=True, choices=("control", "development", "d4", "holdout", "timing")
    )
    parser.add_argument("--variants", type=parse_variants, default=ALL_VARIANTS)
    parser.add_argument("--out", required=True, type=Path, help="atomic final results JSON")
    parser.add_argument("--evidence-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol()
    if args.stage == "holdout":
        # This gate intentionally precedes case construction and source importing.
        verify_freeze_sources(require_holdout_freeze())
    source_before = module_hashes(args.variants)
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    host_before = v2_resources.wait_for_capacity()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    completed = 0
    for variant in args.variants:
        for architecture in ARCHITECTURES:
            group_cases = build_stage_cases(args.stage, protocol, variant, architecture)
            group_rows = []
            for case in group_cases:
                group_rows.append(run_study_case(case, args.evidence_dir))
                completed += 1
                if completed % 10 == 0:
                    print(f"{completed} rows complete", flush=True)
            classify_group(group_rows)
            rows.extend(group_rows)
    source_after = module_hashes(args.variants)
    if source_after != source_before:
        raise RuntimeError("source modules changed while the study was running")
    metadata = {
        "stage": args.stage,
        "official": False,
        "variants": list(args.variants),
        "architectures": list(ARCHITECTURES),
        "max_owned_simulations": 1,
        "case_count": len(rows),
        "protocol_sha256": protocol_sha256,
        "source_sha256": source_before,
        "host_before": host_before,
        "host_after": v2_resources.snapshot(),
        "elapsed_seconds": time.perf_counter() - started,
        "timing_note": "wall/thread CPU are paired per cycle; all >=1 ms wall outliers retained",
    }
    _atomic_json(args.out, {"metadata": metadata, "results": rows})
    print(f"completed {len(rows)} rows; saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
