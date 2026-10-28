#!/usr/bin/env python3
"""Repeatable IC-PBL experiment sweep using :func:`sim.runner.run_episode`.

The simulator is not patched.  Thin subclasses only time the public controller
and detector calls and return their results unchanged.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_integrity import check_integrity as _check_kit_integrity  # noqa: E402

IMPORT_INTEGRITY = _check_kit_integrity(root=ROOT)

from sim.runner import run_episode  # noqa: E402
from sim import trajectory as T  # noqa: E402


PICK_WINDOW = (1.35, 1.6)
PLACE_WINDOW = (3.7, 4.1)
BASELINE_MASSES = (0.0, 4.0, 8.0)
BASELINE_COLLISION_MASSES = (0.0, 8.0)
BASELINE_COLLISION_TIMES = (1.42, 3.0, 3.86)
BASELINE_FORCES = (10.0, 15.0, 40.0, 100.0)
DEFAULT_COM = (0.02, -0.01, 0.03)
MODULE_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected finite comma-separated numbers")
    return values


def parse_vector(text: str) -> tuple[float, float, float]:
    values = parse_float_list(text)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected exactly x,y,z")
    return values  # type: ignore[return-value]


def parse_collision(text: str) -> tuple[float, tuple[float, float, float]]:
    values = parse_float_list(text)
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected time,fx,fy,fz")
    collision_time, *force = values
    if collision_time < 0 or collision_time >= T.T_END or not any(force):
        raise argparse.ArgumentTypeError(
            f"collision time must satisfy 0 <= time < {T.T_END:g} s and force must be nonzero"
        )
    return collision_time, tuple(force)  # type: ignore[return-value]


def module_name(text: str) -> str:
    if not MODULE_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("module must be a dotted Python name, not a path")
    return text


def experiment_name(text: str) -> str:
    if not NAME_RE.fullmatch(text) or text in {".", ".."}:
        raise argparse.ArgumentTypeError("name may contain only letters, digits, '.', '_' and '-'")
    return text


def available_memory_bytes() -> int:
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine MemAvailable from /proc/meminfo")


def timing_stats(seconds: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(tuple(seconds), dtype=float) * 1e6
    if not len(values):
        return {"samples": 0, "median_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
    return {
        "samples": int(len(values)),
        "median_us": float(np.median(values)),
        "p99_us": float(np.percentile(values, 99)),
        "max_us": float(np.max(values)),
    }


def timed_classes(controller_base: type, detector_base: type) -> tuple[type, type]:
    """Return behavior-preserving timing subclasses (constructed inside workers)."""

    class TimedController(controller_base):  # type: ignore[valid-type, misc]
        def reset(self) -> None:
            self._sweep_compute_seconds = []
            reset = getattr(super(), "reset", None)
            if reset is not None:
                reset()

        def compute(self, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = super().compute(*args, **kwargs)
            self._sweep_compute_seconds.append(time.perf_counter() - started)
            return result

    class TimedDetector(detector_base):  # type: ignore[valid-type, misc]
        def reset(self) -> None:
            self._sweep_update_seconds = []
            reset = getattr(super(), "reset", None)
            if reset is not None:
                reset()

        def update(self, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = super().update(*args, **kwargs)
            self._sweep_update_seconds.append(time.perf_counter() - started)
            return result

    return TimedController, TimedDetector


def evaluate_control_metrics(result: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    """Mirror evaluate.py's [1] windows and definitions exactly."""
    t = np.asarray(log["t"])
    false_alarm_times = result["false_alarm_times"]
    pick_false = sum(PICK_WINDOW[0] <= value < PICK_WINDOW[1] for value in false_alarm_times)
    place_false = sum(PLACE_WINDOW[0] <= value < PLACE_WINDOW[1] for value in false_alarm_times)
    torque_step = np.r_[0.0, np.abs(np.diff(log["tau"], axis=0)).max(axis=1)]
    pick = (t >= PICK_WINDOW[0]) & (t < PICK_WINDOW[1])
    place = (t >= PLACE_WINDOW[0]) & (t < PLACE_WINDOW[1])
    return {
        "rms_mm": float(result["rms_mm"]),
        "max_mm": float(result["max_mm"]),
        "saturation_pct": float(result["saturation_pct"]),
        "diverged": bool(result["diverged"]),
        "torque_step_pick_nm": float(torque_step[pick].max()),
        "torque_step_place_nm": float(torque_step[place].max()),
        "table_force_place_max_n": float(np.asarray(log["f_table"])[place].max()),
        "false_alarms_pick": int(pick_false),
        "false_alarms_place": int(place_false),
        "false_alarms_other": int(len(false_alarm_times) - pick_false - place_false),
    }


def classify_collision(
    result: dict[str, Any], reference_detect: np.ndarray, collision_time: float
) -> dict[str, Any]:
    """Mirror evaluate.py's alarm-before/miss/reference ±20 ms precedence."""
    delay = result["detect_delay_ms"]
    detection_index = None if delay is None else int(round((collision_time + delay / 1e3) / 1e-3))
    if result["alarm_before_collision"]:
        outcome = "alarm_before_collision"
        label = "경보 중"
    elif delay is None:
        outcome = "missed"
        label = "놓침"
    elif reference_detect[max(0, detection_index - 20):detection_index + 21].any():
        outcome = "indistinguishable"
        label = "구분 불가"
    else:
        outcome = "detected"
        label = f"{delay:.0f} ms"
    return {
        "collision_outcome": outcome,
        "collision_label": label,
        "detect_delay_ms": None if delay is None else float(delay),
        "alarm_before_collision": bool(result["alarm_before_collision"]),
        "false_alarms": int(result["false_alarms"]),
        "tool_contact_delay_ms": result.get("tool_contact_delay_ms"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_case(payload: dict[str, Any]) -> dict[str, Any]:
    controller_module = importlib.import_module(payload["controller"])
    detector_module = importlib.import_module(payload["detector"])
    timed_controller, timed_detector = timed_classes(
        controller_module.Controller, detector_module.Detector
    )
    controller, detector = timed_controller(), timed_detector()
    collision = payload["collision"]
    result, log = run_episode(
        controller,
        box_mass=payload["mass"],
        box_com=payload["com"],
        detector=detector,
        collision=None if collision is None else (collision[0], collision[1]),
        pick_offset=payload["pick_offset"],
    )
    controller_times = controller._sweep_compute_seconds
    detector_times = detector._sweep_update_seconds
    paired_count = min(len(controller_times), len(detector_times))
    paired = np.asarray(controller_times[:paired_count]) + np.asarray(
        detector_times[:paired_count]
    )
    row = {
        key: _jsonable(value) for key, value in payload.items()
    }
    row.update(evaluate_control_metrics(result, log))
    row["timing"] = {
        "controller": timing_stats(controller_times),
        "detector": timing_stats(detector_times),
        "combined": timing_stats(paired),
    }
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
    })
    row["_detect"] = np.asarray(log["detect"], dtype=bool)
    return row


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    common = {
        "controller": args.controller,
        "detector": args.detector,
        "pick_offset": args.pick_offset,
    }
    cases: list[dict[str, Any]] = []
    sequence = 0

    def add(mass: float, com: tuple[float, float, float], collision: Any, kind: str) -> None:
        nonlocal sequence
        cases.append({
            **common,
            "case_id": sequence,
            "kind": kind,
            "mass": mass,
            "com": com,
            "collision": collision,
        })
        sequence += 1

    if args.baseline:
        for mass in BASELINE_MASSES:
            add(mass, DEFAULT_COM, None, "reference")
        for mass in BASELINE_COLLISION_MASSES:
            for collision_time in BASELINE_COLLISION_TIMES:
                for force in BASELINE_FORCES:
                    add(mass, DEFAULT_COM, (collision_time, (0.0, force, 0.0)), "collision")
        return cases

    for mass in args.masses:
        if mass < 0:
            raise ValueError("box masses must be >= 0")
        for com in args.com:
            add(mass, com, None, "reference")
            for collision in args.collision:
                add(mass, com, collision, "collision")
    return cases


def _reference_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["mass"], tuple(row["com"]), tuple(row["pick_offset"]) if row["pick_offset"] else None)


def classify_rows(rows: list[dict[str, Any]]) -> None:
    references = {_reference_key(row): row["_detect"] for row in rows if row["collision"] is None}
    for row in rows:
        if row["collision"] is not None:
            row.update(classify_collision(row, references[_reference_key(row)], row["collision"][0]))
        row.pop("_detect", None)


def safe_output_dir(name: str, root: Path = ROOT) -> Path:
    """Return the fixed output directory, rejecting symlink-based escapes."""
    root = root.resolve()
    results_root = root / "results"
    sweeps_root = results_root / "sweeps"
    output_dir = sweeps_root / name
    for path in (results_root, sweeps_root, output_dir):
        if path.is_symlink():
            raise ValueError(f"refusing symlink in output path: {path}")
    resolved_sweeps = sweeps_root.resolve()
    if not resolved_sweeps.is_relative_to(root) or output_dir.parent.resolve() != resolved_sweeps:
        raise ValueError("output path must remain under results/sweeps")
    return output_dir


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    timing = row["timing"]
    flat = {key: value for key, value in row.items() if key != "timing"}
    for component, stats in timing.items():
        for statistic, value in stats.items():
            flat[f"{component}_{statistic}"] = value
    for key, value in tuple(flat.items()):
        if isinstance(value, (list, tuple, dict)):
            flat[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        elif value is None:
            flat[key] = ""
    return flat


def _validate_output_targets(output_dir: Path, root: Path) -> dict[str, Path]:
    expected = safe_output_dir(output_dir.name, root)
    if output_dir.absolute() != expected.absolute():
        raise ValueError(f"output directory is not the fixed results/sweeps path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: output_dir / name for name in ("results.json", "results.csv", "SUMMARY.md")}
    for target in targets.values():
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"refusing symlink/non-regular output target: {target}")
    return targets


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            temporary_name.unlink(missing_ok=True)


def write_outputs(
    output_dir: Path, metadata: dict[str, Any], rows: list[dict[str, Any]], root: Path = ROOT
) -> None:
    targets = _validate_output_targets(output_dir, root)
    json_content = json.dumps(
        {"metadata": metadata, "results": rows}, ensure_ascii=False, indent=2
    ) + "\n"
    flat_rows = [flatten_row(row) for row in rows]
    fieldnames = list(dict.fromkeys(key for row in flat_rows for key in row))
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat_rows)
    references = [row for row in rows if row["collision"] is None]
    collisions = [row for row in rows if row["collision"] is not None]
    outcome_counts = {
        outcome: sum(row.get("collision_outcome") == outcome for row in collisions)
        for outcome in ("detected", "missed", "alarm_before_collision", "indistinguishable")
    }
    lines = [
        f"# Sweep: {metadata['name']}",
        "",
        f"- Controller / detector: `{metadata['controller']}` / `{metadata['detector']}`",
        f"- Workers: {metadata['workers']}",
        f"- Cases: {len(rows)} ({len(references)} no-collision references, {len(collisions)} collisions)",
        f"- Collision outcomes: {outcome_counts}",
        "",
        "## No-collision metrics",
        "",
        "| mass kg | COM m | RMS / max mm | pick / place step N m | place table N | false pick/place/other | saturation % |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in references:
        lines.append(
            f"| {row['mass']:g} | {row['com']} | {row['rms_mm']:.2f} / {row['max_mm']:.2f} | "
            f"{row['torque_step_pick_nm']:.1f} / {row['torque_step_place_nm']:.1f} | "
            f"{row['table_force_place_max_n']:.0f} | {row['false_alarms_pick']} / "
            f"{row['false_alarms_place']} / {row['false_alarms_other']} | {row['saturation_pct']:.1f} |"
        )
    lines += ["", "## Timing", "", "Timing distributions for every case are in JSON/CSV (median, p99, max in microseconds)."]
    if rows:
        lines.append("")
        for component in ("controller", "detector", "combined"):
            lines.append(
                f"- Worst case {component}: p99 "
                f"{max(row['timing'][component]['p99_us'] for row in rows):.2f} us, max "
                f"{max(row['timing'][component]['max_us'] for row in rows):.2f} us"
            )
    if collisions:
        lines += [
            "",
            "## Collision classification",
            "",
            "| mass kg | COM m | collision (s, N xyz) | result | false alarms |",
            "|---:|---|---|---|---:|",
        ]
        for row in collisions:
            lines.append(
                f"| {row['mass']:g} | {row['com']} | {row['collision']} | "
                f"{row['collision_label']} | {row['false_alarms']} |"
            )
    _atomic_write_text(targets["results.json"], json_content)
    _atomic_write_text(targets["results.csv"], csv_stream.getvalue())
    _atomic_write_text(targets["SUMMARY.md"], "\n".join(lines) + "\n")


def check_integrity() -> dict[str, Any]:
    return _check_kit_integrity(root=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, type=experiment_name, help="results/sweeps/<name>")
    parser.add_argument("--controller", "--ctrl", default="servo_controller", type=module_name)
    parser.add_argument("--detector", default="collision_detector", type=module_name)
    parser.add_argument("--masses", type=parse_float_list, default=None)
    parser.add_argument("--com", action="append", type=parse_vector, default=None, metavar="X,Y,Z")
    parser.add_argument(
        "--collision", action="append", type=parse_collision, default=[], metavar="TIME,FX,FY,FZ"
    )
    parser.add_argument("--pick-offset", type=parse_vector, default=None, metavar="X,Y,Z")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5), metavar="{1,2,3,4}")
    parser.add_argument(
        "--baseline", action="store_true", help="run evaluate.py's exact default conditions"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.baseline and (
        args.collision
        or args.com is not None
        or args.masses is not None
        or args.pick_offset is not None
    ):
        raise SystemExit(
            "--baseline cannot be combined with --masses, --com, --collision, or --pick-offset"
        )
    args.masses = args.masses or BASELINE_MASSES
    args.com = args.com or [DEFAULT_COM]
    memory = available_memory_bytes()
    minimum = 6 * 1024**3
    if memory < minimum:
        raise SystemExit(f"refusing sweep: MemAvailable {memory / 1024**3:.2f} GiB < 6 GiB")
    integrity = check_integrity()
    cases = build_cases(args)
    try:
        output_dir = safe_output_dir(args.name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    started = time.time()
    if args.workers == 1:
        rows = [run_case(case) for case in cases]
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(run_case, cases)
    classify_rows(rows)
    metadata = {
        "name": args.name,
        "controller": args.controller,
        "detector": args.detector,
        "workers": args.workers,
        "baseline": args.baseline,
        "case_count": len(rows),
        "elapsed_seconds": time.time() - started,
        "mem_available_gib_before": memory / 1024**3,
        "integrity": integrity,
    }
    write_outputs(output_dir, metadata, rows)
    print(f"completed {len(rows)} cases in {metadata['elapsed_seconds']:.1f} s")
    print(f"saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
