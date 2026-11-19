"""Reproduce paired cold detector startup timings on a checked local kit.

This measures detector-module import plus the first detector reset in separate
fresh Python processes. Simulator/controller setup is reported separately and
excluded from the headline. Results describe only the recorded host and runs;
they are not a universal timing guarantee.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
FAST2 = ROOT / "experiments/r9_final/FAST2.py"
CURRENT = ROOT / "my_detector.py"
CONTROLLER = ROOT / "r1_servo.py"
PAYLOAD = (3.141, (-0.017, 0.013, 0.029))
VARIANTS = ("FAST2", "current")
BLAS_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measurement_order(pairs):
    if not isinstance(pairs, int) or isinstance(pairs, bool) or not 1 <= pairs <= 32:
        raise ValueError("pairs must be an integer from 1 through 32")
    return [variant for pair in range(pairs)
            for variant in (VARIANTS if pair % 2 == 0 else VARIANTS[::-1])]


def conservative_summary(rows):
    if not rows or any(row.get("variant") not in VARIANTS for row in rows):
        raise ValueError("rows must contain known variants")
    grouped = {variant: [float(row["headline_ms"]) for row in rows
                         if row["variant"] == variant] for variant in VARIANTS}
    if not grouped["FAST2"] or len(grouped["FAST2"]) != len(grouped["current"]):
        raise ValueError("paired FAST2/current measurements are required")
    extrema = {variant: {"minimum_ms": min(values), "maximum_ms": max(values)}
               for variant, values in grouped.items()}
    extrema["conservative_speedup_FAST2_over_current"] = (
        extrema["FAST2"]["minimum_ms"] / extrema["current"]["maximum_ms"])
    return extrema


def validate_inputs(kit, out, pairs, cpu):
    kit, out = Path(kit).resolve(), Path(out).resolve()
    measurement_order(pairs)
    if not kit.is_dir() or kit.is_symlink():
        raise ValueError("kit must be a real directory")
    for name in ("sim", "model", "evaluate.py"):
        if not (kit / name).exists():
            raise ValueError("kit is missing " + name)
    if out.exists() or not out.parent.is_dir() or out.is_relative_to(kit):
        raise ValueError("out must be a new file in an existing directory outside the kit")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpu is not None and (not isinstance(cpu, int) or cpu < 0
                            or affinity is None or cpu not in affinity):
        raise ValueError("cpu must be a member of the current process affinity")
    return kit, out, affinity


def _clock(name):
    info = time.get_clock_info(name)
    return {"implementation": info.implementation, "resolution_s": info.resolution,
            "monotonic": info.monotonic, "adjustable": info.adjustable}


def _probe(variant, expected_sha, cpu_text):
    """Fresh-process worker; keep every non-stdlib import below the precheck."""
    if "numpy" in sys.modules or "mujoco" in sys.modules:
        raise RuntimeError("numeric modules were imported before the detector probe")
    sys.path.insert(0, str(Path.cwd()))
    if cpu_text != "none":
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("requested CPU affinity is unsupported")
        os.sched_setaffinity(0, {int(cpu_text)})
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    worker_start = time.perf_counter_ns()
    import_start = time.perf_counter_ns()
    detector_module = importlib.import_module("collision_detector")
    import_ns = time.perf_counter_ns() - import_start
    loaded_sha = _sha256(Path(detector_module.__file__))
    if loaded_sha != expected_sha:
        raise RuntimeError("loaded detector source mismatch")

    setup_start = time.perf_counter_ns()
    controller_module = importlib.import_module("servo_controller")
    from sim import config as config
    from sim import trajectory as trajectory
    from sim.nominal import Nominal
    controller = controller_module.Controller()
    detector = detector_module.Detector()
    mass, com = PAYLOAD
    detector.nom, detector.nom_box = Nominal(), Nominal()
    detector.nom_box.set_payload(mass, com)
    detector.box_mass, detector.box_com = mass, com
    detector.ctrl = controller
    setup_ns = time.perf_counter_ns() - setup_start

    reset_start = time.perf_counter_ns()
    detector.reset()
    reset_ns = time.perf_counter_ns() - reset_start
    import numpy
    import mujoco
    result = {
        "variant": variant, "detector_sha256": loaded_sha,
        "controller_sha256": _sha256(Path(controller_module.__file__)),
        "preimport_numeric_modules": [], "affinity": affinity,
        "detector_import_ms": import_ns / 1e6, "setup_ms": setup_ns / 1e6,
        "first_reset_ms": reset_ns / 1e6,
        "headline_ms": (import_ns + reset_ns) / 1e6,
        "worker_total_ms": (time.perf_counter_ns() - worker_start) / 1e6,
        "runtime": {"python": sys.version, "implementation": sys.implementation.name,
                    "numpy": numpy.__version__, "mujoco": mujoco.__version__},
        "config": {"dt_s": config.DT, "trajectory_end_s": trajectory.T_END,
                   "grasp_s": trajectory.T_GRASP, "release_s": trajectory.T_RELEASE},
        "clocks": {"perf_counter": _clock("perf_counter"),
                   "thread_time": _clock("thread_time")},
        "config_sha256": _sha256(Path(config.__file__)),
        "trajectory_sha256": _sha256(Path(trajectory.__file__)),
    }
    print(json.dumps(result, allow_nan=False))


def _validated_measurement(record, variant, detector_sha, controller_sha):
    if (record.get("variant") != variant or record.get("detector_sha256") != detector_sha
            or record.get("controller_sha256") != controller_sha
            or record.get("preimport_numeric_modules") != []):
        raise ValueError("worker identity or pre-import state mismatch")
    for field in ("detector_import_ms", "setup_ms", "first_reset_ms",
                  "headline_ms", "worker_total_ms"):
        if not isinstance(record.get(field), (int, float)) or record[field] < 0:
            raise ValueError("invalid worker timing: " + field)
    expected = record["detector_import_ms"] + record["first_reset_ms"]
    if abs(record["headline_ms"] - expected) > 1e-6:
        raise ValueError("worker headline definition mismatch")
    return record


def _one_measurement(kit, variant, source, cpu, env):
    from tools.check_integrity import check_integrity
    from tools.validate_submission import prepare_overlay

    overlay_start = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="r10-startup-") as folder:
        overlay = Path(folder)
        controller_sha, _ = prepare_overlay(kit, overlay, "r1", "servo")
        shutil.copyfile(source, overlay / "collision_detector.py")
        detector_sha = _sha256(source)
        if _sha256(overlay / "collision_detector.py") != detector_sha:
            raise ValueError("disposable detector copy mismatch")
        integrity = check_integrity(overlay)
        overlay_setup_ms = (time.perf_counter_ns() - overlay_start) / 1e6
        command = [sys.executable, str(Path(__file__).resolve()), "--_probe", variant,
                   detector_sha, "none" if cpu is None else str(cpu)]
        process_start = time.perf_counter_ns()
        completed = subprocess.run(command, cwd=overlay, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process_wall_ms = (time.perf_counter_ns() - process_start) / 1e6
        if completed.returncode:
            raise RuntimeError("startup worker failed: " + completed.stderr.strip())
        try:
            record = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("startup worker emitted non-JSON output") from error
        _validated_measurement(record, variant, detector_sha, controller_sha)
        record.update(overlay_setup_ms=overlay_setup_ms, process_wall_ms=process_wall_ms,
                      kit_integrity=integrity)
        return record


def run(kit, pairs=4, cpu=None):
    sources = {"FAST2": FAST2, "current": CURRENT}
    for path in (*sources.values(), CONTROLLER):
        if not path.is_file():
            raise ValueError("missing repository source: " + str(path.relative_to(ROOT)))
    source_hashes = {variant: _sha256(path) for variant, path in sources.items()}
    controller_sha = _sha256(CONTROLLER)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(BLAS_ENV, PYTHONDONTWRITEBYTECODE="1")
    rows = []
    for sequence, variant in enumerate(measurement_order(pairs)):
        row = _one_measurement(kit, variant, sources[variant], cpu, env)
        if (row["detector_sha256"] != source_hashes[variant]
                or row["controller_sha256"] != controller_sha):
            raise ValueError("repository source changed during measurement")
        row.update(sequence=sequence, pair=sequence // 2,
                   order_in_pair=sequence % 2)
        rows.append(row)
    if (source_hashes != {variant: _sha256(path) for variant, path in sources.items()}
            or controller_sha != _sha256(CONTROLLER)):
        raise ValueError("repository source changed during measurement")
    common = rows[0]
    for row in rows[1:]:
        for field in ("controller_sha256", "config_sha256", "trajectory_sha256",
                      "runtime", "config", "clocks", "kit_integrity"):
            if row[field] != common[field]:
                raise ValueError("measurement provenance mismatch: " + field)
    return {
        "schema": "r10-startup.v1",
        "scope": "finite paired cold startup measurement; not a universal timing guarantee",
        "headline_definition": "detector module import wall + first detector reset wall; setup excluded",
        "payload": {"mass_kg": PAYLOAD[0], "com_m": PAYLOAD[1]},
        "pairs": pairs, "requested_cpu": cpu,
        "inherited_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": BLAS_ENV,
        "sources": {**source_hashes, "r1_servo": controller_sha,
                    "tool": _sha256(Path(__file__))},
        "source_identical": source_hashes["FAST2"] == source_hashes["current"],
        "kit_integrity": common["kit_integrity"],
        "runtime": common["runtime"], "config": common["config"],
        "clocks": common["clocks"],
        "config_sha256": common["config_sha256"],
        "trajectory_sha256": common["trajectory_sha256"],
        "rows": rows, "conservative": conservative_summary(rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new JSON result file")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        kit, out, _ = validate_inputs(args.kit, args.out, args.pairs, args.cpu)
        result = run(kit, args.pairs, args.cpu)
        with out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print("wrote paired startup measurement:", out)
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    if len(sys.argv) == 5 and sys.argv[1] == "--_probe":
        _probe(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        # A directly invoked script has tools/, not the checkout, on sys.path.
        # Keep this parent-only: the cold worker must import from its overlay.
        sys.path.insert(0, str(ROOT))
        raise SystemExit(main())
