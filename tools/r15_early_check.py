"""Authenticated portable reproduction of the stock R15 early-COM check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import r13_grasp_audit as assembly
from tools import r14_registered_check as r14


EARLY_SHA256 = "d07c7c49308680eb50824ef3790596c6f2789abb084fa313f33e7569944575a3"
DEPENDENCIES = (
    "tools/r15_early_check.py",
    "r1_torque.py", "my_detector.py",
    "experiments/r12_hybrid/controller.py",
    "experiments/r14_online/__init__.py", "experiments/r14_online/stream.py",
    "experiments/r14_airborne/registration.py", "experiments/r14_airborne/plan.json",
    "experiments/r14_stock_handoff/__init__.py",
    "experiments/r14_stock_handoff/controller.py",
    "experiments/r14_stock_handoff/detector.py",
    "experiments/r14_stock_handoff/geometry.py",
    "experiments/r14_stock_handoff/qualification.py",
    "experiments/r15_early_com/__init__.py",
    "experiments/r15_early_com/stream.py",
    "experiments/r15_early_com/sensing.py",
    "experiments/r15_early_com/controller.py",
    "experiments/r15_early_com/stock_check.py",
    "tools/r13_grasp_estimator.py", "tools/r11_placement.py",
    "tools/r14_paired_events.py", "tools/r5_metrics.py",
    "models/r12/counterfactual_coefficients.json",
)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def has_symlink_component(path):
    absolute = path.absolute()
    return any(part.is_symlink() for part in (absolute, *absolute.parents) if part.exists())


def validate_paths(kit, height, out):
    if kit.is_symlink() or not kit.is_dir() or not all((kit / p).exists() for p in ("sim", "model", "evaluate.py")):
        raise ValueError("regular ORIGINAL kit required")
    if height.is_symlink() or not height.is_file():
        raise ValueError("regular immutable height calibration required")
    if os.path.lexists(out) or has_symlink_component(out) or not out.parent.is_dir():
        raise ValueError("output must be a new non-symlink file")
    kit, parent = kit.resolve(), out.parent.resolve()
    if parent.is_relative_to(kit) or parent.is_relative_to(ROOT.resolve()):
        raise ValueError("output must be outside kit and input repository")
    return kit, height.resolve(), parent / out.name


def validate_early_calibration(blob):
    if digest(blob) != EARLY_SHA256:
        raise ValueError("early calibration does not match bundled normal-only stock model")
    document = json.loads(blob)
    if document.get("schema") != "r15-early-com-calibration.v1" or set(document) != {
            "schema", "dynamics", "baseline_xy", "normal_guard_mm", "loo_vectors_mm"}:
        raise ValueError("invalid early calibration schema")
    dynamics = document["dynamics"]
    if set(dynamics) != {"bias", "center", "range", "sigma"}:
        raise ValueError("invalid early dynamics fields")
    checks = ((dynamics["bias"], (2, 76, 6)), (dynamics["sigma"], (6,)),
              (dynamics["range"], (2,)), (document["baseline_xy"], (2, 2)),
              (document["loo_vectors_mm"], (12, 2)))
    for value, shape in checks:
        array = np.asarray(value, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("invalid early calibration array")
    if not (2.8 <= dynamics["range"][0] < dynamics["center"] < dynamics["range"][1] <= 8.1):
        raise ValueError("invalid early calibration mass provenance domain")
    guard = float(document["normal_guard_mm"])
    if not np.isfinite(guard) or not 0.0 <= guard <= 2.5:
        raise ValueError("invalid early calibration guard")
    return document


def snapshot_sources():
    paths = {relative: ROOT / relative for relative in DEPENDENCIES}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("regular curated public dependencies required")
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    return blobs, {name: digest(blob) for name, blob in blobs.items()}


def run_checked(command, *, cwd, env):
    completed = subprocess.run(command, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"stock check failed ({completed.returncode}):\n{completed.stderr}")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--height-calibration", required=True, type=Path)
    parser.add_argument("--early-calibration", type=Path,
                        default=ROOT / "results/r15/early_stock_calibration.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        kit, height_path, out = validate_paths(args.kit, args.height_calibration, args.out)
        early_path = args.early_calibration.resolve()
        if early_path.is_symlink() or not early_path.is_file():
            raise ValueError("regular immutable early calibration required")
        height_blob, early_blob = height_path.read_bytes(), early_path.read_bytes()
        plan = json.loads((ROOT / "experiments/r14_stock_handoff/plan.json").read_text())
        r14.validate_plan(plan)
        blobs, pins = snapshot_sources()
        r14_pins = {key: digest((ROOT / relative).read_bytes())
                    for key, relative in r14.SOURCE_PATHS.items()}
        height_document, height = r14.validate_calibration(height_blob, plan, r14_pins)
        early = validate_early_calibration(early_blob)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="disable")
    environment.pop("PYTHONPATH", None)
    try:
        with tempfile.TemporaryDirectory(prefix="r15-early-", dir=out.parent) as folder:
            overlay = Path(folder)
            from tools.validate_submission import prepare_overlay
            prepare_overlay(kit, overlay, "my", "torque")
            for relative, blob in blobs.items():
                destination = overlay / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                assembly.install_snapshot(blob, pins[relative], destination)
            (overlay / "height.json").write_text(json.dumps(height, allow_nan=False))
            (overlay / "early.json").write_text(json.dumps(early, allow_nan=False))
            completed = run_checked(
                [sys.executable, "-B", "-m", "experiments.r15_early_com.stock_check",
                 "--height", "height.json", "--early", "early.json", "--out", "result.json"],
                cwd=overlay, env=environment)
            result = json.loads((overlay / "result.json").read_text())
            if result.get("schema") != "r15-portable-early-stock.v1":
                raise ValueError("invalid neutral runner output")
        for relative, pin in pins.items():
            if digest((ROOT / relative).read_bytes()) != pin:
                raise ValueError("snapshotted dependency mutated during execution")
        if digest(height_path.read_bytes()) != digest(height_blob) or digest(early_path.read_bytes()) != digest(early_blob):
            raise ValueError("calibration mutated during execution")
        document = {
            "schema": "r15-portable-early-check.v1",
            "scope": "stock ORIGINAL-kit reproduction; exposed/development cases only",
            "calibration_provenance": {
                "height_sha256": digest(height_blob),
                "height_source_pins": height_document["pins"],
                "early_sha256": digest(early_blob),
                "early_source": "bundled byte-pinned model derived from 12 stock public normal traces",
                "early_mass_domain_kg": early["dynamics"]["range"],
            },
            "dependencies": list(DEPENDENCIES), "pins": pins,
            "runner_stdout": completed.stdout.strip(), "result": result,
            "summary": result["summary"],
            "limitations": ["reproduction, not new blind evidence", "no external pulses",
                            "no threshold or detector-delay changes",
                            "qualified=false is a valid exit-zero result"],
        }
        with out.open("x") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(document["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
