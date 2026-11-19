"""Run a self-contained R22 detector under original default filenames."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import my_detector
import my_servo
import my_torque
from experiments.r22_combined import candidate as augmented
from experiments.r22_combined import core
from experiments.r22_guarded import candidate as guarded
from experiments.r22_light import candidate as light
from experiments.r22_export import build
from sim.runner import run_episode
from tools.compare_baseline import official_cells
from tools.r21_replay import authenticate_kit
from tools.validate_submission import prepare_overlay


ROOT = Path(__file__).resolve().parents[2]
PHYSICAL = ("t", "q", "q_d", "tau", "e_mm", "f_table", "f_tool", "qpos", "f_ext")
CASES = (
    {"case_id": 174, "mass": 8., "com": (0., 0., -.02), "collision": None,
     "pick_offset": None},
    {"case_id": 319, "mass": 4., "com": (.03, .02, .05),
     "collision": (3.8, [10., 0., 0.]), "pick_offset": None},
    {"case_id": 330, "mass": 1.3, "com": (0., 0., -.02),
     "collision": (3.83, [-15., 0., 0.]), "pick_offset": None},
    {"case_id": 335, "mass": 8., "com": (0., 0., .015),
     "collision": (3.83, [10.606601717798211, 10.606601717798211, 0.]), "pick_offset": None},
    {"case_id": 339, "mass": 3.7, "com": (-.03, 0., -.02),
     "collision": (3.8, [25., 0., 0.]), "pick_offset": None},
    {"case_id": 349, "mass": 8., "com": (-.03, 0., .05),
     "collision": (3.8, [40., 0., 0.]), "pick_offset": None},
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location("collision_detector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_evaluator(overlay, mode, env):
    process = subprocess.run(
        [sys.executable, "-B", "evaluate.py", "--ctrl", mode + "_controller",
         "--detector", "collision_detector", "--out", mode + ".txt"],
        cwd=overlay, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(process.stderr)
    episodes = sum(line.strip().startswith("실행 중: 박스") for line in process.stdout.splitlines())
    if episodes != 29:
        raise ValueError("incomplete original evaluator run")
    control, collisions = official_cells((overlay / (mode + ".txt")).read_text())
    return {"episodes": episodes, "control": {str(k): v for k, v in control.items()},
            "collisions": [{"mass": m, "time": t, "cells": collisions[m, t]}
                           for m, t in sorted(collisions)]}


def cells(report):
    return {key: report[key] for key in ("control", "collisions")}


def count(report):
    return sum(value.isdecimal() for row in report["collisions"] for value in row["cells"][:4])


def expected(mode):
    return json.loads((ROOT / "results/r10/official_regression.json").read_bytes())[mode]


def episode(controller_type, detector_type, case):
    return run_episode(controller_type(), box_mass=case["mass"], box_com=tuple(case["com"]),
                       detector=detector_type(), collision=case.get("collision"),
                       pick_offset=case.get("pick_offset"))


def exact(left, right):
    return {key: bool(np.array_equal(left[key], right[key])) for key in PHYSICAL + ("r", "detect")}


def parity(module, variant):
    modular = {"core": core.Detector, "augmented": augmented.Detector,
               "guarded": guarded.Detector, "light": light.Detector}[variant]
    cases = list(CASES)
    cases.append({"case_id": "weak6", "mass": 0., "com": [0., 0., 0.],
                  "collision": (1.42, [0., 6., 0.]), "pick_offset": None})
    rows = []
    for case in cases:
        mm, ml = episode(my_torque.Controller, modular, case)
        sm, sl = episode(my_torque.Controller, module.Detector, case)
        same = exact(ml, sl)
        row = {"case_id": case["case_id"], "controller": "torque", "samples": len(sl["t"]),
               "exact": same, "metrics_exact": {
                   key: mm[key] == sm[key] for key in ("detect_delay_ms", "alarm_before_collision",
                                                       "false_alarm_times", "tool_contact_delay_ms")}}
        row["passed"] = row["samples"] == 6000 and all(same.values()) and all(row["metrics_exact"].values())
        rows.append(row)
    for case in ({"case_id": "servo_normal", "mass": 8., "com": [0., 0., 0.],
                  "collision": None, "pick_offset": None},
                 {"case_id": "servo_force", "mass": 8., "com": [0., 0., 0.],
                  "collision": (1.42, [0., 10., 0.]), "pick_offset": None}):
        bm, bl = episode(my_servo.Controller, my_detector.Detector, case)
        sm, sl = episode(my_servo.Controller, module.Detector, case)
        same = exact(bl, sl)
        row = {"case_id": case["case_id"], "controller": "servo", "samples": len(sl["t"]),
               "exact": same, "metrics_exact": {
                   key: bm[key] == sm[key] for key in ("detect_delay_ms", "alarm_before_collision",
                                                       "false_alarm_times", "tool_contact_delay_ms")}}
        row["passed"] = row["samples"] == 6000 and all(same.values()) and all(row["metrics_exact"].values())
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("core", "augmented", "guarded", "light"), required=True)
    args = parser.parse_args(argv)
    try:
        kit, out = args.kit.resolve(strict=True), args.out.resolve()
        if (args.kit.is_symlink() or not kit.is_dir() or os.path.lexists(args.out)
                or out.parent.is_relative_to(kit) or out.parent.is_relative_to(ROOT)):
            raise ValueError("regular original kit and new external output required")
        manifest_raw = (ROOT / "tools/kit_manifest.sha256").read_bytes()
        original = authenticate_kit(kit, manifest_raw)
        with tempfile.TemporaryDirectory(prefix="r22-export-", dir=out.parent) as tmp:
            overlay = Path(tmp)
            torque_pin, detector_pin = prepare_overlay(kit, overlay, "my", "torque")
            receipt = build.build(overlay / "standalone.py", args.variant)
            shutil.copyfile(overlay / "standalone.py", overlay / "collision_detector.py")
            if digest(overlay / "collision_detector.py") != receipt["standalone_sha256"]:
                raise ValueError("standalone detector copy differs")
            shutil.copyfile(ROOT / "my_servo.py", overlay / "servo_controller.py")
            if digest(overlay / "servo_controller.py") != expected("servo")["source_sha256"]:
                raise ValueError("R10 servo source differs")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="disable")
            env.pop("PYTHONPATH", None)
            torque = run_evaluator(overlay, "torque", env)
            servo = run_evaluator(overlay, "servo", env)
            if cells(servo) != expected("servo")["control_and_collisions"]:
                raise ValueError("generated servo fallback differs from R10")
            # Wall-clock compute is the only nondeterministic official control cell.
            expected_control = expected("torque")["control_and_collisions"]["control"]
            for mass, row in torque["control"].items():
                if row[:5] + row[6:] != expected_control[mass][:5] + expected_control[mass][6:]:
                    raise ValueError("generated torque control row differs from R10")
            if count(torque) != 24:
                raise ValueError("R22 standalone does not preserve 24/24 official torque cells")
            module = load(overlay / "collision_detector.py")
            paired = parity(module, args.variant)
            if not all(row["passed"] for row in paired):
                raise ValueError("modular/generated or servo fallback parity differs")
            if authenticate_kit(kit, manifest_raw) != original or digest(ROOT / "my_detector.py") != detector_pin:
                raise ValueError("protected original or R10 detector changed")
            payload = {"schema": "r22-release-audit.v1", "variant": args.variant,
                       "status": "research_option", "default_submission_promoted": False,
                       "build": receipt,
                       "kit": {"status": "ok", "files": len(original),
                               "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest()},
                       "evaluator_sha256": digest(overlay / "evaluate.py"),
                       "controllers": {"torque": torque_pin, "servo": expected("servo")["source_sha256"]},
                       "official": {"torque": torque, "servo": servo},
                       "recognized_collision_cells": {"torque": count(torque), "servo": count(servo)},
                       "parity": paired,
                       "gates": {"default_filename": True, "torque_official_24_of_24": True,
                                 "torque_controls_R10_exact_except_compute": True,
                                 "servo_R10_exact": True, "modular_generated_6000_exact": True,
                                 "protected_kit_unchanged": True}}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"variant": args.variant, "torque": count(torque),
                      "servo": count(servo), "parity": len(paired), "promoted": False}))


if __name__ == "__main__":
    main()
