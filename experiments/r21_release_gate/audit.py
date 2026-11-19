"""Run an optional single-file detector under original default filenames."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.r21_release_gate import build
from tools.check_integrity import check_integrity
from tools.compare_baseline import official_cells
from tools.r13_grasp_audit import digest
from tools.validate_submission import prepare_overlay


YAW_COLLISIONS = [
    {"mass": 0.0, "time": 1.42, "cells": ["44", "37", "26", "20", "0회", "10 N"]},
    {"mass": 0.0, "time": 3.0, "cells": ["57", "38", "25", "19", "0회", "10 N"]},
    {"mass": 0.0, "time": 3.86, "cells": ["92", "78", "30", "19", "0회", "10 N"]},
    {"mass": 8.0, "time": 1.42, "cells": ["43", "37", "27", "21", "0회", "10 N"]},
    {"mass": 8.0, "time": 3.0, "cells": ["56", "41", "25", "18", "1회", "10 N"]},
    {"mass": 8.0, "time": 3.86, "cells": ["91", "86", "38", "22", "0회", "10 N"]},
]

R21_COLLISIONS = [
    {"mass": 0.0, "time": 1.42, "cells": ["47", "35", "22", "16", "0회", "10 N"]},
    {"mass": 0.0, "time": 3.0, "cells": ["57", "42", "27", "19", "0회", "10 N"]},
    {"mass": 0.0, "time": 3.86, "cells": ["놓침", "78", "30", "19", "0회", "15 N"]},
    {"mass": 8.0, "time": 1.42, "cells": ["49", "33", "23", "17", "0회", "10 N"]},
    {"mass": 8.0, "time": 3.0, "cells": ["56", "43", "26", "18", "1회", "10 N"]},
    {"mass": 8.0, "time": 3.86, "cells": ["놓침", "91", "42", "22", "0회", "15 N"]},
]


def run_evaluator(overlay, mode, env):
    process = subprocess.run(
        [sys.executable, "-B", "evaluate.py", "--ctrl", mode + "_controller",
         "--detector", "collision_detector", "--out", mode + ".txt"],
        cwd=overlay, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(process.stderr)
    episodes = sum(line.strip().startswith("실행 중: 박스")
                   for line in process.stdout.splitlines())
    if episodes != 29:
        raise ValueError(f"incomplete {mode} official run: {episodes}")
    control, collisions = official_cells((overlay / (mode + ".txt")).read_text())
    return {
        "episodes": episodes,
        "control": {str(k): v for k, v in control.items()},
        "collisions": [{"mass": m, "time": t, "cells": collisions[m, t]}
                       for m, t in sorted(collisions)],
    }


def cells(report):
    return {key: report[key] for key in ("control", "collisions")}


def collision_count(report):
    return sum(value.isdecimal() for row in report["collisions"] for value in row["cells"][:4])


def control_without_compute(report):
    """Ignore only the non-deterministic wall-clock microsecond column."""
    return {mass: row[:5] + row[6:] for mass, row in report["control"].items()}


def expected(mode):
    path = ROOT / "results/r10/official_regression.json"
    return json.loads(path.read_bytes())[mode]["control_and_collisions"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("r20", "r21", "yaw"), default="r20")
    args = parser.parse_args(argv)
    try:
        kit, out = args.kit.resolve(strict=True), args.out.resolve()
        if (args.kit.is_symlink() or not kit.is_dir() or out.exists()
                or out.parent.resolve().is_relative_to(kit)
                or out.parent.resolve().is_relative_to(ROOT)):
            raise ValueError("regular original kit and new external output required")
        with tempfile.TemporaryDirectory(prefix="r21-release-", dir=out.parent) as tmp:
            overlay = Path(tmp)
            controller_pin, detector_pin = prepare_overlay(kit, overlay, "my", "torque")
            build_receipt = build.build(overlay / "standalone.py", variant=args.variant)
            shutil.copyfile(overlay / "standalone.py", overlay / "collision_detector.py")
            if digest(overlay / "collision_detector.py") != build_receipt["standalone_sha256"]:
                raise ValueError("standalone detector copy differs")
            shutil.copyfile(ROOT / "my_servo.py", overlay / "servo_controller.py")
            if digest(overlay / "servo_controller.py") != expected_source("servo"):
                raise ValueError("R10 servo source differs")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="disable")
            env.pop("PYTHONPATH", None)
            torque, servo = run_evaluator(overlay, "torque", env), run_evaluator(overlay, "servo", env)
            if args.variant == "r20":
                research = json.loads((ROOT / "results/r20/horizontal.json").read_bytes())
                torque_exact = cells(torque) == research["candidate"]["cells"]
            else:
                expected_collisions = (R21_COLLISIONS if args.variant == "r21"
                                       else YAW_COLLISIONS)
                torque_exact = (control_without_compute(torque)
                                == control_without_compute(expected("torque"))
                                and torque["collisions"] == expected_collisions)
            if not torque_exact:
                raise ValueError(f"standalone torque table differs from frozen {args.variant}")
            if cells(servo) != expected("servo"):
                raise ValueError("servo fallback differs from R10")
            integrity = check_integrity(overlay)
            if (digest(ROOT / "my_detector.py") != build.R10_SHA256
                    or digest(ROOT / "experiments/r20_horizontal/candidate_detector.py") != build.BRANCH_SHA256):
                raise ValueError("authored input changed during official run")
            payload = {
                "schema": f"r21-{args.variant}-release-audit.v1",
                "variant": args.variant,
                "status": "installable_research_option",
                "default_submission_promoted": False,
                "build": build_receipt,
                "kit": integrity,
                "evaluator_sha256": digest(overlay / "evaluate.py"),
                "controllers": {"torque": controller_pin, "servo": expected_source("servo")},
                "official": {"torque": torque, "servo": servo},
                "recognized_collision_cells": {
                    "torque": collision_count(torque), "servo": collision_count(servo)},
                "gates": {
                    "standalone_default_filename": True,
                    f"frozen_{args.variant}_torque_table_exact": True,
                    "servo_R10_fallback_exact": True,
                    "protected_kit_unchanged": True,
                },
            }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps({"torque": collision_count(torque), "servo": collision_count(servo),
                      "promoted": False}))


def expected_source(mode):
    return json.loads((ROOT / "results/r10/official_regression.json").read_bytes())[mode]["source_sha256"]


if __name__ == "__main__":
    main()
