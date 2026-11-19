"""Authenticated ORIGINAL-kit reproduction of four exposed R15 closed-loop cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import r13_grasp_audit as assembly
from tools import r14_registered_check as r14

DEPENDENCIES = (
    "r1_torque.py", "my_detector.py",
    "experiments/r12_hybrid/controller.py",
    "experiments/r14_online/stream.py",
    "experiments/r14_airborne/registration.py",
    "tools/r13_grasp_estimator.py", "tools/r11_placement.py",
    "tools/r14_paired_events.py", "tools/r5_metrics.py",
    "models/r12/counterfactual_coefficients.json",
    "experiments/r14_stock_handoff/controller.py",
    "experiments/r14_stock_handoff/detector.py",
    "experiments/r14_stock_handoff/qualification.py",
    "experiments/r14_stock_handoff/plan.json",
    "experiments/r15_com_model/sensing.py",
    "experiments/r15_com_model/controller.py",
    "experiments/r15_com_model/handoff.py",
    "experiments/r15_com_model/plan.json",
)


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def summarize(rows):
    if len(rows) != 4 or any(not str(row.get("id", "")).strip() for row in rows):
        raise ValueError("four nonempty exposed case identifiers required")
    r15 = sum(bool(row.get("gates", {}).get("g4")) for row in rows)
    r14_count = sum(bool(row.get("r14_g4", {}).get("all_four_g4")) for row in rows)
    all_gates = all(row.get("passed") is True for row in rows)
    return {
        "cases": 4,
        "r15_g4_cases": r15,
        "r14_g4_cases": r14_count,
        "all_gates": all_gates,
        "qualified": bool(all_gates and r15 > r14_count),
        "failed": [row["id"] for row in rows if row.get("passed") is not True],
    }


def has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(part.is_symlink() for part in (absolute, *absolute.parents) if part.exists())


def validate_paths(kit: Path, calibration: Path, out: Path):
    if kit.is_symlink() or not kit.is_dir() or not all((kit / p).exists() for p in ("sim", "model", "evaluate.py")):
        raise ValueError("regular ORIGINAL kit required")
    if calibration.is_symlink() or not calibration.is_file():
        raise ValueError("regular immutable height calibration required")
    if os.path.lexists(out) or has_symlink_component(out) or not out.parent.is_dir():
        raise ValueError("output must be a new non-symlink file")
    kit_real, root_real, parent = kit.resolve(), ROOT.resolve(), out.parent.resolve()
    if parent.is_relative_to(kit_real) or parent.is_relative_to(root_real):
        raise ValueError("output must be outside kit and input repository")
    return kit_real, calibration.resolve(), parent / out.name


def snapshot_sources():
    paths = {relative: ROOT / relative for relative in DEPENDENCIES}
    paths["tools/r15_payload_audit.py"] = ROOT / "tools/r15_payload_audit.py"
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("regular curated public dependencies required")
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    return blobs, {name: digest(blob) for name, blob in blobs.items()}


def decode_height_calibration(blob: bytes):
    plan_blob = (ROOT / "experiments/r14_stock_handoff/plan.json").read_bytes()
    plan = json.loads(plan_blob)
    r14.validate_plan(plan)
    pins = {
        key: digest((ROOT / relative).read_bytes())
        for key, relative in r14.SOURCE_PATHS.items()
    }
    _, model = r14.validate_calibration(blob, plan, pins)
    return model


def verify_unchanged(path, expected):
    if digest(Path(path).read_bytes()) != expected:
        raise ValueError("snapshotted input mutated during execution")


def run_checked(command, *, cwd, env, label):
    completed = subprocess.run(command, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"{label} failed ({completed.returncode}):\n{completed.stderr}")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--height-calibration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        kit, calibration_path, out = validate_paths(args.kit, args.height_calibration, args.out)
        calibration_blob = calibration_path.read_bytes()
        height_model = decode_height_calibration(calibration_blob)
        blobs, pins = snapshot_sources()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.pop("PYTHONPATH", None)
    try:
        with tempfile.TemporaryDirectory(prefix="r15-handoff-", dir=out.parent) as folder:
            temporary = Path(folder)
            audit = temporary / "audit"
            run_checked(
                [sys.executable, "-B", str(ROOT / "tools/r15_payload_audit.py"),
                 "--kit", str(kit), "--out", str(audit)],
                cwd=temporary, env=environment, label="payload audit",
            )
            verify_unchanged(ROOT / "tools/r15_payload_audit.py", pins["tools/r15_payload_audit.py"])
            verify_unchanged(calibration_path, digest(calibration_blob))
            calibration_audit = json.loads((audit / "calibration.json").read_text())
            sensing_summary = json.loads((audit / "summary.json").read_text())
            if calibration_audit.get("schema") != "r15-payload-calibration-audit.v1" or not sensing_summary.get("metrics", {}).get("qualified"):
                raise ValueError("derived stock R15 sensing calibration is malformed or unqualified")
            xy_model = calibration_audit.get("model")
            if not isinstance(xy_model, dict) or xy_model.get("schema") != "r15-xy-calibration.v1":
                raise ValueError("derived R15 model schema missing")

            overlay = temporary / "overlay"
            from tools.validate_submission import prepare_overlay
            prepare_overlay(kit, overlay, "my", "torque")
            for relative, blob in blobs.items():
                if relative == "tools/r15_payload_audit.py":
                    continue
                destination = overlay / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                assembly.install_snapshot(blob, pins[relative], destination)
            (overlay / "height.json").write_text(json.dumps(height_model, allow_nan=False))
            (overlay / "xy.json").write_text(json.dumps(xy_model, allow_nan=False))
            worker = (
                'import json;from pathlib import Path;'
                'from experiments.r15_com_model.handoff import compare;'
                'p=json.loads(Path("experiments/r15_com_model/plan.json").read_text());'
                'h=json.loads(Path("height.json").read_text());x=json.loads(Path("xy.json").read_text());'
                'r=json.loads(Path("experiments/r14_stock_handoff/plan.json").read_text());'
                'rows=[compare(next(c for c in r["normal_cases"] if c["id"]==i),h,x) '
                'for i in p["exposed_development_ids"]];print(json.dumps(rows,allow_nan=False))'
            )
            completed = run_checked([sys.executable, "-B", "-c", worker], cwd=overlay,
                                    env=environment, label="closed-loop overlay")
            rows = json.loads(completed.stdout)

        result = {
            "schema": "r15-handoff-check.v2",
            "scope": "ORIGINAL stock kit, derived R15 calibration, exposed development only",
            "dependencies": list(DEPENDENCIES), "pins": pins,
            "height_calibration_sha256": digest(calibration_blob),
            "payload_sensing_metrics": sensing_summary["metrics"],
            "rows": rows, "summary": summarize(rows),
            "limitations": ["development reproduction only", "host timings are descriptive",
                            "qualified=false is a valid reproduced scientific result"],
        }
        with out.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
