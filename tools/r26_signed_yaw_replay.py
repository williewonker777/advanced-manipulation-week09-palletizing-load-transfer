"""Authenticate and replay the disclosed R25 long signed-yaw diagnostic only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r21_replay as shared


CLOSURE = "experiments/r26_yaw_portable/closure.json"
REGISTRATION = "experiments/r26_yaw_portable/registration.json"
EVALUATOR = "experiments.r26_yaw_portable.evaluator"


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts
                and "\\" not in name)


def closure():
    path = ROOT / CLOSURE
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular R26 signed-yaw closure required")
    data = json.loads(path.read_bytes())
    files, overlay, public = (data.get(key) for key in
                              ("files", "overlay_files", "public_files"))
    if (data.get("schema") != "r26-known-signed-yaw-closure.v1"
            or not isinstance(files, dict) or not files
            or not isinstance(overlay, list) or not isinstance(public, list)
            or set(files) != set(overlay) | set(public)
            or REGISTRATION not in overlay
            or "experiments/r26_long/candidate.py" not in overlay
            or "experiments/r25_holdout/runner.py" not in overlay):
        raise ValueError("complete R26 signed-yaw closure differs")
    for name, expected in files.items():
        source = ROOT / name
        if (not relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py" or len(expected) != 64
                or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("R26 signed-yaw source pin differs: " + name)
    return data


def source_blobs(data):
    blobs = {name: (ROOT / name).read_bytes() for name in data["overlay_files"]}
    pins = {name: digest(blob) for name, blob in blobs.items()}
    if any(pins[name] != data["files"][name] for name in pins):
        raise ValueError("R26 signed-yaw source changed after closure validation")
    return blobs, pins


def runtime_gate(kit):
    try:
        import mujoco
    except ImportError as exc:
        raise ValueError("MuJoCo 3.17.0 is required in this interpreter") from exc
    if mujoco.__version__ != "3.17.0":
        raise ValueError("course MuJoCo 3.17.0 required")
    requirements = kit / "requirements.txt"
    if requirements.is_symlink() or not requirements.is_file():
        raise ValueError("regular course requirements.txt required")
    raw = requirements.read_bytes()
    if "mujoco==3.17.0" not in raw.decode("utf-8").splitlines():
        raise ValueError("course requirements must pin mujoco==3.17.0")
    return {"mujoco": mujoco.__version__, "requirements_sha256": digest(raw)}


def command(python, output):
    return [python, "-B", "-m", EVALUATOR, "--out", output]


def validate_result(result):
    if (result.get("schema") != "r26-known-signed-yaw-diagnostic.v1"
            or result.get("classification") !=
            "Previously disclosed R25 targeted improvement, rejected on older duration guards; not validated for unopened R26 independent conditions."
            or result.get("independent") is not False
            or result.get("unopened_r26_conditions_used") is not False
            or result.get("original_course_physics") is not True
            or result.get("episodes") != 12
            or result.get("case", {}).get("id") != "s25_long_transfer_signed"
            or [row.get("variant") for row in result.get("rows", [])] !=
            ["stock", "actuator_shift", "contact_shift"]
            or not all(row.get("passed") for row in result["rows"])):
        raise ValueError("R26 signed-yaw scientific result contract differs")
    return result


def describe():
    data = json.loads((ROOT / REGISTRATION).read_bytes())
    return {
        "schema": "r26-known-signed-yaw-replay-catalog.v1",
        "classification": data["classification"],
        "case": data["case"]["id"],
        "mechanics": list(data["variants"]),
        "arms": data["arms"],
        "episodes": 12,
        "default_run": False,
        "original_demo": False,
    }


def replay(kit, out):
    if any((ROOT / name).exists() for name in ("sim", "model", "evaluate.py")):
        raise ValueError("public R26 signed-yaw source snapshot must be kit-free")
    kit, out = shared.validate_paths(kit, out)
    runtime = runtime_gate(kit)
    data = closure()
    blobs, pins = source_blobs(data)
    manifest_raw = (ROOT / shared.KIT_MANIFEST).read_bytes()
    kit_pins = shared.authenticate_kit(kit, manifest_raw)
    environment = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", MUJOCO_GL="disable",
        HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
        ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost",
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="r26-yaw-replay-", dir=out.parent) as folder:
        base = Path(folder)
        overlay = base / "overlay"
        overlay.mkdir()
        shared.install_overlay(kit, overlay, manifest_raw, blobs, pins)
        shared.authenticate_kit(overlay, manifest_raw)
        result_path = overlay / "result.json"
        completed = subprocess.run(
            command(sys.executable, result_path.name), cwd=overlay, env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError((completed.stdout + completed.stderr).strip()
                               or "R26 signed-yaw replay failed")
        scientific = validate_result(json.loads(result_path.read_bytes()))
        shared.authenticate_kit(kit, manifest_raw)
        shared.authenticate_kit(overlay, manifest_raw)
        if any(digest(ROOT / name) != expected for name, expected in data["files"].items()):
            raise ValueError("R26 signed-yaw authored source changed during replay")
        edges = {row["variant"]: {
            "continuous_force": [row["force_interval"]["first"],
                                 row["force_interval"]["last"]],
            "r25_alarm_rises": row["arms"]["r25"]["alarm_rise_ticks"],
            "r26_alarm_rises": row["arms"]["r26_long"]["alarm_rise_ticks"],
        } for row in scientific["rows"]}
        receipt = {
            "schema": "r26-known-signed-yaw-authenticated-replay.v1",
            "classification": scientific["classification"],
            "source_snapshot_kit_free": True,
            "closure_sha256": digest(ROOT / CLOSURE),
            "source_sha256": data["files"],
            "source_closure": sorted(data["files"]),
            "kit_manifest_sha256": digest(manifest_raw),
            "kit_files": kit_pins,
            "runtime": runtime,
            "episodes": scientific["episodes"],
            "edge_summary": edges,
            "scientific_sha256": digest(result_path),
            "passed": scientific["passed"],
            "default_promoted": False,
            "independent_evaluation": False,
            "original_demo": False,
            "limitations": [
                "previously disclosed finite simulator diagnostic",
                "not unopened R26 independent evidence, hardware, deployment, or safety proof",
            ],
        }
        out.mkdir()
        with (out / "result.json").open("xb") as stream:
            stream.write(result_path.read_bytes())
        with (out / "run.log").open("x") as stream:
            stream.write(completed.stdout + completed.stderr)
        with (out / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="describe the fixed disclosed diagnostic; never runs it")
    run = commands.add_parser("replay", help="explicitly run the fixed disclosed diagnostic")
    run.add_argument("--kit", required=True, type=Path,
                     help="authenticated external course kit")
    run.add_argument("--out", required=True, type=Path,
                     help="fresh output directory outside kit and source")
    args = parser.parse_args(argv)
    try:
        output = describe() if args.command == "describe" else replay(args.kit, args.out)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError,
            subprocess.SubprocessError) as exc:
        parser.exit(2, json.dumps({"status": "rejected", "reason": str(exc)}) + "\n")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
