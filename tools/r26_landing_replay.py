"""Authenticate and run either ORIGINAL box-8 or named R26 landing research replays."""
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

from tools import course_example
from tools import r24_transfer_replay as shared


CLOSURE = "experiments/r26_transfer_portable/closure.json"
REGISTRATION = "experiments/r26_transfer_portable/registration.json"
SCENARIOS = ("normal-sh5", "signed-x-positive", "signed-x-negative",
             "signed-z-positive", "signed-z-negative")
ARMS = ("registered-c1", "r24", "baseline")


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
        raise ValueError("regular R26 portable closure required")
    data = json.loads(path.read_bytes())
    files = data.get("files")
    overlay = data.get("overlay_files")
    public = data.get("public_files")
    if (data.get("schema") != "r26-transfer-portable-closure.v1"
            or not isinstance(files, dict) or not files
            or not isinstance(overlay, list) or not isinstance(public, list)
            or set(files) != set(overlay) | set(public)):
        raise ValueError("complete R26 portable closure differs")
    for name, expected in files.items():
        path = ROOT / name
        if (not relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py" or len(expected) != 64
                or path.is_symlink() or not path.is_file() or digest(path) != expected):
            raise ValueError("R26 portable source pin differs: " + name)
    return data


def source_blobs(data):
    blobs = {name: (ROOT / name).read_bytes() for name in data["overlay_files"]}
    if any(digest(blob) != data["files"][name] for name, blob in blobs.items()):
        raise ValueError("R26 source changed after closure validation")
    return blobs


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


def describe():
    registration = json.loads((ROOT / REGISTRATION).read_bytes())
    return {
        "schema": "r26-landing-replay-catalog.v1",
        "original": registration["original"],
        "research_arms": registration["arms"],
        "research_scenarios": list(SCENARIOS),
        "separation": "Only the original subcommand is labeled ORIGINAL. Every replay arm is custom finite simulation research.",
    }


def replay(kit, out, scenario, arms):
    kit, out = shared.validate_paths(kit, out)
    runtime = runtime_gate(kit)
    data = closure()
    blobs = source_blobs(data)
    manifest_raw = (ROOT / shared.MANIFEST).read_bytes()
    kit_pins = shared.authenticate_kit(kit, manifest_raw)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MUJOCO_GL="disable", HTTP_PROXY="http://127.0.0.1:9",
                       HTTPS_PROXY="http://127.0.0.1:9", ALL_PROXY="http://127.0.0.1:9",
                       NO_PROXY="127.0.0.1,localhost")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="r26-landing-replay-", dir=out.parent) as folder:
        overlay = Path(folder) / "overlay"
        overlay.mkdir()
        shared.install(kit, overlay, manifest_raw, blobs,
                       {name: data["files"][name] for name in data["overlay_files"]})
        shared.authenticate_kit(overlay, manifest_raw)
        result = overlay / "result.json"
        command = [sys.executable, "-B", "-m", "experiments.r26_transfer_portable.evaluator",
                   "--scenario", scenario]
        for arm in arms:
            command += ["--arm", arm]
        command += ["--out", result.name]
        completed = subprocess.run(command, cwd=overlay, env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError((completed.stdout + completed.stderr).strip())
        scientific = json.loads(result.read_bytes())
        if (scientific.get("schema") != "r26-transfer-disclosed-replay.v1"
                or scientific.get("scenario") != scenario
                or set(scientific.get("arms", {})) != set(arms)):
            raise ValueError("R26 scientific result contract differs")
        shared.authenticate_kit(kit, manifest_raw)
        shared.authenticate_kit(overlay, manifest_raw)
        if any(digest(ROOT / name) != expected for name, expected in data["files"].items()):
            raise ValueError("R26 source changed during replay")
        receipt = {
            "schema": "r26-transfer-authenticated-replay-receipt.v1",
            "classification": scientific["classification"],
            "source_snapshot_kit_free": True,
            "closure_sha256": digest(ROOT / CLOSURE),
            "source_sha256": data["files"],
            "kit_manifest_sha256": digest(manifest_raw),
            "kit_files": kit_pins,
            "runtime": runtime,
            "scenario": scenario,
            "arms": list(arms),
            "scientific_sha256": digest(result),
            "original_demo": {"included": False, "separate_command": "original --box 8"},
            "limitations": ["finite CPU simulation", "disclosed scenarios",
                            "not ORIGINAL, hidden evaluation, hardware, deployment, or safety proof"],
        }
        out.mkdir()
        (out / "result.json").write_bytes(result.read_bytes())
        (out / "run.log").write_text(completed.stdout + completed.stderr)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                                      allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="show ORIGINAL/custom separation and arm status")
    original = commands.add_parser("original", help="run authenticated unmodified course demo")
    original.add_argument("--kit", type=Path, required=True)
    original.add_argument("--guide", type=Path, required=True)
    original.add_argument("--out", type=Path, required=True)
    original.add_argument("--box", type=int, choices=(8,), default=8)
    research = commands.add_parser("replay", help="run disclosed custom research scenario(s)")
    research.add_argument("--kit", type=Path, required=True)
    research.add_argument("--out", type=Path, required=True)
    research.add_argument("--scenario", choices=SCENARIOS, required=True)
    research.add_argument("--arm", choices=ARMS, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            output = describe()
        elif args.command == "original":
            output = course_example.run(args.kit, args.guide, args.out)
        else:
            output = replay(args.kit, args.out, args.scenario,
                            tuple(dict.fromkeys(args.arm)))
    except (course_example.CourseError, OSError, RuntimeError, ValueError,
            KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        parser.exit(2, json.dumps({"status": "rejected", "reason": str(exc)}) + "\n")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
