"""Authenticate and replay the disclosed signed-yaw acquisition diagnostic."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r26_signed_yaw_replay as sealed


CLOSURE = "experiments/r26_yaw_onset_portable/closure.json"
TARGET = "experiments.r26_yaw_onset_portable.target"
CLASSIFICATION = (
    "Previously disclosed R25 signed-yaw acquisition diagnostic; opt-in only, "
    "with no default promotion or unopened R26 independent evidence."
)


def extension():
    path = ROOT / CLOSURE
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular signed-yaw acquisition closure required")
    data = json.loads(path.read_bytes())
    files = data.get("files")
    overlay = data.get("overlay_files")
    public = data.get("public_files")
    base = data.get("sealed_v2")
    if (data.get("schema") != "r26-known-yaw-acquisition-closure.v1"
            or not isinstance(files, dict) or not files
            or not isinstance(overlay, list) or not isinstance(public, list)
            or set(files) != set(overlay) | set(public)
            or not isinstance(base, dict)
            or base.get("closure_sha256") != sealed.digest(ROOT / sealed.CLOSURE)
            or base.get("registration_sha256") != sealed.digest(ROOT / sealed.REGISTRATION)
            or base.get("evaluator_sha256") != sealed.digest(
                ROOT / "experiments/r26_yaw_portable/evaluator.py")
            or base.get("signed_ownership_sha256") != sealed.digest(
                ROOT / "experiments/r26_long/candidate.py")):
        raise ValueError("sealed v2 acquisition dependency pins differ")
    for name, expected in files.items():
        source = ROOT / name
        if (not sealed.relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py" or len(expected) != 64
                or source.is_symlink() or not source.is_file()
                or sealed.digest(source) != expected):
            raise ValueError("signed-yaw acquisition source pin differs: " + name)
    return data


def blobs(data):
    base = sealed.closure()
    result, pins = sealed.source_blobs(base)
    for name in data["overlay_files"]:
        result[name] = (ROOT / name).read_bytes()
        pins[name] = sealed.digest(result[name])
        if pins[name] != data["files"][name]:
            raise ValueError("signed-yaw acquisition source changed: " + name)
    return result, pins, base


def command(python, output):
    return [python, "-B", "-m", TARGET, "--out", output]


def validate_result(result):
    rows = result.get("rows", [])
    registered = result.get("registered_input", {})
    if (result.get("schema") != "r26-known-signed-yaw-acquisition-diagnostic.v1"
            or result.get("classification") != CLASSIFICATION
            or result.get("independent") is not False
            or result.get("unopened_r26_conditions_used") is not False
            or result.get("default_promoted") is not False
            or result.get("original_course_physics") is not True
            or result.get("episodes") != 12
            or result.get("candidate_sha256") !=
            "212f4e79367cb636c0143a2d1f8aa7ba5584aaaca5ecde6b412d01a4985b6784"
            or registered.get("case", {}).get("id") != "s25_long_transfer_signed"
            or set(registered.get("variants", {})) !=
            {"stock", "actuator_shift", "contact_shift"}
            or [row.get("variant") for row in rows] !=
            ["stock", "actuator_shift", "contact_shift"]
            or not all(row.get("passed") for row in rows)
            or result.get("passed") is not True):
        raise ValueError("signed-yaw acquisition scientific result contract differs")
    return result


def describe():
    registry = json.loads((ROOT / sealed.REGISTRATION).read_bytes())
    return {
        "schema": "r26-known-yaw-acquisition-replay-catalog.v1",
        "classification": CLASSIFICATION,
        "case": registry["case"]["id"],
        "mechanics": list(registry["variants"]),
        "arms": ["r25", "acquisition"],
        "episodes": 12,
        "default_run": False,
        "original_demo": False,
    }


def replay(kit, out):
    if any((ROOT / name).exists() for name in ("sim", "model", "evaluate.py")):
        raise ValueError("public acquisition source snapshot must be kit-free")
    kit, out = sealed.shared.validate_paths(kit, out)
    runtime = sealed.runtime_gate(kit)
    data = extension()
    source_blobs, pins, base = blobs(data)
    manifest_raw = (ROOT / sealed.shared.KIT_MANIFEST).read_bytes()
    kit_pins = sealed.shared.authenticate_kit(kit, manifest_raw)
    environment = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", MUJOCO_GL="disable",
        HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
        ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost",
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="r26-yaw-acquisition-", dir=out.parent) as folder:
        overlay = Path(folder) / "overlay"
        overlay.mkdir()
        sealed.shared.install_overlay(kit, overlay, manifest_raw, source_blobs, pins)
        sealed.shared.authenticate_kit(overlay, manifest_raw)
        result_path = overlay / "result.json"
        completed = subprocess.run(
            command(sys.executable, result_path.name), cwd=overlay, env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError((completed.stdout + completed.stderr).strip()
                               or "signed-yaw acquisition replay failed")
        scientific = validate_result(json.loads(result_path.read_bytes()))
        sealed.shared.authenticate_kit(kit, manifest_raw)
        sealed.shared.authenticate_kit(overlay, manifest_raw)
        extension()
        sealed.closure()
        summary = {row["variant"]: {
            "continuous_force": [row["force_interval"]["first"],
                                 row["force_interval"]["last"]],
            "r25_first_rise": row["arms"]["r25"]["force"]["alarm"]["first_rise_tick"],
            "acquisition_first_rise": row["arms"]["acquisition"]["force"]["alarm"]["first_rise_tick"],
            "acquisition_alarm_active_samples": row["arms"]["acquisition"]["force"]["alarm"]["active_samples"],
            "inferred_event_active_samples": row["arms"]["acquisition"]["force"]["telemetry"]["inferred_continuous_event"]["active_samples"],
            "consumed_reacquisitions": row["arms"]["acquisition"]["force"]["telemetry"]["consumed_same_source_reacquisition"]["count"],
        } for row in scientific["rows"]}
        combined = dict(base["files"])
        combined.update(data["files"])
        receipt = {
            "schema": "r26-known-yaw-acquisition-authenticated-replay.v1",
            "classification": CLASSIFICATION,
            "source_snapshot_kit_free": True,
            "sealed_v2_closure_sha256": sealed.digest(ROOT / sealed.CLOSURE),
            "extension_closure_sha256": sealed.digest(ROOT / CLOSURE),
            "source_sha256": combined,
            "source_closure": sorted(combined),
            "kit_manifest_sha256": sealed.digest(manifest_raw),
            "kit_files": kit_pins,
            "runtime": runtime,
            "registered_input": scientific["registered_input"],
            "episodes": scientific["episodes"],
            "telemetry_summary": summary,
            "scientific_sha256": sealed.digest(result_path),
            "passed": scientific["passed"],
            "default_promoted": False,
            "independent_evaluation": False,
            "original_demo": False,
            "limitations": scientific["limitations"],
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
    commands.add_parser("describe", help="describe the fixed disclosed diagnostic")
    run = commands.add_parser("replay", help="explicitly run the diagnostic")
    run.add_argument("--kit", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
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
