"""Reproduce the frozen R31/R33 vertical-event comparison on an authenticated kit copy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r21_replay as shared
from tools import r31_event_replay as parent

DIR = "experiments/r33_vertical_replay/"
CLOSURE = DIR + "closure.json"
PREFIX = "week09-r33-vertical-events"
SCOPES = {"smoke": 3, "axial19": 19, "full-d4": 180}
CORE = {DIR + name for name in ("__init__.py", "runner.py", "registry.json")}
MEMBERS = CORE | {"tools/r33_vertical_replay.py"}
README = """# R33 수직 충돌 경보 재현

수업 원본 데모나 제출본이 아니라 공개 연구 감지기의 로컬 재현입니다. ZIP에는 수업 키트가 없습니다.
MuJoCo 3.17.0과 본인이 받은 원본 키트를 사용해 새 출력 경로에 실행하세요.

    python -B tools/r33_vertical_replay.py compare --kit /path/to/icpbl_sim --scope smoke --out /new/result

smoke는 집기·운반·놓기 각 1개, axial19는 공개된 수직 미검출 19개, full-d4는 385조건 전체입니다.
각 입력은 R31과 R33가 별도 실행되며, 충돌 입력에는 같은 물성의 무충격 실행이 함께 포함됩니다.
result.json은 물리 해시, 실제 외력, 충돌 전 접두 구간, 힘 구간 내 첫 경보, 중복·오경보를 기록합니다.
공개 조건의 시뮬레이션 재현이며 독립 안전 인증이나 원본 데모 결과가 아닙니다.
"""


def load():
    raw, files, overlay = parent.load()
    path = ROOT / CLOSURE
    data = path.read_bytes()
    reg = json.loads(data)
    extra = set(reg.get("files", {}))
    expected = MEMBERS | extra
    allowed = all(name in MEMBERS or name.startswith("experiments/r33_") for name in extra)
    barred = any(name.startswith(("experiments/r33_vertical_validation/", "results/"))
                 or Path(name).name in {"run.py", "evaluate.py", "servo_controller.py",
                                        "torque_controller.py", "collision_detector.py"}
                 for name in extra)
    if (reg.get("schema") != "r33-vertical-replay-closure.v1"
            or reg.get("kit_included") is not False
            or reg.get("r31_closure_sha256") != shared.digest(raw)
            or set(files) & expected
            or CLOSURE in extra or not MEMBERS <= extra or not allowed or barred):
        raise ValueError("changed R33 replay source contract")
    blobs = {**files, parent.CLOSURE: raw}
    for name, pin in reg["files"].items():
        source = ROOT / name
        if source.is_symlink() or not source.is_file() or shared.digest(source) != pin:
            raise ValueError("changed R33 replay member: " + name)
        blobs[name] = source.read_bytes()
    return data, blobs, sorted(set(overlay) | (extra - {"tools/r33_vertical_replay.py"}))


def bundle(out):
    fn = parent.bundle
    return FunctionType(fn.__code__, {**fn.__globals__, "load": load, "CLOSURE": CLOSURE,
                                      "PREFIX": PREFIX, "README": README, "ROOT": ROOT})(out)


def compare(args):
    import mujoco
    kit, out = shared.validate_paths(args.kit, args.out)
    raw, files, overlay = load()
    manifest = files[shared.KIT_MANIFEST]
    kit_pins = shared.authenticate_kit(kit, manifest)
    if mujoco.__version__ != "3.17.0":
        raise ValueError("MuJoCo 3.17.0 required")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MUJOCO_GL="disable")
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    with TemporaryDirectory(prefix="r33-vertical-replay-", dir=out.parent) as folder:
        work, report = (Path(folder) / name for name in ("work", "report"))
        work.mkdir(); report.mkdir()
        sources = {name: files[name] for name in overlay}
        shared.install_overlay(kit, work, manifest, sources,
                               {name: shared.digest(blob) for name, blob in sources.items()})
        try:
            shared.run([sys.executable, "-B", "-m", "experiments.r33_vertical_replay.runner",
                        args.scope, "--jobs", str(args.jobs), "--out", "result.json"],
                       work, environment, report / "execution.log")
        except RuntimeError as exc:
            lines = (report / "execution.log").read_text().splitlines()
            raise RuntimeError(str(exc) + ("; " + lines[-1] if lines else "")) from exc
        result = (work / "result.json").read_bytes()
        obj = json.loads(result)
        expected_inputs = 385 if args.scope == "full-d4" else 2 * SCOPES[args.scope]
        if (obj.get("schema") != "r33-vertical-physical-replay.v1"
                or obj.get("scope") != args.scope or obj.get("course_original")
                or obj.get("default_changed") or obj.get("arms") != ["r31", "r33"]
                or obj.get("collisions") != SCOPES[args.scope]
                or obj.get("inputs") != expected_inputs
                or obj.get("episodes") != 2 * expected_inputs
                or obj.get("actual_independent_executions_per_input") != 2
                or obj.get("input_failures")
                or shared.authenticate_kit(work, manifest) != kit_pins
                or shared.authenticate_kit(kit, manifest) != kit_pins
                or load()[0] != raw):
            raise ValueError("incomplete, unauthenticated, or changed R33 replay")
        (report / "result.json").write_bytes(result)
        receipt = {
            "schema": "r33-vertical-replay-receipt.v1",
            "scope": args.scope,
            "inputs": expected_inputs,
            "episodes": 2 * expected_inputs,
            "candidate": obj["candidate"],
            "result_sha256": shared.digest(result),
            "kit_files_sha256": kit_pins,
            "kit_manifest_sha256": shared.digest(manifest),
            "r33_closure_sha256": shared.digest(raw),
            "kit_unchanged": True,
        }
        (report / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        shared.publish(report, out)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("describe")
    archive = sub.add_parser("build-zip")
    archive.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("compare")
    run.add_argument("--kit", type=Path, required=True)
    run.add_argument("--scope", choices=SCOPES, default="smoke")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--jobs", type=int, choices=range(1, 9), default=2)
    args = parser.parse_args(argv)
    try:
        obj = (compare(args) if args.command == "compare" else bundle(args.out)
               if args.command == "build-zip" else {
                   "scopes": SCOPES, "course_original": False,
                   "closure_sha256": shared.digest(load()[0])})
    except (KeyError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(obj, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
