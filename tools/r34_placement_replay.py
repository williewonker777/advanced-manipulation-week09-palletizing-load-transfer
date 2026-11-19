"""Replay frozen R31/R33 and an optional public R34 on an authenticated kit copy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.r34_archive_replay import runner
from tools import r21_replay as shared
from tools import r33_vertical_replay as parent

DIR = "experiments/r34_archive_replay/"
CLOSURE = DIR + "closure.json"
RUNTIME_MEMBERS = {DIR + "__init__.py", DIR + "runner.py", "tools/r34_placement_replay.py"}
SOURCE_MEMBERS = RUNTIME_MEMBERS | {DIR + "test_replay.py", DIR + "RESULT.md"}
SCOPES = {"smoke": (2, 4), "placement9": (9, 18), "full-place": (60, 120)}
OPTIONAL_NAME = DIR + "r34-entry.json"


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular source required: " + str(path))
    return path.read_bytes()


def load() -> tuple[bytes, dict[str, bytes], list[str]]:
    r33_raw, files, overlay = parent.load()
    data = _read_regular(ROOT / CLOSURE)
    registry = json.loads(data)
    pins = registry.get("files", {})
    if (registry.get("schema") != "r34-placement-replay-closure.v1"
            or registry.get("kit_included") is not False
            or registry.get("packaging_enabled") is not False
            or registry.get("r33_closure_sha256") != shared.digest(r33_raw)
            or set(pins) != SOURCE_MEMBERS
            or set(files) & SOURCE_MEMBERS):
        raise ValueError("changed R34 placement replay source contract")
    blobs = {**files, parent.CLOSURE: r33_raw}
    for name, pin in pins.items():
        raw = _read_regular(ROOT / name)
        if shared.digest(raw) != pin:
            raise ValueError("changed R34 replay member: " + name)
        blobs[name] = raw
    return data, blobs, sorted(set(overlay) | (RUNTIME_MEMBERS - {"tools/r34_placement_replay.py"}))


def load_optional(path: Path | None, root: Path = ROOT) -> tuple[dict[str, Any] | None, dict[str, bytes]]:
    obj = runner.optional_registration(path, root)
    if obj is None:
        return None, {}
    blobs: dict[str, bytes] = {}
    for name, pin in obj["files"].items():
        raw = _read_regular(root / name)
        if shared.digest(raw) != pin:
            raise ValueError("changed R34 optional source: " + name)
        blobs[name] = raw
    canonical = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()
    blobs[OPTIONAL_NAME] = canonical
    return obj, blobs


def describe(registry: Path | None) -> dict[str, Any]:
    raw, _, _ = load()
    optional, _ = load_optional(registry)
    return {
        "schema": "r34-placement-replay-description.v1",
        "scopes": {name: {"collisions": counts[0], "inputs": counts[1]}
                   for name, counts in SCOPES.items()},
        "arms": ["r31", "r33"] + (["r34"] if optional else []),
        "course_original": False,
        "kit_included": False,
        "packaging_enabled": False,
        "optional_r34": None if optional is None else optional["candidate"],
        "closure_sha256": shared.digest(raw),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco

    kit, out = shared.validate_paths(args.kit, args.out)
    raw, files, overlay = load()
    optional, optional_blobs = load_optional(args.r34_registry)
    if set(files) & set(optional_blobs):
        raise ValueError("R34 optional source overlaps frozen replay closure")
    manifest = files[shared.KIT_MANIFEST]
    kit_pins = shared.authenticate_kit(kit, manifest)
    if mujoco.__version__ != "3.17.0":
        raise ValueError("MuJoCo 3.17.0 required")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MUJOCO_GL="disable")
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    with TemporaryDirectory(prefix="r34-placement-replay-", dir=out.parent) as folder:
        work, report = (Path(folder) / name for name in ("work", "report"))
        work.mkdir()
        report.mkdir()
        names = set(overlay) | set(optional_blobs)
        sources = {name: optional_blobs[name] if name in optional_blobs else files[name]
                   for name in names}
        shared.install_overlay(kit, work, manifest, sources,
                               {name: shared.digest(blob) for name, blob in sources.items()})
        command = [sys.executable, "-B", "-m", "experiments.r34_archive_replay.runner",
                   args.scope, "--jobs", str(args.jobs), "--out", "result.json"]
        if optional:
            command.extend(("--r34-registry", OPTIONAL_NAME))
        try:
            shared.run(command, work, environment, report / "execution.log")
        except RuntimeError as exc:
            lines = (report / "execution.log").read_text().splitlines()
            raise RuntimeError(str(exc) + ("; " + lines[-1] if lines else "")) from exc
        result = (work / "result.json").read_bytes()
        obj = json.loads(result)
        collisions, inputs = SCOPES[args.scope]
        arms = ["r31", "r33"] + (["r34"] if optional else [])
        if (obj.get("schema") != "r34-placement-physical-replay.v1"
                or obj.get("scope") != args.scope or obj.get("course_original")
                or obj.get("default_changed") or obj.get("kit_changed")
                or obj.get("arms") != arms or obj.get("collisions") != collisions
                or obj.get("inputs") != inputs or obj.get("episodes") != inputs * len(arms)
                or obj.get("actual_independent_executions_per_input") != len(arms)
                or obj.get("input_failures")
                or shared.authenticate_kit(work, manifest) != kit_pins
                or shared.authenticate_kit(kit, manifest) != kit_pins
                or load()[0] != raw):
            raise ValueError("incomplete, unauthenticated, or changed R34 placement replay")
        (report / "result.json").write_bytes(result)
        receipt = {
            "schema": "r34-placement-replay-receipt.v1",
            "scope": args.scope,
            "arms": arms,
            "inputs": inputs,
            "episodes": inputs * len(arms),
            "registrations": obj["registrations"],
            "result_sha256": shared.digest(result),
            "kit_files_sha256": kit_pins,
            "kit_manifest_sha256": shared.digest(manifest),
            "r34_closure_sha256": shared.digest(raw),
            "kit_unchanged": True,
        }
        (report / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        shared.publish(report, out)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("describe")
    info.add_argument("--r34-registry", type=Path)
    run = sub.add_parser("compare")
    run.add_argument("--kit", type=Path, required=True)
    run.add_argument("--scope", choices=SCOPES, default="smoke")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--jobs", type=int, choices=range(1, 9), default=2)
    run.add_argument("--r34-registry", type=Path)
    args = parser.parse_args(argv)
    try:
        obj = compare(args) if args.command == "compare" else describe(args.r34_registry)
    except (ImportError, KeyError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(obj, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
