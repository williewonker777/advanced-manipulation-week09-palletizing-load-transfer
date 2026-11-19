"""Build or run a kit-free disclosed R30 G4 contact comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import r21_replay as shared
from tools import r29_g4_matrix as base


CLOSURE = "experiments/r30_g4_replay/closure.json"
PREFIX = "week09-r30-g4-disclosed-contacts"
BASE_CLOSURE_SHA256 = "77f7f72d4891ec28f452c31c39fe355624440dd2db8b7c7bd948a58da20fa811"
PLAN = "experiments/r30_g4_replay/plan.json"
PLAN_SHA256 = "d5e7d5e38b8876b52e57cda62d1bb0643762ee5f070c19a12cfc7e45b93d369d"
RESULT_SCHEMA = "r30-g4-disclosed-replay-result.v1"
CANDIDATES = {
    "memory": "85c4107c3939c7624dd77040ad322a99595c8e6941c7b75751c516a33be7a07b",
    "early_onset_rejected": "27add785807587380fb0fc7f2ba58037d2e3a48c0c9edef77786f4a3775e1e8a",
    "rising_owner_rejected": "83f8b97514fffe04275d96897e0ec197535b5f38d667c118f5a333181c8e80ef",
    "unload_notice": "db394659e0f61b159bb740d4fe9e9acea4d1f3be068aabc613fc396bca606d42",
}
FORBIDDEN_PARTS = {"results", "result", "evidence", "fresh", "prospect", "holdout",
                   "sim", "model", ".git", ".omx", "__pycache__"}
FORBIDDEN_SUFFIXES = {".png", ".npz", ".xml", ".stl", ".pyc", ".zip"}
README = """# R30 disclosed G4 contact comparison

This is a **known, disclosed research replay**, not the original course demo and
not blind, prospective, enrolled, promoted, default, hardware, or safety
qualified. It runs four passive detector arms with the same frozen landing
controller on four previously disclosed physical cases: the 24.32 N weak tool
contact, its exact zero-offset mechanics twin, and the low/high offset pair that
exposed an 8 ms edge-ownership displacement.

The arms are frozen R28 observation memory, the rejected early-onset witness,
the rejected rising-owner witness, and the R30 physical-unload notice. The first
two remain explicitly rejected. Output includes all alarm edges, inclusive true
tool-contact bounds (`f_tool > 1 N`), edge/contact relation, active samples,
force checks, G4 support, and full physical hashes.

No licensed kit, simulator, model, evaluator, previous output, or unopened case
is included. With the unmodified course kit and MuJoCo 3.17.0:

```bash
OUT=$(mktemp -d)
python -B tools/r30_g4_replay.py replay \\
  --kit /path/to/original/icpbl_sim --out "$OUT/r30-g4"
```
"""


def result_differences(left, right, path=""):
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            child = path + "/" + str(key)
            if key not in left or key not in right:
                out.append(child)
            else:
                out.extend(result_differences(left[key], right[key], child))
        return out
    if isinstance(left, list):
        if len(left) != len(right):
            return [path or "/"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out.extend(result_differences(a, b, path + "/" + str(index)))
        return out
    return [] if left == right else [path or "/"]


def compare_result(actual, expected, actual_bytes=None, expected_bytes=None):
    differences = result_differences(actual, expected)
    return {
        "byte_exact": (actual_bytes == expected_bytes if actual_bytes is not None
                       and expected_bytes is not None else not differences),
        "scientific_exact": not differences,
        "difference_paths": differences,
    }


def _file(name):
    relative = PurePosixPath(name)
    path = ROOT / name
    if (not name or name != relative.as_posix() or relative.is_absolute() or "\\" in name
            or ".." in relative.parts or set(relative.parts) & FORBIDDEN_PARTS
            or relative.suffix in FORBIDDEN_SUFFIXES or not path.is_file()
            or shared.has_symlink_component(path)):
        raise ValueError("forbidden or missing R30 member: " + name)
    return path.read_bytes()


def load():
    base_raw, base_closure, base_blobs = base.load()
    if shared.digest(base_raw) != BASE_CLOSURE_SHA256:
        raise ValueError("pinned portable R29 G4 closure changed")
    raw = _file(CLOSURE)
    closure = json.loads(raw)
    if (closure.get("schema") != "r30-g4-disclosed-export.v1"
            or closure.get("kit_included") is not False
            or closure.get("results_included") is not False
            or closure.get("unopened_conditions_included") is not False
            or closure.get("base_closure_sha256") != BASE_CLOSURE_SHA256
            or closure.get("plan_sha256") != PLAN_SHA256
            or closure.get("candidate_sha256") != CANDIDATES):
        raise ValueError("wrong R30 disclosed export identity")
    blobs = dict(base_blobs)
    for name, wanted in closure["files"].items():
        blob = _file(name)
        if shared.digest(blob) != wanted:
            raise ValueError("locked R30 member changed: " + name)
        blobs[name] = blob
    overlay = list(base_closure["overlay"]) + list(closure["overlay_additions"])
    if (len(overlay) != len(set(overlay)) or not set(overlay) <= set(blobs)
            or closure["files"].get(PLAN) != PLAN_SHA256):
        raise ValueError("incomplete or duplicate R30 overlay")
    blobs[base.CLOSURE] = base_raw
    return raw, closure, blobs, overlay


def bundle(out):
    if (os.path.lexists(out) or not out.parent.is_dir()
            or shared.has_symlink_component(out.parent)
            or out.resolve().is_relative_to(ROOT.resolve())):
        raise ValueError("fresh external archive required")
    raw, _, blobs, _ = load()
    payload = dict(blobs)
    payload[CLOSURE] = raw
    payload["README.md"] = README.encode()
    payload["bundle.json"] = (json.dumps({
        "schema": "r30-g4-disclosed-bundle.v1",
        "kit_included": False,
        "results_included": False,
        "unopened_conditions_included": False,
        "classification": "known_disclosed_research_only_no_promotion",
        "members_sha256": {name: shared.digest(blob) for name, blob in payload.items()},
    }, sort_keys=True, indent=2) + "\n").encode()
    with zipfile.ZipFile(out, "x") as archive:
        for name, blob in sorted(payload.items()):
            info = zipfile.ZipInfo(PREFIX + "/" + name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, blob, compresslevel=9)
    return {"zip_sha256": shared.digest(out.read_bytes()), "entries": len(payload)}


def replay(args):
    import mujoco

    kit, out = shared.validate_paths(args.kit, args.out)
    requirements = kit / "requirements.txt"
    if (mujoco.__version__ != "3.17.0" or shared.has_symlink_component(requirements)
            or "mujoco==3.17.0" not in requirements.read_text().splitlines()):
        raise ValueError("authenticated licensed MuJoCo 3.17.0 kit required")
    raw, closure, blobs, overlay = load()
    manifest = blobs[shared.KIT_MANIFEST]
    kit_pins = shared.authenticate_kit(kit, manifest)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", MUJOCO_GL="disable")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="r30-g4-disclosed-", dir=out.parent) as folder:
        temporary = Path(folder)
        work = temporary / "work"
        report = temporary / "report"
        work.mkdir()
        report.mkdir()
        sources = {name: blobs[name] for name in overlay}
        source_pins = {name: shared.digest(blob) for name, blob in sources.items()}
        shared.install_overlay(kit, work, manifest, sources, source_pins)
        shared.authenticate_kit(work, manifest)
        shared.run([sys.executable, "-B", "-m", "experiments.r30_g4_replay.study",
                    "--out", "result.json"], work, environment, report / "run.log")
        result_bytes = (work / "result.json").read_bytes()
        result = json.loads(result_bytes)
        if (result.get("schema") != RESULT_SCHEMA or result.get("episodes") != 16
                or result.get("candidate_sha256") != CANDIDATES
                or result.get("physics_exact_cases") != 4
                or result.get("force_exact_episodes") != 16
                or result.get("claim_status") != "disclosed_replay_only_no_promotion"):
            raise ValueError("wrong or incomplete R30 disclosed replay")
        comparison = None
        if args.expected:
            if not args.expected.is_file() or shared.has_symlink_component(args.expected):
                raise ValueError("regular expected disclosed replay required")
            expected_bytes = args.expected.read_bytes()
            comparison = compare_result(result, json.loads(expected_bytes),
                                        result_bytes, expected_bytes)
            if not comparison["scientific_exact"]:
                raise ValueError("R30 replay differs: " +
                                 ",".join(comparison["difference_paths"][:20]))
        if (shared.authenticate_kit(kit, manifest) != kit_pins
                or shared.authenticate_kit(work, manifest) != kit_pins
                or any(shared.digest((work / name).read_bytes()) != pin
                       or shared.digest((ROOT / name).read_bytes()) != pin
                       for name, pin in source_pins.items())
                or load()[0] != raw):
            raise ValueError("kit or locked R30 source changed during replay")
        (report / "result.json").write_bytes(result_bytes)
        receipt = {
            "schema": "r30-g4-disclosed-replay-receipt.v1",
            "runtime": mujoco.__version__,
            "kit_source_sha256": kit_pins,
            "base_closure_sha256": BASE_CLOSURE_SHA256,
            "closure_sha256": shared.digest(raw),
            "plan_sha256": PLAN_SHA256,
            "candidate_sha256": CANDIDATES,
            "result_sha256": shared.digest(result_bytes),
            "episodes": 16,
            "physics_exact_cases": 4,
            "force_exact_episodes": 16,
            "expected_result_checked": bool(args.expected),
            "result_comparison": comparison,
            "classification": "known_disclosed_research_only_no_promotion",
        }
        (report / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        shared.publish(report, out)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build-zip")
    build.add_argument("--out", type=Path, required=True)
    commands.add_parser("describe")
    run = commands.add_parser("replay")
    run.add_argument("--kit", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--expected", type=Path)
    args = parser.parse_args(argv)
    try:
        value = (bundle(args.out) if args.action == "build-zip"
                 else replay(args) if args.action == "replay" else load()[1])
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
