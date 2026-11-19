"""Build or run the deterministic rejected R29 G4 release-control replay."""
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
from tools import r28_landing_replay as landing


CLOSURE = "experiments/r29_g4_export/closure.json"
PREFIX = "week09-r29-g4-release-controls"
BASE_CLOSURE_SHA256 = "3e266e9314f341b419d3edde98458f522cd24fd133f4e0cdc51ef59fb1525a70"
CANDIDATE = "experiments.r29_g4_support_release.candidate:Detector"
CANDIDATE_FILE = "experiments/r29_g4_support_release/candidate.py"
CANDIDATE_SHA256 = "edc89c20eeeb469bd16096ed978bbde0d17dd4a35ddd666ba0c3c298d8b7c6c9"
RESULT_SCHEMA = "r29-g4-empty-inverse-release-result.v1"
README = """# R29 rejected G4 release-control replay

This archive reproduces a **known, disclosed, rejected research mechanism**.
It is not blind, prospective, enrolled, promoted, default, hardware, or safety
qualified. The candidate misses a genuine matched `-Z 15 N` impulse and is
included for auditability, not use.

The archive contains no course kit, simulator, model, evaluator, sealed input,
raw log, or prior result. Supply the unmodified licensed kit and MuJoCo 3.17.0:

```bash
OUT=$(mktemp -d)
python -B tools/r29_g4_replay.py release-controls \\
  --kit /path/to/original/icpbl_sim --out "$OUT/r29-release"
```

The fixed 54-episode panel compares C1, frozen R28 observation-memory, and the
frozen rejected R29 candidate on disclosed +29 mm/offset/load normals, mass
0/8 +Y controls, and fixed near/late signed X/Z impulses. Exact force, physical
signatures, prefixes, misses, strictness, duplicates, delay, and duration are
preserved in `result.json`.

The separately committed 90-case R29 prospect is intentionally not included in
this archive and must remain under its original owner and fixed-arm protocol.
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


def load():
    base_raw, base, base_blobs = landing.load()
    if shared.digest(base_raw) != BASE_CLOSURE_SHA256:
        raise ValueError("pinned licensed R28 replay closure changed")
    source = ROOT / CLOSURE
    if not source.is_file() or shared.has_symlink_component(source):
        raise ValueError("regular R29 closure required")
    raw = source.read_bytes()
    closure = json.loads(raw)
    if (closure.get("schema") != "r29-g4-release-export.v1"
            or closure.get("kit_included") is not False
            or closure.get("results_included") is not False
            or closure.get("base_closure_sha256") != BASE_CLOSURE_SHA256
            or closure.get("candidate_sha256") != CANDIDATE_SHA256
            or closure.get("prospect_included") is not False):
        raise ValueError("wrong R29 export identity")
    blobs = dict(base_blobs)
    overlay = set(base["overlay"])
    for name, wanted in closure["files"].items():
        relative = PurePosixPath(name)
        path = ROOT / name
        if (not name or "\\" in name or name != relative.as_posix() or relative.is_absolute()
                or ".." in relative.parts or not path.is_file()
                or shared.has_symlink_component(path)
                or any(part in {"r29_g4_prospect", "r27_loaded_holdout", "r22_normal",
                                "sim", "model", ".git", "__pycache__"}
                       for part in relative.parts)):
            raise ValueError("forbidden or missing R29 member: " + name)
        blob = path.read_bytes()
        if shared.digest(blob) != wanted:
            raise ValueError("locked R29 member changed: " + name)
        blobs[name] = blob
    if closure["files"].get(CANDIDATE_FILE) != CANDIDATE_SHA256:
        raise ValueError("candidate pin missing")
    overlay.update(closure["overlay_additions"])
    if not overlay <= set(blobs):
        raise ValueError("incomplete R29 overlay")
    blobs[landing.CLOSURE] = base_raw
    return raw, closure, blobs, sorted(overlay)


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
        "kit_included": False,
        "results_included": False,
        "prospect_included": False,
        "classification": "known_disclosed_rejected_research_only",
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


def release_controls(args):
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
    with tempfile.TemporaryDirectory(prefix="r29-g4-release-", dir=out.parent) as folder:
        temporary = Path(folder)
        work = temporary / "work"
        report = temporary / "report"
        work.mkdir()
        report.mkdir()
        sources = {name: blobs[name] for name in overlay}
        pins = {name: shared.digest(blob) for name, blob in sources.items()}
        shared.install_overlay(kit, work, manifest, sources, pins)
        shared.authenticate_kit(work, manifest)
        shared.run([sys.executable, "-B", "-m", "experiments.r29_g4_support_release.study",
                    "--out", "result.json"], work, environment, report / "run.log")
        result_bytes = (work / "result.json").read_bytes()
        result = json.loads(result_bytes)
        if (result.get("schema") != RESULT_SCHEMA or result.get("episodes") != 54
                or result.get("pins", {}).get("candidate") != CANDIDATE_SHA256
                or result.get("passed") is not False
                or result.get("claim_status") != "disclosed_postmortem_only_no_promotion"):
            raise ValueError("wrong, incomplete, or non-rejected R29 result")
        comparison = None
        if args.expected:
            if not args.expected.is_file() or shared.has_symlink_component(args.expected):
                raise ValueError("regular expected R29 result required")
            expected_bytes = args.expected.read_bytes()
            comparison = compare_result(result, json.loads(expected_bytes),
                                        result_bytes, expected_bytes)
            if not comparison["scientific_exact"]:
                raise ValueError("R29 scientific result differs: "
                                 + ",".join(comparison["difference_paths"][:20]))
        if (shared.authenticate_kit(kit, manifest) != kit_pins
                or shared.authenticate_kit(work, manifest) != kit_pins
                or any(shared.digest((work / name).read_bytes()) != pin
                       or shared.digest((ROOT / name).read_bytes()) != pin
                       for name, pin in pins.items())
                or load()[0] != raw):
            raise ValueError("kit or locked R29 source changed during replay")
        (report / "result.json").write_bytes(result_bytes)
        receipt = {
            "schema": "r29-g4-release-replay-receipt.v1",
            "runtime": mujoco.__version__,
            "kit_source_sha256": kit_pins,
            "closure_sha256": shared.digest(raw),
            "candidate_sha256": CANDIDATE_SHA256,
            "result_sha256": shared.digest(result_bytes),
            "episodes": 54,
            "rejected": True,
            "expected_result_checked": bool(args.expected),
            "result_comparison": comparison,
            "prospect_included": False,
            "classification": "known_disclosed_rejected_research_only",
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
    run = commands.add_parser("release-controls")
    run.add_argument("--kit", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--expected", type=Path)
    args = parser.parse_args(argv)
    try:
        value = (bundle(args.out) if args.action == "build-zip"
                 else release_controls(args) if args.action == "release-controls"
                 else load()[1])
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
