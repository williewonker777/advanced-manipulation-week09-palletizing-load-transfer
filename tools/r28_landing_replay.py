"""Build and run the kit-free R28 known landing replay handoff."""
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


CLOSURE = "experiments/r28_landing_replay/closure.json"
PLAN = "experiments/r28_landing_replay/plan.json"
PREFIX = "week09-r28-landing-known"
ARTIFACT_VERSION = "r28-known-landing-v1"
CANDIDATE = "experiments.r28_finite_contact.candidate:Detector"
CANDIDATE_FILE = "experiments/r28_finite_contact/candidate.py"
CANDIDATE_SHA256 = "9402c6de58a8948dd0461bea65c2300bcf8b872ba7c96b98cf5dcff8e7e6202b"
DEPENDENCY_FILE = "experiments/r28_landing_contact/candidate.py"
DEPENDENCY_SHA256 = "f764e6ebd13cfb67d41dad869e963ee56ccfd571d40efa58fa17f995e0daf4f7"
DIAGNOSTIC = "experiments.r28_finite_contact.pulse_diagnostic:Detector"
DIAGNOSTIC_FILE = "experiments/r28_finite_contact/pulse_diagnostic.py"
DIAGNOSTIC_SHA256 = "bb87916d443cc96dbbb012bcb53a08fd68c57c7ad4ab6d39e9b0ba1b70d6c680"
CANDIDATE_PLAN = "experiments/r28_landing_contact/plan.json"
CANDIDATE_PLAN_SHA256 = "b9f76ac4302b4ec23d229b391a0bad2e483caa532f2218d050ef42c14971e4dc"
FORBIDDEN_PARTS = {"sim", "model", ".git", "__pycache__", "r27_loaded_holdout", "r22_normal"}
FORBIDDEN_NAMES = {"evaluate.py", "servo_controller.py", "torque_controller.py", "collision_detector.py"}
README = """# R28 known landing replay (opt-in research)

This is **not the original course demo** and does not change any default. It is a
known, disclosed, finite MuJoCo comparison of the same frozen R24 landing
controller with C1, frozen R27 V3, and the frozen R28 detector.

The archive contains no course `sim/`, `model/`, or `evaluate.py`. Supply the
unaltered original course kit and the original MuJoCo 3.17.0 environment:

```bash
KIT=/path/to/original/icpbl_sim
OUT=$(mktemp -d)
python -B tools/r28_landing_replay.py compare --kit "$KIT" --out "$OUT/replay"
```

The fixed panel contains the disclosed 5.25 kg / +29 mm real tool contact, its
friction-matched zero-offset twin, and 0 kg / 8 kg +Y pickup positive controls.
The fourth arm is explicitly labelled `pulse_diagnostic`; it is a disclosed
bounded intervention, not an enrolled candidate or default.
`result.json` reports first real-tool latency, late false rises, exact force, and
full physical-trace hashes. A failed protection gate is retained as a reproducible
failure; it is never converted into a promotion claim.
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
        "byte_exact": (actual_bytes == expected_bytes if actual_bytes is not None and
                       expected_bytes is not None else not differences),
        "physical_result_exact": not differences,
        "difference_paths": differences,
    }


def load():
    source = ROOT / CLOSURE
    if shared.has_symlink_component(source) or not source.is_file():
        raise ValueError("regular R28 closure required")
    raw = source.read_bytes()
    document = json.loads(raw)
    if (document.get("schema") != "r28-landing-replay-export.v1"
            or document.get("artifact_version") != ARTIFACT_VERSION
            or document.get("kit_included") is not False
            or document.get("claim") != "opt-in-research-not-validated-default"
            or document.get("candidate", {}).get("entrypoint") != CANDIDATE
            or document.get("candidate", {}).get("source_sha256") != CANDIDATE_SHA256
            or document.get("candidate", {}).get("frozen_dependency_sha256") != DEPENDENCY_SHA256
            or document.get("candidate", {}).get("pulse_diagnostic_sha256") != DIAGNOSTIC_SHA256
            or document.get("candidate", {}).get("preregistration_sha256") != CANDIDATE_PLAN_SHA256):
        raise ValueError("wrong R28 closure identity")
    files = document.get("files")
    overlay = document.get("overlay")
    if not isinstance(files, dict) or not files or not isinstance(overlay, list) or not overlay:
        raise ValueError("complete R28 closure required")
    blobs = {}
    for name, wanted in files.items():
        relative = PurePosixPath(name)
        path = ROOT / name
        if (not name or "\\" in name or name != relative.as_posix() or relative.is_absolute()
                or ".." in relative.parts or any(part in FORBIDDEN_PARTS for part in relative.parts)
                or relative.name in FORBIDDEN_NAMES or not path.is_file()
                or shared.has_symlink_component(path)):
            raise ValueError("forbidden or missing R28 source member: " + name)
        blob = path.read_bytes()
        if shared.digest(blob) != wanted:
            raise ValueError("locked R28 source changed: " + name)
        blobs[name] = blob
    if (not set(overlay) <= set(blobs)
            or shared.KIT_MANIFEST not in blobs
            or shared.digest(blobs[shared.KIT_MANIFEST]) != shared.MANIFEST_SHA256
            or document["files"].get(CANDIDATE_FILE) != CANDIDATE_SHA256
            or document["files"].get(DEPENDENCY_FILE) != DEPENDENCY_SHA256
            or document["files"].get(DIAGNOSTIC_FILE) != DIAGNOSTIC_SHA256
            or document["files"].get(CANDIDATE_PLAN) != CANDIDATE_PLAN_SHA256):
        raise ValueError("incomplete R28 overlay or prerequisite pin")
    return raw, document, blobs


def bundle(out):
    if (os.path.lexists(out) or not out.parent.is_dir()
            or shared.has_symlink_component(out.parent)
            or out.resolve().is_relative_to(ROOT.resolve())):
        raise ValueError("fresh archive outside source root required")
    raw, _, blobs = load()
    payload = dict(blobs)
    payload[CLOSURE] = raw
    payload["README.md"] = README.encode()
    payload["bundle.json"] = (json.dumps({
        "artifact_version": ARTIFACT_VERSION,
        "kit_included": False,
        "results_included": False,
        "claim": "opt-in-research-not-validated-default",
        "members_sha256": {name: shared.digest(blob) for name, blob in payload.items()},
    }, sort_keys=True, indent=2) + "\n").encode()
    with zipfile.ZipFile(out, "x") as archive:
        for name, blob in sorted(payload.items()):
            info = zipfile.ZipInfo(PREFIX + "/" + name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, blob, compresslevel=9)
    return {"artifact_version": ARTIFACT_VERSION,
            "zip_sha256": shared.digest(out.read_bytes()), "entries": len(payload)}


def compare(args):
    import mujoco

    kit, out = shared.validate_paths(args.kit, args.out)
    requirements = kit / "requirements.txt"
    if (mujoco.__version__ != "3.17.0" or shared.has_symlink_component(requirements)
            or "mujoco==3.17.0" not in requirements.read_text().splitlines()):
        raise ValueError("original MuJoCo 3.17.0 course environment required")
    raw, closure, blobs = load()
    manifest = blobs[shared.KIT_MANIFEST]
    kit_pins = shared.authenticate_kit(kit, manifest)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", MUJOCO_GL="disable")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="r28-landing-", dir=out.parent) as folder:
        temporary = Path(folder)
        work = temporary / "work"
        result = temporary / "result"
        work.mkdir()
        result.mkdir()
        sources = {name: blobs[name] for name in closure["overlay"]}
        source_pins = {name: closure["files"][name] for name in sources}
        shared.install_overlay(kit, work, manifest, sources, source_pins)
        shared.authenticate_kit(work, manifest)
        shared.run([sys.executable, "-B", "-m", "experiments.r28_landing_replay.evaluator",
                    "--out", "result.json"], work, environment, result / "run.log")
        result_source = work / "result.json"
        result_bytes = result_source.read_bytes()
        result_document = json.loads(result_bytes)
        if (result_document.get("schema") != "r28-landing-known-replay-result.v1"
                or result_document.get("episodes") != 24
                or result_document.get("arms", {}).get("r28") != CANDIDATE
                or result_document.get("arms", {}).get("pulse_diagnostic") != DIAGNOSTIC
                or result_document.get("claim_status") != "known_replay_only_no_promotion"):
            raise ValueError("incomplete or wrong R28 replay result")
        comparison = None
        if args.expected:
            if not args.expected.is_file() or shared.has_symlink_component(args.expected):
                raise ValueError("regular expected R28 result required")
            expected_bytes = args.expected.read_bytes()
            comparison = compare_result(result_document, json.loads(expected_bytes),
                                        result_bytes, expected_bytes)
            if not comparison["physical_result_exact"]:
                raise ValueError("R28 physical result differs: " +
                                 ",".join(comparison["difference_paths"][:20]))
        if (shared.authenticate_kit(kit, manifest) != kit_pins
                or shared.authenticate_kit(work, manifest) != kit_pins
                or any(shared.digest((work / name).read_bytes()) != pin
                       or shared.digest((ROOT / name).read_bytes()) != pin
                       for name, pin in source_pins.items())
                or load()[0] != raw):
            raise ValueError("R28 kit or source changed during replay")
        (result / "result.json").write_bytes(result_bytes)
        receipt = {
            "schema": "r28-landing-known-replay-receipt.v1",
            "artifact_version": ARTIFACT_VERSION,
            "runtime": mujoco.__version__,
            "kit_source_sha256": kit_pins,
            "closure_sha256": shared.digest(raw),
            "candidate_sha256": CANDIDATE_SHA256,
            "frozen_dependency_sha256": DEPENDENCY_SHA256,
            "pulse_diagnostic_sha256": DIAGNOSTIC_SHA256,
            "candidate_preregistration_sha256": CANDIDATE_PLAN_SHA256,
            "result_sha256": shared.digest(result_bytes),
            "episodes": result_document["episodes"],
            "protection_passed": result_document["protection_passed"],
            "pulse_diagnostic_passed": result_document["pulse_diagnostic_passed"],
            "expected_result_checked": bool(args.expected),
            "result_comparison": comparison,
            "claim": "opt-in-research-not-validated-default",
        }
        (result / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        shared.publish(result, out)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build-zip")
    build.add_argument("--out", type=Path, required=True)
    commands.add_parser("describe")
    run = commands.add_parser("compare")
    run.add_argument("--kit", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--expected", type=Path)
    args = parser.parse_args(argv)
    try:
        value = (compare(args) if args.action == "compare" else bundle(args.out)
                 if args.action == "build-zip" else load()[1]["description"])
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
