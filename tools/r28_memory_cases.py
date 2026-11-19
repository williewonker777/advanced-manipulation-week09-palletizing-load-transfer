"""Build or execute the kit-free R28 observation-memory known-case handoff."""
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


CLOSURE = "experiments/r28_memory_cases/closure.json"
PREFIX = "week09-r28-memory-cases"
BASE_CLOSURE_SHA256 = "3e266e9314f341b419d3edde98458f522cd24fd133f4e0cdc51ef59fb1525a70"
MEMORY_ENTRYPOINT = "experiments.r28_contact_memory.candidate:Detector"
MEMORY_FILE = "experiments/r28_contact_memory/candidate.py"
MEMORY_SHA256 = "85c4107c3939c7624dd77040ad322a99595c8e6941c7b75751c516a33be7a07b"
README = """# R28 observation-memory known-case replay

This is a disclosed deterministic postmortem replay, **not** a blind result,
enrollment, default, hardware, or safety claim. The archive does not contain the
course kit. Supply the unmodified original kit and MuJoCo 3.17.0 environment:

```bash
OUT=$(mktemp -d)
python -B tools/r28_memory_cases.py compare \\
  --kit /path/to/original/icpbl_sim --out "$OUT/memory-cases"
```

The fixed authored R28 panel compares C1, V3, frozen observation-memory, and the
rejected finite arm on the disclosed +29 mm tool/quiet twin and +Y 0/8 kg
force/no-force pairs. The result retains every authored edge, exact-force gate,
and physical-array signature.
"""


def load():
    base_raw, base, base_blobs = landing.load()
    if shared.digest(base_raw) != BASE_CLOSURE_SHA256:
        raise ValueError("pinned R28 landing replay closure changed")
    path = ROOT / CLOSURE
    if not path.is_file() or shared.has_symlink_component(path):
        raise ValueError("regular memory-case closure required")
    raw = path.read_bytes()
    closure = json.loads(raw)
    if (closure.get("schema") != "r28-memory-case-export.v1"
            or closure.get("kit_included") is not False
            or closure.get("base_closure_sha256") != BASE_CLOSURE_SHA256
            or closure.get("memory_candidate_sha256") != MEMORY_SHA256):
        raise ValueError("wrong memory-case closure identity")
    blobs = dict(base_blobs)
    overlay = set(base["overlay"])
    for name, wanted in closure["files"].items():
        relative = PurePosixPath(name)
        source = ROOT / name
        if (not name or "\\" in name or name != relative.as_posix() or relative.is_absolute()
                or ".." in relative.parts or not source.is_file()
                or shared.has_symlink_component(source)):
            raise ValueError("invalid memory-case source: " + name)
        blob = source.read_bytes()
        if shared.digest(blob) != wanted:
            raise ValueError("locked memory-case source changed: " + name)
        blobs[name] = blob
    if closure["files"].get(MEMORY_FILE) != MEMORY_SHA256:
        raise ValueError("memory candidate pin missing")
    overlay.update(closure["overlay_additions"])
    if not overlay <= set(blobs):
        raise ValueError("incomplete memory-case overlay")
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
        "classification": "known_disclosed_postmortem_only",
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


def compare(args):
    import mujoco

    kit, out = shared.validate_paths(args.kit, args.out)
    requirements = kit / "requirements.txt"
    if (mujoco.__version__ != "3.17.0" or shared.has_symlink_component(requirements)
            or "mujoco==3.17.0" not in requirements.read_text().splitlines()):
        raise ValueError("authenticated original MuJoCo 3.17.0 kit required")
    raw, closure, blobs, overlay = load()
    manifest = blobs[shared.KIT_MANIFEST]
    kit_pins = shared.authenticate_kit(kit, manifest)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", MUJOCO_GL="disable")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="r28-memory-cases-", dir=out.parent) as folder:
        temporary = Path(folder)
        work = temporary / "work"
        report = temporary / "report"
        work.mkdir()
        report.mkdir()
        sources = {name: blobs[name] for name in overlay}
        pins = {name: shared.digest(blob) for name, blob in sources.items()}
        shared.install_overlay(kit, work, manifest, sources, pins)
        shared.authenticate_kit(work, manifest)
        shared.run([sys.executable, "-B", "-m", "experiments.r28_memory_cases.evaluator",
                    "--out", "result.json"], work, environment, report / "run.log")
        result_bytes = (work / "result.json").read_bytes()
        result = json.loads(result_bytes)
        if (result.get("schema") != "r28-memory-known-case-delivery.v1"
                or result.get("episodes") != 24
                or result.get("memory_candidate_sha256") != MEMORY_SHA256
                or result.get("claim_status") != "known_replay_only_no_fresh_enrollment"):
            raise ValueError("wrong or incomplete memory-case result")
        if (shared.authenticate_kit(kit, manifest) != kit_pins
                or shared.authenticate_kit(work, manifest) != kit_pins
                or any(shared.digest((work / name).read_bytes()) != pin
                       or shared.digest((ROOT / name).read_bytes()) != pin
                       for name, pin in pins.items())
                or load()[0] != raw):
            raise ValueError("kit or authored source changed during replay")
        (report / "result.json").write_bytes(result_bytes)
        receipt = {
            "schema": "r28-memory-known-case-receipt.v1",
            "runtime": mujoco.__version__,
            "kit_source_sha256": kit_pins,
            "closure_sha256": shared.digest(raw),
            "memory_candidate_sha256": MEMORY_SHA256,
            "result_sha256": shared.digest(result_bytes),
            "episodes": result["episodes"],
            "core_question_passed": result["summary"]["core_question_passed"],
            "classification": "known_disclosed_postmortem_only",
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
    run = commands.add_parser("compare")
    run.add_argument("--kit", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = (bundle(args.out) if args.action == "build-zip" else compare(args)
                 if args.action == "compare" else load()[1])
    except (ValueError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
