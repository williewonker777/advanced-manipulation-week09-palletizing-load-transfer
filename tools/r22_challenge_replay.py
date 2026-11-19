"""Offline replay of a released R22 research lock against an authenticated local kit.

This is a laboratory simulation replay, not the original course demo, a hidden
evaluation, hardware validation, or a safety certificate.  It never writes to
the supplied course kit or the source/distribution repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r21_replay as shared


CHALLENGE = (
    "experiments/r22_challenge/__init__.py",
    "experiments/r22_challenge/plan.json",
    "experiments/r22_challenge/protocol.json",
    "experiments/r22_challenge/evaluator.py",
    "experiments/r22_challenge/verify.py",
)

# The evaluator imports one function from the R14 tool, whose module-level
# dependency graph is already the authenticated public R21 surface replay set.
PUBLIC_SUPPORT = tuple(dict.fromkeys(shared.COMMON + shared.SURFACE_EXTRA))


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def contains_absolute_string(value):
    if isinstance(value, dict):
        return any(contains_absolute_string(key) or contains_absolute_string(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(contains_absolute_string(item) for item in value)
    return isinstance(value, str) and (value.startswith("/") or "\\Users\\" in value)


def read_lock(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular archived candidate lock required")
    raw = path.read_bytes()
    lock = json.loads(raw)
    if (lock.get("schema") != "r22-candidate-lock.v1"
            or lock.get("status") != "LOCKED_BEFORE_CHALLENGE_OUTCOMES"
            or not lock.get("entrypoint", "").startswith("experiments.r22_")
            or not lock.get("entrypoint", "").endswith(":Detector")
            or contains_absolute_string(lock)):
        raise ValueError("invalid or nonportable candidate lock")
    if lock.get("plan_sha256") != digest(ROOT / CHALLENGE[1]) \
            or lock.get("evaluator_sha256") != digest(ROOT / CHALLENGE[3]):
        raise ValueError("candidate lock belongs to another challenge revision")
    sources = lock.get("candidate_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("candidate source manifest required")
    for name, expected in sources.items():
        rel = Path(name)
        source = ROOT / rel
        if (rel.is_absolute() or ".." in rel.parts or not name.startswith("experiments/r22_")
                or len(expected) != 64 or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("candidate source pin differs: " + name)
    return raw, lock


def snapshot(lock):
    protocol = json.loads((ROOT / CHALLENGE[2]).read_bytes())
    repository_protocol = tuple(name for name in protocol["source_pins"]
                                if not name.startswith(("sim/", "model/")))
    names = tuple(dict.fromkeys(PUBLIC_SUPPORT + CHALLENGE +
                                repository_protocol +
                                tuple(lock["candidate_sources"])))
    blobs, pins = {}, {}
    for name in names:
        rel = Path(name)
        path = ROOT / rel
        if rel.is_absolute() or ".." in rel.parts or path.is_symlink() or not path.is_file():
            raise ValueError("regular public dependency required: " + name)
        blobs[name] = path.read_bytes()
        pins[name] = digest(blobs[name])
    for name in repository_protocol:
        expected = protocol["source_pins"][name]
        if pins[name] != expected:
            raise ValueError("sealed protocol dependency differs: " + name)
    for name, expected in lock["candidate_sources"].items():
        if pins[name] != expected:
            raise ValueError("candidate changed after lock: " + name)
    return blobs, pins


def commands(python, repeat):
    base = [python, "-B", "-m", "experiments.r22_challenge.evaluator",
            "--lock", "challenge.lock.json", "--jobs"]
    result = [(base + ["{jobs}", "--out", "first.json"], "first.log")]
    if repeat:
        result.append((base + ["{jobs}", "--out", "repeat.json"], "repeat.log"))
    verify = [python, "-B", "-m", "experiments.r22_challenge.verify",
              "--lock", "challenge.lock.json", "--first", "first.json"]
    if repeat:
        verify += ["--repeat", "repeat.json"]
    verify += ["--out", "verification.json"]
    result.append((verify, "verify.log"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path,
                        help="licensed original icpbl_sim directory (read-only)")
    parser.add_argument("--lock", required=True, type=Path,
                        help="released portable candidate lock JSON")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh result directory outside kit and repository")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeat", action="store_true",
                        help="run all 486 episodes twice and require exact non-timing equality")
    args = parser.parse_args(argv)
    try:
        kit, out = shared.validate_paths(args.kit, args.out)
        lock_raw, lock = read_lock(args.lock)
        manifest_raw = (ROOT / shared.KIT_MANIFEST).read_bytes()
        kit_pins = shared.authenticate_kit(kit, manifest_raw)
        protocol = json.loads((ROOT / CHALLENGE[2]).read_bytes())
        for name, expected in protocol["source_pins"].items():
            if name.startswith(("sim/", "model/")) and digest(kit / name) != expected:
                raise ValueError("authenticated kit runtime differs from sealed protocol: " + name)
        blobs, pins = snapshot(lock)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable",
                           HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                           ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost")
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="r22-replay-", dir=out.parent) as folder:
            work = Path(folder)
            overlay, result = work / "overlay", work / "result"
            overlay.mkdir(); result.mkdir()
            shared.install_overlay(kit, overlay, manifest_raw, blobs, pins)
            shared.authenticate_kit(overlay, manifest_raw)
            (overlay / "challenge.lock.json").write_bytes(lock_raw)
            for command, log_name in commands(sys.executable, args.repeat):
                command = [str(args.jobs) if value == "{jobs}" else value for value in command]
                shared.run(command, overlay, environment, result / log_name)
            outputs = ("first.json", "verification.json") + (("repeat.json",) if args.repeat else ())
            for name in outputs:
                shutil.copyfile(overlay / name, result / name)
            (result / "candidate.lock.json").write_bytes(lock_raw)
            shared.authenticate_kit(kit, manifest_raw)
            shared.authenticate_kit(overlay, manifest_raw)
            if any(digest(ROOT / name) != expected for name, expected in pins.items()):
                raise ValueError("source dependency changed during replay")
            receipt = {"schema": "r22-authenticated-offline-replay.v1",
                       "classification": "finite laboratory simulation; not official course demo or safety proof",
                       "entrypoint": lock["entrypoint"], "candidate_lock_sha256": digest(lock_raw),
                       "kit_manifest_sha256": digest(manifest_raw), "kit_files": kit_pins,
                       "evaluator_sha256": shared.EVALUATOR_SHA256,
                       "challenge_plan_sha256": digest(ROOT / CHALLENGE[1]),
                       "source_dependencies": list(blobs), "source_sha256": pins,
                       "repeat_requested": args.repeat,
                       "result_sha256": {path.name: digest(path) for path in result.iterdir() if path.is_file()},
                       "limitations": ["predeclared finite CPU simulation", "no network required",
                                       "no hard real-time, hardware, hidden-test or safety claim"]}
            (result / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                                             allow_nan=False) + "\n")
            shared.publish(result, out)
        print(json.dumps({"out": str(out), "entrypoint": lock["entrypoint"],
                          "repeat": args.repeat}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
