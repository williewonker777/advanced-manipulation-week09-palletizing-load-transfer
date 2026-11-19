"""Offline replay of a released R23 holdout lock against an authenticated kit.

The repository snapshot stays kit-free. This is finite laboratory simulation,
not the original course demo, hidden production evaluation, hardware evidence,
hard-real-time proof, or a safety qualification.
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


HOLDOUT = (
    "experiments/r23_holdout/__init__.py",
    "experiments/r23_holdout/plan.json",
    "experiments/r23_holdout/protocol.json",
    "experiments/r23_holdout/evaluator.py",
    "experiments/r23_holdout/verify.py",
)


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
    if (lock.get("schema") != "r23-candidate-lock.v1"
            or lock.get("status") != "LOCKED_BEFORE_HOLDOUT_OUTCOMES"
            or not lock.get("entrypoint", "").startswith("experiments.r23_")
            or not lock.get("entrypoint", "").endswith(":Detector")
            or contains_absolute_string(lock)):
        raise ValueError("invalid or nonportable R23 candidate lock")
    if (lock.get("plan_sha256") != digest(ROOT / HOLDOUT[1])
            or lock.get("protocol_sha256") != digest(ROOT / HOLDOUT[2])
            or lock.get("evaluator_sha256") != digest(ROOT / HOLDOUT[3])):
        raise ValueError("candidate lock belongs to another holdout revision")
    protocol = json.loads((ROOT / HOLDOUT[2]).read_bytes())
    if lock.get("frozen_protocol_sources") != protocol["source_pins"]:
        raise ValueError("candidate lock has another protocol source closure")
    sources = lock.get("transitive_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("candidate transitive source manifest required")
    for name, expected in sources.items():
        rel = Path(name)
        source = ROOT / rel
        if (rel.is_absolute() or ".." in rel.parts
                or not name.startswith("experiments/r23_")
                or len(expected) != 64 or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("candidate source pin differs: " + name)
    return raw, lock


def snapshot(lock):
    protocol = json.loads((ROOT / HOLDOUT[2]).read_bytes())
    repository_protocol = tuple(name for name in protocol["source_pins"]
                                if not name.startswith(("sim/", "model/")))
    names = tuple(dict.fromkeys(HOLDOUT + repository_protocol
                                + tuple(lock["transitive_sources"])))
    blobs, pins = {}, {}
    for name in names:
        rel = Path(name)
        path = ROOT / rel
        if rel.is_absolute() or ".." in rel.parts or path.is_symlink() or not path.is_file():
            raise ValueError("regular public dependency required: " + name)
        blobs[name] = path.read_bytes()
        pins[name] = digest(blobs[name])
    for name in repository_protocol:
        if pins[name] != protocol["source_pins"][name]:
            raise ValueError("sealed protocol dependency differs: " + name)
    for name, expected in lock["transitive_sources"].items():
        if pins[name] != expected:
            raise ValueError("candidate changed after lock: " + name)
    return blobs, pins


def commands(python, repeat):
    base = [python, "-B", "-m", "experiments.r23_holdout.evaluator",
            "--lock", "holdout.lock.json", "--jobs"]
    result = [(base + ["{jobs}", "--out", "first.json"], "first.log")]
    if repeat:
        result.append((base + ["{jobs}", "--out", "repeat.json"], "repeat.log"))
    verify = [python, "-B", "-m", "experiments.r23_holdout.verify",
              "--lock", "holdout.lock.json", "--first", "first.json"]
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
                        help="released portable R23 candidate lock")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh result directory outside kit and repository")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeat", action="store_true",
                        help="run all 216 episodes twice and require exact equality")
    args = parser.parse_args(argv)
    try:
        kit, out = shared.validate_paths(args.kit, args.out)
        lock_raw, lock = read_lock(args.lock)
        manifest_raw = (ROOT / shared.KIT_MANIFEST).read_bytes()
        kit_pins = shared.authenticate_kit(kit, manifest_raw)
        protocol = json.loads((ROOT / HOLDOUT[2]).read_bytes())
        for name, expected in protocol["source_pins"].items():
            if name.startswith(("sim/", "model/")) and digest(kit / name) != expected:
                raise ValueError("authenticated kit runtime differs from sealed protocol: " + name)
        blobs, pins = snapshot(lock)
        environment = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable",
            HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
            ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost",
        )
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="r23-replay-", dir=out.parent) as folder:
            work = Path(folder)
            overlay, result = work / "overlay", work / "result"
            overlay.mkdir()
            result.mkdir()
            shared.install_overlay(kit, overlay, manifest_raw, blobs, pins)
            shared.authenticate_kit(overlay, manifest_raw)
            (overlay / "holdout.lock.json").write_bytes(lock_raw)
            for command, log_name in commands(sys.executable, args.repeat):
                command = [str(args.jobs) if value == "{jobs}" else value
                           for value in command]
                shared.run(command, overlay, environment, result / log_name)
            outputs = (("first.json", "verification.json", "repeat.json")
                       if args.repeat else ("first.json", "verification.json"))
            for name in outputs:
                shutil.copyfile(overlay / name, result / name)
            (result / "candidate.lock.json").write_bytes(lock_raw)
            shared.authenticate_kit(kit, manifest_raw)
            shared.authenticate_kit(overlay, manifest_raw)
            if any(digest(ROOT / name) != expected for name, expected in pins.items()):
                raise ValueError("source dependency changed during replay")
            receipt = {
                "schema": "r23-authenticated-offline-replay.v1",
                "classification": "finite laboratory simulation; not original course demo or safety qualification",
                "entrypoint": lock["entrypoint"],
                "candidate_lock_sha256": digest(lock_raw),
                "kit_manifest_sha256": digest(manifest_raw), "kit_files": kit_pins,
                "holdout_plan_sha256": digest(ROOT / HOLDOUT[1]),
                "holdout_protocol_sha256": digest(ROOT / HOLDOUT[2]),
                "source_dependencies": list(blobs), "source_sha256": pins,
                "repeat_requested": args.repeat,
                "result_sha256": {path.name: digest(path)
                                  for path in result.iterdir() if path.is_file()},
                "limitations": ["predeclared finite CPU simulation", "no network required",
                                "no hard real-time, hardware, hidden production, or safety claim"],
            }
            (result / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
            shared.publish(result, out)
        print(json.dumps({"out": str(out), "entrypoint": lock["entrypoint"],
                          "repeat": args.repeat}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
