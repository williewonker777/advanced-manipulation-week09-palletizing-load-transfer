"""Deterministic offline replay for a finalized R26 independent challenge.

The public source stays kit-free. Execution requires a finalized exposed choice,
candidate lock, authored transitive closure, and a separately supplied authentic
course kit. This is finite laboratory simulation, not hardware or safety proof.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.r26_challenge_portable import support
from tools import r21_replay as shared


def commands(python, repeat):
    base = [python, "-B", "-m", "experiments.r26_holdout.evaluator",
            "--lock", "holdout.lock.json", "--jobs"]
    result = [(base + ["{jobs}", "--out", "first.json"], "first.log")]
    if repeat:
        result.append((base + ["{jobs}", "--out", "repeat.json"], "repeat.log"))
    verify = [python, "-B", "-m", "experiments.r26_holdout.verify",
              "--lock", "holdout.lock.json", "--first", "first.json"]
    if repeat:
        verify += ["--repeat", "repeat.json"]
    verify += ["--out", "verification.json"]
    result.append((verify, "verify.log"))
    return result


def regular_expected(path):
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular authoritative result required")
    raw = path.read_bytes()
    document = json.loads(raw)
    if (document.get("schema") != "r26-independent-long-close-result.v1"
            or document.get("episodes") != 156):
        raise ValueError("authoritative R26 result contract differs")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path,
                        help="licensed original course kit directory (read-only)")
    parser.add_argument("--selection", required=True, type=Path,
                        help="finalized exposed R26 choice inside this source root")
    parser.add_argument("--lock", required=True, type=Path,
                        help="finalized portable R26 candidate lock")
    parser.add_argument("--closure", required=True, type=Path,
                        help="finalized authored R26 source closure")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh result directory outside kit and source")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeat", action="store_true",
                        help="run all 156 episodes twice and require exact equality")
    parser.add_argument("--expected-result", type=Path,
                        help="optional authoritative result requiring byte identity")
    args = parser.parse_args(argv)
    try:
        kit, out = shared.validate_paths(args.kit, args.out)
        versions = support.runtime_versions()
        requirements = kit / "requirements.txt"
        if requirements.is_symlink() or not requirements.is_file():
            raise ValueError("regular course requirements.txt required")
        requirements_raw = requirements.read_bytes()
        if "mujoco==3.17.0" not in requirements_raw.decode("utf-8").splitlines():
            raise ValueError("course requirements do not pin MuJoCo 3.17.0")
        release = support.read_release(args.selection, args.lock, args.closure)
        expected_raw = regular_expected(args.expected_result)
        manifest_raw = (ROOT / shared.KIT_MANIFEST).read_bytes()
        kit_pins = shared.authenticate_kit(kit, manifest_raw)
        protocol = json.loads((ROOT / support.HOLDOUT[4]).read_bytes())
        for name, expected in protocol["source_pins"].items():
            if name.startswith(("sim/", "model/")) and support.digest(kit / name) != expected:
                raise ValueError("authenticated kit runtime differs from R26 protocol: " + name)
        blobs, pins = support.snapshot(release)
        environment = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable",
            HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
            ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost",
        )
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="r26-replay-", dir=out.parent) as folder:
            work = Path(folder)
            overlay, result = work / "overlay", work / "result"
            overlay.mkdir()
            result.mkdir()
            shared.install_overlay(kit, overlay, manifest_raw, blobs, pins)
            shared.authenticate_kit(overlay, manifest_raw)
            (overlay / "holdout.lock.json").write_bytes(release["lock_raw"])
            for command, log_name in commands(sys.executable, args.repeat):
                command = [str(args.jobs) if value == "{jobs}" else value
                           for value in command]
                shared.run(command, overlay, environment, result / log_name)
            outputs = (("first.json", "verification.json", "repeat.json")
                       if args.repeat else ("first.json", "verification.json"))
            for name in outputs:
                shutil.copyfile(overlay / name, result / name)
            (result / "candidate.lock.json").write_bytes(release["lock_raw"])
            (result / "selection.json").write_bytes(release["selection_raw"])
            if expected_raw is not None and (result / "first.json").read_bytes() != expected_raw:
                raise ValueError("clean R26 replay differs from authoritative result")
            shared.authenticate_kit(kit, manifest_raw)
            shared.authenticate_kit(overlay, manifest_raw)
            if any(support.digest(ROOT / name) != expected
                   for name, expected in pins.items()):
                raise ValueError("R26 source dependency changed during replay")
            scientific = json.loads((result / "first.json").read_bytes())
            verification = json.loads((result / "verification.json").read_bytes())
            receipt = {
                "schema": "r26-authenticated-offline-replay.v1",
                "classification": "finite laboratory simulation; not hardware or safety qualification",
                "entrypoint": release["entrypoint"],
                "selection_sha256": support.digest(release["selection_raw"]),
                "candidate_lock_sha256": support.digest(release["lock_raw"]),
                "authored_closure_sha256": support.digest(args.closure),
                "kit_manifest_sha256": support.digest(manifest_raw),
                "kit_files": kit_pins,
                "course_requirements_sha256": support.digest(requirements_raw),
                "runtime_versions": versions,
                "source_dependencies": list(blobs),
                "source_sha256": pins,
                "episodes": scientific["episodes"],
                "scientific_passed": scientific["passed"],
                "scientific_claims": scientific["claims"],
                "verified_summary": verification["summary"],
                "process_exit_success": True,
                "repeat_requested": args.repeat,
                "expected_result_checked": expected_raw is not None,
                "result_sha256": {path.name: support.digest(path)
                                  for path in result.iterdir() if path.is_file()},
                "limitations": ["predeclared finite CPU simulation", "no network required",
                                "no hard real-time, hardware, production, or safety claim"],
            }
            (result / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
            shared.publish(result, out)
        print(json.dumps({"out": str(out), "entrypoint": release["entrypoint"],
                          "episodes": scientific["episodes"],
                          "scientific_passed": scientific["passed"],
                          "repeat": args.repeat}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
