"""Authenticated external-kit replay of the frozen R24 physical controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "tools/kit_manifest.sha256"
MANIFEST_SHA256 = "7ea083e92b7fa9fd303ce91a69271f58dd6fb9ace22613039f469a84e2867294"
EVALUATOR_SHA256 = "ee7effc27ac0006c7d2f392523e3e1e4a45626d1113116cba40e00978bd2786f"
CLOSURE = "experiments/r24_transfer_replay/closure.json"


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def has_symlink_component(path):
    absolute = path.absolute()
    return any(item.is_symlink() for item in (absolute, *absolute.parents)
               if item.exists())


def relative_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts
                and "\\" not in name)


def closure():
    path = ROOT / CLOSURE
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular authored closure manifest required")
    document = json.loads(path.read_bytes())
    if document.get("schema") != "r24-transfer-authored-closure.v1":
        raise ValueError("closure schema differs")
    files = document.get("files")
    overlay = document.get("overlay_files")
    public = document.get("public_files")
    if not isinstance(files, dict) or not files or not isinstance(overlay, list) or not public:
        raise ValueError("complete closure lists required")
    if set(files) != set(overlay) | set(public):
        raise ValueError("closure lists do not cover the pinned files exactly")
    for name, expected in files.items():
        source = ROOT / name
        if (not relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py" or len(expected) != 64
                or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("authored closure pin differs: " + name)
    return document


def source_blobs(document):
    blobs = {name: (ROOT / name).read_bytes() for name in document["overlay_files"]}
    pins = {name: digest(value) for name, value in blobs.items()}
    if any(pins[name] != document["files"][name] for name in pins):
        raise ValueError("overlay source changed after closure lock")
    return blobs, pins


def manifest_entries(raw):
    entries = {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        name = name.strip()
        if len(expected) != 64 or not relative_name(name):
            raise ValueError("invalid external-kit manifest")
        entries[name] = expected
    if not entries:
        raise ValueError("empty external-kit manifest")
    return entries


def authenticate_kit(kit, manifest_raw):
    if digest(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("kit manifest identity differs")
    pins = manifest_entries(manifest_raw)
    for name, expected in pins.items():
        path = kit / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("external kit authentication failed: " + name)
    expected_runtime = {name for name in pins if name.startswith(("sim/", "model/"))}
    actual_runtime = {path.relative_to(kit).as_posix()
                      for tree in (kit / "sim", kit / "model")
                      for path in tree.rglob("*") if path.is_file()}
    if expected_runtime != actual_runtime:
        raise ValueError("external kit runtime tree differs")
    evaluator = kit / "evaluate.py"
    if evaluator.is_symlink() or not evaluator.is_file() or digest(evaluator) != EVALUATOR_SHA256:
        raise ValueError("external kit evaluator identity differs")
    return pins


def validate_paths(kit, out):
    if (kit.is_symlink() or not kit.is_dir() or has_symlink_component(kit)
            or not (kit / "sim").is_dir() or not (kit / "model").is_dir()):
        raise ValueError("regular external stock kit required")
    if os.path.lexists(out) or has_symlink_component(out) or not out.parent.is_dir():
        raise ValueError("fresh regular output directory required")
    kit = kit.resolve()
    out = out.parent.resolve() / out.name
    if out.parent.is_relative_to(kit) or out.parent.is_relative_to(ROOT.resolve()):
        raise ValueError("output must be outside kit and public source snapshot")
    return kit, out


def install(kit, overlay, manifest_raw, blobs, pins):
    for tree in ("sim", "model"):
        shutil.copytree(kit / tree, overlay / tree, symlinks=False)
    target_manifest = overlay / MANIFEST
    target_manifest.parent.mkdir(parents=True)
    target_manifest.write_bytes(manifest_raw)
    shutil.copyfile(kit / "evaluate.py", overlay / "evaluate.py")
    for name, blob in blobs.items():
        target = overlay / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        if digest(target) != pins[name]:
            raise ValueError("overlay source copy differs: " + name)


def compact(data):
    normal = []
    for row in data["known_replay"] + data["new_normal"]:
        normal.append({
            "id": row["id"], "kind": row["kind"], "g4": row["g4"],
            "false_alarm_times": {name: row["arms"][name]["false_alarm_times"]
                                  for name in ("r10", "r19", "r24")},
            "r24_support": row["arms"]["r24"].get("support"),
            "max_cost_ratio": {name: max(row["ratios"][name].values())
                               for name in ("r10", "r19")},
            "disclosure": row["disclosure"],
            "known_reproduction": row.get("known_reproduction"),
        })
    impacts = []
    for row in data["late_impulses"]:
        impacts.append({
            "id": row["id"], "collision": row["collision"],
            "strict_pass": row["strict_pass"], "input_integrity": row["input_integrity"],
            "arms": {name: {key: row["arms"][name][key] for key in
                            ("all_rises_s", "twin_all_rises_s", "genuine_rises_s",
                             "first_genuine_delay_ms", "forced_false_rises_s",
                             "twin_false_rises_s", "peak_signed_force_n")}
                     for name in ("r10", "r19", "r24")},
            "candidate_vs": row["candidate_vs"],
        })
    failures = {
        "normal_disclosure": [{"id": row["id"], "failed": [name for name, value
                                in row["disclosure"].items() if not value]}
                               for row in data["new_normal"]],
        "impulse_strict": [{"id": row["id"], "r10": [name for name, value in
                             row["candidate_vs"]["r10"]["gates"].items() if not value],
                             "r19": [name for name, value in
                             row["candidate_vs"]["r19"]["gates"].items() if not value]}
                            for row in data["late_impulses"]],
        "inherited_true_detection": [
            {"id": row["id"],
             "arms_without_genuine_event": [name for name in ("r10", "r19")
                                              if not row["arms"][name]["genuine_rises_s"]]}
            for row in data["late_impulses"]
        ],
        "candidate_true_detection": [
            {"id": row["id"],
             "candidate_missing_genuine_event": not bool(row["arms"]["r24"]["genuine_rises_s"])}
            for row in data["late_impulses"]
        ],
    }
    return {"schema": "r24-transfer-external-replay-summary.v1",
            "classification": data["classification"],
            "known_reproduction_passed": data["known_reproduction_passed"],
            "new_challenge_strict_passed": data["new_challenge_strict_passed"],
            "promotion_eligible": False, "normal": normal, "late_impulses": impacts,
            "failures": failures}


def publish(source, out):
    out.mkdir()
    created = []
    try:
        for path in source.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("regular result file required")
            target = out / path.name
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
            created.append(target)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        out.rmdir()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True,
                        help="external authenticated stock-kit directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="fresh result directory outside source and kit")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    try:
        if any((ROOT / name).exists() for name in ("sim", "model", "evaluate.py")):
            raise ValueError("public source snapshot must be kit-free")
        kit, out = validate_paths(args.kit, args.out)
        document = closure()
        blobs, pins = source_blobs(document)
        manifest_raw = (ROOT / MANIFEST).read_bytes()
        kit_pins = authenticate_kit(kit, manifest_raw)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable",
                           HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                           ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost")
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        with tempfile.TemporaryDirectory(prefix="r24-transfer-replay-", dir=out.parent) as folder:
            work = Path(folder)
            overlay, result = work / "overlay", work / "result"
            overlay.mkdir(); result.mkdir()
            install(kit, overlay, manifest_raw, blobs, pins)
            authenticate_kit(overlay, manifest_raw)
            command = [sys.executable, "-B", "-m", "experiments.r24_transfer_replay.evaluator",
                       "--workers", str(args.workers), "--out", "result.json"]
            completed = subprocess.run(command, cwd=overlay, env=environment, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (result / "run.log").write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError("external replay failed")
            shutil.copyfile(overlay / "result.json", result / "result.json")
            data = json.loads((result / "result.json").read_bytes())
            if data.get("schema") != "r24-transfer-external-replay-result.v1":
                raise ValueError("incomplete replay result")
            summary = compact(data)
            (result / "summary.json").write_text(json.dumps(summary, indent=2,
                                                              allow_nan=False) + "\n")
            authenticate_kit(kit, manifest_raw)
            authenticate_kit(overlay, manifest_raw)
            if any(digest(ROOT / name) != expected for name, expected in document["files"].items()):
                raise ValueError("public authored source changed during replay")
            receipt = {
                "schema": "r24-transfer-authenticated-external-replay.v1",
                "classification": data["classification"], "source_snapshot_kit_free": True,
                "closure_sha256": digest(ROOT / CLOSURE),
                "source_sha256": document["files"],
                "kit_manifest_sha256": digest(manifest_raw), "kit_files": kit_pins,
                "evaluator_sha256": EVALUATOR_SHA256,
                "workers": args.workers,
                "result_sha256": {path.name: digest(path) for path in result.iterdir()
                                  if path.is_file()},
                "known_reproduction_passed": data["known_reproduction_passed"],
                "new_challenge_strict_passed": data["new_challenge_strict_passed"],
                "promotion_eligible": False,
                "limitations": ["finite CPU simulation", "no network required",
                                "not original demo, hidden evaluation, hardware, promotion, or safety proof"],
            }
            (result / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True,
                                                              allow_nan=False) + "\n")
            publish(result, out)
        print(json.dumps({"written": True,
                          "known_reproduction_passed": data["known_reproduction_passed"],
                          "new_challenge_strict_passed": data["new_challenge_strict_passed"]},
                         sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
