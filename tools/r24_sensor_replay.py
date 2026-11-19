"""Replay frozen R24 loaded-sensor panels from a kit-free authored source tree."""
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
CLOSURE = "experiments/r24_portable/sensor_closure.json"
SCIENCE = "experiments/r24_portable/dc_science.sha256"
R23_FREE = "experiments/r24_portable/r23_free_sources.json"
STUDIES = {"target": "negative", "nearby": "nearby", "disclosed": "disclosed"}
ARMS = {
    "cone": "experiments.r24_loaded.memory:Cone",
    "response": "experiments.r24_loaded.memory:Response",
    "dc": "experiments.r24_loaded.dc:Detector",
    "joint": "experiments.r24_composed.detector:Detector",
}


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name)


def has_symlink_component(path):
    absolute = path.absolute()
    return any(item.is_symlink() for item in (absolute, *absolute.parents) if item.exists())


def parse_sha256(raw):
    output = {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        name = name.strip()
        if len(expected) != 64 or not relative_name(name):
            raise ValueError("invalid relative SHA-256 manifest")
        output[name] = expected
    if not output:
        raise ValueError("empty SHA-256 manifest")
    return output


def closure():
    path = ROOT / CLOSURE
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular sensor closure required")
    document = json.loads(path.read_bytes())
    if document.get("schema") != "r24-sensor-authored-closure.v1":
        raise ValueError("sensor closure schema differs")
    files, overlay, public, overlay_map = (document.get(key) for key in
                                           ("files", "overlay_files", "public_files",
                                            "overlay_map"))
    if not isinstance(files, dict) or not files or not isinstance(overlay, list) or not public:
        raise ValueError("complete sensor closure required")
    if set(files) != set(overlay) | set(public):
        raise ValueError("sensor closure roles do not cover pins")
    if not isinstance(overlay_map, dict) or set(overlay_map.values()) - set(overlay):
        raise ValueError("invalid sensor overlay map")
    if (len(set(overlay_map.values())) != len(overlay_map)
            or any(not relative_name(target) for target in overlay_map)
            or any(target.startswith(("sim/", "model/")) or target == "evaluate.py"
                   for target in overlay_map)):
        raise ValueError("unsafe or ambiguous sensor overlay map")
    for name, expected in files.items():
        source = ROOT / name
        if (not relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py" or len(expected) != 64
                or source.is_symlink() or not source.is_file() or digest(source) != expected):
            raise ValueError("sensor authored source pin differs: " + name)
    science = parse_sha256((ROOT / SCIENCE).read_bytes())
    if any(name not in files or files[name] != expected for name, expected in science.items()):
        raise ValueError("frozen R24 DC science differs")
    r23 = json.loads((ROOT / R23_FREE).read_bytes())
    if (r23.get("schema") != "r24-portable-r23-free-source-manifest.v1"
            or r23.get("entrypoint") != "experiments.r23_detached.free:Detector"
            or any(name not in files or files[name] != expected
                   for name, expected in r23.get("sources", {}).items())):
        raise ValueError("R23 FREE source reference differs")
    return document


def source_blobs(document):
    targets = {source: target for target, source in document["overlay_map"].items()}
    blobs = {targets.get(name, name): (ROOT / name).read_bytes()
             for name in document["overlay_files"]}
    return blobs, {name: digest(value) for name, value in blobs.items()}


def authenticate_kit(kit, manifest_raw):
    if digest(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("course-kit manifest identity differs")
    pins = parse_sha256(manifest_raw)
    for name, expected in pins.items():
        path = kit / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("external course kit authentication failed: " + name)
    expected_runtime = {name for name in pins if name.startswith(("sim/", "model/"))}
    actual_runtime = {path.relative_to(kit).as_posix()
                      for tree in (kit / "sim", kit / "model")
                      for path in tree.rglob("*") if path.is_file()}
    if actual_runtime != expected_runtime:
        raise ValueError("external course-kit runtime tree differs")
    evaluator = kit / "evaluate.py"
    if evaluator.is_symlink() or not evaluator.is_file() or digest(evaluator) != EVALUATOR_SHA256:
        raise ValueError("external evaluator identity differs")
    return pins


def validate_paths(kit, out):
    if (kit.is_symlink() or not kit.is_dir() or has_symlink_component(kit)
            or not (kit / "sim").is_dir() or not (kit / "model").is_dir()):
        raise ValueError("regular external course kit required")
    if os.path.lexists(out) or out.parent.is_symlink() or not out.parent.is_dir():
        raise ValueError("fresh output file and regular parent required")
    kit = kit.resolve(); out = out.parent.resolve() / out.name
    if out.parent.is_relative_to(kit) or out.parent.is_relative_to(ROOT.resolve()):
        raise ValueError("output must be outside kit and authored source")
    return kit, out


def install(kit, overlay, manifest_raw, blobs, pins):
    for tree in ("sim", "model"):
        shutil.copytree(kit / tree, overlay / tree, symlinks=False)
    manifest = overlay / MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(manifest_raw)
    shutil.copyfile(kit / "evaluate.py", overlay / "evaluate.py")
    for name, blob in blobs.items():
        target = overlay / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        if digest(target) != pins[name]:
            raise ValueError("sensor overlay copy differs: " + name)


def command(python, study, arm, workers):
    return [python, "-B", "-m", "experiments.r24_loaded.panel",
            "--study", STUDIES[study], "--arm", ARMS[arm],
            "--workers", str(workers), "--out", "panel.json"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True,
                        help="external authenticated course kit")
    parser.add_argument("--out", type=Path, required=True,
                        help="fresh result JSON outside source and kit")
    parser.add_argument("--study", choices=tuple(STUDIES), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), default="dc")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    try:
        if any((ROOT / name).exists() for name in ("sim", "model", "evaluate.py")):
            raise ValueError("authored source snapshot must be kit-free")
        kit, out = validate_paths(args.kit, args.out)
        document = closure(); blobs, pins = source_blobs(document)
        manifest_raw = (ROOT / MANIFEST).read_bytes(); kit_pins = authenticate_kit(kit, manifest_raw)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable",
                           HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                           ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost")
        environment.pop("PYTHONPATH", None); environment.pop("PYTHONHOME", None)
        with tempfile.TemporaryDirectory(prefix="r24-sensor-replay-", dir=out.parent) as folder:
            overlay = Path(folder) / "overlay"; overlay.mkdir()
            install(kit, overlay, manifest_raw, blobs, pins)
            authenticate_kit(overlay, manifest_raw)
            process = subprocess.run(command(sys.executable, args.study, args.arm, args.workers),
                                     cwd=overlay, env=environment, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.returncode:
                raise RuntimeError(process.stderr.strip() or "R24 sensor replay failed")
            data = json.loads((overlay / "panel.json").read_bytes())
            if (data.get("schema") != "week09-r24-observed-comparison.v1"
                    or data.get("study") != STUDIES[args.study]
                    or data.get("arms") != [ARMS[args.arm]]):
                raise ValueError("R24 panel result contract differs")
            authenticate_kit(kit, manifest_raw); authenticate_kit(overlay, manifest_raw)
            if any(digest(ROOT / name) != expected for name, expected in document["files"].items()):
                raise ValueError("authored source changed during sensor replay")
            result = {
                "schema": "r24-authenticated-sensor-replay.v1",
                "classification": "exposed finite research replay; not independent selection",
                "study": args.study, "panel_study": STUDIES[args.study], "arm": args.arm,
                "entrypoint": ARMS[args.arm], "original_course_demo": False,
                "independent_evaluation": False, "default_promoted": False,
                "closure_sha256": digest(ROOT / CLOSURE),
                "source_sha256": document["files"],
                "kit_manifest_sha256": digest(manifest_raw), "kit_files": kit_pins,
                "evaluator_sha256": EVALUATOR_SHA256, "workers": args.workers,
                "panel": data,
            }
            with out.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps({"written": True, "study": args.study, "arm": args.arm,
                          "episodes": data["episodes"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
