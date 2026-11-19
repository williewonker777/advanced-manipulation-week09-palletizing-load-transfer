"""Replay exposed R25 detector studies with an authenticated external kit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import r21_replay as shared


CLOSURE = "experiments/r25_portable/exposed_closure.json"
COHORTS = ("target", "stress", "nearby", "r23", "r24")
ARMS = {
    "stream": "experiments.r25_fragment.stream:Detector",
    "pickup": "experiments.r25_pickup.candidate:Detector",
    "joint": "experiments.r25_composed.detector:Detector",
}


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts
                and "\\" not in name)


def closure():
    path = ROOT / CLOSURE
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular R25 exposed closure required")
    document = json.loads(path.read_bytes())
    files, overlay, public = (document.get(key) for key in
                              ("files", "overlay_files", "public_files"))
    if (document.get("schema") != "r25-exposed-authored-closure.v1"
            or not isinstance(files, dict) or not files
            or not isinstance(overlay, list) or not isinstance(public, list)
            or set(files) != set(overlay) | set(public)):
        raise ValueError("R25 exposed closure differs")
    for name, expected in files.items():
        source = ROOT / name
        if (not relative_name(name) or name.startswith(("sim/", "model/"))
                or name == "evaluate.py" or len(expected) != 64
                or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("R25 exposed source pin differs: " + name)
    return document


def source_blobs(document):
    blobs = {name: (ROOT / name).read_bytes() for name in document["overlay_files"]}
    pins = {name: digest(value) for name, value in blobs.items()}
    if any(pins[name] != document["files"][name] for name in pins):
        raise ValueError("R25 source changed after closure validation")
    return blobs, pins


def command(python, cohort, arm, workers):
    return [python, "-B", "-m", "experiments.r25_fragment.study",
            "--cohort", cohort, "--arm", ARMS[arm],
            "--workers", str(workers), "--out", "study.json"]


def validate_study(study, cohort, candidate):
    """Validate provenance/completeness, never turn a scientific loss into CLI failure."""
    if (study.get("schema") != "r25-disclosed-original-simulator-comparison.v1"
            or study.get("cohort") != cohort
            or study.get("arms") != [ARMS[candidate]]
            or study.get("independent") is not False
            or type(study.get("episodes")) is not int or study["episodes"] <= 0
            or not isinstance(study.get("regressions"), dict)):
        raise ValueError("R25 exposed study result contract differs")
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path,
                        help="authenticated external course kit")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh result JSON outside source and kit")
    parser.add_argument("--study", required=True, choices=COHORTS)
    parser.add_argument("--candidate", required=True, choices=tuple(ARMS))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    try:
        if any((ROOT / name).exists() for name in ("sim", "model", "evaluate.py")):
            raise ValueError("public R25 source snapshot must be kit-free")
        kit, out = shared.validate_paths(args.kit, args.out)
        document = closure()
        blobs, pins = source_blobs(document)
        manifest_raw = (ROOT / shared.KIT_MANIFEST).read_bytes()
        kit_pins = shared.authenticate_kit(kit, manifest_raw)
        environment = dict(
            os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable",
            HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
            ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost",
        )
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        with tempfile.TemporaryDirectory(prefix="r25-replay-", dir=out.parent) as folder:
            overlay = Path(folder) / "overlay"
            overlay.mkdir()
            shared.install_overlay(kit, overlay, manifest_raw, blobs, pins)
            shared.authenticate_kit(overlay, manifest_raw)
            completed = subprocess.run(
                command(sys.executable, args.study, args.candidate, args.workers),
                cwd=overlay, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip() or "R25 study replay failed")
            study = validate_study(json.loads((overlay / "study.json").read_bytes()),
                                   args.study, args.candidate)
            shared.authenticate_kit(kit, manifest_raw)
            shared.authenticate_kit(overlay, manifest_raw)
            if any(digest(ROOT / name) != expected
                   for name, expected in document["files"].items()):
                raise ValueError("R25 authored source changed during replay")
            result = {
                "schema": "r25-authenticated-exposed-replay.v1",
                "classification": "finite exposed research replay; unselected and not independent",
                "study": args.study, "candidate": args.candidate,
                "entrypoint": ARMS[args.candidate], "episodes": study["episodes"],
                "selected": False, "independent_evaluation": False,
                "default_promoted": False, "original_course_demo": False,
                "closure_sha256": digest(ROOT / CLOSURE),
                "source_sha256": document["files"],
                "kit_manifest_sha256": digest(manifest_raw), "kit_files": kit_pins,
                "workers": args.workers, "study_result": study,
            }
            with out.open("x") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        print(json.dumps({"written": True, "study": args.study,
                          "candidate": args.candidate, "episodes": study["episodes"]},
                         sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
