"""Portable authenticated progressive R16 ORIGINAL-kit qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import r13_grasp_audit as assembly
from tools import r14_registered_check as r14
from tools.r15_early_check import validate_early_calibration, validate_paths


COMMON_DEPENDENCIES = (
    "tools/r16_handoff_check.py",
    "r1_torque.py", "my_detector.py",
    "experiments/r12_hybrid/controller.py",
    "experiments/r14_online/__init__.py", "experiments/r14_online/stream.py",
    "experiments/r14_airborne/registration.py", "experiments/r14_airborne/plan.json",
    "experiments/r14_stock_handoff/__init__.py",
    "experiments/r14_stock_handoff/controller.py",
    "experiments/r14_stock_handoff/detector.py",
    "experiments/r14_stock_handoff/geometry.py",
    "experiments/r14_stock_handoff/qualification.py",
    "experiments/r15_early_com/__init__.py",
    "experiments/r15_early_com/stream.py",
    "experiments/r15_early_com/sensing.py",
    "experiments/r15_early_com/controller.py",
    "experiments/r16_handoff_check/__init__.py",
    "experiments/r16_handoff_check/qualification.py",
    "tools/r13_grasp_estimator.py", "tools/r11_placement.py",
    "tools/r14_paired_events.py", "tools/r5_metrics.py",
    "models/r12/counterfactual_coefficients.json",
)
LAW_CONFIG = {
    "release": {
        "controller": "experiments/r16_release/controller.py",
        "candidate_plan": "experiments/r16_release/plan.json",
        "plan": "experiments/r16_handoff_check/plan_release.json",
    },
    "normal-phase": {
        "controller": "experiments/r16_normal_phase/controller.py",
        "candidate_plan": "experiments/r16_normal_phase/plan.json",
        "plan": "experiments/r16_handoff_check/plan_normal_phase.json",
    },
    "balanced": {
        "controller": "experiments/r16_balanced/controller.py",
        "candidate_plan": "experiments/r16_balanced/plan.json",
        "plan": "experiments/r16_handoff_check/plan_balanced.json",
        "extra_dependencies": (
            "experiments/r16_balanced/coefficients.json",
            "experiments/r16_release/controller.py",
            "experiments/r16_release/plan.json",
            "experiments/r16_normal_phase/controller.py",
            "experiments/r16_normal_phase/plan.json",
        ),
    },
    "payload-contract": {
        "controller": "experiments/r16_payload_contract/controller.py",
        "candidate_plan": "experiments/r16_payload_contract/plan.json",
        "plan": "experiments/r16_handoff_check/plan_payload_contract.json",
        "expected_counts": (28, 6, 7),
        "extra_dependencies": (
            "experiments/r16_balanced/controller.py",
            "experiments/r16_balanced/coefficients.json",
            "experiments/r16_balanced/plan.json",
            "experiments/r16_release/controller.py",
            "experiments/r16_release/plan.json",
            "experiments/r16_normal_phase/controller.py",
            "experiments/r16_normal_phase/plan.json",
        ),
    },
}
SCOPE = "ORIGINAL stock progressive qualification; negative result exits zero"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def dependencies(law):
    config = LAW_CONFIG[law]
    values = (COMMON_DEPENDENCIES + (config["controller"], config["candidate_plan"], config["plan"])
              + tuple(config.get("extra_dependencies", ())))
    if len(values) != len(set(values)) or any(Path(value).name == "run.py" for value in values):
        raise ValueError("public dependency list must be unique and neutrally named")
    return values


def snapshot_sources(law):
    paths = {relative: ROOT / relative for relative in dependencies(law)}
    missing = [name for name, path in paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError("regular frozen public dependencies required: " + ", ".join(missing))
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    return blobs, {name: digest(blob) for name, blob in blobs.items()}


def make_auth(law, pins, height_blob, early_blob):
    config = LAW_CONFIG[law]
    return {"schema": "r16-progressive-auth.v1", "source_pins": pins,
            "law": law, "candidate_path": config["controller"],
            "candidate_sha256": pins[config["controller"]],
            "candidate_plan_path": config["candidate_plan"],
            "candidate_plan_sha256": pins[config["candidate_plan"]],
            "supporting_pins": {path: pins[path]
                                for path in config.get("extra_dependencies", ())},
            "plan_path": config["plan"], "plan_sha256": pins[config["plan"]],
            "height_sha256": digest(height_blob), "early_sha256": digest(early_blob)}


def validate_law_plan(document, law):
    expected_keys = {"schema", "status", "scope", "thresholds", "exposed_cases",
                     "fresh_cases", "impact_cases", "law", "candidate_module"}
    expected_module = LAW_CONFIG[law]["controller"].removesuffix(".py").replace("/", ".")
    if (set(document) != expected_keys
            or document.get("schema") != "r16-progressive-handoff-plan.v1"
            or document.get("status") != "FROZEN_BEFORE_R16_CANDIDATE_OUTCOMES"
            or document.get("law") != law
            or document.get("candidate_module") != expected_module
            or not isinstance(document.get("thresholds"), dict)
            or not document["thresholds"]):
        raise ValueError("invalid law-specific frozen plan")
    identifiers = []
    counts = LAW_CONFIG[law].get("expected_counts", (22, 6, 7))
    expected_counts = dict(zip(("exposed_cases", "fresh_cases", "impact_cases"), counts))
    for key in ("exposed_cases", "fresh_cases", "impact_cases"):
        cases = document.get(key)
        if not isinstance(cases, list) or len(cases) != expected_counts[key]:
            raise ValueError("incomplete law-specific case stage")
        identifiers.extend(case.get("id") for case in cases if isinstance(case, dict))
    total = sum(len(document[key]) for key in ("exposed_cases", "fresh_cases", "impact_cases"))
    if (len(identifiers) != total
            or any(not isinstance(value, str) or not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError("missing or duplicate law-specific case id")
    return document


def validate_upstream_wrapper(document, expected_stage, auth):
    result = document.get("result")
    plan_path = ROOT / auth["plan_path"]
    if digest(plan_path.read_bytes()) != auth["plan_sha256"]:
        raise ValueError("authenticated plan pin changed")
    plan = json.loads(plan_path.read_text())
    expected_ids = {case["id"] for case in plan[expected_stage + "_cases"]}
    rows = result.get("rows") if isinstance(result, dict) else None
    if (document.get("schema") != "r16-portable-handoff-check.v1"
            or document.get("scope") != SCOPE
            or document.get("law") != auth["law"]
            or document.get("pins") != auth["source_pins"]
            or not isinstance(result, dict)
            or result.get("schema") != "r16-progressive-stage.v1"
            or result.get("stage") != expected_stage
            or result.get("complete") is not True or result.get("passed") is not True
            or result.get("auth") != auth
            or not isinstance(rows, list) or len(rows) != len(expected_ids)
            or len({row.get("id") for row in rows}) != len(rows)
            or {row.get("id") for row in rows} != expected_ids
            or any(row.get("passed") is not True
                   or not isinstance(row.get("gates"), dict)
                   or not row["gates"]
                   or not all(value is True for value in row["gates"].values()) for row in rows)
            or (expected_stage == "fresh" and
                (not isinstance(result.get("unloaded"), dict)
                 or result["unloaded"].get("passed") is not True
                 or not isinstance(result.get("upstream_sha256"), str)
                 or not isinstance(document.get("upstream_wrapper_sha256"), str)))):
        raise ValueError("complete passing authenticated upstream wrapper required")
    return result


def run_checked(command, cwd, env):
    completed = subprocess.run(command, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"authored R16 stage failed ({completed.returncode}):\n{completed.stderr}")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("exposed", "fresh", "impact"), required=True)
    parser.add_argument("--law", choices=tuple(LAW_CONFIG), default="release")
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--height-calibration", required=True, type=Path)
    parser.add_argument("--early-calibration", type=Path,
                        default=ROOT / "results/r15/early_stock_calibration.json")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        kit, height_path, out = validate_paths(args.kit, args.height_calibration, args.out)
        early_path = args.early_calibration.resolve()
        if early_path.is_symlink() or not early_path.is_file():
            raise ValueError("regular immutable early calibration required")
        height_blob, early_blob = height_path.read_bytes(), early_path.read_bytes()
        plan = json.loads((ROOT / "experiments/r14_stock_handoff/plan.json").read_text())
        r14.validate_plan(plan)
        r14_pins = {key: digest((ROOT / relative).read_bytes())
                    for key, relative in r14.SOURCE_PATHS.items()}
        height_document, height = r14.validate_calibration(height_blob, plan, r14_pins)
        early = validate_early_calibration(early_blob)
        blobs, pins = snapshot_sources(args.law)
        validate_law_plan(json.loads(blobs[LAW_CONFIG[args.law]["plan"]]), args.law)
        auth = make_auth(args.law, pins, height_blob, early_blob)
        upstream = None
        if args.stage == "exposed":
            if args.upstream is not None:
                raise ValueError("exposed stage accepts no upstream")
        else:
            if args.upstream is None or args.upstream.is_symlink() or not args.upstream.is_file():
                raise ValueError("regular upstream result required")
            upstream_document = json.loads(args.upstream.read_text())
            upstream = validate_upstream_wrapper(
                upstream_document, "exposed" if args.stage == "fresh" else "fresh", auth)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="disable")
    environment.pop("PYTHONPATH", None)
    try:
        with tempfile.TemporaryDirectory(prefix=f"r16-{args.stage}-", dir=out.parent) as folder:
            overlay = Path(folder)
            from tools.validate_submission import prepare_overlay
            prepare_overlay(kit, overlay, "my", "torque")
            for relative, blob in blobs.items():
                destination = overlay / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                assembly.install_snapshot(blob, pins[relative], destination)
            (overlay / "height.json").write_text(json.dumps(height, allow_nan=False))
            (overlay / "early.json").write_text(json.dumps(early, allow_nan=False))
            (overlay / "auth.json").write_text(json.dumps(auth, allow_nan=False))
            command = [sys.executable, "-B", "-m", "experiments.r16_handoff_check.qualification",
                       "--stage", args.stage, "--law", args.law, "--plan", auth["plan_path"],
                       "--height", "height.json", "--early", "early.json",
                       "--auth", "auth.json", "--out", "stage.json"]
            if upstream is not None:
                (overlay / "upstream.json").write_text(json.dumps(upstream, allow_nan=False))
                command.extend(("--upstream", "upstream.json"))
            if args.case is not None:
                command.extend(("--case", args.case))
            completed = run_checked(command, overlay, environment)
            result = json.loads((overlay / "stage.json").read_text())
            if result.get("auth") != auth or result.get("stage") != args.stage:
                raise ValueError("authored stage output failed authentication")
        for relative, pin in pins.items():
            if digest((ROOT / relative).read_bytes()) != pin:
                raise ValueError("snapshotted dependency mutated during execution")
        document = {
            "schema": "r16-portable-handoff-check.v1", "stage": args.stage, "law": args.law,
            "scope": SCOPE,
            "pins": pins,
            "calibration_provenance": {
                "height_sha256": digest(height_blob), "height_source_pins": height_document["pins"],
                "early_sha256": digest(early_blob),
                "early_source": "byte-pinned model derived from 12 stock public normal traces"},
            "upstream_wrapper_sha256": None if args.upstream is None else digest(args.upstream.read_bytes()),
            "runner_stdout": completed.stdout.strip(), "result": result,
            "limitations": ["finite stock simulation qualification", "diagnostic --case is incomplete",
                            "no external impact stage unless both full normal stages pass",
                            "qualified=false is a valid exit-zero result"],
        }
        with out.open("x") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"stage": args.stage, "complete": result["complete"],
                      "passed": result["passed"], "failed": result["failed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
