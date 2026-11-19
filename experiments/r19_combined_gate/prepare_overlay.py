"""Create one persistent authenticated stock overlay for the R19 regression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import r13_grasp_audit as assembly
from tools import r14_registered_check as r14
from tools import r17_release_replay as prior
from tools import r18_recovery_audit as r18
from tools.r15_early_check import validate_early_calibration, validate_paths
from tools.validate_submission import prepare_overlay


ROOT = Path(__file__).resolve().parents[2]
R19_FILES = (
    "experiments/r19_event_state/__init__.py",
    "experiments/r19_event_state/state.py",
    "experiments/r19_event_state/physical_detector.py",
    "experiments/r19_combined_gate/__init__.py",
    "experiments/r19_combined_gate/policy.py",
    "experiments/r19_combined_gate/study.py",
    "experiments/r19_combined_gate/plan.json",
)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--kit", type=Path, required=True)
    cli.add_argument("--height", type=Path, required=True)
    cli.add_argument("--early", type=Path, required=True)
    cli.add_argument("--overlay", type=Path, required=True)
    cli.add_argument("--auth", type=Path, required=True)
    args = cli.parse_args(argv)
    if args.overlay.exists() or args.auth.exists():
        cli.error("fresh overlay and auth output required")
    kit, height_path, _ = validate_paths(args.kit, args.height, args.auth)
    if args.early.is_symlink() or not args.early.is_file():
        cli.error("regular early calibration required")
    h_blob, e_blob = height_path.read_bytes(), args.early.read_bytes()
    r14_plan = json.loads((ROOT / "experiments/r14_stock_handoff/plan.json").read_bytes())
    r14.validate_plan(r14_plan)
    source_pins = {key: r18.sha((ROOT / value).read_bytes()) for key, value in r14.SOURCE_PATHS.items()}
    _, height = r14.validate_calibration(h_blob, r14_plan, source_pins)
    early = validate_early_calibration(e_blob)
    blobs, pins = prior.snapshots()
    additions = set(r18.ADDITIONS + r18.PROGRESSIVE_FILES + R19_FILES)
    for rel in sorted(additions):
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            raise ValueError("regular source required: " + rel)
        blobs[rel] = path.read_bytes()
        pins[rel] = r18.sha(blobs[rel])
    r18.verify_source(blobs)
    args.overlay.mkdir(parents=True)
    prepare_overlay(kit, args.overlay, "my", "torque")
    for rel, blob in blobs.items():
        target = args.overlay / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        assembly.install_snapshot(blob, pins[rel], target)
    (args.overlay / "height.json").write_text(json.dumps(height, allow_nan=False))
    (args.overlay / "early.json").write_text(json.dumps(early, allow_nan=False))
    (args.overlay / "run_pins.json").write_text(json.dumps(pins, indent=2, allow_nan=False) + "\n")
    auth = {
        "schema": "r19-authenticated-stock-overlay.v1",
        "research_only": True,
        "source_pins": pins,
        "height_source_sha256": r18.sha(h_blob),
        "early_source_sha256": r18.sha(e_blob),
        "kit_manifest_sha256": assembly.digest(args.overlay / "tools/kit_manifest.sha256"),
        "evaluator_sha256": assembly.digest(args.overlay / "evaluate.py"),
        "overlay": str(args.overlay),
    }
    args.auth.parent.mkdir(parents=True, exist_ok=True)
    with args.auth.open("x") as stream:
        json.dump(auth, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sources": len(pins), "kit_manifest": auth["kit_manifest_sha256"],
                      "evaluator": auth["evaluator_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
