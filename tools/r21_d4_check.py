"""Authenticated offline replay of the exposed full R21 D4 protection census.

This is a replay, not a blind benchmark or safety certificate.  It installs
only a pinned relative dependency set beside an authenticated local ORIGINAL
kit, runs the frozen full-protection and exact-input audit modules with ``-B``,
and compares deterministic scientific accounting (never timing) with the
checked-in exposed result summary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from tools import r21_replay as shared


ROOT = Path(__file__).resolve().parents[1]
PLAN = "experiments/r21_full_protection/plan_v2.json"
RUNNER = "experiments/r21_full_protection/runner_v2.py"
AUDIT_PLAN = "experiments/r21_full_protection/force_audit_plan.json"
AUDIT = "experiments/r21_full_protection/force_audit.py"
EXPECTED = "experiments/r21_full_protection/summary.json"
DEPENDENCIES = (
    "my_detector.py", "my_torque.py",
    "tools/check_integrity.py", "tools/d4_robustness.py", "tools/sweep.py",
    "experiments/r20_horizontal/__init__.py",
    "experiments/r20_horizontal/candidate_detector.py",
    "experiments/r21_yaw_impulse/__init__.py",
    "experiments/r21_yaw_impulse/state.py",
    "experiments/r21_yaw_impulse/detector.py",
    "experiments/r21_yaw_impulse/frozen.py",
    "experiments/r21_yaw_impulse/coarse.py",
    "experiments/r21_yaw_impulse/normal_bins.json",
    "experiments/r21_full_protection/__init__.py",
    PLAN, RUNNER, AUDIT_PLAN, AUDIT, EXPECTED,
)
SCIENCE_KEYS = (
    "census", "raw_full_census_not_compound_gate_certified",
    "confirmed_policy_comparison_failures", "collision_false_event_audit",
    "original_compound_gate_failures", "fully_v2_gate_passing_collision_subset",
    "separate_exact_stock_input_audit", "prefix_and_boundary_facts",
)


def snapshot():
    blobs = {}
    for relative in DEPENDENCIES:
        path = ROOT / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("dependency manifest must be portable and relative")
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular D4 dependency required: " + relative)
        blobs[relative] = path.read_bytes()
    return blobs, {name: shared.digest(blob) for name, blob in blobs.items()}


def commands(python, workers):
    return (
        ([python, "-B", "-m", "experiments.r21_full_protection.runner_v2",
          "--plan", PLAN, "--out", "d4_raw.json", "--workers", str(workers)],
         "d4_raw.log"),
        ([python, "-B", "-m", "experiments.r21_full_protection.force_audit",
          "--plan", AUDIT_PLAN, "--out", "force_audit.json", "--workers", str(workers)],
         "force_audit.log"),
    )


def _detail(row):
    case = row["case"]
    return {
        "case_id": case["case_id"], "kind": case["kind"], "mass_kg": case["mass"],
        "com_m": case["com"], "phase": case["phase"], "direction": case["direction"],
        "force_n": case["force_n"], "collision": case["collision"],
        "r10": {"strict": row["arms"]["r10"]["strict_outcome"],
                "rises": row["arms"]["r10"]["events"],
                "duplicates": row["arms"]["r10"]["duplicate_attributable_rises"]},
        "yaw": {"strict": row["arms"]["yaw"]["strict_outcome"],
                "rises": row["arms"]["yaw"]["events"],
                "duplicates": row["arms"]["yaw"]["duplicate_attributable_rises"]},
        "failed_comparison_gates": [name for name, passed in row["comparison"].items()
                                    if not passed],
    }


def scientific_accounting(document, audit):
    if (document.get("schema") != "r21-full-d4-protection-result.v1"
            or document.get("cases_per_arm") != 385 or document.get("episodes") != 770
            or document.get("arms") != ["r10", "yaw"]):
        raise ValueError("incomplete full D4 result")
    if audit.get("schema") != "r21-full-d4-force-audit.v1" or audit.get("cases") != 180:
        raise ValueError("incomplete exact-input audit")
    rows = document["rows"]
    if len(rows) != 385 or {row["case"]["case_id"] for row in rows} != set(range(385)):
        raise ValueError("D4 case matrix/result IDs differ")
    collisions = [row for row in rows if row["case"]["kind"] == "collision"]
    fully_gated = [row for row in collisions
                   if all(all(gates.values()) for gates in row["arm_gates"].values())]
    diagnostic_failed = {row["case"]["case_id"] for row in rows
                         if any(not all(gates.values()) for gates in row["arm_gates"].values())}
    arm_failures = Counter()
    for row in rows:
        for arm, gates in row["arm_gates"].items():
            for gate, passed in gates.items():
                if not passed:
                    arm_failures[f"{arm}.{gate}"] += 1
    regressions = [row for row in rows if not all(row["comparison"].values())]
    kinds = Counter(row["case"]["kind"] for row in rows)
    source_unchanged = document["source_pins_before"] == document["source_pins_after"]
    return {
        "census": {
            "cases_per_arm": 385, "episodes": 770, "reference": kinds["reference"],
            "collision": kinds["collision"], "offset": kinds["offset"],
            "source_pins_unchanged": source_unchanged,
            "kit_integrity": document["kit_integrity"],
        },
        "raw_full_census_not_compound_gate_certified": {
            "r10": document["summary"]["r10"], "yaw": document["summary"]["yaw"],
            "strict_candidate_only_gain_case_ids": document["summary"]["strict_candidate_only_gains"],
            "strict_loss_case_ids": document["summary"]["strict_losses"],
        },
        "confirmed_policy_comparison_failures": {
            "case_count": len(regressions),
            "case_ids": [row["case"]["case_id"] for row in regressions],
            "new_no_force_false_case_ids": document["summary"]["new_false_case_ids"],
            "new_duplicate_case_ids": document["summary"]["new_duplicate_case_ids"],
            "details": [_detail(row) for row in regressions],
        },
        "collision_false_event_audit": {
            arm: [{"case_id": row["case"]["case_id"],
                   "events": row["arms"][arm]["false_events_vs_matched_no_force"],
                   "delayed_indices": row["arms"][arm]["delayed_false_indices"]}
                  for row in collisions if row["arms"][arm]["false_rise_indices"]]
            for arm in ("r10", "yaw")
        },
        "original_compound_gate_failures": {
            "failed_case_rows": len(document["summary"]["failed_case_ids"]),
            "diagnostic_failed_case_rows": len(diagnostic_failed),
            "per_arm_gate_failure_counts": dict(sorted(arm_failures.items())),
        },
        "fully_v2_gate_passing_collision_subset": {
            "cases": len(fully_gated),
            "case_ids": [row["case"]["case_id"] for row in fully_gated],
            "r10_strict": dict(Counter(row["arms"]["r10"]["strict_outcome"]
                                       for row in fully_gated)),
            "yaw_strict": dict(Counter(row["arms"]["yaw"]["strict_outcome"]
                                       for row in fully_gated)),
            "strict_gain_case_ids": [row["case"]["case_id"] for row in fully_gated
                                     if row["arms"]["r10"]["strict_outcome"] != "detected"
                                     and row["arms"]["yaw"]["strict_outcome"] == "detected"],
            "strict_loss_case_ids": [row["case"]["case_id"] for row in fully_gated
                                     if row["arms"]["r10"]["strict_outcome"] == "detected"
                                     and row["arms"]["yaw"]["strict_outcome"] != "detected"],
            "new_false_case_ids": [row["case"]["case_id"] for row in fully_gated
                                   if not row["comparison"]["no_new_false_vs_r10"]],
            "new_duplicate_case_ids": [row["case"]["case_id"] for row in fully_gated
                                       if not row["comparison"]["no_new_duplicate_attributable_rises"]],
            "protection_pass_cases": sum(all(row["comparison"].values())
                                         for row in fully_gated),
        },
        "separate_exact_stock_input_audit": {
            "cases": audit["cases"],
            "all_signed_input_exact": audit["all_signed_input_exact"],
            "all_stock_scalar_exact_same_per_tick_order": audit["all_stock_scalar_exact"],
            "onset_exact_zero_cases": audit["onset_exact_zero_cases"],
            "first_nonzero_offset_counts": audit["first_nonzero_offset_counts"],
            "last_nonzero_offset_counts": audit["last_nonzero_offset_counts"],
            "interpretation": ("Exact, no tolerance. It certifies applied force inputs, "
                               "not the raw full policy outcome counts."),
        },
        "prefix_and_boundary_facts": {
            "before_onset_exact_cases": sum(
                all(all(row["arms"][arm]["matched_no_force_prefix_exact"]
                        ["before_onset"].values()) for arm in ("r10", "yaw"))
                for row in collisions),
            "through_onset_exact_cases": sum(
                all(all(row["arms"][arm]["matched_no_force_prefix_exact"]
                        ["through_zero_force_onset"].values()) for arm in ("r10", "yaw"))
                for row in collisions),
            "true_window_last_offset_counts": dict(Counter(
                str(row["arms"]["r10"]["true_window_last_index"]
                    - row["arms"]["r10"]["onset_index"]) for row in collisions)),
        },
    }


def first_difference(left, right, prefix="science"):
    if type(left) is not type(right):
        return f"{prefix}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{prefix}: keys differ"
        for key in left:
            difference = first_difference(left[key], right[key], f"{prefix}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{prefix}: list lengths differ"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(a, b, f"{prefix}[{index}]")
            if difference:
                return difference
        return None
    return None if left == right else f"{prefix}: {left!r} != {right!r}"


def validate_science(document, audit, expected):
    actual = scientific_accounting(document, audit)
    frozen = {key: expected[key] for key in SCIENCE_KEYS}
    difference = first_difference(actual, frozen)
    if difference:
        raise ValueError("deterministic D4 accounting differs: " + difference)
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path,
                        help="authenticated local ORIGINAL kit directory")
    parser.add_argument("--out", required=True, type=Path,
                        help="fresh result directory outside kit/source repository")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args(argv)
    try:
        kit, out = shared.validate_paths(args.kit, args.out)
        manifest_raw = (ROOT / shared.KIT_MANIFEST).read_bytes()
        kit_pins = shared.authenticate_kit(kit, manifest_raw)
        blobs, pins = snapshot()
        expected = json.loads(blobs[EXPECTED])
        expected_artifacts = expected["artifacts"]
        for relative, name in ((PLAN, "plan_v2_sha256"), (RUNNER, "runner_v2_sha256"),
                               (AUDIT_PLAN, "force_audit_plan_sha256"),
                               (AUDIT, "force_audit_source_sha256")):
            if pins[relative] != expected_artifacts[name]:
                raise ValueError("frozen D4 dependency/summary pin differs: " + relative)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable",
                           HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                           ALL_PROXY="http://127.0.0.1:9", NO_PROXY="")
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="r21-d4-check-", dir=out.parent) as folder:
            work = Path(folder); overlay, result = work / "overlay", work / "result"
            overlay.mkdir(); result.mkdir()
            shared.install_overlay(kit, overlay, manifest_raw, blobs, pins)
            shared.authenticate_kit(overlay, manifest_raw)
            for command, log_name in commands(sys.executable, args.workers):
                shared.run(command, overlay, environment, result / log_name)
            for name in ("d4_raw.json", "force_audit.json"):
                shutil.copyfile(overlay / name, result / name)
            document = json.loads((result / "d4_raw.json").read_bytes())
            audit = json.loads((result / "force_audit.json").read_bytes())
            science = validate_science(document, audit, expected)
            proof = {
                "schema": "r21-public-d4-check.v1", "verdict": "FAIL",
                "classification": expected["classification"],
                "exposure": "full pre-existing exposed D4; not blind evidence or a safety certificate",
                "deterministic_science_matches_frozen_summary": True,
                "timing_equality_required": False, "science": science,
            }
            (result / "proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
            shared.authenticate_kit(kit, manifest_raw)
            shared.authenticate_kit(overlay, manifest_raw)
            if any(shared.digest(ROOT / name) != pin for name, pin in pins.items()):
                raise ValueError("D4 source dependency changed during check")
            receipt = {
                "schema": "r21-authenticated-d4-check.v1",
                "kit_manifest_sha256": shared.digest(manifest_raw), "kit_files": kit_pins,
                "evaluator_sha256": shared.EVALUATOR_SHA256,
                "source_dependencies": list(blobs), "source_sha256": pins,
                "result_files": {path.name: shared.digest(path) for path in result.iterdir()
                                 if path.is_file()},
                "execution": {"python_no_bytecode": True, "workers": args.workers,
                              "network_required": False, "output_exclusive": True},
                "limitations": ["replay of exposed finite D4", "not blind evidence",
                                "not a safety certificate", "timing excluded from equality"],
            }
            (result / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            shared.publish(result, out)
        print(json.dumps({"out": str(out), "verdict": "FAIL", "regression_case_ids":
                          science["confirmed_policy_comparison_failures"]["case_ids"],
                          "deterministic_match": True}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
