"""Create a deterministic, audited R22 research-only torque package."""
from __future__ import annotations

import argparse
import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.r22_export import build
from tools.r21_d4_check import DEPENDENCIES as D4_DEPENDENCIES
from tools.r21_replay import authenticate_kit


TORQUE = ROOT / "my_torque.py"
TORQUE_SHA256 = "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371"
LIGHT_HOLDOUT = ROOT / "results/r22/holdout_selected.json"
LIGHT_HOLDOUT_SHA256 = "6320d9c3d2971eed367756cafc97c4c6b187e7d8b13897d7f1719cd41692b338"
AUGMENTED_HOLDOUT = ROOT / "results/r22/holdout_augmented.json"
AUGMENTED_HOLDOUT_SHA256 = "a6f658927d30a0c2d82956075fe571c8223f3b27ce006959827a52c4ed372c8d"
NAMES = ("torque_controller.py", "collision_detector.py", "README.md", "receipt.json")
PARITY_CASES = (174, 319, 330, 335, 339, 349, "weak6", "servo_normal", "servo_force")
INDEPENDENT_VERDICT = {
    "status": "bounded_not_generalized",
    "rows_passed": 51,
    "rows_total": 66,
    "negative_rows_passed": 18,
    "negative_rows_total": 18,
    "single_rows_passed": 18,
    "single_rows_total": 24,
    "twin_slots_one_to_one": 79,
    "twin_slots_total": 96,
    "pair_rows_complete": 15,
    "pair_rows_total": 24,
    "strict_gain_vs_r21": 0,
    "strict_loss_vs_r21": 0,
    "genuine_offset_tool_contacts_missed_by_all": 2,
    "hard_realtime_proven": False,
    "promoted": False,
    "published_holdout_sha256": LIGHT_HOLDOUT_SHA256,
}
AUGMENTED_VERDICT = {
    "status": "failed_not_generalized", "rows_passed": 52, "rows_total": 66,
    "negative_rows_passed": 18, "negative_rows_total": 18,
    "single_rows_passed": 21, "single_rows_total": 24,
    "twin_slots_one_to_one": 78, "twin_slots_total": 96,
    "pair_rows_complete": 14, "pair_rows_total": 24, "added_false_event": 1,
    "hard_realtime_proven": False, "promoted": False,
    "published_holdout_sha256": AUGMENTED_HOLDOUT_SHA256,
}
UNASSERTED_VERDICT = {"status": "not_asserted_by_this_receipt", "promoted": False}


def literal_tuple(relative, name):
    tree = ast.parse((ROOT / relative).read_bytes(), filename=relative)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
                return value
    raise ValueError(f"missing bounded dependency tuple: {relative}:{name}")


PUBLIC_DEPENDENCIES = tuple(dict.fromkeys(
    D4_DEPENDENCIES
    + literal_tuple("tools/r22_studies.py", "RUNTIME")
    + literal_tuple("tools/r22_studies.py", "REVISED")
    + tuple(build.PINS)
    + (
        "my_servo.py", "experiments/r22_export/__init__.py",
        "experiments/r22_export/audit.py", "experiments/r22_export/build.py",
        "tools/check_integrity.py", "tools/compare_baseline.py", "tools/r21_replay.py",
        "tools/validate_submission.py", "tools/kit_manifest.sha256",
        "results/r10/official_regression.json", "results/v2_protocol/PROTOCOL_R7.json",
    )))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def validate_paths(kit, output):
    if kit.is_symlink() or not kit.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("regular kit and fresh non-symlink output required")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("regular existing output directory required")
    kit = kit.resolve(strict=True); parent = output.parent.resolve(strict=True)
    if parent.is_relative_to(ROOT.resolve()) or parent.is_relative_to(kit):
        raise ValueError("output must be external to repository and kit")
    return kit


def run_audit(kit, output, variant):
    manifest_raw = (ROOT / "tools/kit_manifest.sha256").read_bytes()
    authenticate_kit(kit, manifest_raw)
    with tempfile.TemporaryDirectory(prefix="r22-audit-runtime-", dir=output.parent) as tmp:
        runtime = Path(tmp)
        for relative in PUBLIC_DEPENDENCIES:
            source = ROOT / relative
            if (Path(relative).is_absolute() or ".." in Path(relative).parts
                    or source.is_symlink() or not source.is_file()):
                raise ValueError("regular bounded public dependency required: " + relative)
            target = runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
        for tree in ("sim", "model"):
            shutil.copytree(kit / tree, runtime / tree, symlinks=False)
        shutil.copyfile(kit / "evaluate.py", runtime / "evaluate.py")
        authenticate_kit(runtime, manifest_raw)
        bootstrap = ("import runpy,sys;"
                     "root=sys.argv.pop(1);sys.path.insert(0,root);"
                     "runpy.run_module('experiments.r22_export.audit',run_name='__main__')")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", bootstrap, str(runtime),
             "--kit", str(kit), "--out", str(output), "--variant", variant],
            cwd=runtime, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "R22 audit failed")


def validate_audit(doc, variant, detector_sha):
    required = {"default_filename", "torque_official_24_of_24",
                "torque_controls_R10_exact_except_compute", "servo_R10_exact",
                "modular_generated_6000_exact", "protected_kit_unchanged"}
    parity = doc.get("parity")
    official = doc.get("official")
    parity_ok = (isinstance(parity, list)
                 and [row.get("case_id") for row in parity] == list(PARITY_CASES)
                 and all(row.get("samples") == 6000 and row.get("passed") is True
                         for row in parity))
    official_ok = (isinstance(official, dict)
                   and all(isinstance(official.get(controller), dict)
                           and official[controller].get("episodes") == 29
                           and bool(official[controller].get("control"))
                           and bool(official[controller].get("collisions"))
                           for controller in ("torque", "servo")))
    if (doc.get("schema") != "r22-release-audit.v1" or doc.get("variant") != variant
            or doc.get("default_submission_promoted") is not False
            or doc.get("build", {}).get("standalone_sha256") != detector_sha
            or doc.get("recognized_collision_cells") != {"torque": 24, "servo": 23}
            or not required.issubset(doc.get("gates", {}))
            or not all(doc["gates"][name] for name in required)
            or not parity_ok or not official_ok):
        raise ValueError("audit does not prove the selected frozen R22 standalone")


def readme(variant):
    descriptions = {
        "core": "The core variant contains the isolated-spike correction and release-momentum continuity. ",
        "augmented": "The augmented variant adds an unrestricted delayed, normal-only calibrated planar-force witness. ",
        "guarded": "The guarded variant uses sensor evidence to select release continuity. ",
        "light": "The light variant adds planar-force evidence only during the lift of a payload no heavier than 10 N. ",
    }
    extra = descriptions[variant]
    if variant == "light":
        independent = ("Independent evaluation passed 51/66 rows and all 18 negative rows, but only 18/24 "
                       "single rows and 15/24 complete pair rows; it found no strict gain or loss versus R21. "
                       "Thus the apparent known weak/D4 improvements did not generalize. ")
    elif variant == "augmented":
        independent = ("Independent evaluation failed 14/66 rows despite passing all 18 negative rows; "
                       "it reached 21/24 single rows, 78/96 paired slots and 14/24 complete pair rows, "
                       "and added one false event. Its exposed improvements did not generalize. ")
    else:
        independent = ("This receipt asserts no locked blind result for this variant; its exposed "
                       "improvements must not be treated as general. ")
    return ("# Optional R22 torque detector (" + variant + ")\n\n"
            "Research/education simulator option only; not the default submission and not for hardware.\n\n"
            + extra +
            "The generator authenticated the local original evaluator, verified 24/24 torque cells, "
            "exact R10 servo fallback, and 6,000-sample modular/standalone parity for the recorded cases. "
            + independent + "These finite checks "
            "are not a hidden-test, broad safety, or hard real-time guarantee and do not justify promotion. "
            "Use the two Python files only in a separate licensed simulator copy.\n").encode()


def info(name):
    value = ZipInfo(name, (1980, 1, 1, 0, 0, 0)); value.compress_type = ZIP_DEFLATED
    value.create_system = 3; value.external_attr = 0o100644 << 16
    return value


def create_package(kit, output, variant, *, audit_runner=run_audit):
    kit, output = Path(kit), Path(output)
    kit = validate_paths(kit, output)
    torque = TORQUE.read_bytes()
    if sha(torque) != TORQUE_SHA256:
        raise ValueError("frozen R10 torque controller differs")
    if variant == "light" and sha(LIGHT_HOLDOUT.read_bytes()) != LIGHT_HOLDOUT_SHA256:
        raise ValueError("published LIGHT independent holdout record differs")
    if variant == "augmented" and sha(AUGMENTED_HOLDOUT.read_bytes()) != AUGMENTED_HOLDOUT_SHA256:
        raise ValueError("published augmented independent holdout record differs")
    with tempfile.TemporaryDirectory(prefix="r22-package-", dir=output.parent) as tmp:
        temp = Path(tmp); detector_path = temp / "collision_detector.py"
        build_receipt = build.build(detector_path, variant); detector = detector_path.read_bytes()
        audit_path = temp / "audit.json"; audit_runner(kit, audit_path, variant)
        if not audit_path.is_file() or audit_path.is_symlink():
            raise ValueError("audit did not create a regular receipt")
        audit_raw = audit_path.read_bytes(); audit = json.loads(audit_raw)
        validate_audit(audit, variant, sha(detector)); guide = readme(variant)
        receipt = {"schema": "r22-torque-package.v1", "variant": variant,
                   "status": "research_option", "default_submission_promoted": False,
                   "independent_evaluation": ({"light": INDEPENDENT_VERDICT,
                                               "augmented": AUGMENTED_VERDICT}.get(
                                                   variant, UNASSERTED_VERDICT)),
                   "audit_sha256": sha(audit_raw), "build": build_receipt,
                   "kit_manifest_sha256": audit["kit"]["manifest_sha256"],
                   "evaluator_sha256": audit["evaluator_sha256"],
                   "recognized_collision_cells": audit["recognized_collision_cells"],
                   "parity_case_ids": [row["case_id"] for row in audit["parity"]],
                   "gates": audit["gates"], "files": {
                       "torque_controller.py": sha(torque), "collision_detector.py": sha(detector),
                       "README.md": sha(guide)}}
        receipt_raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
        buffer = BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
            for name, blob in zip(NAMES, (torque, detector, guide, receipt_raw), strict=True):
                archive.writestr(info(name), blob, compresslevel=9)
        with output.open("xb") as stream:
            stream.write(buffer.getvalue())
    return {"archive_sha256": sha(output.read_bytes()), "audit_sha256": receipt["audit_sha256"],
            "standalone_sha256": build_receipt["standalone_sha256"], "entries": list(NAMES)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("core", "augmented", "guarded", "light"), required=True)
    args = parser.parse_args(argv)
    try: result = create_package(args.kit, args.out, args.variant)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error: parser.error(str(error))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
