"""Create a deterministic research-only R21 yaw torque submission package."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.r21_release_gate import build


R10_TORQUE = ROOT / "my_torque.py"
R10_TORQUE_SHA256 = "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371"
D4_RECORD = ROOT / "experiments/r21_full_protection/summary.json"
D4_SHA256 = "f88efb7c4417d19a4974e4875d7df8b4db4bdf956bb9e4bc29220c805f4c2c71"
ARCHIVE_NAMES = (
    "torque_controller.py", "collision_detector.py", "README.md", "receipt.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _validate_paths(kit, output):
    if kit.is_symlink() or not kit.is_dir():
        raise ValueError("regular local kit directory required")
    if output.exists() or output.is_symlink():
        raise ValueError("fresh non-symlink output required")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("regular existing output directory required")
    kit_real = kit.resolve(strict=True)
    parent_real = output.parent.resolve(strict=True)
    if parent_real.is_relative_to(ROOT.resolve()) or parent_real.is_relative_to(kit_real):
        raise ValueError("output must be external to repository and kit")
    return kit_real


def _run_audit(kit, output):
    process = subprocess.run(
        [sys.executable, "-B", "-m", "experiments.r21_release_gate.audit",
         "--variant", "yaw", "--kit", str(kit), "--out", str(output)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "yaw release audit failed")


def _validate_audit(document, detector_sha):
    required_gates = {
        "standalone_default_filename", "frozen_yaw_torque_table_exact",
        "servo_R10_fallback_exact", "protected_kit_unchanged"}
    if (document.get("schema") != "r21-yaw-release-audit.v1"
            or document.get("variant") != "yaw"
            or document.get("status") != "installable_research_option"
            or document.get("default_submission_promoted") is not False
            or document.get("recognized_collision_cells", {}).get("torque") != 24
            or document.get("build", {}).get("standalone_sha256") != detector_sha
            or document.get("controllers", {}).get("torque") != R10_TORQUE_SHA256
            or document.get("kit", {}).get("status") != "ok"
            or not required_gates.issubset(document.get("gates", {}))
            or not all(document["gates"][name] for name in required_gates)):
        raise ValueError("audit does not prove the frozen yaw standalone package")


def _readme():
    return (
        "# Optional R21 torque yaw detector\n\n"
        "Status: installable research opt-in; not the promoted default submission.\n\n"
        "For a pristine licensed Week 9 simulator copy, place the two Python files "
        "at its top level under their existing default names, then run the original "
        "torque evaluation. The detector embeds its frozen 2 x 60 normal bounds and "
        "has no runtime model or repository dependency. Unsupported, servo, and "
        "unregistered controllers retain the R10 detector behavior.\n\n"
        "`receipt.json` records source pins and the SHA-256 of the fresh original-kit "
        "audit of these two files. Protection on the full exposed 385-case D4 grid failed: "
        "a new false alarm in a no-impact 8 kg motion (case 174) and more duplicate "
        "rises in three impact cases (330, 339, 349). Do not use as the default. "
        "Known limitations also include "
        "duplicate-at-release responses at 40/129 ms and misses around 6--6.5 N. This "
        "artifact is for education and simulator evaluation, not hardware use.\n"
    ).encode()


def _zip_info(name):
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def create_package(kit, output, *, audit_runner=_run_audit):
    kit, output = Path(kit), Path(output)
    kit_real = _validate_paths(kit, output)
    d4_raw = D4_RECORD.read_bytes()
    if sha(d4_raw) != D4_SHA256:
        raise ValueError("frozen D4 protection receipt differs")
    d4 = json.loads(d4_raw)
    torque = R10_TORQUE.read_bytes()
    if sha(torque) != R10_TORQUE_SHA256:
        raise ValueError("frozen R10 torque controller differs")
    with tempfile.TemporaryDirectory(prefix="r21-yaw-package-", dir=output.parent) as tmp:
        temporary = Path(tmp)
        detector_path = temporary / "collision_detector.py"
        build_receipt = build.build(detector_path, variant="yaw")
        detector = detector_path.read_bytes()
        audit_path = temporary / "audit.json"
        audit_runner(kit_real, audit_path)
        if not audit_path.is_file() or audit_path.is_symlink():
            raise ValueError("audit runner did not create a regular receipt")
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        _validate_audit(audit, sha(detector))
        readme = _readme()
        receipt = {
            "schema": "r21-torque-yaw-portable-package.v1",
            "status": "installable_research_option",
            "default_submission_promoted": False,
            "audit_sha256": sha(audit_bytes),
            "full_d4": {"verdict": d4["verdict"], "summary_sha256": D4_SHA256,
                        "exposed": True, "episodes": d4["census"]["episodes"],
                        "policy_regression_case_ids": d4["confirmed_policy_comparison_failures"]["case_ids"]},
            "build": build_receipt,
            "kit_manifest_sha256": audit["kit"]["manifest_sha256"],
            "evaluator_sha256": audit["evaluator_sha256"],
            "recognized_collision_cells": audit["recognized_collision_cells"],
            "gates": audit["gates"],
            "files": {
                "torque_controller.py": sha(torque),
                "collision_detector.py": sha(detector),
                "README.md": sha(readme),
            },
        }
        receipt_bytes = (json.dumps(
            receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        contents = (torque, detector, readme, receipt_bytes)
        buffer = BytesIO()
        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED,
                     compresslevel=9, strict_timestamps=True) as archive:
            for name, data in zip(ARCHIVE_NAMES, contents, strict=True):
                archive.writestr(_zip_info(name), data, compresslevel=9)
        # Exclusive creation happens only after the complete archive exists in
        # memory. A file raced into place is never opened, replaced, or removed.
        with output.open("xb") as stream:
            stream.write(buffer.getvalue())
    return {
        "archive_sha256": sha(output.read_bytes()),
        "audit_sha256": receipt["audit_sha256"],
        "standalone_sha256": build_receipt["standalone_sha256"],
        "entries": list(ARCHIVE_NAMES),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = create_package(args.kit, args.out)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
