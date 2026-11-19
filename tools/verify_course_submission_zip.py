"""Verify a three-file Week09 submission against the authentic course evaluator.

This is a verifier, not a packager.  It never imports the submitted files in
the repository process.  Every official evaluation runs in a fresh copy of
the authenticated licensed kit, and host-dependent timing cells are excluded
from comparisons exactly as in ``tools.compare_baseline.official_cells``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = {"my_servo.py", "my_torque.py", "my_detector.py"}
KIT_MANIFEST_SHA256 = "7ea083e92b7fa9fd303ce91a69271f58dd6fb9ace22613039f469a84e2867294"
EVALUATE_SHA256 = "ee7effc27ac0006c7d2f392523e3e1e4a45626d1113116cba40e00978bd2786f"
R10_REFERENCE_SHA256 = "5abef9110e707884b0ed46ec6e329351b293ede8beb82c007ce42f66ad993a5c"
R31_REFERENCE_SHA256 = "18fe83daa41644086f1aa2165428076399b8d9074c9f7d4aa20a75b5656bd8ac"
R10_CONTROLLER_PINS = {
    "my_servo.py": "40706968fb72ed8d7a864e44e828787ae8db0365c2927e5166e4a5b2d83bba8b",
    "my_torque.py": "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371",
}


def sha256(data_or_path):
    if isinstance(data_or_path, (bytes, bytearray)):
        return hashlib.sha256(data_or_path).hexdigest()
    with Path(data_or_path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authenticate_kit(kit: Path) -> dict[str, str]:
    manifest = ROOT / "tools/kit_manifest.sha256"
    if sha256(manifest) != KIT_MANIFEST_SHA256:
        raise ValueError("repository kit manifest changed")
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    actual = {}
    discovered = set()
    for tree in ("sim", "model"):
        folder = kit / tree
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("missing or unsafe licensed kit tree: " + tree)
        for path in folder.rglob("*"):
            relative = path.relative_to(kit).as_posix()
            if path.is_symlink():
                raise ValueError("unexpected kit symlink: " + relative)
            if path.is_file():
                discovered.add(relative)
    if discovered != set(expected):
        raise ValueError("licensed kit file set differs")
    for name, digest in expected.items():
        path = kit / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError("licensed kit differs: " + name)
        actual[name] = digest
    evaluator = kit / "evaluate.py"
    if evaluator.is_symlink() or not evaluator.is_file() or sha256(evaluator) != EVALUATE_SHA256:
        raise ValueError("evaluate.py differs from the authentic course evaluator")
    return actual


def inspect_zip(path: Path, expected_sha: str | None) -> tuple[dict[str, bytes], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular ZIP required")
    digest = sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("submission ZIP SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != EXPECTED_NAMES:
            raise ValueError("ZIP must contain exactly three top-level my_*.py files")
        if archive.testzip() is not None:
            raise ValueError("corrupt ZIP member")
        files = {}
        for info in infos:
            name = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (len(name.parts) != 1 or name.name not in EXPECTED_NAMES or info.is_dir()
                    or info.file_size > 2_000_000 or (mode and (mode & 0o170000) == 0o120000)):
                raise ValueError("unsafe ZIP member: " + info.filename)
            data = archive.read(info)
            source = data.decode("utf-8")
            tree = ast.parse(source, info.filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (
                        node.level or (node.module or "").split(".")[0] in {"experiments", "tools"}):
                    raise ValueError(info.filename + " depends on non-submission source")
                if isinstance(node, ast.Import) and any(
                        alias.name.split(".")[0] in {"experiments", "tools"} for alias in node.names):
                    raise ValueError(info.filename + " depends on non-submission source")
            compile(source, info.filename, "exec")
            files[info.filename] = data
    for name, pin in R10_CONTROLLER_PINS.items():
        if sha256(files[name]) != pin:
            raise ValueError(name + " is not the verified R10 controller")
    return files, digest


def reference_cells() -> tuple[dict, dict, dict]:
    r10_path = ROOT / "results/r10/official_regression.json"
    r31_path = ROOT / "results/r31/official.json"
    if sha256(r10_path) != R10_REFERENCE_SHA256 or sha256(r31_path) != R31_REFERENCE_SHA256:
        raise ValueError("official reference record changed")
    r10 = json.loads(r10_path.read_bytes())
    r31 = json.loads(r31_path.read_bytes())["research_torque"]
    expected = {"servo": r10["servo"]["control_and_collisions"], "torque": r31["cells"]}
    pins = {"r10_servo": R10_REFERENCE_SHA256, "r31_torque": R31_REFERENCE_SHA256,
            "r31_detector_source": r31["source_sha256"]}
    return expected, pins, r31["counts"]


def normalized(text: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from tools.compare_baseline import official_cells
    control, collisions = official_cells(text)
    return {
        "control": {str(m): cells for m, cells in control.items()},
        "collisions": [dict(mass=m, time=t, cells=cells)
                       for (m, t), cells in collisions.items()],
    }


def detected_count(cells: dict) -> int:
    return sum(str(value).isdigit() for row in cells["collisions"] for value in row["cells"][:4])


def install(kit: Path, work: Path, files: dict[str, bytes], aliases: bool) -> None:
    for tree in ("sim", "model"):
        source = kit / tree
        if source.is_symlink() or not source.is_dir():
            raise ValueError("unsafe licensed kit tree: " + tree)
        shutil.copytree(source, work / tree, symlinks=False)
    shutil.copyfile(kit / "evaluate.py", work / "evaluate.py")
    for name, data in files.items():
        (work / name).write_bytes(data)
    if aliases:
        shutil.copyfile(work / "my_servo.py", work / "servo_controller.py")
        shutil.copyfile(work / "my_torque.py", work / "torque_controller.py")
        shutil.copyfile(work / "my_detector.py", work / "collision_detector.py")


def one_run(kit, files, naming, mode, expected, parent) -> dict:
    with tempfile.TemporaryDirectory(prefix="week09-submit-", dir=parent) as folder:
        work = Path(folder)
        aliases = naming == "course-default-aliases"
        install(kit, work, files, aliases)
        ctrl = mode + "_controller" if aliases else "my_" + mode
        detector = "collision_detector" if aliases else "my_detector"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   MUJOCO_GL="disable")
        env.pop("PYTHONPATH", None)
        before = time.monotonic()
        process = subprocess.run(
            [sys.executable, "-B", "evaluate.py", "--ctrl", ctrl,
             "--detector", detector, "--out", "evaluation.txt"],
            cwd=work, env=env, capture_output=True, text=True)
        elapsed = time.monotonic() - before
        if process.returncode:
            raise RuntimeError(f"{naming}/{mode} failed: {process.stderr.strip()}")
        episodes = sum(line.strip().startswith("실행 중: 박스")
                       for line in process.stdout.splitlines())
        cells = normalized((work / "evaluation.txt").read_text())
        if episodes != 29 or cells != expected[mode]:
            raise ValueError(f"{naming}/{mode} does not match the pinned non-timing cells")
        imported = sorted(path.relative_to(work).as_posix()
                          for path in work.rglob("*.py")
                          if path.parts[-1] not in {"evaluate.py"})
        allowed = EXPECTED_NAMES | ({"servo_controller.py", "torque_controller.py",
                                     "collision_detector.py"} if aliases else set())
        root_python = {name for name in imported if "/" not in name}
        if not root_python <= allowed:
            raise ValueError("unexpected authored runtime source: " + repr(root_python - allowed))
        return dict(naming=naming, mode=mode, episodes=episodes,
                    detected_of_24=detected_count(cells), wall_seconds=elapsed,
                    cells=cells, source_hashes={name: sha256(data) for name, data in files.items()})


def verify(args) -> dict:
    kit = args.kit.resolve()
    out = args.out.resolve()
    if out.exists() or out.is_relative_to(kit):
        raise ValueError("fresh output outside the licensed kit required")
    kit_before = authenticate_kit(kit)
    files, zip_sha = inspect_zip(args.zip.resolve(), args.sha256)
    expected, references, r31_counts = reference_cells()
    namings = ("explicit-student-names", "course-default-aliases") if args.names == "both" else (args.names,)
    out.mkdir(parents=True)
    records = []
    try:
        for naming in namings:
            for mode in ("servo", "torque"):
                record = one_run(kit, files, naming, mode, expected, out)
                records.append(record)
                (out / f"{naming}-{mode}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        shutil.rmtree(out)
        raise
    if authenticate_kit(kit) != kit_before:
        shutil.rmtree(out)
        raise ValueError("input kit changed")
    for mode in ("servo", "torque"):
        same = [r["cells"] for r in records if r["mode"] == mode]
        if any(value != same[0] for value in same[1:]):
            raise ValueError(mode + " differs between explicit and alias names")
    report = dict(
        schema="week09-combined-submission-official-verification.v1", status="pass",
        submission_zip=str(args.zip.resolve()), submission_sha256=zip_sha,
        files={name: sha256(data) for name, data in files.items()},
        authentic_evaluate_sha256=EVALUATE_SHA256, kit_manifest_sha256=KIT_MANIFEST_SHA256,
        input_kit_unchanged=True, repository_experiments_or_tools_required_by_submission=False,
        references=references, selected={"servo":"R10", "torque":"R31"},
        official_scores={"servo":23, "torque":24}, r31_recorded_counts=r31_counts,
        actual_independent_executions=len(records) * 29, records=records)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status":"pass", "scores":report["official_scores"],
                      "executions":report["actual_independent_executions"],
                      "zip_sha256":zip_sha}, sort_keys=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--names", choices=("both", "explicit-student-names", "course-default-aliases"),
                        default="both")
    args = parser.parse_args(argv)
    try:
        verify(args)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
