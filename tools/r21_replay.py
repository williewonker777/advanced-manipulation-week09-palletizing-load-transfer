"""Authenticated, self-contained replay of the frozen R21 weak/surface studies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
KIT_MANIFEST = "tools/kit_manifest.sha256"
MANIFEST_SHA256 = "7ea083e92b7fa9fd303ce91a69271f58dd6fb9ace22613039f469a84e2867294"
EVALUATOR_SHA256 = "ee7effc27ac0006c7d2f392523e3e1e4a45626d1113116cba40e00978bd2786f"

COMMON = (
    "my_detector.py", "my_torque.py",
    "experiments/r18_paired_impulses/__init__.py",
    "experiments/r18_paired_impulses/runner.py",
    "experiments/r20_horizontal/__init__.py",
    "experiments/r20_horizontal/candidate_detector.py",
    "experiments/r21_pick_observer/__init__.py",
    "experiments/r21_pick_observer/candidate.py",
    "experiments/r21_pick_observer/detector.py",
    "experiments/r21_pick_observer/model.json",
    "experiments/r21_yaw_impulse/__init__.py",
    "experiments/r21_yaw_impulse/state.py",
    "experiments/r21_yaw_impulse/detector.py",
    "experiments/r21_yaw_impulse/frozen.py",
    "experiments/r21_yaw_impulse/coarse.py",
    "experiments/r21_yaw_impulse/normal_bins.json",
)

WEAK_EXTRA = (
    "experiments/r21_weak_challenge/__init__.py",
    "experiments/r21_weak_challenge/plan.json",
    "experiments/r21_weak_challenge/evaluator.py",
    "experiments/r21_weak_challenge/yaw_driver.py",
    "experiments/r21_weak_challenge/verify.py",
)

SURFACE_EXTRA = (
    "r1_torque.py", "models/r12/counterfactual_coefficients.json",
    "experiments/r12_hybrid/controller.py",
    "experiments/r14_airborne/registration.py",
    "experiments/r14_online/stream.py",
    "experiments/r14_stock_handoff/__init__.py",
    "experiments/r14_stock_handoff/controller.py",
    "experiments/r14_stock_handoff/detector.py",
    "experiments/r14_stock_handoff/geometry.py",
    "experiments/r14_stock_handoff/qualification.py",
    "experiments/r15_early_com/controller.py",
    "experiments/r15_early_com/sensing.py",
    "experiments/r15_early_com/stream.py",
    "experiments/r16_balanced/controller.py",
    "experiments/r16_normal_phase/controller.py",
    "experiments/r16_payload_contract/controller.py",
    "experiments/r16_release/controller.py",
    "experiments/r17_release_prediction/detector.py",
    "experiments/r17_release_coincidence/portable_detector.py",
    "experiments/r18_paired_impulses/scoring.py",
    "experiments/r18_single_pass/__init__.py",
    "experiments/r18_single_pass/detector.py",
    "experiments/r18_single_pass/paired_replay.py",
    "experiments/r18_single_pass/recovery.py",
    "experiments/r18_single_pass/reference_recovery.py",
    "experiments/r19_close_pairs/__init__.py",
    "experiments/r19_close_pairs/assay.py",
    "experiments/r20_force_certificate/__init__.py",
    "experiments/r20_force_certificate/certificate.py",
    "experiments/r21_surface_pairs/__init__.py",
    "experiments/r21_surface_pairs/plan.json",
    "experiments/r21_surface_pairs/study.py",
    "tools/r5_metrics.py", "tools/r11_placement.py",
    "tools/r13_grasp_estimator.py", "tools/r14_paired_events.py",
)

HISTORY_EXTRA = (
    "experiments/r21_yaw_impulse/calibrate.py",
    "experiments/r21_yaw_impulse/coarse_calibration.py",
    "experiments/r21_yaw_continuity/__init__.py",
    "experiments/r21_yaw_continuity/assay.py",
    "experiments/r21_yaw_continuity/candidate.py",
    "experiments/r21_yaw_continuity/state.py",
    "experiments/r21_yaw_continuity/model.json",
    "experiments/r21_yaw_continuity/plan.json",
)

PICK_PIN = "30e56f7ac752c6072ef88869378147be3534484b097ee23044c328f787f50364"


def digest(blob):
    if isinstance(blob, Path):
        blob = blob.read_bytes()
    return hashlib.sha256(blob).hexdigest()


def has_symlink_component(path):
    absolute = path.absolute()
    return any(part.is_symlink() for part in (absolute, *absolute.parents) if part.exists())


def validate_paths(kit, out):
    if kit.is_symlink() or not kit.is_dir() or has_symlink_component(kit):
        raise ValueError("regular ORIGINAL kit directory required")
    for relative in ("sim", "model", "evaluate.py"):
        path = kit / relative
        if path.is_symlink() or not path.exists():
            raise ValueError("incomplete ORIGINAL kit: " + relative)
    if os.path.lexists(out) or has_symlink_component(out) or not out.parent.is_dir():
        raise ValueError("output must be a fresh non-symlink directory")
    kit, parent = kit.resolve(), out.parent.resolve()
    if parent.is_relative_to(kit) or parent.is_relative_to(ROOT.resolve()):
        raise ValueError("output must be outside kit and source repository")
    return kit, parent / out.name


def manifest_entries(raw):
    entries = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        pin, relative = line.split(None, 1)
        relative = relative.strip()
        if len(pin) != 64 or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("invalid kit manifest")
        entries[relative] = pin
    if not entries:
        raise ValueError("empty kit manifest")
    return entries


def authenticate_kit(kit, manifest_raw):
    if digest(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("R7/R20 kit manifest identity differs")
    pins = manifest_entries(manifest_raw)
    for relative, expected in pins.items():
        path = kit / relative
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("ORIGINAL kit authentication failed: " + relative)
    for tree in (kit / "sim", kit / "model"):
        if any(path.is_symlink() for path in tree.rglob("*")):
            raise ValueError("symlink inside ORIGINAL kit tree")
    expected = {name for name in pins if name == "sim" or name == "model"
                or name.startswith("sim/") or name.startswith("model/")}
    actual = {path.relative_to(kit).as_posix() for tree in (kit / "sim", kit / "model")
              for path in tree.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("unexpected or missing regular file in ORIGINAL kit tree")
    evaluator = kit / "evaluate.py"
    if evaluator.is_symlink() or not evaluator.is_file() or digest(evaluator) != EVALUATOR_SHA256:
        raise ValueError("R7/R20 evaluator identity differs")
    return pins


def snapshot(study):
    extras = {"weak": WEAK_EXTRA, "surface": SURFACE_EXTRA,
              "history": SURFACE_EXTRA + HISTORY_EXTRA}[study]
    relative = tuple(dict.fromkeys(COMMON + extras))
    blobs = {}
    for name in relative:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("regular replay dependency required: " + name)
        blobs[name] = path.read_bytes()
    return blobs, {name: digest(blob) for name, blob in blobs.items()}


def install_overlay(kit, overlay, manifest_raw, blobs, pins):
    for tree in ("sim", "model"):
        shutil.copytree(kit / tree, overlay / tree, symlinks=False)
    manifest = overlay / KIT_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(manifest_raw)
    shutil.copyfile(kit / "evaluate.py", overlay / "evaluate.py")
    for relative, blob in blobs.items():
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        if digest(target) != pins[relative]:
            raise ValueError("overlay copy differs: " + relative)


def publish(result, out):
    """Publish without POSIX rename-over-empty-directory replacement semantics."""
    out.mkdir()
    created = []
    try:
        for source in result.iterdir():
            if not source.is_file() or source.is_symlink():
                raise ValueError("regular replay result required")
            target = out / source.name
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
            created.append(target)
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        try:
            out.rmdir()
        except OSError:
            pass
        raise


def run(command, overlay, environment, log):
    completed = subprocess.run(command, cwd=overlay, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"replay command failed ({completed.returncode}): {command[3]}")


def weak_commands(python):
    plan = "experiments/r21_weak_challenge/plan.json"
    return (
        ([python, "-B", "-m", "experiments.r21_weak_challenge.evaluator", "--plan", plan,
          "--out", "pick.json", "--arms", "r10,r20,r21_pick", "--candidate-pin", PICK_PIN], "pick.log"),
        ([python, "-B", "-m", "experiments.r21_weak_challenge.yaw_driver", "--plan", plan,
          "--out", "yaw.json"], "yaw.log"),
        ([python, "-B", "-m", "experiments.r21_weak_challenge.verify", "--plan", plan,
          "--pick", "pick.json", "--yaw", "yaw.json", "--out", "summary.json"], "verify.log"),
    )


def surface_proof(document):
    if document.get("schema") != "r21-surface-pair-assay.v1" or document.get("episodes") != 128:
        raise ValueError("incomplete surface result")
    rows = []
    for row in document.get("rows", []):
        modes = {}
        for mode in ("stock", "changed"):
            arms = row["variants"][mode]["arms"]
            if set(arms) != {"r10", "r20", "pick", "yaw"}:
                raise ValueError("surface arm set differs")
            modes[mode] = {arm: {"strict_exact_force": value["strict_exact_force"],
                                  "force_exact": value["force_exact"],
                                  "retained_vs": value["retained_vs"],
                                  "score_sha256": value["score_sha256"]}
                           for arm, value in arms.items()}
        rows.append({"id": row["id"], "material_effect_observed": row["material_effect_observed"],
                     "variants": modes})
    if len(rows) != 4:
        raise ValueError("surface cohort differs")
    return {"schema": "r21-surface-curatable-proof.v1", "source_result_sha256": None,
            "four_arms": ["r10", "r20", "pick", "yaw"], "four_twins": True,
            "stock_and_changed_materials": True, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--study", required=True, choices=("weak", "surface", "history"))
    parser.add_argument("--out", required=True, type=Path, help="new external result directory")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    try:
        kit, out = validate_paths(args.kit, args.out)
        manifest_raw = (ROOT / KIT_MANIFEST).read_bytes()
        kit_pins = authenticate_kit(kit, manifest_raw)
        blobs, pins = snapshot(args.study)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", MUJOCO_GL="disable")
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(prefix="r21-replay-", dir=out.parent) as folder:
            work = Path(folder)
            overlay, result = work / "overlay", work / "result"
            overlay.mkdir(); result.mkdir()
            install_overlay(kit, overlay, manifest_raw, blobs, pins)
            authenticate_kit(overlay, manifest_raw)
            if args.study == "weak":
                for command, name in weak_commands(sys.executable):
                    run(command, overlay, environment, result / name)
                for name in ("pick.json", "yaw.json", "summary.json"):
                    shutil.copyfile(overlay / name, result / name)
                summary = json.loads((result / "summary.json").read_bytes())
                if summary.get("schema") != "r21-independent-weak-summary.v1" or summary.get("episodes") != 198:
                    raise ValueError("incomplete weak aggregation")
            elif args.study == "surface":
                command = [sys.executable, "-B", "-m", "experiments.r21_surface_pairs.study",
                           "--out", "surface.json", "--jobs", str(args.jobs)]
                run(command, overlay, environment, result / "surface.log")
                shutil.copyfile(overlay / "surface.json", result / "surface.json")
                surface = json.loads((result / "surface.json").read_bytes())
                proof = surface_proof(surface)
                proof["source_result_sha256"] = digest(result / "surface.json")
                (result / "proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
                summary = {"schema": proof["schema"], "episodes": surface["episodes"],
                           "rows": len(proof["rows"])}
            else:
                command = [sys.executable, "-B", "-m", "experiments.r21_yaw_continuity.assay",
                           "--out", "history.json"]
                run(command, overlay, environment, result / "history.log")
                shutil.copyfile(overlay / "history.json", result / "history.json")
                history = json.loads((result / "history.json").read_bytes())
                if set(history) - {"exposed", "fresh"} or "exposed" not in history:
                    raise ValueError("invalid continuity history result")
                for stage, value in history.items():
                    if set(value) != {"passed", "rows"} or not isinstance(value["rows"], list):
                        raise ValueError("invalid continuity history stage: " + stage)
                summary = {"schema": "r21-yaw-continuity-history-replay.v1",
                           "stages": {stage: {"passed": value["passed"], "rows": len(value["rows"])}
                                      for stage, value in history.items()},
                           "episodes": sum(row["episodes"] for value in history.values()
                                           for row in value["rows"])}
            authenticate_kit(kit, manifest_raw)
            authenticate_kit(overlay, manifest_raw)
            if any(digest(ROOT / name) != pin for name, pin in pins.items()):
                raise ValueError("source dependency changed during replay")
            receipt = {"schema": "r21-authenticated-replay.v1", "study": args.study,
                       "kit_manifest_sha256": digest(manifest_raw), "kit_files": kit_pins,
                       "evaluator_sha256": EVALUATOR_SHA256,
                       "source_dependencies": list(blobs), "source_sha256": pins,
                       "result_files": {p.name: digest(p) for p in result.iterdir() if p.is_file()},
                       "summary": summary, "limitations": ["replay, not new blind evidence",
                       "no kit or source modification", "no network access required"]}
            (result / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            publish(result, out)
        print(json.dumps({"study": args.study, "out": str(out), "summary": summary}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
