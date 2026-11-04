#!/usr/bin/env python3
"""Serial R2 development, D4, sealed-holdout and perturbation studies.

This module contains builders and execution gates. Importing it never opens the
sealed holdout and never constructs its seeded conditions.  The real holdout
seed is first touched only after a committed freeze has been verified and an
exclusive, durable opened marker has been created.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from sim.runner import run_episode
from tools import d4_robustness, sweep, v2_resources as resources, v2_study as V
from tools.check_integrity import check_integrity
from tools import r2_study as R

D = ROOT / "results/v2_protocol"
PROTOCOL = D / "PROTOCOL_R2.json"
FREEZE = D / "R2_FREEZE.json"
HOLDOUT_MARKER = D / "R2_HOLDOUT_OPENED.json"

DIRECTIONS = {
    "+x": (1., 0., 0.), "-x": (-1., 0., 0.),
    "+y": (0., 1., 0.), "-y": (0., -1., 0.),
    "+z": (0., 0., 1.), "-z": (0., 0., -1.),
    "+xy": (1., 1., 0.), "-xy": (-1., -1., 0.),
    "+xyz": (1., 1., 1.), "-xyz": (-1., -1., -1.),
}
HOLDOUT_PHASES = {
    "pick": (1.375, 1.515),
    "transport": (2.05, 3.35),
    "place": (3.775, 3.945),
}


def source_hashes() -> dict[str, str]:
    """R2 sources plus this stage runner and its fixed design/integrity gates."""
    hashes = R.source_hashes()
    for relative in ("tools/r2_extended_study.py", "tools/d4_robustness.py",
                     "tools/check_integrity.py", "tools/kit_manifest.sha256",
                     "sim/config.py", "sim/trajectory.py"):
        hashes[relative] = R.sha(ROOT / relative)
    return hashes


def _force(direction: str, magnitude: float) -> list[float]:
    vector = np.asarray(DIRECTIONS[direction], dtype=float)
    return (float(magnitude) * vector / np.linalg.norm(vector)).tolist()


def load_selection(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = value if isinstance(value, list) else value.get("chosen_specs")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection must be a nonempty list or chosen_specs list")
    canonical = {item["id"]: item for item in R.specs()}
    output = []
    for item in selected:
        if not isinstance(item, dict) or item.get("id") not in canonical:
            raise ValueError("selection contains an unknown candidate")
        if item != canonical[item["id"]]:
            raise ValueError(f"selection parameters differ from preregistered spec: {item['id']}")
        output.append(item)
    if len({item["id"] for item in output}) != len(output):
        raise ValueError("selection contains duplicate candidate IDs")
    return output


def development_cases(protocol: dict[str, Any], spec: dict[str, Any], mode: str):
    """Two references plus 24 balanced, sign-symmetric collision cases."""
    design = protocol["development"]
    conditions = design["conditions"]
    rows = [
        R.case_row(spec, mode, condition, index, stage="development",
                   phase="none", direction="none", force_n=0.)
        for index, condition in enumerate(conditions)
    ]
    # Each of eight directions occurs once at each phase. Aggregate force count
    # is 12/12; each opposite-sign pair has the same phase/force multiset.
    directions = design["directions"]
    times = design["times_s"]
    forces = design["forces_N"]
    for phase_index, collision_time in enumerate(times):
        for direction_index, direction in enumerate(directions):
            pair_index = direction_index // 2
            force = forces[(phase_index + pair_index) % 2]
            # Orthogonalize the two conditions from force: every force/condition
            # cell has six rows, while opposite signs retain matched conditions.
            condition = conditions[(phase_index + pair_index // 2) % len(conditions)]
            rows.append(R.case_row(
                spec, mode, condition, len(rows),
                collision=(float(collision_time), _force(direction, force)),
                stage="development", phase=("pick", "transport", "place")[phase_index],
                direction=direction, force_n=float(force),
            ))
    if len(rows) != 26:
        raise AssertionError("development design must have 2 references + 24 collisions")
    return rows


def d4_cases(spec: dict[str, Any], mode: str):
    """Convert the unchanged 385-case D4 design to R2 runner rows."""
    legacy = d4_robustness.build_cases("unused", "unused")
    rows = []
    for item in legacy:
        condition = {"mass": item["mass"], "com": item["com"]}
        rows.append(R.case_row(
            spec, mode, condition, item["case_id"],
            collision=item["collision"], pick_offset=item["pick_offset"],
            stage="d4", phase=item["phase"], direction=item["direction"],
            force_n=float(item["force_n"]),
        ))
    if len(rows) != 385:
        raise AssertionError("D4 design drifted from 385 cases")
    return rows


def build_holdout_design(protocol: dict[str, Any], rng) -> list[dict[str, Any]]:
    """Construct one shared condition design from an already-created RNG.

    Tests pass a fake-seed RNG. Production must call this only through
    :func:`open_holdout`, after the exclusive marker has been written.
    """
    design = protocol["holdout"]
    masses = np.asarray(design["masses_kg"], dtype=float)
    coms = np.column_stack((
        rng.uniform(-.028, .028, len(masses)),
        rng.uniform(-.018, .018, len(masses)),
        rng.uniform(-.012, .048, len(masses)),
    ))
    conditions = [dict(mass=float(mass), com=com.tolist())
                  for mass, com in zip(masses, coms, strict=True)]
    rows: list[dict[str, Any]] = [dict(kind="reference", condition=c)
                                 for c in conditions]
    phases = np.repeat(np.asarray(tuple(HOLDOUT_PHASES), dtype=object), 10)
    forces = np.tile(np.asarray(design["forces_N"], dtype=float), 6)
    directions = np.tile(np.asarray(design["directions"], dtype=object), 3)
    condition_indices = np.resize(np.arange(len(conditions)), 30)
    for values in (phases, forces, directions, condition_indices):
        rng.shuffle(values)
    for index in range(30):
        phase = str(phases[index])
        force = float(forces[index])
        direction = str(directions[index])
        rows.append(dict(
            kind="collision", condition=conditions[int(condition_indices[index])],
            collision=(float(rng.uniform(*HOLDOUT_PHASES[phase])),
                       _force(direction, force)),
            phase=phase, direction=direction, force_n=force,
        ))
    for index, offset in enumerate(design["offsets_m"]):
        rows.append(dict(kind="offset", condition=conditions[index],
                         pick_offset=[float(x) for x in offset], phase="pick",
                         direction="offset", force_n=0.))
    if len(rows) != 38:
        raise AssertionError("holdout design must have 4 references + 30 collisions + 4 offsets")
    return rows


def _git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def _required_freeze_paths(selection: list[dict[str, Any]], freeze) -> set[str]:
    required = {
        "results/v2_protocol/PROTOCOL_R2.json",
        "tools/r2_extended_study.py", "tools/r2_study.py", "tools/v2_study.py",
        "tools/v2_resources.py", "tools/check_integrity.py",
        "tools/kit_manifest.sha256",
        "sim/config.py", "sim/trajectory.py",
        "r1_servo.py", "r1_torque.py", "r1_detector.py",
    }
    families = {spec["family"] for spec in selection}
    if families & {"C1", "C2", "C6"}:
        required.add("experiments/r2_contact.py")
    if families & {"C3", "C4"}:
        required.add("experiments/r2_robust.py")
    if "C5" in families:
        required.add("experiments/r2_prne.py")
    if "base" in families:
        required.update(("servo_controller.py", "torque_controller.py",
                         "collision_detector.py"))
    if "v1" in families:
        required.update(("v1_servo.py", "v1_torque.py", "v1_detector.py"))
    if freeze.get("ridge_model"):
        required.add(str(freeze["ridge_model"]))
    return required


def verify_freeze(selection: list[dict[str, Any]], freeze_path: Path = FREEZE):
    """Verify protocol, selected specs, bytes and their local commit proof."""
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze_path.resolve() == FREEZE.resolve():
        relative_freeze = str(FREEZE.relative_to(ROOT))
        try:
            committed_freeze = _git_bytes("HEAD", relative_freeze)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("holdout freeze is not committed at HEAD") from exc
        if committed_freeze != FREEZE.read_bytes():
            raise RuntimeError("working-tree holdout freeze differs from committed HEAD bytes")
    if freeze.get("protocol_sha256") != R.sha(PROTOCOL):
        raise RuntimeError("holdout freeze protocol hash mismatch")
    if freeze.get("chosen_specs") != selection:
        raise RuntimeError("holdout selection differs from committed freeze")
    commit = freeze.get("local_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("holdout freeze lacks full local commit ID")
    subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = freeze.get("source_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("holdout freeze lacks source/config/weight hashes")
    missing = _required_freeze_paths(selection, freeze) - set(files)
    if missing:
        raise RuntimeError("holdout freeze lacks required source/config/weight hashes: "
                           + ", ".join(sorted(missing)))
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (ROOT / path).is_file():
            raise RuntimeError(f"unsafe/missing frozen path: {relative}")
        current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        committed = hashlib.sha256(_git_bytes(commit, relative)).hexdigest()
        if current != expected or committed != expected:
            raise RuntimeError(f"current/committed freeze mismatch: {relative}")
    ridge_path = freeze.get("ridge_model")
    if ridge_path:
        ridge = json.loads((ROOT / ridge_path).read_text(encoding="utf-8"))
        R.validate_ridge(ridge)
    if any(spec["id"] == "C5_prne_plus_ridge" for spec in selection) and not ridge_path:
        raise RuntimeError("pRNE+ridge holdout selection lacks a frozen ridge model")
    return freeze


def _exclusive_marker(marker: Path, freeze: dict[str, Any], selection):
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema="R2_HOLDOUT_OPENED.v1",
                   opened_utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=R.sha(PROTOCOL), local_commit=freeze["local_commit"],
                   chosen_specs=selection, batch_attempt=1,
                   statement="exclusive marker created before seeded conditions")
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError("holdout is already opened; a second batch is forbidden") from exc
    return payload


def open_holdout(protocol, selection, marker: Path = HOLDOUT_MARKER,
                 freeze_path: Path = FREEZE, ridge=None):
    """Verify freeze, durably mark opened, then and only then touch real seed."""
    freeze = verify_freeze(selection, freeze_path)
    ridge_path = freeze.get("ridge_model")
    if ridge_path:
        frozen_ridge = json.loads((ROOT / ridge_path).read_text(encoding="utf-8"))
        if ridge != frozen_ridge:
            raise RuntimeError("CLI ridge model differs from the frozen model")
    elif ridge is not None:
        raise RuntimeError("CLI ridge model was not included in the holdout freeze")
    opened = _exclusive_marker(marker, freeze, selection)
    # Do not move this line above marker creation: the seed is sealed evidence.
    rng = np.random.default_rng(int(protocol["holdout"]["seed"]))
    return opened, build_holdout_design(protocol, rng), freeze


def _execute_rows(rows, private_dir, ridge):
    completed = []
    for row in rows:
        result = R.run_case(row, private_dir, ridge)
        completed.append(result)
        if len(completed) % 10 == 0:
            print(row["stage"], row["variant"], row["architecture"],
                  len(completed), "/", len(rows), flush=True)
    sweep.classify_rows(completed)
    return completed


def run_regular_stage(stage, protocol, selection, private_dir, ridge):
    hashes = source_hashes()
    for spec in selection:
        target = D / f"R2_{stage.upper()}_{spec['id']}.json"
        if target.exists():
            raise RuntimeError(f"refuse to overwrite completed study: {target.name}")
        rows = []
        for mode in ("servo", "torque"):
            design = (development_cases(protocol, spec, mode) if stage == "development"
                      else d4_cases(spec, mode))
            rows.extend(_execute_rows(design, private_dir, ridge))
        if hashes != source_hashes():
            raise RuntimeError("source changed during study; no valid result written")
        R.write_group(target, R.metadata(stage, hashes), rows)


def holdout_rows(shared_design, spec, mode):
    rows = []
    for index, design in enumerate(shared_design):
        rows.append(R.case_row(
            spec, mode, design["condition"], index,
            collision=design.get("collision"), pick_offset=design.get("pick_offset"),
            stage="holdout", phase=design.get("phase", "none"),
            direction=design.get("direction", "none"),
            force_n=float(design.get("force_n", 0.)),
        ))
    return rows


def run_holdout(protocol, selection, private_dir, ridge):
    target = D / "R2_HOLDOUT.json"
    if target.exists():
        raise RuntimeError("holdout result already exists; second batch forbidden")
    opened, shared, freeze = open_holdout(protocol, selection, ridge=ridge)
    hashes = source_hashes()
    if freeze["source_files"].get("tools/r2_study.py") != hashes["tools/r2_study.py"]:
        raise RuntimeError("frozen runner source differs before holdout execution")
    rows = []
    for spec in selection:
        for mode in ("servo", "torque"):
            rows.extend(_execute_rows(holdout_rows(shared, spec, mode), private_dir, ridge))
    if hashes != source_hashes():
        raise RuntimeError("source changed during holdout; opened attempt retained, no rewrite allowed")
    meta = R.metadata("holdout", hashes)
    meta.update(opened_marker_sha256=R.sha(HOLDOUT_MARKER), freeze_sha256=R.sha(FREEZE),
                one_batch=True, cases_per_spec_architecture=38)
    R.write_group(target, meta, rows)


def _load_perturb_harness(path: Path):
    path = R.require_private(path)
    spec = importlib.util.spec_from_file_location("r2_private_perturb_harness", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load perturbation harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("run_perturbed", "matrix_cases"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"perturbation harness lacks {name}")
    return module


def _parity_result(first, second, log1, log2):
    keys = ("t", "e_mm", "q", "q_d", "tau", "f_table", "f_tool",
            "f_ext", "r", "detect")
    arrays = {key: bool(np.array_equal(log1[key], log2[key])) for key in keys}
    result = all(first[key] == second[key] for key in first
                 if key != "ctrl_time_us" and not isinstance(first[key], np.ndarray))
    return dict(arrays=arrays, results=bool(result), passed=bool(all(arrays.values()) and result))


def run_perturb(protocol, selection, private_dir, ridge, harness_path):
    """Analysis-only instance perturbations, gated by nominal parity."""
    target = D / "R2_PERTURBATIONS.json"
    if target.exists():
        raise RuntimeError("refuse to overwrite perturbation study")
    private_dir = R.require_private(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    harness_path = R.require_private(harness_path)
    harness_hash = R.sha(harness_path)
    harness = _load_perturb_harness(harness_path)
    hashes = source_hashes()
    parity, rows = [], []
    for spec in selection:
        for mode in ("servo", "torque"):
            ctrl, det = R.candidate_classes(mode, spec, ridge)
            resources.wait_for_capacity()
            first, log1 = run_episode(ctrl(), box_mass=4.3, detector=det())
            resources.wait_for_capacity()
            second, log2 = harness.run_perturbed(
                ctrl(), box_mass=4.3, detector=det(), perturb={"seed": 7})
            evidence = _parity_result(first, second, log1, log2)
            evidence.update(variant=spec["id"], architecture=mode)
            parity.append(evidence)
            if not evidence["passed"]:
                raise RuntimeError("perturbation runner nominal parity failed; no perturbation evidence is valid")

            cases = list(harness.matrix_cases())
            for seed in protocol["seeds"]["analysis_noise"]:
                for mass in (0., 4.3, 8.):
                    cases.append(dict(family="new_noise", seed=int(seed), mass=mass,
                                      com=(.02, -.01, .03), viscous_scale=1.,
                                      stiffness_scale=1., coulomb_scale=1., collision=None))
            group = []
            for index, case in enumerate(cases):
                TC, TD = V.timed_classes(ctrl, det)
                controller, detector = TC(), TD()
                perturb = {key: case[key] for key in
                           ("seed", "viscous_scale", "stiffness_scale", "coulomb_scale")}
                before = resources.wait_for_capacity()
                result, log = harness.run_perturbed(
                    controller, box_mass=case["mass"],
                    box_com=case.get("com", (.02, -.01, .03)), detector=detector,
                    collision=case.get("collision"), perturb=perturb)
                row = dict(case, variant=spec["id"], specification=spec,
                           architecture=mode, official=False, case_id=index, stage="perturb")
                row.update(sweep.evaluate_control_metrics(result, log))
                row.update(g4=V.g4_window_metrics(log, case["mass"]),
                           timing=V.timing_summary(controller, detector),
                           host_before=before, host_after=resources.snapshot(),
                           detect_delay_ms=result.get("detect_delay_ms"),
                           alarm_before_collision=bool(result.get("alarm_before_collision", False)),
                           false_alarms=int(result["false_alarms"]),
                           tool_contact_delay_ms=result.get("tool_contact_delay_ms"))
                row["_detect"] = log["detect"].copy()
                raw = private_dir / (f"perturb_{spec['id']}_{mode}_{index:04}.npz")
                if raw.exists():
                    raise RuntimeError(f"refuse to overwrite raw perturbation evidence: {raw.name}")
                np.savez_compressed(raw, **{name: value for name, value in log.items()
                                           if isinstance(value, np.ndarray)})
                row.update(private_raw_basename=raw.name,
                           private_raw_sha256=R.sha(raw))
                group.append(row)
            reference_detect = {}
            for row in group:
                key = tuple(row.get(name) for name in
                            ("mass", "seed", "viscous_scale", "stiffness_scale",
                             "coulomb_scale"))
                if row.get("collision") is None:
                    reference_detect[key] = row["_detect"]
            for row in group:
                if row.get("collision") is not None:
                    key = tuple(row.get(name) for name in
                                ("mass", "seed", "viscous_scale", "stiffness_scale",
                                 "coulomb_scale"))
                    if key not in reference_detect:
                        raise RuntimeError("perturbation collision lacks its matched reference")
                    row.update(sweep.classify_collision(
                        row, reference_detect[key], row["collision"][0]))
                row.pop("_detect")
            rows.extend(group)
    if hashes != source_hashes() or harness_hash != R.sha(harness_path):
        raise RuntimeError("source changed during perturbation study")
    meta = R.metadata("perturb", hashes)
    meta.update(classification="analysis-only instance perturbations; never official or fit data",
                harness_sha256=harness_hash, nominal_parity=parity)
    R.write_group(target, meta, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("development", "d4", "holdout", "perturb"))
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--ridge", type=Path)
    parser.add_argument("--perturb-harness", type=Path)
    args = parser.parse_args(argv)
    args.private_dir = R.require_private(args.private_dir)
    selection = load_selection(args.selection)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    ridge = json.loads(args.ridge.read_text(encoding="utf-8")) if args.ridge else None
    check_integrity(ROOT)
    if args.stage in ("development", "d4"):
        run_regular_stage(args.stage, protocol, selection, args.private_dir, ridge)
    elif args.stage == "holdout":
        run_holdout(protocol, selection, args.private_dir, ridge)
    else:
        if args.perturb_harness is None:
            parser.error("--perturb-harness is required for perturb")
        run_perturb(protocol, selection, args.private_dir, ridge, args.perturb_harness)
    check_integrity(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
