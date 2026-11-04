#!/usr/bin/env python3
"""R3 development, D4, perturbation, and sealed-holdout orchestration.

Importing this module is side-effect free: it neither constructs holdout
conditions nor touches the registered holdout seed.  Real holdout design
generation is reachable only through :func:`open_holdout`, after freeze
verification and durable exclusive-marker creation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from sim.runner import run_episode
from tools import d4_robustness
from tools.check_integrity import check_integrity
from tools.r2_extended_study import _load_perturb_harness as _load_r2_harness
from tools import r3_study as R


D = ROOT / "results" / "v2_protocol"
PROTOCOL = D / "PROTOCOL_R3.json"
FREEZE = D / "R3_FREEZE.json"
HOLDOUT_MARKER = D / "R3_HOLDOUT_OPENED.json"
HOLDOUT_RESULT = D / "R3_HOLDOUT.json"
PERTURB_RESULT = D / "R3_PERTURBATIONS.json"
PERTURB_PARITY = D / "R3_PERTURB_PARITY.json"

DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
    "+xy": (1.0, 1.0, 0.0),
    "-xy": (-1.0, -1.0, 0.0),
    "+xyz": (1.0, 1.0, 1.0),
    "-xyz": (-1.0, -1.0, -1.0),
}


def _force(direction, magnitude):
    vector = np.asarray(DIRECTIONS[direction], dtype=float)
    return (float(magnitude) * vector / np.linalg.norm(vector)).tolist()


def _canonical_specs():
    values = {
        (spec["architecture"], spec["id"]): spec
        for spec in R.specifications()
    }
    for mode in ("servo", "torque"):
        spec = R.baseline_spec(mode)
        values[(mode, "R1")] = spec
    return values


def _validate_selection(selected):
    canonical = _canonical_specs()
    output = []
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("each chosen spec must be an object")
        key = (item.get("architecture"), item.get("id"))
        if key not in canonical or item != canonical[key]:
            raise ValueError("selection contains an altered or unknown spec")
        output.append(item)
    modes = [item["architecture"] for item in output]
    if sorted(modes) != ["servo", "torque"] or len(set(modes)) != 2:
        raise ValueError("selection must choose exactly one servo and one torque source")
    return output


def load_selection(path):
    """Load exactly one canonical chosen source for each architecture."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = value.get("chosen_specs") if isinstance(value, dict) else None
    if not isinstance(selected, list):
        raise ValueError("selection must contain a chosen_specs list")
    return _validate_selection(selected)


def selected_with_r1(selection, include_r1=False):
    output = list(selection)
    if include_r1:
        present = {(spec["architecture"], spec["id"]) for spec in output}
        for mode in ("servo", "torque"):
            if (mode, "R1") not in present:
                output.append(R.baseline_spec(mode))
    return output


def orchestration_hashes():
    return {
        relative: R.sha(ROOT / relative)
        for relative in (
            "tools/r3_extended.py",
            "tools/d4_robustness.py",
            "tools/r2_extended_study.py",
            "results/v2_protocol/PROTOCOL_R3.json",
        )
    }


def development_cases(protocol, spec):
    """Build the registered two-reference plus 24 balanced collision design."""
    design = protocol["development"]
    conditions = design["conditions"]
    rows = [
        R.case_row(
            spec,
            condition,
            index,
            stage="development",
            phase="none",
            direction="none",
            force_n=0.0,
        )
        for index, condition in enumerate(conditions)
    ]
    phases = ("pick", "transport", "place")
    for phase_index, collision_time in enumerate(design["times_s"]):
        for direction_index, direction in enumerate(design["directions"]):
            pair_index = direction_index // 2
            force = float(design["forces_N"][(phase_index + pair_index) % 2])
            condition = conditions[
                (phase_index + pair_index // 2) % len(conditions)
            ]
            rows.append(
                R.case_row(
                    spec,
                    condition,
                    len(rows),
                    stage="development",
                    collision=(float(collision_time), _force(direction, force)),
                    phase=phases[phase_index],
                    direction=direction,
                    force_n=force,
                )
            )
    if len(rows) != int(design["rows_per_architecture"]):
        raise AssertionError("R3 development design must contain exactly 26 rows")
    return rows


def d4_cases(spec):
    """Adapt the unchanged read-only 385-row D4 design to R3 case rows."""
    rows = []
    for item in d4_robustness.build_cases("unused", "unused"):
        condition = {"mass": item["mass"], "com": item["com"]}
        rows.append(
            R.case_row(
                spec,
                condition,
                item["case_id"],
                stage="d4",
                collision=item["collision"],
                pick_offset=item["pick_offset"],
                phase=item["phase"],
                direction=item["direction"],
                force_n=float(item["force_n"]),
            )
        )
    if len(rows) != 385:
        raise AssertionError("D4 design drifted from the registered 385 rows")
    return rows


def _neutral_perturb(case):
    return {
        "seed": int(case.get("seed", 0)),
        "stiffness_scale": float(case.get("stiffness_scale", 1.0)),
        "viscous_scale": float(case.get("viscous_scale", 1.0)),
        "coulomb_scale": float(case.get("coulomb_scale", 1.0)),
    }


def perturb_cases(protocol, spec, harness):
    """Build the frozen physical matrix plus the nine registered noise rows."""
    matrix = list(harness.matrix_cases())
    for seed in protocol["perturbations"]["noise_seeds"]:
        for mass in protocol["perturbations"]["new_noise_masses_kg"]:
            matrix.append(
                {
                    "family": "new_noise",
                    "seed": int(seed),
                    "mass": float(mass),
                    "com": (0.02, -0.01, 0.03),
                    "stiffness_scale": 1.0,
                    "viscous_scale": 1.0,
                    "coulomb_scale": 1.0,
                    "collision": None,
                }
            )
    rows = []
    for index, case in enumerate(matrix):
        perturb = _neutral_perturb(case)
        condition = {
            "mass": float(case["mass"]),
            "com": list(case.get("com", (0.02, -0.01, 0.03))),
        }
        collision = case.get("collision")
        rows.append(
            R.case_row(
                spec,
                condition,
                index,
                stage="perturb",
                collision=collision,
                perturb=perturb,
                perturb_family=case.get("family", "physical_matrix"),
                seed=perturb["seed"],
                stiffness_scale=perturb["stiffness_scale"],
                viscous_scale=perturb["viscous_scale"],
                coulomb_scale=perturb["coulomb_scale"],
            )
        )
    return rows


def _load_perturb_harness(path):
    """Load the read-only private R2 perturbation harness by explicit path."""
    path = R.require_private(path)
    return _load_r2_harness(path)


def _parity_result(first, second, log1, log2):
    """Compare every non-timing result and every observation/physical array."""
    result_keys = sorted(set(first) | set(second))
    result_equal = {}
    for key in result_keys:
        if key == "ctrl_time_us":
            continue
        left, right = first.get(key), second.get(key)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            result_equal[key] = bool(np.array_equal(left, right))
        else:
            result_equal[key] = bool(left == right)
    array_keys = sorted(
        key
        for key in set(log1) | set(log2)
        if key != 't_ctrl' and (isinstance(log1.get(key), np.ndarray)
                               or isinstance(log2.get(key), np.ndarray))
    )
    arrays = {
        key: bool(np.array_equal(log1.get(key), log2.get(key)))
        for key in array_keys
    }
    passed = bool(all(result_equal.values()) and all(arrays.values()))
    return {"arrays": arrays, "results": result_equal, "passed": passed}


def nominal_parity(spec, harness):
    """Prove neutral harness parity at 4.3 kg before perturbation evidence."""
    controller_class, detector_class = R.candidate_classes(
        spec["architecture"], spec
    )
    R.resources.wait_for_capacity()
    first, log1 = run_episode(
        controller_class(), box_mass=4.3, detector=detector_class()
    )
    neutral = {
        "seed": 7,
        "stiffness_scale": 1.0,
        "viscous_scale": 1.0,
        "coulomb_scale": 1.0,
    }
    R.resources.wait_for_capacity()
    second, log2 = harness.run_perturbed(
        controller_class(),
        box_mass=4.3,
        detector=detector_class(),
        perturb=neutral,
    )
    evidence = _parity_result(first, second, log1, log2)
    evidence.update(
        variant=spec["id"],
        architecture=spec["architecture"],
        mass_kg=4.3,
        neutral_seed=7,
    )
    return evidence


def run_regular_stage(stage, protocol, selection, private_dir, include_r1=False):
    if stage not in ("development", "d4"):
        raise ValueError("regular stage must be development or d4")
    hashes = orchestration_hashes()
    results = []
    for spec in selected_with_r1(selection, include_r1):
        cases = development_cases(protocol, spec) if stage == "development" else d4_cases(spec)
        results.append(R.run_group(stage, spec, cases, private_dir))
    if hashes != orchestration_hashes():
        raise RuntimeError("R3 orchestration source changed during stage")
    return results


def run_perturb(protocol, selection, private_dir, harness_path, include_r1=False):
    """Run parity for every source first, then analysis-only perturbations."""
    if PERTURB_RESULT.exists() or PERTURB_PARITY.exists():
        raise RuntimeError("R3 perturbation evidence already exists")
    harness_path = R.require_private(harness_path)
    harness = _load_perturb_harness(harness_path)
    specs = selected_with_r1(selection, include_r1)
    harness_hash = R.sha(harness_path)
    source_hashes = orchestration_hashes()

    parity = [nominal_parity(spec, harness) for spec in specs]
    if not all(row["passed"] for row in parity):
        raise RuntimeError(
            "perturbation runner nominal parity failed before perturbation data"
        )
    R.write(
        PERTURB_PARITY,
        {
            "schema": "R3_PERTURB_PARITY.v1",
            "official": False,
            "harness_sha256": harness_hash,
            "results": parity,
        },
    )

    groups = []
    for spec in specs:
        cases = perturb_cases(protocol, spec, harness)
        result = R.run_group(
            "perturb", spec, cases, private_dir, harness=harness
        )
        target = D / f"R3_PERTURB_{spec['id']}_{spec['architecture']}.json"
        groups.append(
            {
                "variant": spec["id"],
                "architecture": spec["architecture"],
                "result_file": target.name,
                "result_sha256": R.sha(target),
                "rows": len(result["results"]),
            }
        )
    if harness_hash != R.sha(harness_path):
        raise RuntimeError("private perturbation harness changed during study")
    if source_hashes != orchestration_hashes():
        raise RuntimeError("R3 orchestration source changed during perturbations")
    aggregate = {
        "schema": "R3_PERTURBATIONS.v1",
        "official": False,
        "classification": "analysis-only; never official or fit data",
        "harness_sha256": harness_hash,
        "orchestration_sha256": source_hashes,
        "nominal_parity_file": PERTURB_PARITY.name,
        "nominal_parity_sha256": R.sha(PERTURB_PARITY),
        "groups": groups,
    }
    _assert_portable(aggregate)
    R.write(PERTURB_RESULT, aggregate)
    return aggregate


def build_holdout_design(protocol, rng):
    """Build the single 38-row shared design from an already-created RNG."""
    design = protocol["holdout"]
    excluded = {float(value) for value in design["exclude_masses_kg"]}
    digits = int(design["mass_round_digits"])
    masses = []
    for lower, upper in design["mass_ranges_kg"]:
        for _ in range(1000):
            mass = round(float(rng.uniform(lower, upper)), digits)
            if mass not in excluded and mass not in masses:
                masses.append(mass)
                break
        else:
            raise RuntimeError("could not draw a fresh holdout mass")

    low = np.asarray(design["com_ranges_m"][0], dtype=float)
    high = np.asarray(design["com_ranges_m"][1], dtype=float)
    conditions = [
        {"mass": mass, "com": rng.uniform(low, high).tolist()}
        for mass in masses
    ]
    rows = [{"kind": "reference", "condition": condition} for condition in conditions]

    phases = tuple(design["phase_time_ranges_s"])
    directions = list(design["directions"])
    if len(directions) != 10 or any(
        directions[index][0] != "+" or directions[index + 1][0] != "-"
        for index in range(0, len(directions), 2)
    ):
        raise ValueError("holdout directions must remain five signed pairs")
    for pair_index in range(5):
        for phase_index, phase in enumerate(phases):
            collision_time = float(
                rng.uniform(*design["phase_time_ranges_s"][phase])
            )
            force = float(rng.uniform(*design["force_range_N"]))
            condition = conditions[(pair_index + phase_index) % len(conditions)]
            for direction in directions[2 * pair_index : 2 * pair_index + 2]:
                rows.append(
                    {
                        "kind": "collision",
                        "condition": condition,
                        "collision": (
                            collision_time,
                            _force(direction, force),
                        ),
                        "phase": phase,
                        "direction": direction,
                        "force_n": force,
                    }
                )
    for condition, offset in zip(
        conditions, design["pick_offsets_m"], strict=True
    ):
        rows.append(
            {
                "kind": "offset",
                "condition": condition,
                "pick_offset": [float(value) for value in offset],
                "phase": "pick",
                "direction": "offset",
                "force_n": 0.0,
            }
        )
    counts = {
        kind: sum(row["kind"] == kind for row in rows)
        for kind in ("reference", "collision", "offset")
    }
    expected = design["rows_per_source_architecture"]
    if counts != {
        "reference": int(expected["references"]),
        "collision": int(expected["collisions"]),
        "offset": int(expected["offsets"]),
    }:
        raise AssertionError("holdout design row counts changed")
    return rows


def holdout_rows(shared_design, spec):
    rows = []
    for index, design in enumerate(shared_design):
        rows.append(
            R.case_row(
                spec,
                design["condition"],
                index,
                stage="holdout",
                collision=design.get("collision"),
                pick_offset=design.get("pick_offset"),
                phase=design.get("phase", "none"),
                direction=design.get("direction", "none"),
                force_n=float(design.get("force_n", 0.0)),
            )
        )
    return rows


def _git_bytes(commit, relative):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )


def _required_freeze_paths(selection):
    required = {
        "results/v2_protocol/PROTOCOL_R3.json",
        "tools/r3_extended.py",
        "tools/r3_study.py",
        "tools/r3_resources.py",
        "tools/r3_build.py",
        "tools/d4_robustness.py",
        "tools/r2_extended_study.py",
        "tools/r2_study.py",
        "tools/check_integrity.py",
        "tools/kit_manifest.sha256",
        "sim/config.py",
        "sim/trajectory.py",
        "sim/runner.py",
        "sim/nominal.py",
        "r1_servo.py",
        "r1_torque.py",
        "r1_detector.py",
    }
    for spec in selection:
        required.update(R.source_hashes(spec["architecture"], spec))
    return required


def verify_freeze(selection, freeze_path=FREEZE):
    """Verify committed freeze bytes, canonical selection, and Git objects."""
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "R3_FREEZE.v1":
        raise RuntimeError("unexpected R3 freeze schema")
    if freeze_path.resolve() == FREEZE.resolve():
        relative = FREEZE.relative_to(ROOT).as_posix()
        try:
            committed = _git_bytes("HEAD", relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("R3 freeze is not committed at HEAD") from exc
        if committed != freeze_path.read_bytes():
            raise RuntimeError("working R3 freeze differs from committed HEAD bytes")
    if freeze.get("protocol_sha256") != R.sha(PROTOCOL):
        raise RuntimeError("R3 freeze protocol hash mismatch")
    if freeze.get("chosen_specs") != selection:
        raise RuntimeError("R3 freeze selection mismatch")
    try:
        _validate_selection(freeze["chosen_specs"])
    except ValueError as exc:
        raise RuntimeError("R3 freeze contains a noncanonical selection") from exc
    commit = freeze.get("local_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("R3 freeze lacks a full local commit ID")
    subprocess.run(
        ["git", "cat-file", "-e", commit + "^{commit}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    fallback = freeze.get("fallback_R1_sha256")
    if fallback != R.protocol()["fallback_R1_sha256"]:
        raise RuntimeError("R3 fallback R1 hashes differ from protocol")
    files = freeze.get("source_files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("R3 freeze lacks source_files")
    missing = _required_freeze_paths(selection) - set(files)
    if missing:
        raise RuntimeError(
            "R3 freeze lacks required sources: " + ", ".join(sorted(missing))
        )
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (ROOT / path).is_file():
            raise RuntimeError(f"unsafe or missing frozen source: {relative}")
        current = R.sha(ROOT / path)
        committed = hashlib.sha256(_git_bytes(commit, relative)).hexdigest()
        if current != expected or committed != expected:
            raise RuntimeError(f"current/committed source mismatch: {relative}")
    for name, expected in fallback.items():
        if files.get(name) != expected:
            raise RuntimeError(f"fallback source missing from source_files: {name}")
    verify_final_selection(selection, freeze)
    return freeze


def verify_final_selection(selection, freeze):
    """A frozen canonical spec alone never authorizes diagnostic revival."""
    proof = freeze.get('selection_evidence', {})
    relative = Path(proof.get('path', '/missing-selection-proof'))
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('freeze lacks safe final selection evidence')
    path = ROOT / relative
    expected = proof.get('sha256')
    if (not path.is_file() or R.sha(path) != expected
            or freeze['source_files'].get(relative.as_posix()) != expected):
        raise RuntimeError('final selection evidence is not source-frozen')
    record = json.loads(path.read_text())
    if (record.get('schema') != 'R3_PREFREEZE_DECISION.v1'
            or record.get('chosen_specs') != selection
            or record.get('no_runner_up_or_diagnostic_revival') is not True):
        raise RuntimeError('final pre-holdout decision differs from frozen selection')
    for spec in selection:
        row = record.get('architectures', {}).get(spec['architecture'], {})
        if row.get('effective_selection') != spec:
            raise RuntimeError('effective final selection is inconsistent')
        if spec['id'] != 'R1' and (
                row.get('preholdout_adoption_eligible') is not True
                or row.get('irreversibly_ineligible_for_adoption') is not False):
            raise RuntimeError('rejected diagnostic source cannot be revived')
    sources = record.get('source_evidence_sha256')
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError('final decision lacks source-bound gate evidence')
    for name, digest in sources.items():
        source = Path(name)
        if source.is_absolute() or '..' in source.parts or R.sha(ROOT / source) != digest:
            raise RuntimeError('pre-holdout decision evidence changed')
    return record


def _exclusive_marker(marker, freeze, selection):
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "R3_HOLDOUT_OPENED.v1",
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": R.sha(PROTOCOL),
        "local_commit": freeze["local_commit"],
        "chosen_specs": selection,
        "batch_attempt": 1,
        "statement": "exclusive durable marker created before seeded conditions",
    }
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError("R3 holdout is already opened") from exc
    directory = os.open(marker.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def _write_private_design(path, payload):
    path = R.require_private(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(R.json_safe(payload), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError("private holdout design already exists") from exc
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def _resume_design(marker, design_path, freeze, selection):
    opened = json.loads(Path(marker).read_text(encoding="utf-8"))
    saved = json.loads(Path(design_path).read_text(encoding="utf-8"))
    if (
        opened.get("schema") != "R3_HOLDOUT_OPENED.v1"
        or opened.get("batch_attempt") != 1
        or opened.get("chosen_specs") != selection
        or opened.get("local_commit") != freeze["local_commit"]
        or saved.get("schema") != "R3_HOLDOUT_DESIGN.v1"
        or saved.get("marker_sha256") != R.sha(marker)
        or saved.get("protocol_sha256") != R.sha(PROTOCOL)
        or saved.get("local_commit") != freeze["local_commit"]
    ):
        raise RuntimeError("holdout resume evidence differs from the opened batch")
    _verify_design_seal(marker, design_path, freeze)
    return opened, saved["design"]


def design_seal_path(marker):
    return Path(marker).with_name('R3_HOLDOUT_DESIGN_SEAL.json')


def _design_digest(design):
    data = json.dumps(R.json_safe(design), sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _seal_design(marker, design_path, freeze):
    saved = json.loads(Path(design_path).read_text())
    seal = dict(schema='R3_HOLDOUT_DESIGN_SEAL.v1',
                marker_sha256=R.sha(marker), protocol_sha256=R.sha(PROTOCOL),
                local_commit=freeze['local_commit'], design_sha256=_design_digest(saved['design']))
    path = design_seal_path(marker)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(seal, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return seal


def _verify_design_seal(marker, design_path, freeze):
    seal = json.loads(design_seal_path(marker).read_text())
    saved = json.loads(Path(design_path).read_text())
    if (seal.get('schema') != 'R3_HOLDOUT_DESIGN_SEAL.v1'
            or seal.get('marker_sha256') != R.sha(marker)
            or seal.get('protocol_sha256') != R.sha(PROTOCOL)
            or seal.get('local_commit') != freeze['local_commit']
            or seal.get('design_sha256') != _design_digest(saved['design'])):
        raise RuntimeError('sealed holdout design changed; never reseed or repair')
    return seal


def open_holdout(protocol, selection, private_dir, marker=HOLDOUT_MARKER, freeze_path=FREEZE):
    """Verify freeze, mark once, then generate or resume the same sealed design."""
    private_dir = R.require_private(private_dir)
    design_path = private_dir / "R3_HOLDOUT_DESIGN.json"
    if protocol != json.loads(PROTOCOL.read_text(encoding="utf-8")):
        raise RuntimeError("holdout protocol object differs from registered bytes")
    freeze = verify_freeze(selection, freeze_path)
    marker = Path(marker)
    if marker.exists():
        if not design_path.is_file():
            raise RuntimeError("opened marker exists without resumable private design")
        opened, design = _resume_design(marker, design_path, freeze, selection)
        return opened, design, freeze

    opened = _exclusive_marker(marker, freeze, selection)
    # The real seed must remain below durable marker creation.
    rng = np.random.default_rng(int(protocol["holdout"]["seed"]))
    design = R.json_safe(build_holdout_design(protocol, rng))
    _write_private_design(
        design_path,
        {
            "schema": "R3_HOLDOUT_DESIGN.v1",
            "protocol_sha256": R.sha(PROTOCOL),
            "local_commit": freeze["local_commit"],
            "marker_sha256": R.sha(marker),
            "design": design,
        },
    )
    _seal_design(marker, design_path, freeze)
    return opened, design, freeze


def _assert_portable(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"private_dir", "private_path", "harness_path"}:
                raise RuntimeError("public aggregate exposes a private path")
            _assert_portable(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable(item)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise RuntimeError("public aggregate contains an absolute path")


def run_holdout(protocol, selection, private_dir):
    if HOLDOUT_RESULT.exists():
        raise RuntimeError("R3 holdout result already exists; second batch forbidden")
    opened, shared_design, freeze = open_holdout(
        protocol, selection, private_dir
    )
    # The predeclared experiment compares every final candidate to immutable
    # R1 in the same opened batch.  Exact R1 aliases are evaluated only once.
    specs = selected_with_r1(selection, include_r1=True)
    _verify_design_seal(HOLDOUT_MARKER, Path(private_dir) / 'R3_HOLDOUT_DESIGN.json', freeze)
    seal_sha256 = R.sha(design_seal_path(HOLDOUT_MARKER))
    hashes_before = {
        (spec["architecture"], spec["id"]): R.runtime_hashes(
            spec["architecture"], spec
        )
        for spec in specs
    }
    orchestration_before = orchestration_hashes()
    rows = []
    for spec in specs:
        group = [
            R.run_case(row, private_dir)
            for row in holdout_rows(shared_design, spec)
        ]
        R.classify_group(group)
        rows.extend(group)
    hashes_after = {
        key: R.runtime_hashes(key[0], next(
            spec for spec in specs
            if (spec["architecture"], spec["id"]) == key
        ))
        for key in hashes_before
    }
    if hashes_after != hashes_before:
        raise RuntimeError("source changed during opened R3 holdout")
    if orchestration_before != orchestration_hashes():
        raise RuntimeError("R3 orchestration source changed during opened holdout")
    _verify_design_seal(HOLDOUT_MARKER, Path(private_dir) / 'R3_HOLDOUT_DESIGN.json', freeze)
    result = {
        "schema": "R3_HOLDOUT.v1",
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "official": False,
            "one_batch": True,
            "batch_attempt": opened["batch_attempt"],
            "protocol_sha256": R.sha(PROTOCOL),
            "freeze_sha256": R.sha(FREEZE),
            "opened_marker_sha256": R.sha(HOLDOUT_MARKER),
            "design_seal_sha256": seal_sha256,
            "local_commit": freeze["local_commit"],
            "chosen_specs": selection,
            "evaluated_specs": specs,
            "cases_per_source_architecture": 38,
            "source_sha256": {
                f"{mode}:{variant}": hashes
                for (mode, variant), hashes in hashes_before.items()
            },
            "orchestration_sha256": orchestration_before,
        },
        "results": rows,
    }
    _assert_portable(result)
    R.write(HOLDOUT_RESULT, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("development", "d4", "perturb", "holdout"),
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--include-r1", action="store_true")
    args = parser.parse_args(argv)
    private_dir = R.require_private(args.private_dir)
    selection = load_selection(args.selection)
    protocol = R.protocol()
    check_integrity(ROOT)
    if args.stage in ("development", "d4"):
        run_regular_stage(
            args.stage, protocol, selection, private_dir, args.include_r1
        )
    elif args.stage == "perturb":
        if args.harness is None:
            parser.error("--harness is required for perturb")
        run_perturb(
            protocol,
            selection,
            private_dir,
            args.harness,
            args.include_r1,
        )
    else:
        if args.include_r1:
            parser.error("holdout uses only the immutable chosen sources")
        run_holdout(protocol, selection, private_dir)
    check_integrity(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
