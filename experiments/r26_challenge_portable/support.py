"""Fail-closed validation for a finalized R26 public replay release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = (
    "experiments/r26_holdout/__init__.py",
    "experiments/r26_holdout/plan.json",
    "experiments/r26_holdout/runner.py",
    "experiments/r26_holdout/evaluator.py",
    "experiments/r26_holdout/protocol.json",
    "experiments/r26_holdout/verify.py",
    "experiments/r26_holdout/closure.json",
)
SELECTION_SCHEMA = "r26-finalized-exposed-choice.v1"
LOCK_SCHEMA = "r26-candidate-lock.v1"
CLOSURE_SCHEMA = "r26-independent-authored-closure.v1"
EXPECTED_ENTRY_PREFIX = "experiments.r26_"
EXPECTED_SOURCE_PREFIX = "experiments/r26_"


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def contains_absolute_string(value):
    if isinstance(value, dict):
        return any(contains_absolute_string(key) or contains_absolute_string(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(contains_absolute_string(item) for item in value)
    return isinstance(value, str) and (value.startswith("/") or "\\Users\\" in value)


def runtime_versions():
    import mujoco
    import numpy

    versions = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "mujoco": mujoco.__version__,
    }
    if versions["mujoco"] != "3.17.0":
        raise ValueError("R26 replay requires the course MuJoCo 3.17.0 runtime")
    return versions


def regular_json(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular finalized {label} required")
    raw = path.read_bytes()
    value = json.loads(raw)
    if contains_absolute_string(value):
        raise ValueError(f"nonportable {label}")
    return raw, value


def repository_relative(path):
    try:
        relative = Path(path).resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("release contract files must be inside the public source root") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("repository-relative release contract required")
    return relative.as_posix()


def read_release(selection_path, lock_path, closure_path):
    selection_raw, selection = regular_json(selection_path, "selection")
    lock_raw, lock = regular_json(lock_path, "candidate lock")
    _, closure = regular_json(closure_path, "source closure")
    entrypoint = selection.get("entrypoint")
    if (selection.get("schema") != SELECTION_SCHEMA
            or selection.get("status") != "FINALIZED_BEFORE_INDEPENDENT_OUTPUT"
            or selection.get("selected_for_independent") is None
            or selection.get("default_promoted") is not False
            or not isinstance(entrypoint, str)
            or not entrypoint.startswith(EXPECTED_ENTRY_PREFIX)
            or entrypoint.startswith("experiments.r26_holdout")
            or not entrypoint.endswith(":Detector")
            or selection.get("plan_sha256") != digest(ROOT / HOLDOUT[1])
            or selection.get("holdout_closure_sha256") != digest(ROOT / HOLDOUT[6])):
        raise ValueError("R26 selection is absent, stale, or not finalized")
    selected_pins = selection.get("selected_source_pins")
    if not isinstance(selected_pins, dict) or not selected_pins:
        raise ValueError("finalized R26 selected source pins required")
    protocol = json.loads((ROOT / HOLDOUT[4]).read_bytes())
    if (lock.get("schema") != LOCK_SCHEMA
            or lock.get("status") != "LOCKED_BEFORE_HOLDOUT_OUTCOMES"
            or lock.get("entrypoint") != entrypoint
            or lock.get("plan_sha256") != digest(ROOT / HOLDOUT[1])
            or lock.get("protocol_sha256") != digest(ROOT / HOLDOUT[4])
            or lock.get("evaluator_sha256") != digest(ROOT / HOLDOUT[3])
            or lock.get("holdout_closure_sha256") != digest(ROOT / HOLDOUT[6])
            or lock.get("selection_sha256") != digest(selection_raw)
            or lock.get("candidate_was_selected_without_holdout_output") is not True
            or lock.get("challenge_detail_was_not_disclosed_by_verifier_before_lock") is not True
            or lock.get("frozen_protocol_sources") != protocol["source_pins"]):
        raise ValueError("R26 candidate lock is absent, stale, or not final")
    sources = lock.get("transitive_sources")
    if (not isinstance(sources, dict) or not sources
            or not set(selected_pins) <= set(sources)):
        raise ValueError("complete selected R26 transitive closure required")
    for name, expected in sources.items():
        rel = Path(name)
        source = ROOT / rel
        if (rel.is_absolute() or ".." in rel.parts
                or not name.startswith(EXPECTED_SOURCE_PREFIX)
                or name.startswith("experiments/r26_holdout")
                or len(expected) != 64 or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("candidate source pin differs: " + name)
    for name, expected in selected_pins.items():
        if sources.get(name) != expected:
            raise ValueError("selected source pin differs from candidate lock: " + name)
    selection_rel = repository_relative(selection_path)
    lock_rel = repository_relative(lock_path)
    closure_rel = repository_relative(closure_path)
    files = closure.get("files")
    if (closure.get("schema") != CLOSURE_SCHEMA
            or closure.get("status") != "FINALIZED_BEFORE_INDEPENDENT_OUTPUT"
            or closure.get("entrypoint") != entrypoint
            or closure.get("selection_path") != selection_rel
            or closure.get("selection_sha256") != digest(selection_raw)
            or closure.get("candidate_lock_path") != lock_rel
            or closure.get("candidate_lock_sha256") != digest(lock_raw)
            or closure.get("holdout_closure_sha256") != digest(ROOT / HOLDOUT[6])
            or closure.get("kit_included") is not False
            or not isinstance(files, dict) or not files):
        raise ValueError("R26 authored source closure is absent, stale, or incomplete")
    if closure_rel in files:
        raise ValueError("self-referential closure pin is not permitted")
    for name, expected in files.items():
        rel = Path(name)
        source = ROOT / rel
        if (rel.is_absolute() or ".." in rel.parts
                or name == "evaluate.py" or name.startswith(("sim/", "model/"))
                or len(expected) != 64 or source.is_symlink() or not source.is_file()
                or digest(source) != expected):
            raise ValueError("public source closure differs: " + name)
    overlay = closure.get("overlay_files")
    public = closure.get("public_files")
    if (not isinstance(overlay, list) or not overlay
            or not isinstance(public, list) or not public
            or len(overlay) != len(set(overlay))
            or len(public) != len(set(public))
            or not set(overlay + public) <= set(files)):
        raise ValueError("R26 closure file accounting differs")
    repository_protocol = tuple(name for name in protocol["source_pins"]
                                if not name.startswith(("sim/", "model/")))
    required_overlay = set(HOLDOUT) | set(repository_protocol) | set(sources)
    if set(overlay) != required_overlay:
        raise ValueError("R26 overlay closure is not exact")
    return {
        "selection_raw": selection_raw,
        "selection": selection,
        "lock_raw": lock_raw,
        "lock": lock,
        "closure": closure,
        "entrypoint": entrypoint,
        "files": files,
        "overlay_files": overlay,
    }


def snapshot(release):
    blobs = {name: (ROOT / name).read_bytes() for name in release["overlay_files"]}
    pins = {name: digest(blob) for name, blob in blobs.items()}
    expected = release["files"]
    if any(pins[name] != expected[name] for name in pins):
        raise ValueError("R26 public dependency changed during snapshot")
    return blobs, pins
