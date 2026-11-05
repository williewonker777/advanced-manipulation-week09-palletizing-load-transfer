"""Seal R4 science, open one holdout batch, then run one official regression.

Importing this module is side-effect free: in particular, it does not construct
the holdout RNG or inspect official outputs.  Publication-only rendering is
deliberately outside the science freeze.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from tools.check_integrity import check_integrity
from tools import r3_official_regression as O3
from tools import r3_resources as resources
from tools import r3_study as R
from tools import r4_redesign as X
from tools import r4_review as Q
from tools import r4_study as S


D = ROOT / "results" / "v2_protocol"
PROTOCOL = D / "PROTOCOL_R4.json"
RESEAL = D / "PROTOCOL_R4_HOLDOUT_RESEAL.json"
REDESIGN = D / "PROTOCOL_R4_REDESIGN_W40.json"
DECISION = D / "R4_PREFREEZE_DECISION.json"
FREEZE = D / "R4_FREEZE.json"
HOLDOUT_MARKER = D / "R4_HOLDOUT_OPENED.json"
HOLDOUT = D / "R4_HOLDOUT.json"
HOLDOUT_REVIEW = D / "R4_HOLDOUT_REVIEW.json"
OFFICIAL_MARKER = D / "R4_OFFICIAL_OPENED.json"
OFFICIAL_DIR = D / "R4_OFFICIAL_DEFAULT"
OFFICIAL_RESULT = D / "R4_OFFICIAL_REGRESSION.json"
R1_OFFICIAL = D / "S4_OFFICIAL.json"

DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
    "+xy": (1.0, 1.0, 0.0), "-xy": (-1.0, -1.0, 0.0),
    "+xyz": (1.0, 1.0, 1.0), "-xyz": (-1.0, -1.0, -1.0),
}

# Test seam: called only after the durable holdout-open marker exists.
_rng_factory = np.random.default_rng


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def _commit_id(value: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", value + "^{commit}"], cwd=ROOT, text=True
    ).strip()


def _verify_bindings(files: dict[str, str], commit: str) -> None:
    """Require every current byte and named Git-object byte to match."""
    for relative, expected in files.items():
        path = Path(relative)
        current = ROOT / path
        if path.is_absolute() or ".." in path.parts or not current.is_file():
            raise RuntimeError("unsafe or missing frozen source: " + relative)
        if (_sha(current) != expected
                or hashlib.sha256(_git_bytes(commit, relative)).hexdigest() != expected):
            raise RuntimeError("current/committed source mismatch: " + relative)


def _exclusive_json(path: Path, value: dict, exists_message: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(R.json_safe(value), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError(exists_message) from exc
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_candidates() -> dict[str, dict]:
    return {row["id"]: row for row in S.protocol()["candidate_grid"]}


def _reseal() -> dict:
    value = _read(RESEAL)
    protocol = S.protocol()
    if (value.get("schema") != "R4_HOLDOUT_RESEAL.v1"
            or value.get("original_protocol_sha256") != _sha(PROTOCOL)
            or value.get("invalidated_seed") != protocol["holdout"]["seed"]
            or value.get("replacement_seed") == value.get("invalidated_seed")
            or not isinstance(value.get("replacement_seed"), int)
            or value.get("actual_holdout_episodes_before_reseal") != 0
            or value.get("candidate_collision_episodes_before_reseal") != 0
            or value.get("official_episodes_before_reseal") != 0
            or value.get("replacement_conditions_generated_before_freeze") is not False
            or value.get("all_other_protocol_gates_and_design_ranges_unchanged") is not True
            or not value.get("reason") or not value.get("test_rule")
            or not value.get("effective_rule")):
        raise RuntimeError("R4 holdout reseal addendum is malformed")
    return value


def _effective_protocol() -> dict:
    protocol = copy.deepcopy(S.protocol())
    protocol["holdout"]["seed"] = _reseal()["replacement_seed"]
    redesign = _redesign_protocol()
    excluded = protocol["holdout"]["exclude_masses_kg"]
    additions = redesign["new_holdout_exclusion_masses"]
    if any(value in excluded for value in additions):
        raise RuntimeError("W40 revalidation masses were already in the original exclusion set")
    excluded.extend(additions)
    return protocol


def _redesign_protocol() -> dict:
    value = _read(REDESIGN)
    revalidation = value.get("revalidation", {})
    conditions = revalidation.get("conditions")
    cases = revalidation.get("cases")
    additions = value.get("new_holdout_exclusion_masses")
    if (value.get("schema") != "R4_REDESIGN_W40.v1"
            or value.get("original_protocol_sha256") != _sha(PROTOCOL)
            or value.get("validation_review_sha256") != _sha(D / "R4_VALIDATION_REVIEW.json")
            or value.get("original_redesign_decision_sha256") != _sha(D / "R4_REDESIGN_DECISION.json")
            or value.get("holdout_reseal_sha256") != _sha(RESEAL)
            or value.get("family") != "W40" or value.get("variant") != "W40_ACTIVE"
            or value.get("redesign_number") != 1
            or value.get("architecture") != ["servo", "torque"]
            or value.get("original_candidates_literal_store_free") is not False
            or value.get("redesigns_used") != {"W40": 1, "CUSUM_ACTIVE": 0, "CUSUM_PHASE": 0}
            or value.get("first_holdout_generated") is not False
            or value.get("official_runs") != 0
            or revalidation.get("rows_per_mode") != 38
            or revalidation.get("full_loop") is not True
            or revalidation.get("all_numeric_gates_unchanged") is not True
            or not isinstance(conditions, list) or len(conditions) != 4
            or not isinstance(cases, list) or len(cases) != 38
            or {row.get("case_id") for row in cases} != set(range(38))
            or not isinstance(additions, list) or len(additions) != 4
            or len(set(additions)) != 4
            or additions != [row.get("mass") for row in conditions]
            or {row.get("mass") for row in cases} != set(additions)
            or not value.get("reason") or not value.get("science_gate")):
        raise RuntimeError("R4 W40 redesign addendum is malformed")
    return value


def _resolved_path(name: str, *args) -> Path:
    resolver = getattr(S, name, None)
    if not callable(resolver):
        raise RuntimeError("missing sealed R4 resolver: " + name)
    path = Path(resolver(*args))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise RuntimeError("R4 resolver returned an unsafe or missing path: " + name)
    return path


def _literal_store_free(spec: dict) -> bool:
    helper = getattr(S, "literal_store_free", None)
    if not callable(helper):
        raise RuntimeError("missing sealed R4 literal-store proof resolver")
    result = helper(spec)
    if not isinstance(result, bool):
        raise RuntimeError("literal-store proof resolver did not return bool")
    if spec.get("id") != "W40_ACTIVE":
        return result
    return result and _redesign_semantic_proof()


def _redesign_semantic_proof() -> bool:
    """Independently check the repaired source, archive, and normal-only proof."""
    build = _read(D / "R4_REDESIGN_BUILD.json")
    audit = _read(D / "R4_REDESIGN_NORMAL_AUDIT.json")
    original = _read(D / "R4_BUILT.json")
    calibration = _read(D / "R4_CALIBRATION.json")
    entries = {Path(row["path"]).name: row for row in original.get("generated", [])}
    expected_names = {"W40_ACTIVE.py", "CUSUM_ACTIVE_K05.py",
                      "CUSUM_ACTIVE_K10.py", "CUSUM_PHASE_K05.py"}
    original_hashes = {name: row.get("sha256") for name, row in entries.items()}
    archived = build.get("archived_sha256")
    expected_archive = {f"experiments/r4_original/{name}": digest
                        for name, digest in original_hashes.items()}
    literal = build.get("literal_store_audit", {})
    production = build.get("production_audit", {})
    revised = ROOT / "experiments" / "r4_generated" / "W40_ACTIVE.py"
    archived_w40 = ROOT / "experiments" / "r4_original" / "W40_ACTIVE.py"
    w40_parameters = entries.get("W40_ACTIVE.py", {}).get("parameters")
    reconstructed = None
    recomputed_literal = None
    recomputed_production = None
    if archived_w40.is_file() and isinstance(w40_parameters, dict):
        reconstructed = X.revised_source(
            archived_w40.read_text(encoding="utf-8"), w40_parameters).encode()
        recomputed_literal = X.literal_audit(reconstructed.decode())
        recomputed_production = X.audit_text(
            reconstructed.decode(), "experiments/r4_generated/W40_ACTIVE.py")
    rows = audit.get("rows")
    row_keys = {(row.get("architecture"), row.get("case_id")) for row in rows or []}
    archive_bytes_match = (isinstance(archived, dict) and archived == expected_archive
                           and all((ROOT / path).is_file() and _sha(ROOT / path) == digest
                                   for path, digest in archived.items()))
    untouched_match = all(
        (ROOT / "experiments" / "r4_generated" / name).is_file()
        and _sha(ROOT / "experiments" / "r4_generated" / name) == original_hashes[name]
        for name in expected_names - {"W40_ACTIVE.py"})
    valid = (
        set(entries) == expected_names
        and isinstance(w40_parameters, dict)
        and build.get("schema") == "R4_REDESIGN_BUILD.v1"
        and build.get("passed") is True
        and build.get("redesign_protocol_sha256") == _sha(REDESIGN)
        and build.get("original_generated_sha256") == original_hashes
        and archive_bytes_match and untouched_match
        and build.get("original_w40_sha256") == original_hashes["W40_ACTIVE.py"]
        and build.get("revised_path") == "experiments/r4_generated/W40_ACTIVE.py"
        and revised.is_file() and revised.read_bytes() == reconstructed
        and build.get("revised_sha256") == _sha(revised)
        and build.get("calibration_sha256") == _sha(D / "R4_CALIBRATION.json")
        and original.get("calibration_sha256") == _sha(D / "R4_CALIBRATION.json")
        and build.get("parameters") == w40_parameters
        and all(calibration.get("parameters", {}).get(f"{mode}:W40_ACTIVE")
                == w40_parameters.get(mode) for mode in ("servo", "torque"))
        and build.get("thresholds_unchanged") is True
        and build.get("exact_replacement_count") == 1
        and production == recomputed_production and production.get("errors") == []
        and literal == recomputed_literal and literal.get("passed") is True
        and literal.get("super_update_calls") == 1
        and literal.get("forbidden_mutator_calls") == []
        and literal.get("nonowned_self_stores") == []
        and audit.get("schema") == "R4_REDESIGN_NORMAL_AUDIT.v1"
        and audit.get("passed") is True
        and audit.get("redesign_protocol_sha256") == _sha(REDESIGN)
        and audit.get("calibration_sha256") == _sha(D / "R4_CALIBRATION.json")
        and audit.get("revised_source_sha256") == _sha(revised)
        and audit.get("collision_episodes_read") == 0
        and audit.get("official_episodes_read") == 0
        and audit.get("holdout_episodes_read") == 0
        and audit.get("cached_normal_rows") == 32
        and audit.get("extra_true_samples") == 0
        and audit.get("thresholds_unchanged") is True
        and isinstance(rows, list) and len(rows) == 32
        and row_keys == {(mode, case_id) for mode in ("servo", "torque")
                         for case_id in range(16)}
        and all(row.get("passed") is True and row.get("extra_true_samples") == 0
                for row in rows))
    if not valid:
        raise RuntimeError("R4 W40 semantic read-only proof is inconsistent")
    return True


def _exact_reused_equivalence(old_rows: list[dict], reused: list[dict],
                              source_sha256: str) -> bool:
    """Accept only byte-semantics-equivalent old rows plus explicit rejection metadata."""
    extra = {"source_reused", "source_certificate_sha256", "literal_store_free",
             "adoption_eligible", "reuse_reason"}
    old_canonical = [{key: item for key, item in row.items() if key not in extra}
                     for row in old_rows if row.get("variant") != "W40_ACTIVE"]
    reused_canonical = [{key: item for key, item in row.items() if key not in extra}
                        for row in reused]
    return (len(reused) == 12 and reused_canonical == old_canonical
            and all(row.get("source_reused") is True
                    and row.get("source_certificate_sha256") == source_sha256
                    and row.get("literal_store_free") is False
                    and row.get("adoption_eligible") is False
                    and bool(row.get("reuse_reason")) for row in reused))


def _equivalence_certificate() -> tuple[dict, bool]:
    """Validate fresh W40 parity and exact diagnostic reuse in the revised proof."""
    path = _resolved_path("equivalence_path")
    value = _read(path)
    rows = value.get("results")
    expected_keys = {(mode, variant) for mode in ("servo", "torque")
                     for variant in _canonical_candidates()}
    keys = {(row.get("architecture"), row.get("variant")) for row in rows or []}
    if value.get("schema") == "R4_REDESIGN_EQUIVALENCE.v1":
        old_path = D / "R4_EQUIVALENCE.json"
        old = _read(old_path)
        old_rows = old.get("results")
        fresh = [row for row in rows or [] if row.get("variant") == "W40_ACTIVE"]
        reused = [row for row in rows or [] if row.get("variant") != "W40_ACTIVE"]
        fresh_ok = (len(fresh) == 4
                    and {(row.get("architecture"), row.get("case_id")) for row in fresh}
                        == {(mode, case_id) for mode in ("servo", "torque")
                            for case_id in (0, 4)}
                    and all(row.get("source_reused") is False
                            and row.get("literal_store_free") is True
                            and row.get("adoption_eligible") is True for row in fresh))
        reused_ok = _exact_reused_equivalence(old_rows or [], reused, _sha(old_path))
        envelope_ok = (
            value.get("redesign_protocol_sha256") == _sha(REDESIGN)
            and value.get("original_equivalence_sha256") == _sha(old_path)
            and value.get("revised_w40_sha256")
                == _sha(ROOT / "experiments/r4_generated/W40_ACTIVE.py")
            and value.get("fresh_full_loop_W40_rows") == 76
            and value.get("fresh_representative_rows") == 4
            and value.get("unchanged_source_reused_rows") == 12
            and value.get("revalidation_documents_sha256") == {
                mode: _sha(_resolved_path("validation_document_path", "W40_ACTIVE", mode))
                for mode in ("servo", "torque")})
        semantic_ok = fresh_ok and reused_ok and envelope_ok
    else:
        semantic_ok = False
    parity = (isinstance(rows, list) and len(rows) == 16 and keys == expected_keys
              and all(sum(row.get("architecture") == mode and row.get("variant") == variant
                          for row in rows) == 2 for mode, variant in expected_keys)
              and all(row.get("passed") is True
                      and row.get("online_offline_decisions_equal") is True
                      and bool(row.get("physical_arrays_bit_equal"))
                      and all(row["physical_arrays_bit_equal"].values()) for row in rows))
    if value.get("passed") != parity or not semantic_ok:
        raise RuntimeError("R4 full-loop equivalence certificate is inconsistent")
    return value, parity


def _candidate_auxiliary_evidence(protocol: dict) -> dict[tuple[str, str], dict]:
    calibration = _read(D / "R4_CALIBRATION.json")
    equivalence, parity_recomputed = _equivalence_certificate()
    candidates = _canonical_candidates()
    expected_keys = {(mode, variant) for mode in ("servo", "torque") for variant in candidates}
    audits = calibration.get("audit")
    if (calibration.get("schema") != "R4_CALIBRATION.v1"
            or calibration.get("protocol_sha256") != _sha(PROTOCOL)
            or not isinstance(audits, list) or len(audits) != 128):
        raise RuntimeError("R4 normal-only calibration audit is incomplete")
    output = {}
    for mode, variant in expected_keys:
        audit_rows = [row for row in audits
                      if row.get("architecture") == mode and row.get("variant") == variant]
        if (len(audit_rows) != 16 or len({row.get("case_id") for row in audit_rows}) != 16
                or any(row.get("passed") != (row.get("extra_true_samples") == 0
                                              and row.get("extra_rises") == 0)
                       for row in audit_rows)):
            raise RuntimeError("R4 candidate normal audit rows are malformed")
        audit_pass = all(row["passed"] for row in audit_rows)
        online = _read(_resolved_path("validation_document_path", variant, mode))
        measured = online.get("results")
        physical = (isinstance(measured, list) and len(measured) == 38
                    and len({row.get("case_id") for row in measured}) == 38
                    and all(row.get("equivalence_passed") is True
                            and row.get("online_offline_decisions_equal") is True
                            and bool(row.get("physical_arrays_bit_equal"))
                            and all(row["physical_arrays_bit_equal"].values())
                            for row in measured))
        expected_schema = ("R4_ONLINE_REVALIDATION.v1" if variant == "W40_ACTIVE"
                           else "R4_ONLINE_VALIDATION.v1")
        if (online.get("schema") != expected_schema
                or online.get("architecture") != mode or online.get("variant") != variant
                or online.get("all_physical_bits_equal") != physical
                or online.get("fresh_simulation") is not True or not physical):
            raise RuntimeError("R4 online validation/equivalence rows are incomplete")
        if variant == "W40_ACTIVE" and (
                online.get("redesign_protocol_sha256") != _sha(REDESIGN)
                or online.get("runtime_bundle_sha256") != equivalence.get("runtime_bundle_sha256")
                or any(row.get("fresh_simulation") is not True
                       or row.get("physical_stream_reused") is not False for row in measured)):
            raise RuntimeError("R4 W40 revalidation is not fresh or redesign-bound")
        p99 = max(row["timing"]["combined"]["wall"]["p99_us"] for row in measured)
        maximum = max(row["timing"]["combined"]["wall"]["max_us"] for row in measured)
        output[mode, variant] = {
            "normal_audit_pass": audit_pass,
            "full_loop_parity_pass": parity_recomputed,
            "timing_p99_us": p99, "timing_max_us": maximum,
            "timing_pass": p99 <= protocol["timing"]["selection_combined_p99_us_max"],
            "literal_store_free": _literal_store_free(candidates[variant]),
        }
    if calibration.get("audit_passed") != all(row["normal_audit_pass"] for row in output.values()):
        raise RuntimeError("R4 aggregate normal-audit flag is inconsistent")
    return output


def _require_auxiliary_fields(row: dict, expected: dict) -> None:
    for name, value in expected.items():
        if row.get(name) != value:
            raise RuntimeError("R4 stage auxiliary evidence differs: " + name)


def _stage_reviews() -> dict[str, dict]:
    protocol = S.protocol()
    protocol_sha = _sha(PROTOCOL)
    candidates = set(_canonical_candidates())
    expected = {(mode, variant) for mode in ("servo", "torque") for variant in candidates}
    auxiliary = _candidate_auxiliary_evidence(protocol)
    reviews = {}
    for stage in ("VALIDATION", "DEVELOPMENT", "D4", "PERTURB"):
        path = (_resolved_path("validation_review_path") if stage == "VALIDATION"
                else D / f"R4_{stage}_REVIEW.json")
        value = _read(path)
        rows = value.get("results")
        keys = {(row.get("architecture"), row.get("variant")) for row in rows or []}
        expected_stage = "revalidation" if stage == "VALIDATION" else stage.lower()
        if (value.get("schema") != "R4_STAGE_REVIEW.v1"
                or value.get("stage") != expected_stage
                or value.get("protocol_sha256") != protocol_sha
                or not isinstance(rows, list) or len(rows) != 8 or keys != expected):
            raise RuntimeError("incomplete R4 stage review: " + stage)
        if stage == "VALIDATION":
            required_sources = set()
            for mode, variant in expected:
                required_sources.add(_resolved_path("validation_document_path", variant, mode).name)
                required_sources.add(_resolved_path("validation_baseline_path", variant, mode).name)
            recorded_sources = value.get("source_files")
            if (not isinstance(recorded_sources, dict)
                    or required_sources != set(recorded_sources)
                    or any(not (D / name).is_file() or _sha(D / name) != digest
                           for name, digest in recorded_sources.items())):
                raise RuntimeError("R4 combined validation review source hash set differs")
        else:
            actual_sources = {item.name: _sha(item) for item in D.glob(f"R4_{stage}_*.json")
                              if not item.name.endswith("_REVIEW.json")}
            if value.get("source_files") != actual_sources:
                raise RuntimeError("R4 stage review source hash set differs: " + stage)
        minimum_key = {"VALIDATION": "validation_min_additional_detected",
                       "DEVELOPMENT": "development_min_additional_detected",
                       "D4": "D4_min_additional_detected", "PERTURB": None}[stage]
        minimum = protocol["detector_gates"][minimum_key] if minimum_key else 0
        for row in rows:
            mode, variant = row["architecture"], row["variant"]
            if stage == "VALIDATION":
                baseline = _read(_resolved_path("validation_baseline_path", variant, mode))
                candidate = _read(_resolved_path("validation_document_path", variant, mode))
            else:
                baseline = _read(D / f"R4_BASE_{stage.lower()}_{mode}.json")
                candidate = _read(D / f"R4_{stage}_{variant}_{mode}.json")
            if row.get("review") != Q.review(baseline, candidate, minimum):
                raise RuntimeError("R4 stage review differs from recomputed case gates")
            _require_auxiliary_fields(row, auxiliary[mode, variant])
        reviews[stage] = value
    # A diagnostic stage may retain a rejected row but can never revive it.
    for key in expected:
        eligible = True
        for stage in ("VALIDATION", "DEVELOPMENT", "D4", "PERTURB"):
            row = next(item for item in reviews[stage]["results"]
                       if (item["architecture"], item["variant"]) == key)
            evidence = auxiliary[key]
            expected_eligible = (eligible and row["review"]["passed"]
                                 and evidence["normal_audit_pass"]
                                 and evidence["full_loop_parity_pass"]
                                 and evidence["timing_pass"]
                                 and evidence["literal_store_free"])
            if (row.get("adoption_eligible") != expected_eligible
                    or row.get("earlier_disqualification") != (not eligible)):
                raise RuntimeError("R4 stage eligibility differs from recomputed gates")
            if row.get("permanently_disqualified") != (not expected_eligible):
                raise RuntimeError("R4 permanent-disqualification flag is inconsistent")
            eligible = expected_eligible
    return reviews


def _validate_redesign(reviews: dict[str, dict]) -> dict:
    protocol = _redesign_protocol()
    path = _resolved_path("redesign_decision_path")
    value = _read(path)
    decisions = value.get("decisions")
    if (value.get("schema") != "R4_REDESIGN_DECISION.v2"
            or value.get("supersedes_without_edit") != "R4_REDESIGN_DECISION.json"
            or value.get("original_record_sha256") != protocol["original_redesign_decision_sha256"]
            or value.get("validation_sha256") != protocol["validation_review_sha256"]
            or value.get("redesigns_used") != 1
            or not isinstance(decisions, list)
            or {row.get("family") for row in decisions} != {"W40", "CUSUM_ACTIVE", "CUSUM_PHASE"}
            or sum(row.get("used") is True for row in decisions) != 1
            or next(row for row in decisions if row["family"] == "W40").get("used") is not True
            or next(row for row in decisions if row["family"] == "W40").get("redesign_number") != 1
            or any(not row.get("reason") for row in decisions)):
        raise RuntimeError("R4 redesign decision is incomplete or post-dated")
    return value


def _ranked_choice(reviews: dict[str, dict], mode: str, order: list[str]) -> str:
    eligible = [row for row in reviews["PERTURB"]["results"]
                if row["architecture"] == mode and row["adoption_eligible"]]
    def stage_row(stage, variant):
        return next(row for row in reviews[stage]["results"]
                    if row["architecture"] == mode and row["variant"] == variant)
    def rank(row):
        variant = row["variant"]
        return (-stage_row("DEVELOPMENT", variant)["review"]["additional_detected"],
                -stage_row("D4", variant)["review"]["additional_detected"],
                row["timing_p99_us"], order.index(variant))
    return min(eligible, key=rank)["variant"] if eligible else "R1"


def _chosen() -> dict[str, str]:
    record = _read(DECISION)
    chosen = record.get("chosen")
    canonical_map = _canonical_candidates()
    canonical = set(canonical_map) | {"R1"}
    reviews = _stage_reviews()
    _validate_redesign(reviews)
    if (record.get("schema") != "R4_PREFREEZE_DECISION.v1"
            or not isinstance(chosen, dict) or set(chosen) != {"servo", "torque"}
            or any(value not in canonical for value in chosen.values())
            or record.get("protocol_sha256") != _sha(PROTOCOL)
            or record.get("holdout_reseal_sha256") != _sha(RESEAL)
            or record.get("redesign_protocol_sha256") != _sha(REDESIGN)
            or record.get("redesign_decision_sha256") != _sha(_resolved_path("redesign_decision_path"))
            or record.get("source_evidence_sha256") != {
                stage: _sha(_resolved_path("validation_review_path") if stage == "VALIDATION"
                            else D / f"R4_{stage}_REVIEW.json") for stage in reviews}):
        raise RuntimeError("R4 pre-freeze decision is malformed or noncanonical")
    if record.get("architectures", {}).keys() != chosen.keys():
        raise RuntimeError("R4 decision lacks independent architecture records")
    for mode, variant in chosen.items():
        row = record["architectures"][mode]
        eligible = [item["variant"] for item in reviews["PERTURB"]["results"]
                    if item["architecture"] == mode and item["adoption_eligible"]]
        if (row.get("chosen") != variant or row.get("all_stages_required") is not True
                or row.get("no_later_promotion") is not True
                or row.get("eligible") != eligible):
            raise RuntimeError("R4 decision permits an unregistered later promotion")
    order = list(canonical_map)
    for mode in ("servo", "torque"):
        recomputed = _ranked_choice(reviews, mode, order)
        if chosen[mode] != recomputed:
            raise RuntimeError("R4 pre-freeze choice differs from the registered ranking")
    return chosen


def _selected_parameters(chosen: dict[str, str]) -> dict:
    calibration = _read(D / "R4_CALIBRATION.json")["parameters"]
    return {
        mode: (None if variant == "R1" else calibration[f"{mode}:{variant}"])
        for mode, variant in chosen.items()
    }


def _selected_detector_bytes(chosen: dict[str, str]) -> bytes:
    """Assemble the selected read-only detector without the obsolete wrapper."""
    if all(value == "R1" for value in chosen.values()):
        return (ROOT / "r1_detector.py").read_bytes()
    parameters = X.B._validated(_selected_parameters(chosen))
    original = (X.B.R1.read_text(encoding="utf-8") + "\n\n"
                + X.B.KERNEL.read_text(encoding="utf-8")
                + X.B._wrapper(parameters))
    repaired = X.revised_source(original, parameters)
    literal = X.literal_audit(repaired)
    production = X.audit_text(repaired, "selected_r4_detector.py")
    if (not literal.get("passed") or production.get("errors")
            or repaired.count(X.OLD_GATE) != 0 or repaired.count(X.NEW_GATE) != 1):
        raise RuntimeError("selected R4 detector is not the qualified read-only assembly")
    output = repaired.encode()
    if set(chosen.values()) == {"W40_ACTIVE"}:
        registered = ROOT / "experiments" / "r4_generated" / "W40_ACTIVE.py"
        if not registered.is_file() or output != registered.read_bytes():
            raise RuntimeError("two-mode W40 assembly differs from the qualified source")
    return output


def _expected_detector_bytes(chosen: dict[str, str]) -> bytes:
    return _selected_detector_bytes(chosen)


def _module_file(name: str) -> str | None:
    path = Path(*name.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if (ROOT / candidate).is_file():
            return candidate.as_posix()
    return None


def _local_imports(relative: str) -> set[str]:
    """Resolve repository-local static imports for freeze-closure checks."""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
    module = Path(relative).with_suffix("").as_posix().replace("/", ".")
    package = module.rsplit(".", 1)[0] if "." in module else ""
    found = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                base = parts[:max(0, len(parts) - node.level + 1)]
                if node.module:
                    base += node.module.split(".")
                prefix = ".".join(base)
            else:
                prefix = node.module or ""
            names = [prefix]
            if prefix in ("tools", "sim", "experiments"):
                names.extend(prefix + "." + alias.name for alias in node.names)
        for name in names:
            resolved = _module_file(name)
            if resolved is not None:
                found.add(resolved)
    return found


def runtime_import_closure() -> set[str]:
    seeds = {
        "tools/r4_sealed.py", "tools/r4_study.py", "tools/r4_review.py",
        "tools/r4_build.py", "experiments/r4_sequential.py",
        "tools/r3_official_regression.py", "tools/r3_extended.py",
        "tools/r3_study.py", "tools/r3_resources.py",
        "tools/r2_extended_study.py", "tools/r2_study.py",
        "tools/v2_study.py", "tools/validate_submission.py",
        "tools/check_integrity.py", "tools/production_audit.py", "tools/sweep.py",
        "sim/runner.py", "r1_detector.py", "r1_servo.py", "r1_torque.py",
    }
    closure, pending = set(), list(seeds)
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        if not (ROOT / relative).is_file():
            raise RuntimeError("missing runtime import seed: " + relative)
        closure.add(relative)
        pending.extend(_local_imports(relative) - closure)
    return closure


def source_paths() -> list[str]:
    """Return the transitive science/runtime freeze, excluding publication."""
    names = set(_read(D / "R3_FREEZE.json")["source_files"])
    for directory in ("sim", "model"):
        names.update(path.relative_to(ROOT).as_posix()
                     for path in (ROOT / directory).rglob("*") if path.is_file())
    names.update(path.relative_to(ROOT).as_posix()
                 for path in (ROOT / "models").glob("*.json"))
    names.update(path.relative_to(ROOT).as_posix()
                 for path in (ROOT / "experiments").glob("r4_*.py"))
    generated = ROOT / "experiments" / "r4_generated"
    if generated.is_dir():
        names.update(path.relative_to(ROOT).as_posix()
                     for path in generated.glob("*.py"))
    archived = ROOT / "experiments" / "r4_original"
    if archived.is_dir():
        names.update(path.relative_to(ROOT).as_posix()
                     for path in archived.rglob("*.py"))
    for pattern in ("r4_*.py", "test_r4_*.py"):
        names.update(path.relative_to(ROOT).as_posix()
                     for path in (ROOT / "tools").glob(pattern)
                     if not path.name.startswith(("r4_report", "r4_public", "test_r4_report")))
    # Every R4 pre-freeze observation and decision is science evidence.  Future
    # exposure/result/report files are excluded by exact prefix.
    for path in D.glob("R4_*.json"):
        if (path.name.startswith(("R4_FREEZE", "R4_HOLDOUT", "R4_OFFICIAL", "R4_REPORT"))
                or "holdout" in path.name.lower()):
            continue
        names.add(path.relative_to(ROOT).as_posix())
    names.update({
        "results/v2_protocol/PROTOCOL_R4.json",
        "results/v2_protocol/PROTOCOL_R4_HOLDOUT_RESEAL.json",
        "results/v2_protocol/PROTOCOL_R4_REDESIGN_W40.json",
        "my_servo.py", "my_torque.py", "my_detector.py",
        "r1_servo.py", "r1_torque.py", "r1_detector.py",
        "evaluate.py", "tools/validate_submission.py", "tools/compare_baseline.py",
        "tools/check_integrity.py", "tools/kit_manifest.sha256",
        "tools/production_audit.py", "tools/sweep.py",
        "results/v2_protocol/S4_OFFICIAL.json",
    })
    names.update(runtime_import_closure())
    return sorted(names)


def create_freeze(source_commit: str) -> dict:
    """Bind the final pre-holdout decision and every science byte to Git."""
    if FREEZE.exists() or HOLDOUT_MARKER.exists():
        raise RuntimeError("R4 freeze already exists or holdout is exposed")
    chosen = _chosen()
    if (ROOT / "my_servo.py").read_bytes() != (ROOT / "r1_servo.py").read_bytes():
        raise RuntimeError("R4 has no servo-control candidate; my_servo must remain R1")
    if (ROOT / "my_torque.py").read_bytes() != (ROOT / "r1_torque.py").read_bytes():
        raise RuntimeError("R4 has no torque-control candidate; my_torque must remain R1")
    if (ROOT / "my_detector.py").read_bytes() != _expected_detector_bytes(chosen):
        raise RuntimeError("my_detector differs from the selected frozen assembly")
    files = {name: _sha(ROOT / name) for name in source_paths()}
    commit = _commit_id(source_commit)
    _verify_bindings(files, commit)
    record = {
        "schema": "R4_FREEZE.v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "local_commit": commit,
        "protocol_sha256": _sha(PROTOCOL),
        "holdout_reseal_sha256": _sha(RESEAL),
        "redesign_protocol_sha256": _sha(REDESIGN),
        "redesign_decision_sha256": _sha(_resolved_path("redesign_decision_path")),
        "validation_review_sha256": _sha(_resolved_path("validation_review_path")),
        "equivalence_sha256": _sha(_resolved_path("equivalence_path")),
        "effective_holdout_seed": _reseal()["replacement_seed"],
        "chosen": chosen,
        "selection_evidence": {
            "path": DECISION.relative_to(ROOT).as_posix(), "sha256": _sha(DECISION)
        },
        "calibration_sha256": _sha(D / "R4_CALIBRATION.json"),
        "fallback_R1_sha256": _read(D / "R4_BASELINE.json")["R1_sha256"],
        "submission_sha256": {
            name: _sha(ROOT / name)
            for name in ("my_servo.py", "my_torque.py", "my_detector.py")
        },
        "source_files": files,
        "scope": "runtime, calibration, candidates, all pre-freeze evidence, selection and tests; publication-only processors excluded",
        "candidate_and_R1_fallback_both_frozen": True,
        "tuning_after_holdout_or_official": False,
    }
    _exclusive_json(FREEZE, record, "R4 freeze already exists")
    return record


def verify_freeze() -> dict:
    """Verify the committed freeze, current files, source commit and selection."""
    freeze = _read(FREEZE)
    if freeze.get("schema") != "R4_FREEZE.v1":
        raise RuntimeError("unexpected R4 freeze schema")
    try:
        committed_freeze = _git_bytes("HEAD", FREEZE.relative_to(ROOT).as_posix())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("R4 freeze is not committed at HEAD") from exc
    if committed_freeze != FREEZE.read_bytes():
        raise RuntimeError("working R4 freeze differs from committed HEAD bytes")
    if (freeze.get("protocol_sha256") != _sha(PROTOCOL)
            or freeze.get("holdout_reseal_sha256") != _sha(RESEAL)
            or freeze.get("redesign_protocol_sha256") != _sha(REDESIGN)
            or freeze.get("redesign_decision_sha256") != _sha(_resolved_path("redesign_decision_path"))
            or freeze.get("validation_review_sha256") != _sha(_resolved_path("validation_review_path"))
            or freeze.get("equivalence_sha256") != _sha(_resolved_path("equivalence_path"))
            or freeze.get("effective_holdout_seed") != _reseal()["replacement_seed"]
            or freeze.get("chosen") != _chosen()):
        raise RuntimeError("R4 freeze protocol or selection mismatch")
    commit = freeze.get("local_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("R4 freeze lacks a full source commit")
    subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = freeze.get("source_files")
    if not isinstance(files, dict) or set(source_paths()) - set(files):
        raise RuntimeError("R4 freeze lacks required transitive sources")
    _verify_bindings(files, commit)
    proof = freeze["selection_evidence"]
    if (proof != {"path": DECISION.relative_to(ROOT).as_posix(), "sha256": _sha(DECISION)}
            or files.get(proof["path"]) != proof["sha256"]):
        raise RuntimeError("selection evidence is not source-frozen")
    if (ROOT / "my_servo.py").read_bytes() != (ROOT / "r1_servo.py").read_bytes() \
            or (ROOT / "my_torque.py").read_bytes() != (ROOT / "r1_torque.py").read_bytes():
        raise RuntimeError("R4 control aliases changed after freeze")
    if (ROOT / "my_detector.py").read_bytes() != _expected_detector_bytes(freeze["chosen"]):
        raise RuntimeError("selected detector alias changed after freeze")
    baseline = _read(D / "R4_BASELINE.json")["R1_sha256"]
    for mode in ("servo", "torque", "detector"):
        relative = f"r1_{mode}.py"
        if freeze["fallback_R1_sha256"].get(mode) != baseline[mode] \
                or files.get(relative) != baseline[mode]:
            raise RuntimeError("frozen R1 fallback differs from the fixed baseline: " + mode)
    return freeze


def _force(direction: str, magnitude: float) -> list[float]:
    vector = np.asarray(DIRECTIONS[direction], dtype=float)
    return (float(magnitude) * vector / np.linalg.norm(vector)).tolist()


def _build_holdout_design(protocol: dict, rng) -> list[dict]:
    design = protocol["holdout"]
    excluded = {float(value) for value in design["exclude_masses_kg"]}
    digits = int(design["mass_round_digits"])
    conditions = []
    for lower, upper in design["mass_ranges_kg"]:
        if len(conditions) == 4:
            break
        for _ in range(10000):
            mass = round(float(rng.uniform(lower, upper)), digits)
            if mass not in excluded and all(row["mass"] != mass for row in conditions):
                break
        else:
            raise RuntimeError("unable to generate a novel holdout mass")
        low, high = np.asarray(design["com_ranges_m"], dtype=float)
        conditions.append({"mass": mass, "com": rng.uniform(low, high).tolist()})
    if len(conditions) != 4:
        raise RuntimeError("holdout protocol does not provide four usable ranges")
    rows = [dict(condition=row, kind="reference") for row in conditions]
    phases = tuple(design["phase_time_ranges_s"])
    directions = list(design["directions"])
    if (len(phases) != 3 or len(directions) != 10
            or any(directions[index][0] != "+" or directions[index + 1][0] != "-"
                   for index in range(0, 10, 2))):
        raise RuntimeError("holdout balance contract changed")
    for pair in range(5):
        for phase_index, phase in enumerate(phases):
            when = float(rng.uniform(*design["phase_time_ranges_s"][phase]))
            magnitude = float(rng.uniform(*design["force_range_N"]))
            condition = conditions[(pair + phase_index) % 4]
            for direction in directions[2 * pair:2 * pair + 2]:
                rows.append(dict(condition=condition, kind="collision",
                                 collision=[when, _force(direction, magnitude)],
                                 phase=phase, direction=direction, force_n=magnitude))
    for condition, offset in zip(conditions, design["pick_offsets_m"], strict=True):
        rows.append(dict(condition=condition, kind="offset", pick_offset=list(offset),
                         phase="pick", direction="offset", force_n=0.0))
    counts = {kind: sum(row["kind"] == kind for row in rows)
              for kind in ("reference", "collision", "offset")}
    expected = design["rows_per_source_architecture"]
    if counts != {"reference": expected["references"],
                  "collision": expected["collisions"], "offset": expected["offsets"]}:
        raise RuntimeError("holdout balance differs from protocol")
    return rows


def _explicit_cases(design: list[dict]) -> list[dict]:
    rows = []
    for case_id, row in enumerate(design):
        rows.append({**row["condition"], "case_id": case_id, "kind": row["kind"],
                     "collision": row.get("collision"), "pick_offset": row.get("pick_offset"),
                     "phase": row.get("phase", "none"),
                     "direction": row.get("direction", "none"),
                     "force_n": float(row.get("force_n", 0.0))})
    return rows


def _effective_after_holdout(chosen: str, review: dict | None) -> tuple[str, str]:
    """Pure immutable fallback rule: failure never promotes another candidate."""
    if chosen == "R1":
        return "R1", "PRESERVED"
    if review is not None and review.get("passed") is True:
        return chosen, "PASS"
    return "R1", "FAIL"


def _validated_effective_map(freeze: dict, review: dict) -> dict[str, str]:
    architectures = review.get("architectures")
    if not isinstance(architectures, dict) or set(architectures) != {"servo", "torque"}:
        raise RuntimeError("R4 holdout review lacks both architecture decisions")
    derived = {}
    for mode, chosen in freeze["chosen"].items():
        row = architectures[mode]
        effective = row.get("effective_selection")
        if effective not in {chosen, "R1"}:
            raise RuntimeError("R4 holdout attempted a runner-up promotion")
        if chosen == "R1":
            if row.get("status") != "PRESERVED":
                raise RuntimeError("R1-only holdout status is not preservation")
            expected = "R1"
        else:
            if row.get("status") not in {"PASS", "FAIL"}:
                raise RuntimeError("candidate holdout status is invalid")
            expected = chosen if row.get("status") == "PASS" else "R1"
        if effective != expected:
            raise RuntimeError("R4 holdout effective source differs from its status")
        derived[mode] = expected
    if review.get("effective_selection") != derived:
        raise RuntimeError("R4 top-level holdout effective map differs from validated modes")
    return derived


def open_holdout(private_dir: Path) -> dict:
    """Open and complete the single sealed R4 holdout batch."""
    if HOLDOUT_MARKER.exists() or HOLDOUT.exists() or HOLDOUT_REVIEW.exists():
        raise RuntimeError("R4 holdout is already opened or completed; no rerun")
    private = R.require_private(private_dir)
    private.mkdir(parents=True, exist_ok=True)
    check_integrity(ROOT)
    freeze = verify_freeze()
    protocol = _effective_protocol()
    marker = {
        "schema": "R4_HOLDOUT_OPENED.v1",
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "batch_attempt": 1, "seed": int(protocol["holdout"]["seed"]),
        "local_commit": freeze["local_commit"], "freeze_sha256": _sha(FREEZE),
        "protocol_sha256": _sha(PROTOCOL), "holdout_reseal_sha256": _sha(RESEAL),
        "redesign_protocol_sha256": _sha(REDESIGN),
        "chosen": freeze["chosen"],
        "conditions_generated_before_marker": False,
        "tuning_or_source_changes_after_exposure": False,
    }
    _exclusive_json(HOLDOUT_MARKER, marker, "R4 holdout already opened; no rerun")
    # The first construction/access of the sealed RNG is deliberately here.
    rng = _rng_factory(marker["seed"])
    design = _build_holdout_design(protocol, rng)
    private_design = private / "R4_HOLDOUT_DESIGN.json"
    _exclusive_json(private_design, {"marker_sha256": _sha(HOLDOUT_MARKER),
                    "design": design}, "private R4 holdout design already exists")
    explicit = _explicit_cases(design)
    S.collect("holdout", private, explicit=explicit)
    calibration = _read(D / "R4_CALIBRATION.json")["parameters"]
    candidates, modes, rows = {}, {}, []
    for mode in ("servo", "torque"):
        baseline = _read(D / f"R4_BASE_holdout_{mode}.json")
        rows.extend(baseline["results"])
        variant = freeze["chosen"][mode]
        if variant == "R1":
            effective, status = _effective_after_holdout(variant, None)
            modes[mode] = {"frozen_selection": variant, "status": status,
                           "review": None, "effective_selection": effective,
                           "fallback_to_R1": False, "efficacy_claim": False,
                           "no_runner_up_promotion": True, "retest_permitted": False}
            continue
        spec = _canonical_candidates()[variant]
        candidate, _ = S.replay_document(
            baseline, private, spec, calibration[f"{mode}:{variant}"]["calibration"])
        candidates[mode] = candidate
        rows.extend(candidate["results"])
        review = Q.review(baseline, candidate,
                          S.protocol()["detector_gates"]["holdout_min_additional_detected"])
        effective, status = _effective_after_holdout(variant, review)
        modes[mode] = {"frozen_selection": variant, "status": status,
                       "review": review, "effective_selection": effective,
                       "fallback_to_R1": effective == "R1",
                       "efficacy_claim": True, "no_runner_up_promotion": True,
                       "retest_permitted": False}
    result = {
        "schema": "R4_HOLDOUT.v1", "official": False, "one_batch": True,
        "marker_sha256": _sha(HOLDOUT_MARKER), "freeze_sha256": _sha(FREEZE),
        "protocol_sha256": _sha(PROTOCOL), "holdout_reseal_sha256": _sha(RESEAL),
        "redesign_protocol_sha256": _sha(REDESIGN),
        "effective_holdout_seed": protocol["holdout"]["seed"], "chosen": freeze["chosen"],
        "cases_per_source_architecture": 38, "results": rows,
        "deduplicated_R1_modes": [m for m, v in freeze["chosen"].items() if v == "R1"],
        "candidate_replay_qualification": {
            "equivalence_sha256": freeze["equivalence_sha256"],
            "static_argument": "detector output is logged but never feeds unchanged controllers; full-loop online/offline and physical bit parity were frozen",
            "fresh_candidate_full_loop": False,
        },
        "tuning_or_source_changes_after_exposure": False,
        "second_holdout_permitted": False,
    }
    _exclusive_json(HOLDOUT, result, "R4 holdout result already exists")
    review_record = {
        "schema": "R4_HOLDOUT_REVIEW.v1", "holdout_sha256": _sha(HOLDOUT),
        "freeze_sha256": _sha(FREEZE), "marker_sha256": _sha(HOLDOUT_MARKER),
        "protocol_sha256": _sha(PROTOCOL), "holdout_reseal_sha256": _sha(RESEAL),
        "redesign_protocol_sha256": _sha(REDESIGN),
        "effective_holdout_seed": protocol["holdout"]["seed"], "architectures": modes,
        "effective_selection": {mode: row["effective_selection"] for mode, row in modes.items()},
        "no_tuning_permitted": True, "no_runner_up_promotion": True,
        "second_holdout_permitted": False,
    }
    _exclusive_json(HOLDOUT_REVIEW, review_record, "R4 holdout review already exists")
    verify_freeze()
    check_integrity(ROOT)
    return review_record


def _holdout_preflight() -> tuple[dict, dict]:
    freeze = verify_freeze()
    marker, result, review = _read(HOLDOUT_MARKER), _read(HOLDOUT), _read(HOLDOUT_REVIEW)
    reseal = _reseal()
    for path in (HOLDOUT_MARKER, HOLDOUT, HOLDOUT_REVIEW,
                 D / "R4_BASE_holdout_servo.json", D / "R4_BASE_holdout_torque.json"):
        try:
            committed = _git_bytes("HEAD", path.relative_to(ROOT).as_posix())
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("completed holdout evidence is not committed: " + path.name) from exc
        if committed != path.read_bytes():
            raise RuntimeError("completed holdout differs from committed HEAD: " + path.name)
    if (marker.get("schema") != "R4_HOLDOUT_OPENED.v1" or marker.get("batch_attempt") != 1
            or marker.get("freeze_sha256") != _sha(FREEZE)
            or marker.get("local_commit") != freeze["local_commit"]
            or marker.get("protocol_sha256") != _sha(PROTOCOL)
            or marker.get("holdout_reseal_sha256") != _sha(RESEAL)
            or marker.get("redesign_protocol_sha256") != _sha(REDESIGN)
            or marker.get("seed") != reseal["replacement_seed"]
            or marker.get("chosen") != freeze["chosen"]
            or marker.get("conditions_generated_before_marker") is not False
            or marker.get("tuning_or_source_changes_after_exposure") is not False
            or result.get("schema") != "R4_HOLDOUT.v1" or result.get("one_batch") is not True
            or result.get("official") is not False
            or result.get("marker_sha256") != _sha(HOLDOUT_MARKER)
            or result.get("freeze_sha256") != _sha(FREEZE)
            or result.get("protocol_sha256") != _sha(PROTOCOL)
            or result.get("holdout_reseal_sha256") != _sha(RESEAL)
            or result.get("redesign_protocol_sha256") != _sha(REDESIGN)
            or result.get("effective_holdout_seed") != reseal["replacement_seed"]
            or result.get("chosen") != freeze["chosen"]
            or result.get("cases_per_source_architecture") != 38
            or result.get("tuning_or_source_changes_after_exposure") is not False
            or result.get("second_holdout_permitted") is not False
            or review.get("schema") != "R4_HOLDOUT_REVIEW.v1"
            or review.get("holdout_sha256") != _sha(HOLDOUT)
            or review.get("freeze_sha256") != _sha(FREEZE)
            or review.get("marker_sha256") != _sha(HOLDOUT_MARKER)
            or review.get("protocol_sha256") != _sha(PROTOCOL)
            or review.get("holdout_reseal_sha256") != _sha(RESEAL)
            or review.get("redesign_protocol_sha256") != _sha(REDESIGN)
            or review.get("effective_holdout_seed") != reseal["replacement_seed"]
            or review.get("no_tuning_permitted") is not True
            or review.get("no_runner_up_promotion") is not True
            or review.get("second_holdout_permitted") is not False):
        raise RuntimeError("R4 holdout evidence is incomplete or unbound")
    qualification = result.get("candidate_replay_qualification", {})
    if (qualification.get("equivalence_sha256") != freeze["equivalence_sha256"]
            or qualification.get("fresh_candidate_full_loop") is not False
            or not qualification.get("static_argument")):
        raise RuntimeError("R4 holdout replay lacks the frozen equivalence qualification")
    if result.get("deduplicated_R1_modes") != [
            mode for mode, variant in freeze["chosen"].items() if variant == "R1"]:
        raise RuntimeError("R4 holdout R1 deduplication record is inconsistent")
    grouped = {}
    for row in result.get("results", []):
        digest = row.get("private_raw_sha256")
        basename = row.get("private_raw_basename")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
                or not isinstance(basename, str) or Path(basename).is_absolute()
                or "/" in basename):
            raise RuntimeError("R4 holdout row lacks a valid retained raw SHA reference")
        grouped.setdefault((row.get("architecture"), row.get("variant")), []).append(row)
    expected = 0
    shared_conditions = None
    for mode, chosen in freeze["chosen"].items():
        base = grouped.get((mode, "R1"), [])
        if len(base) != 38 or {row["case_id"] for row in base} != set(range(38)):
            raise RuntimeError("R4 holdout lacks 38 unique R1 cases")
        base_document = _read(D / f"R4_BASE_holdout_{mode}.json")
        if (base_document.get("schema") != "R4_BASE_STREAMS.v1"
                or base_document.get("stage") != "holdout"
                or base_document.get("architecture") != mode
                or base_document.get("protocol_sha256") != _sha(PROTOCOL)
                or base_document.get("fresh_simulation") is not True
                or base_document.get("results") != base):
            raise RuntimeError("R4 holdout baseline stream is incomplete or rebound")
        counts = {kind: sum(row.get("kind") == kind for row in base)
                  for kind in ("reference", "collision", "offset")}
        if counts != {"reference": 4, "collision": 30, "offset": 4}:
            raise RuntimeError("R4 holdout case-kind balance changed")
        conditions = [json.dumps({name: row.get(name) for name in
                      ("case_id", "mass", "com", "kind", "collision", "pick_offset",
                       "phase", "direction", "force_n")}, sort_keys=True)
                      for row in sorted(base, key=lambda value: value["case_id"])]
        if shared_conditions is None:
            shared_conditions = conditions
        elif conditions != shared_conditions:
            raise RuntimeError("R4 holdout architectures do not share identical conditions")
        expected += 38
        review_row = review.get("architectures", {}).get(mode, {})
        if chosen != "R1":
            candidate = grouped.get((mode, chosen), [])
            if len(candidate) != 38 or {row["case_id"] for row in candidate} != set(range(38)):
                raise RuntimeError("R4 holdout lacks 38 unique candidate cases")
            expected += 38
            recomputed = Q.review({"results": base}, {"results": candidate},
                                  S.protocol()["detector_gates"]["holdout_min_additional_detected"])
            effective, status = _effective_after_holdout(chosen, recomputed)
            if (review_row.get("review") != recomputed or review_row.get("status") != status
                    or review_row.get("effective_selection") != effective
                    or review_row.get("no_runner_up_promotion") is not True
                    or review_row.get("retest_permitted") is not False):
                raise RuntimeError("R4 holdout review differs from recomputed gates")
        elif (review_row.get("status") != "PRESERVED"
              or review_row.get("effective_selection") != "R1"
              or review_row.get("review") is not None
              or review_row.get("efficacy_claim") is not False):
            raise RuntimeError("R1-only holdout mode makes an efficacy claim")
    if len(result["results"]) != expected:
        raise RuntimeError("R4 holdout has extra or duplicate source rows")
    _validated_effective_map(freeze, review)
    return freeze, review


def _stage_effective(private: Path, freeze: dict, holdout_review: dict) -> tuple[Path, dict]:
    stage = private / "R4_OFFICIAL_STAGE"
    stage.mkdir(parents=True, exist_ok=False)
    prefix = stage / "my"
    for mode in ("servo", "torque"):
        source = ROOT / f"r1_{mode}.py"
        target = Path(str(prefix) + f"_{mode}.py")
        target.write_bytes(source.read_bytes())
    effective = _validated_effective_map(freeze, holdout_review)
    Path(str(prefix) + "_detector.py").write_bytes(_selected_detector_bytes(effective))
    for path in stage.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    descriptor = os.open(stage, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return prefix, {"effective_selection": dict(effective),
                    "servo_controller_sha256": _sha(Path(str(prefix) + "_servo.py")),
                    "torque_controller_sha256": _sha(Path(str(prefix) + "_torque.py")),
                    "detector_sha256": _sha(Path(str(prefix) + "_detector.py"))}


def _verify_staged(prefix: Path, expected: dict) -> None:
    actual = {"effective_selection": expected["effective_selection"],
              "servo_controller_sha256": _sha(Path(str(prefix) + "_servo.py")),
              "torque_controller_sha256": _sha(Path(str(prefix) + "_torque.py")),
              "detector_sha256": _sha(Path(str(prefix) + "_detector.py"))}
    if actual != expected:
        raise RuntimeError("private official stage bytes changed")


def _official_effective(frozen: str, holdout_effective: str, passed: bool) -> str:
    """An official failure only falls back; it cannot promote or be retried."""
    if holdout_effective not in (frozen, "R1"):
        raise ValueError("holdout attempted an alternative promotion")
    return holdout_effective if passed else "R1"


def _official_payload(gate: dict, records: dict, freeze: dict, holdout_review: dict,
                      marker: dict, guard: dict, staged: dict) -> dict:
    modes = {}
    for mode in ("servo", "torque"):
        row = gate["architectures"][mode]
        passed = row["status"] == "PASS"
        effective = _official_effective(freeze["chosen"][mode],
                                        holdout_review["effective_selection"][mode], passed)
        modes[mode] = {**row, "effective_selection": effective,
                       "fallback_to_R1": effective == "R1" and
                           holdout_review["effective_selection"][mode] != "R1",
                       "no_runner_up_promotion": True, "retest_permitted": False}
    return {
        "schema": "R4_OFFICIAL_REGRESSION.v1", "official": True,
        "status": "PASS" if all(row["status"] == "PASS" for row in modes.values()) else "FAIL",
        "architectures": modes, "results": records,  # literal evaluator strings retained
        "source": "unchanged evaluate.py; pristine default-name overlays; one A+B batch",
        "expected_episodes": 58, "freeze_sha256": _sha(FREEZE),
        "holdout_sha256": _sha(HOLDOUT), "holdout_review_sha256": _sha(HOLDOUT_REVIEW),
        "protocol_sha256": _sha(PROTOCOL), "marker": marker,
        "holdout_reseal_sha256": _sha(RESEAL),
        "redesign_protocol_sha256": _sha(REDESIGN),
        "marker_sha256": _sha(OFFICIAL_MARKER), "guard": guard,
        "staged_effective_sources": staged, "tuning_after_exposure": False,
        "repair_after_exposure": False, "alternative_promotion_after_failure": False,
        "second_official_permitted": False,
    }


def _run_owned_group(command: list[str], log_path: Path) -> dict:
    """Run one owned evaluator while never signalling foreign processes."""
    while True:
        before = resources.snapshot()
        if before["mem_available_gib"] >= 6 and len(before["foreign_heavy_python_pids"]) <= 3:
            break
        time.sleep(5)
    states, pauses = [], 0
    with Path(log_path).open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=O3.official_environment(),
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        stopped = False
        try:
            while child.poll() is None:
                state = O3._foreign_state(resources.snapshot(), child.pid)
                states.append(state)
                safe = (state["mem_available_gib"] >= 6
                        and len(state["foreign_heavy_python_pids"]) <= 3)
                if not safe and not stopped:
                    os.killpg(child.pid, signal.SIGSTOP)
                    stopped, pauses = True, pauses + 1
                elif safe and stopped:
                    os.killpg(child.pid, signal.SIGCONT)
                    stopped = False
                time.sleep(.25)
            if stopped:
                os.killpg(child.pid, signal.SIGCONT)
        except Exception:
            if stopped:
                os.killpg(child.pid, signal.SIGCONT)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    if not states:
        states.append(O3._foreign_state(resources.snapshot(), child.pid))
    return {"returncode": int(child.returncode), "host_before": before,
            "samples": len(states), "owned_group_pause_count": pauses,
            "min_available_RAM_GiB": min(row["mem_available_gib"] for row in states),
            "maximum_foreign_heavy_python_count": max(
                len(row["foreign_heavy_python_pids"]) for row in states),
            "foreign_processes_signalled": 0, "cpu_affinity": [0], "gpu_jobs": 0}


def official(private_dir: Path) -> dict:
    """Run unchanged official evaluation once, last, with immutable fallback."""
    if OFFICIAL_MARKER.exists() or OFFICIAL_RESULT.exists() or OFFICIAL_DIR.exists():
        raise RuntimeError("R4 official batch is already opened or completed; no rerun")
    private = R.require_private(private_dir)
    private.mkdir(parents=True, exist_ok=True)
    check_integrity(ROOT)
    freeze, holdout_review = _holdout_preflight()
    prefix, staged = _stage_effective(private, freeze, holdout_review)
    marker = {
        "schema": "R4_OFFICIAL_OPENED.v1", "opened_utc": datetime.now(timezone.utc).isoformat(),
        "batch_attempt": 1, "expected_episodes": 58,
        "freeze_sha256": _sha(FREEZE), "holdout_sha256": _sha(HOLDOUT),
        "holdout_review_sha256": _sha(HOLDOUT_REVIEW),
        "protocol_sha256": _sha(PROTOCOL), "frozen_selection": freeze["chosen"],
        "holdout_reseal_sha256": _sha(RESEAL),
        "redesign_protocol_sha256": _sha(REDESIGN),
        "effective_selection": holdout_review["effective_selection"],
        "staged_effective_sources": staged,
        "tuning_or_repair_after_exposure": False,
    }
    _exclusive_json(OFFICIAL_MARKER, marker, "R4 official already opened; no rerun")
    command = ["taskset", "-c", "0", sys.executable, "tools/validate_submission.py",
               "--prefix", str(prefix), "--out", OFFICIAL_DIR.relative_to(ROOT).as_posix()]
    records = None
    try:
        _verify_staged(prefix, staged)
        guard = _run_owned_group(command, private / "r4_official_default.log")
    except Exception as exc:
        try:
            _verify_staged(prefix, staged)
            stage_post_verified = True
        except Exception as stage_exc:
            stage_post_verified = False
            exc = RuntimeError(f"{exc}; post-failure stage verification: {stage_exc}")
        guard = {"launch_or_monitor_exception": type(exc).__name__, "message": str(exc),
                 "foreign_processes_signalled": 0, "gpu_jobs": 0, "cpu_affinity": [0],
                 "staged_sources_verified_after_failure": stage_post_verified}
        modes = {mode: {"status": "FAIL", "errors": [str(exc)],
                        "frozen_selection": freeze["chosen"][mode],
                        "holdout_effective": holdout_review["effective_selection"][mode],
                        "effective_selection": "R1", "fallback_to_R1": True,
                        "no_runner_up_promotion": True, "retest_permitted": False}
                 for mode in ("servo", "torque")}
        output = {"schema": "R4_OFFICIAL_REGRESSION.v1", "official": True,
                  "status": "FAIL", "shared_unidentified_failure": True,
                  "reason": "official launch/monitor failure: " + str(exc),
                  "architectures": modes, "results": None,
                  "freeze_sha256": _sha(FREEZE), "holdout_sha256": _sha(HOLDOUT),
                  "holdout_review_sha256": _sha(HOLDOUT_REVIEW),
                  "protocol_sha256": _sha(PROTOCOL), "marker": marker,
                  "holdout_reseal_sha256": _sha(RESEAL),
                  "redesign_protocol_sha256": _sha(REDESIGN),
                  "marker_sha256": _sha(OFFICIAL_MARKER), "guard": guard,
                  "staged_effective_sources": staged, "tuning_after_exposure": False,
                  "repair_after_exposure": False, "alternative_promotion_after_failure": False,
                  "second_official_permitted": False}
        _exclusive_json(OFFICIAL_RESULT, output, "R4 official result already exists")
        return output
    try:
        _verify_staged(prefix, staged)
        post_freeze, post_review = _holdout_preflight()
        if post_freeze != freeze or post_review != holdout_review:
            raise RuntimeError("frozen pre-official evidence changed after exposure")
        if guard["returncode"] != 0:
            raise RuntimeError("unchanged official evaluator process failed")
        records = _read(OFFICIAL_DIR / "summary.json")
        for mode in ("servo", "torque"):
            expected_controller = staged[f"{mode}_controller_sha256"]
            if records.get(mode, {}).get("source_sha256") != expected_controller:
                raise RuntimeError("official evaluator controller hash differs from staged bytes")
        specs = [R.baseline_spec(mode) if holdout_review["effective_selection"][mode] == "R1"
                 else {"id": holdout_review["effective_selection"][mode],
                       "family": _canonical_candidates()[holdout_review["effective_selection"][mode]]["family"],
                       "architecture": mode}
                 for mode in ("servo", "torque")]
        source_hashes = {mode: _sha(ROOT / f"r1_{mode}.py") for mode in ("servo", "torque")}
        gate = O3.assess(records, _read(R1_OFFICIAL), S.protocol(), specs, source_hashes)
        output = _official_payload(gate, records, freeze, holdout_review,
                                   marker, guard, staged)
    except Exception as exc:
        modes = {}
        for mode in ("servo", "torque"):
            modes[mode] = {"status": "FAIL", "errors": [str(exc)],
                           "frozen_selection": freeze["chosen"][mode],
                           "holdout_effective": holdout_review["effective_selection"][mode],
                           "effective_selection": "R1", "fallback_to_R1": True,
                           "no_runner_up_promotion": True, "retest_permitted": False}
        output = {"schema": "R4_OFFICIAL_REGRESSION.v1", "official": True,
                  "status": "FAIL", "shared_unidentified_failure": True,
                  "reason": str(exc), "architectures": modes, "results": records,
                  "freeze_sha256": _sha(FREEZE), "holdout_sha256": _sha(HOLDOUT),
                  "holdout_review_sha256": _sha(HOLDOUT_REVIEW),
                  "protocol_sha256": _sha(PROTOCOL), "marker": marker,
                  "holdout_reseal_sha256": _sha(RESEAL),
                  "redesign_protocol_sha256": _sha(REDESIGN),
                  "marker_sha256": _sha(OFFICIAL_MARKER), "guard": guard,
                  "staged_effective_sources": staged, "tuning_after_exposure": False,
                  "repair_after_exposure": False, "alternative_promotion_after_failure": False,
                  "second_official_permitted": False}
    _exclusive_json(OFFICIAL_RESULT, output, "R4 official result already exists")
    check_integrity(ROOT)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--source-commit", required=True)
    for action in ("holdout", "official"):
        command = sub.add_parser(action)
        command.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = create_freeze(args.source_commit)
    elif args.action == "holdout":
        result = open_holdout(args.private_dir)
    else:
        result = official(args.private_dir)
    print(json.dumps({"schema": result["schema"], "status": result.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
