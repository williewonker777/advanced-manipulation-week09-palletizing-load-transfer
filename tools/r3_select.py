#!/usr/bin/env python3
"""Conservative, pure R3 result review and challenger selection.

This module never runs an episode and never changes controller parameters.  All
review functions accept already-loaded dictionaries so tests and later stages
can inject immutable evidence.  File discovery/writing is confined to the CLI.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import tempfile


PROTOCOL_SHA256 = "dcb76b495d8c2e73a5047f253a7a43a9fc46b642653e4054c5d947109f8f6c39"
COST_METRICS = (
    "rms_mm", "max_mm", "torque_step_pick_nm", "torque_step_place_nm",
    "table_force_place_max_n",
)
PRIORITY = {
    "servo": ("A_handoff_long", "A_handoff_short"),
    "torque": ("B_mass_phase", "B_phase", "B_mass"),
}


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _row_id(row):
    return row.get("case_id", row.get("caseid"))


def condition_key(row):
    """Exact matched-condition key, including collision/perturbation phase."""
    if _row_id(row) is None:
        raise ValueError("row is missing case_id")
    com = row.get("com")
    if not isinstance(com, (list, tuple)) or len(com) != 3:
        raise ValueError("row com must have three elements")
    tags = {
        name: row.get(name) for name in (
            "kind", "collision", "pick_offset", "perturb", "perturb_seed",
            "perturb_family", "perturb_params", "phase", "collision_phase",
            "collision_time_s", "collision_force_n", "collision_direction",
        ) if name in row
    }
    return (
        row.get("architecture"), _row_id(row), _finite(row["mass"], "mass"),
        tuple(_finite(v, "com") for v in com), _canonical(tags),
    )


def pair_results(baseline_document, candidate_document):
    """Pair rows one-to-one; any missing, duplicate, or changed condition fails."""
    baseline_rows = list(baseline_document.get("results", ()))
    candidate_rows = list(candidate_document.get("results", ()))
    baseline = {}
    candidate = {}
    for destination, rows, label in ((baseline, baseline_rows, "baseline"),
                                     (candidate, candidate_rows, "candidate")):
        for row in rows:
            key = condition_key(row)
            if key in destination:
                raise ValueError("duplicate " + label + " condition")
            destination[key] = row
    if baseline.keys() != candidate.keys():
        missing = len(baseline.keys() - candidate.keys())
        extra = len(candidate.keys() - baseline.keys())
        raise ValueError(f"matched conditions differ: missing={missing}, extra={extra}")
    return [(baseline[key], candidate[key]) for key in sorted(baseline,
                                                                key=repr)]


def ratio(candidate, baseline):
    candidate = _finite(candidate, "candidate metric")
    baseline = _finite(baseline, "baseline metric")
    if baseline < 0.0 or candidate < 0.0:
        raise ValueError("cost metrics must be nonnegative")
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return candidate / baseline


def review_costs(pairs, limit=1.05, references_only=False):
    rows = []
    for baseline, candidate in pairs:
        is_reference = (candidate.get("kind", "reference") == "reference"
                        and candidate.get("collision") is None
                        and candidate.get("pick_offset") is None)
        if references_only and not is_reference:
            continue
        metrics = {}
        for name in COST_METRICS:
            value = ratio(candidate[name], baseline[name])
            metrics[name] = {"ratio": value, "pass_le_1_05": value <= limit}
        rows.append({"case_id": _row_id(candidate), "mass": candidate["mass"],
                     "condition_key": repr(condition_key(candidate)),
                     "metrics": metrics,
                     "pass": all(item["pass_le_1_05"]
                                 for item in metrics.values())})
    if not rows:
        return {"status": "FAIL", "reason": "no matched reference rows",
                "rows": [], "worst_ratio": math.inf}
    worst = max(item["ratio"] for row in rows
                for item in row["metrics"].values())
    return {"status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
            "rows": rows, "worst_ratio": worst}


def review_timing(rows, p99_limit=800.0, max_goal=1000.0):
    records = []
    for row in rows:
        try:
            wall = row["timing"]["combined"]["wall"]
            p99 = _finite(wall["p99_us"], "combined wall p99")
            maximum = _finite(wall["max_us"], "combined wall max")
            outliers = list(row["timing"].get("outliers", ()))
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "FAIL", "reason": "missing timing: " + str(error),
                    "rows": records}
        records.append({"case_id": _row_id(row), "p99_us": p99,
                        "max_us": maximum, "outliers": outliers,
                        "p99_pass_strict_lt_800": p99 < p99_limit,
                        "max_1000_goal_met": maximum <= max_goal})
    return {"status": "PASS" if records and all(
                record["p99_pass_strict_lt_800"] for record in records) else "FAIL",
            "rows": records,
            "max_goal_is_record_only": True}


def review_normal_safety(rows):
    records = []
    for row in rows:
        try:
            saturation = _finite(row["saturation_pct"], "saturation_pct")
            false_alarms = int(row["false_alarms"])
            diverged = bool(row["diverged"])
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "FAIL", "reason": "missing safety field: " + str(error)}
        passed = saturation == 0.0 and false_alarms == 0 and not diverged
        records.append({"case_id": _row_id(row), "saturation_pct": saturation,
                        "false_alarms": false_alarms, "diverged": diverged,
                        "pass": passed})
    return {"status": "PASS" if records and all(r["pass"] for r in records)
            else "FAIL", "rows": records,
            "normal_recalibration_used": False,
            "false_alarm_policy": "reject; no recalibration implemented"}


def review_g4(rows, thresholds):
    records = []
    for row in rows:
        g4 = row.get("g4", {})
        mean = g4.get("table_mean_over_mg")
        minimum = g4.get("table_min_over_mg")
        peak = g4.get("place_peak_over_mg")
        increase = g4.get("mean_increase_over_v1")
        records.append({
            "case_id": _row_id(row), "mass": row.get("mass"),
            "mean_support_over_mg": mean,
            "mean_pass": mean is not None and mean >= thresholds["mean_support_over_mg_at_least"],
            "minimum_support_over_mg": minimum,
            "minimum_pass": minimum is not None and minimum >= thresholds["minimum_support_over_mg_at_least"],
            "mean_increase_over_v1": increase,
            "increase_pass": increase is not None and increase >= thresholds["mean_increase_over_v1_at_least"],
            "peak_place_support_over_mg": peak,
            "peak_pass": peak is not None and peak <= thresholds["peak_place_support_over_mg_at_most"],
        })
    return {"thresholds": dict(thresholds), "rows": records,
            "adoption_gate": False,
            "note": "reported under original G4 thresholds; not a C3/C4 adoption gate"}


def _stability_label(row):
    for name in ("candidate", "variant", "spec_id", "label", "id"):
        if name in row:
            return row[name]
    return None


def _protocol_forced_points(protocol, architecture):
    if protocol is None:
        return None
    stability = protocol["stability_R3"]
    if architecture == "servo":
        masses = protocol["stability"]["masses_kg"]
        times = stability["A_times_s"]
    else:
        masses = list(protocol["stability"]["masses_kg"])
        masses.extend(stability["B_mass_knots_kg"])
        masses = sorted(set(float(value) for value in masses))
        times = stability["B_times_s"]
    return {(float(mass), float(time)) for mass in masses for time in times}


def _capture_actual_points(document, candidate_id, architecture):
    expected = set()
    matching_captures = 0
    for capture in document.get("captures", ()):
        if (capture.get("candidate") != candidate_id
                or capture.get("architecture") != architecture
                or capture.get("source_kind") != "actual_candidate"):
            continue
        matching_captures += 1
        mass = float(capture["mass_kg"])
        for sample in capture.get("sample_indices", ()):
            expected.add((mass, int(sample)))
    return expected, matching_captures


def review_stability(document, candidate_id, architecture, source_kind=None,
                     required_points=None, protocol=None, expected_source_sha256=None):
    if not document or not isinstance(document.get("analyses"), list):
        return {"status": "FAIL" if source_kind == "forced" else "pending",
                "reason": "stability analyses unavailable", "rows": []}
    emitted_kind = {"forced": "forced_r1", "actual": "actual_candidate"}.get(
        source_kind, source_kind)
    matching = [row for row in document["analyses"]
                if _stability_label(row) == candidate_id
                and row.get("architecture") == architecture
                and (emitted_kind is None or row.get("source_kind") == emitted_kind)]
    records = []
    for row in matching:
        try:
            closed = row["closed_loop"]
            diagnostics = closed["jacobian_step_diagnostics"]
            radius_flag = closed["protocol_radius_pass_lt_1_plus_1e_minus_6"] is True
            radius = _finite(closed["spectral_radius"], "spectral_radius")
            relative = _finite(diagnostics["A_relative_step_change"],
                               "A_relative_step_change")
            mechanical_fd = row["mechanical_fd"]
            mechanical = mechanical_fd["step_input"]["step_converged"] is True
            previous = mechanical_fd["previous_applied_input_for_jts"]["step_converged"] is True
            poles = closed["poles"]
            if not isinstance(poles, list) or not poles:
                raise ValueError("poles missing")
        except (KeyError, TypeError, ValueError) as error:
            records.append({"pass": False, "reason": "missing/invalid: " + str(error)})
            continue
        passed = radius_flag and radius < 1.0 + 1e-6 and relative <= 0.001 \
            and mechanical and previous
        # The retained preliminary artifact is historical evidence, not an
        # adoption-safe substitute for the full corrected A/B qualification.
        passed = passed and row.get("fd_convergence", {}).get("pass") is True
        source_matches = (row.get("module_sha256") == expected_source_sha256
                          if expected_source_sha256 is not None else None)
        if expected_source_sha256 is not None:
            passed = passed and source_matches
        records.append({"mass_kg": row.get("mass_kg"), "time_s": row.get("time_s"),
                        "sample_index": row.get("sample_index"),
                        "source_kind": row.get("source_kind"),
                        "spectral_radius": radius,
                        "A_relative_step_change": relative,
                        "mechanical_step_converged": mechanical,
                        "previous_applied_step_converged": previous,
                        "full_fd_convergence": row.get("fd_convergence"),
                        "source_sha256_matches": source_matches,
                        "pass": passed})
    coverage = None
    if required_points is not None:
        actual = {(float(row.get("mass_kg")), float(row.get("time_s")))
                  for row in matching}
        expected = {(float(point[0]), float(point[1])) for point in required_points}
        coverage = {"required": sorted(expected), "missing": sorted(expected - actual),
                    "extra": sorted(actual - expected),
                    "duplicates": len(matching) - len(actual),
                    "pass": actual == expected and len(matching) == len(actual)}
    elif emitted_kind == "forced_r1" and protocol is not None:
        expected = _protocol_forced_points(protocol, architecture)
        actual = {(float(row.get("mass_kg")), float(row.get("time_s")))
                  for row in matching}
        coverage = {"required": sorted(expected), "missing": sorted(expected - actual),
                    "extra": sorted(actual - expected),
                    "duplicates": len(matching) - len(actual),
                    "pass": actual == expected and len(matching) == len(actual)}
    elif emitted_kind == "actual_candidate":
        expected, captures = _capture_actual_points(document, candidate_id, architecture)
        actual = {(float(row.get("mass_kg")), int(row.get("sample_index")))
                  for row in matching if row.get("sample_index") is not None}
        coverage = {"required_from_captures": sorted(expected),
                    "missing": sorted(expected - actual), "extra": sorted(actual - expected),
                    "matching_captures": captures,
                    "duplicates": len(matching) - len(actual),
                    "pass": (captures > 0 and bool(expected) and actual == expected
                             and len(matching) == len(actual))}
    passed = bool(records) and all(row["pass"] for row in records)
    if coverage is not None:
        passed = passed and coverage["pass"]
    return {"status": "PASS" if passed else "FAIL", "rows": records,
            "coverage": coverage}


def verify_protocol_gates(r3_protocol, r2_protocol):
    keys = ("G4", "safety_gates", "cost_gates", "timing", "stability")
    comparison = {key: r3_protocol.get(key) == r2_protocol.get(key) for key in keys}
    return {"status": "PASS" if all(comparison.values()) else "FAIL",
            "exact_R2_gate_match": comparison}


def _tracking_efficacy(pairs):
    positive = [(base, cand) for base, cand in pairs if float(cand["mass"]) > 0.0]
    if not positive:
        return {"status": "FAIL", "reason": "no positive-mass matched cases"}
    rms = [ratio(cand["rms_mm"], base["rms_mm"]) for base, cand in positive]
    maximum = [ratio(cand["max_mm"], base["max_mm"]) for base, cand in positive]
    score = min(sum(rms) / len(rms), sum(maximum) / len(maximum))
    return {"status": "PASS" if score <= 0.90 else "FAIL",
            "mean_rms_ratio": sum(rms) / len(rms),
            "mean_max_ratio": sum(maximum) / len(maximum), "score": score}


def _validation_conditions_match(pairs, expected_conditions):
    expected = {index: (float(condition["mass"]),
                        tuple(float(value) for value in condition["com"]))
                for index, condition in enumerate(expected_conditions)}
    actual = {_row_id(candidate): (float(candidate["mass"]),
                                   tuple(float(value) for value in candidate["com"]))
              for _, candidate in pairs}
    return expected == actual


def review_validation_candidate(protocol, baseline_document, candidate_document,
                                forced_stability, actual_stability=None,
                                expected_spec=None, expected_conditions=None):
    rows = list(candidate_document.get("results", ()))
    candidate_id = rows[0].get("variant") if rows else None
    architecture = rows[0].get("architecture") if rows else None
    if not candidate_id or architecture not in PRIORITY:
        return {"candidate": candidate_id, "architecture": architecture,
                "validation_status": "FAIL", "reason": "missing candidate rows"}
    if any(row.get("variant") != candidate_id or
           row.get("architecture") != architecture for row in rows):
        return {"candidate": candidate_id, "architecture": architecture,
                "validation_status": "FAIL", "reason": "mixed candidate document"}
    if expected_spec is not None and any(
            row.get("specification") != expected_spec for row in rows):
        return {"candidate": candidate_id, "architecture": architecture,
                "validation_status": "FAIL",
                "reason": "row specification differs from registered spec"}
    try:
        pairs = pair_results(baseline_document, candidate_document)
        conditions = (protocol["validation"] if expected_conditions is None
                      else expected_conditions)
        if not _validation_conditions_match(pairs, conditions):
            raise ValueError("validation rows differ from registered conditions")
        costs = review_costs(pairs)
    except (KeyError, TypeError, ValueError) as error:
        return {"candidate": candidate_id, "architecture": architecture,
                "validation_status": "FAIL", "reason": str(error)}
    safety = review_normal_safety(rows)
    timing = review_timing(rows)
    expected_source = candidate_document.get("metadata", {}).get(
        "source_sha256", {}).get(
            f"experiments/r3_generated/{candidate_id}_{architecture}.py")
    source_binding = {"status": "PASS" if isinstance(expected_source, str)
                      and len(expected_source) == 64 else "FAIL",
                      "validation_controller_sha256": expected_source}
    forced = review_stability(forced_stability, candidate_id, architecture, "forced",
                              protocol=protocol, expected_source_sha256=expected_source)
    efficacy = (_tracking_efficacy(pairs) if architecture == "torque" else
                {"status": "deferred", "stage": "matched analysis perturbations"})
    eligible = (costs["status"] == safety["status"] == timing["status"] == "PASS"
                and forced["status"] == "PASS"
                and source_binding["status"] == "PASS"
                and (architecture == "servo" or efficacy["status"] == "PASS"))
    actual = review_stability(actual_stability, candidate_id, architecture, "actual",
                              protocol=protocol, expected_source_sha256=expected_source)
    return {
        "candidate": candidate_id, "architecture": architecture,
        "validation_status": "eligible" if eligible else "rejected",
        "validation_eligible": eligible,
        "provisional_only": True,
        "development_ready": eligible and actual["status"] == "PASS",
        "costs": costs, "normal_safety": safety, "timing": timing,
        "tracking_efficacy": efficacy, "forced_stability": forced,
        "actual_stability": actual, "g4": review_g4(rows, protocol["G4"]),
        "source_binding": source_binding,
        "cost_score": costs["worst_ratio"],
    }


def review_redesign_candidate(protocol, baseline_document, candidate_document,
                              forced_stability, amendment,
                              actual_stability=None):
    """Review one root-authorized redesign without selecting or promoting it."""
    if not isinstance(amendment, dict):
        return {"validation_status": "FAIL", "reason": "redesign amendment missing"}
    reason = amendment.get("reason")
    spec = amendment.get("spec")
    seed = amendment.get("seed", amendment.get("validation_seed"))
    if amendment.get("redesign_number") != 1 or not isinstance(reason, str) \
            or not reason.strip() or not isinstance(spec, dict) or seed is None:
        return {"validation_status": "FAIL",
                "reason": "redesign needs number=1, prior reason, spec, and new seed"}
    rows = list(candidate_document.get("results", ()))
    if not rows or rows[0].get("variant") != spec.get("id"):
        return {"validation_status": "FAIL",
                "reason": "redesign result does not match amendment spec"}
    result = review_validation_candidate(
        protocol, baseline_document, candidate_document, forced_stability,
        actual_stability, expected_spec=spec,
        expected_conditions=protocol["redesign_validation"])
    result.update(redesign=True, amendment_reason=reason,
                  amendment_seed=seed, automatic_promotion=False,
                  root_decision_required=True)
    return result


def build_validation_review(protocol, r2_protocol, baselines, candidates,
                            forced_stability, actual_stability=None):
    gate_check = verify_protocol_gates(protocol, r2_protocol)
    reviews = []
    specs = {spec["id"]: spec for spec in protocol["candidate_grid"]}
    for candidate_id in PRIORITY["servo"] + PRIORITY["torque"]:
        spec = specs.get(candidate_id)
        document = candidates.get(candidate_id)
        architecture = spec.get("architecture") if spec else None
        if spec is None or document is None or architecture not in baselines:
            reviews.append({"candidate": candidate_id, "architecture": architecture,
                            "validation_status": "missing", "validation_eligible": False,
                            "provisional_only": True, "development_ready": False})
            continue
        reviews.append(review_validation_candidate(
            protocol, baselines[architecture], document, forced_stability,
            actual_stability, expected_spec=spec,
            expected_conditions=protocol["validation"]))
    return {"schema": "R3_VALIDATION_REVIEW.v1",
            "protocol_sha256": PROTOCOL_SHA256, "protocol_gate_check": gate_check,
            "status": "complete" if gate_check["status"] == "PASS" else "invalid_protocol",
            "candidates": reviews,
            "scope": "validation eligibility/provisional selection; not final adoption"}


def build_challengers(validation_review):
    output = {}
    protocol_valid = (validation_review.get("protocol_gate_check", {})
                      .get("status", "PASS") == "PASS")
    for architecture, priority in PRIORITY.items():
        reviews = {row["candidate"]: row for row in validation_review["candidates"]
                   if row.get("architecture") == architecture}
        eligible = [candidate for candidate in priority
                    if reviews.get(candidate, {}).get("validation_eligible")]
        if not protocol_valid:
            eligible = []
            reviews = {}
        if eligible:
            selected = eligible[0]
            row = reviews[selected]
            output[architecture] = {
                "selected": selected, "role": "provisional_challenger",
                "validation_eligible": True,
                "development_ready": bool(row.get("development_ready")),
                "actual_stability_status": row.get("actual_stability", {}).get("status", "pending"),
                "irreversibly_ineligible_for_adoption": False,
            }
            continue
        completed = [row for row in reviews.values()
                     if row.get("validation_status") == "rejected"
                     and math.isfinite(float(row.get("cost_score", math.inf)))]
        diagnostic = min(completed, key=lambda row: row["cost_score"], default=None)
        output[architecture] = ({"selected": diagnostic["candidate"],
                                 "role": "diagnostic_challenger",
                                 "validation_eligible": False,
                                 "development_ready": False,
                                 "irreversibly_ineligible_for_adoption": True,
                                 "no_later_revival": True}
                                if diagnostic else
                                {"selected": None, "role": "none",
                                 "validation_eligible": False,
                                 "development_ready": False,
                                 "irreversibly_ineligible_for_adoption": True})
    return {"schema": "R3_CHALLENGERS.v1", "architectures": output,
            "one_challenger_per_architecture": True,
            "no_promotion_after_development": True,
            "final_adoption": False}


def review_runtime_safety(pairs, require_zero_saturation=False):
    records = []
    for baseline, candidate in pairs:
        try:
            baseline_saturation = _finite(
                baseline.get("saturation_pct", 0.0), "baseline saturation_pct")
            candidate_saturation = _finite(
                candidate["saturation_pct"], "candidate saturation_pct")
            diverged = bool(candidate["diverged"])
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "FAIL", "reason": "missing safety field: " + str(error)}
        saturation_regression = candidate_saturation > baseline_saturation + 1e-15
        zero_failure = require_zero_saturation and candidate_saturation != 0.0
        passed = not diverged and not saturation_regression and not zero_failure
        records.append({"case_id": _row_id(candidate), "diverged": diverged,
                        "baseline_saturation_pct": baseline_saturation,
                        "candidate_saturation_pct": candidate_saturation,
                        "saturation_regression": saturation_regression,
                        "zero_saturation_required": require_zero_saturation,
                        "pass": passed})
    return {"status": "PASS" if records and all(row["pass"] for row in records)
            else "FAIL", "rows": records}


def review_detection(pairs):
    records = []
    for baseline, candidate in pairs:
        reference = (candidate.get("kind", "reference") == "reference"
                     and candidate.get("collision") is None
                     and candidate.get("pick_offset") is None)
        baseline_outcome = baseline.get("collision_outcome")
        candidate_outcome = candidate.get("collision_outcome")
        lost = baseline_outcome == "detected" and candidate_outcome != "detected"
        extra_fa = int(candidate.get("false_alarms", 0)) > int(
            baseline.get("false_alarms", 0))
        reference_fa = reference and int(candidate.get("false_alarms", 0)) != 0
        tool_loss = (baseline.get("tool_contact_delay_ms") is not None
                     and candidate.get("tool_contact_delay_ms") is None)
        records.append({"case_id": _row_id(candidate), "lost_R1_detection": lost,
                        "extra_false_alarm": extra_fa,
                        "reference_false_alarm": reference_fa,
                        "tool_contact_loss": tool_loss,
                        "pass": not (lost or extra_fa or reference_fa or tool_loss)})
    return {"status": "PASS" if records and all(row["pass"] for row in records)
            else "FAIL", "rows": records}


def review_development(baseline_document, candidate_document):
    try:
        pairs = pair_results(baseline_document, candidate_document)
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "FAIL", "reason": str(error)}
    costs = review_costs(pairs, references_only=True)
    timing = review_timing(list(candidate_document.get("results", ())))
    detection = review_detection(pairs)
    safety = review_runtime_safety(pairs)
    passed = (costs["status"] == "PASS" and timing["status"] == "PASS"
              and detection["status"] == "PASS" and safety["status"] == "PASS")
    return {"status": "PASS" if passed else "FAIL", "costs": costs,
            "detection": detection["rows"], "detection_review": detection,
            "safety": safety, "timing": timing}


def review_d4(baseline_document, candidate_document, architecture,
              expected_rows=385):
    try:
        pairs = pair_results(baseline_document, candidate_document)
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "FAIL", "reason": str(error)}
    if len(pairs) != expected_rows:
        return {"status": "FAIL", "reason": f"expected {expected_rows} rows, got {len(pairs)}"}
    rows = [candidate for _, candidate in pairs]
    if architecture not in PRIORITY or any(row.get("architecture") != architecture
                                           for row in rows):
        return {"status": "FAIL", "reason": "D4 architecture mismatch"}
    costs = review_costs(pairs, references_only=True)
    timing = review_timing(rows)
    safety = review_runtime_safety(pairs)
    detected = sum(row.get("collision_outcome") == "detected" for row in rows)
    missed = sum(row.get("collision_outcome") == "missed" for row in rows)
    reference_fa = sum(int(row.get("false_alarms", 0)) for row in rows
                       if row.get("kind") == "reference")
    saturation_episodes = sum(float(row.get("saturation_pct", 0.0)) > 0.0 for row in rows)
    max_saturation = max(float(row.get("saturation_pct", 0.0)) for row in rows)
    extra_per_case = [condition_key(candidate) for baseline, candidate in pairs
                      if int(candidate.get("false_alarms", 0))
                      > int(baseline.get("false_alarms", 0))]
    total_extra = (sum(int(c.get("false_alarms", 0)) for _, c in pairs)
                   > sum(int(b.get("false_alarms", 0)) for b, _ in pairs))
    episode_limit = 2 if architecture == "servo" else 0
    pct_limit = 0.15 if architecture == "servo" else 0.0
    passed = (costs["status"] == timing["status"] == safety["status"] == "PASS"
              and detected >= 111 and missed <= 69 and reference_fa == 0
              and saturation_episodes <= episode_limit and max_saturation <= pct_limit
              and not extra_per_case and not total_extra)
    return {"status": "PASS" if passed else "FAIL", "detected": detected,
            "missed": missed, "reference_false_alarms": reference_fa,
            "saturation_episodes": saturation_episodes,
            "max_saturation_pct": max_saturation,
            "extra_false_alarm_cases": len(extra_per_case),
            "total_extra_false_alarms": total_extra,
            "costs": costs, "timing": timing, "safety": safety}


def review_perturbations(baseline_document, candidate_document, architecture):
    try:
        pairs = pair_results(baseline_document, candidate_document)
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "FAIL", "reason": str(error)}
    costs = review_costs(pairs, references_only=True)
    reference_pairs = [(base, candidate) for base, candidate in pairs
                       if candidate.get("kind", "reference") == "reference"
                       and candidate.get("collision") is None
                       and candidate.get("pick_offset") is None]
    efficacy = (_tracking_efficacy(reference_pairs) if architecture == "servo" else
                {"status": "not_required_after_validation"})
    timing = review_timing(list(candidate_document.get("results", ())))
    detection = review_detection(pairs)
    safety = review_runtime_safety(pairs)
    passed = (costs["status"] == "PASS" and timing["status"] == "PASS"
              and detection["status"] == "PASS" and safety["status"] == "PASS"
              and (architecture != "servo" or efficacy["status"] == "PASS"))
    return {"status": "PASS" if passed else "FAIL", "costs": costs,
            "tracking_efficacy": efficacy, "timing": timing,
            "detection": detection, "safety": safety}


def review_holdout(baseline_document, candidate_document):
    try:
        pairs = pair_results(baseline_document, candidate_document)
    except (KeyError, TypeError, ValueError) as error:
        return {"status": "FAIL", "reason": str(error),
                "no_tuning_permitted": True}
    costs = review_costs(pairs)
    timing = review_timing(list(candidate_document.get("results", ())))
    detection = review_detection(pairs)
    safety = review_runtime_safety(pairs)
    passed = all(review["status"] == "PASS"
                 for review in (costs, timing, detection, safety))
    return {"status": "PASS" if passed else "FAIL", "costs": costs,
            "timing": timing, "detection": detection, "safety": safety,
            "no_tuning_permitted": True}


def finalize_architectures(challengers, poststage_reviews):
    """Independent immutable selection: failed challenger falls back to R1."""
    final = {}
    for architecture in ("servo", "torque"):
        selected = challengers["architectures"][architecture]
        reviews = poststage_reviews.get(architecture, {})
        required = ("development", "d4", "perturbations", "holdout")
        passes = all(reviews.get(stage, {}).get("status") == "PASS"
                     for stage in required)
        adoptable = (selected.get("validation_eligible") is True
                     and not selected.get("irreversibly_ineligible_for_adoption", True)
                     and selected.get("actual_stability_status") == "PASS"
                     and passes)
        final[architecture] = {
            "selection": selected.get("selected") if adoptable else "R1",
            "adopted": adoptable,
            "fallback": None if adoptable else "immutable R1",
            "challenger_cannot_be_replaced_or_revived": True,
            "stage_status": {stage: reviews.get(stage, {}).get("status", "missing")
                             for stage in required},
        }
    return {"schema": "R3_FINAL_SELECTION.v1", "architectures": final,
            "architecture_decisions_independent": True}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_portable(value), stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_validation_inputs(results_dir, protocol):
    results_dir = Path(results_dir)
    baselines = {}
    candidates = {}
    for architecture in PRIORITY:
        path = results_dir / f"R3_VALIDATION_R1_{architecture}.json"
        if path.is_file():
            baselines[architecture] = _read(path)
    for spec in protocol["candidate_grid"]:
        path = results_dir / f"R3_VALIDATION_{spec['id']}_{spec['architecture']}.json"
        if path.is_file():
            candidates[spec["id"]] = _read(path)
    return baselines, candidates


def _portable(value):
    """Keep aggregate JSON standards-compliant when an internal ratio is infinite."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--r2-protocol", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--forced-stability", type=Path, required=True)
    parser.add_argument("--actual-stability", type=Path)
    parser.add_argument("--write", action="store_true",
                        help="atomically write the two registered review artifacts")
    args = parser.parse_args(argv)
    protocol_bytes = args.protocol.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("PROTOCOL_R3 bytes differ from registered protocol")
    protocol = json.loads(protocol_bytes)
    r2_protocol = _read(args.r2_protocol)
    baselines, candidates = load_validation_inputs(args.results_dir, protocol)
    forced = _read(args.forced_stability)
    actual = _read(args.actual_stability) if args.actual_stability else None
    review = build_validation_review(protocol, r2_protocol, baselines, candidates,
                                     forced, actual)
    challengers = build_challengers(review)
    if args.write:
        _atomic_write(args.results_dir / "R3_VALIDATION_REVIEW.json", review)
        _atomic_write(args.results_dir / "R3_CHALLENGERS.json", challengers)
    else:
        print(json.dumps(_portable({"review": review, "challengers": challengers}),
                         indent=2, sort_keys=True, allow_nan=False))
    return 0 if review["protocol_gate_check"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
