"""Generate the v2 public report bundle from frozen JSON and private traces.

This program is deliberately post-processing only: it does not import the
simulator or production modules.  Official evaluator output and supplemental
non-official studies are kept separate in every public artifact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "v2_protocol"

REQUIRED_JSON = {
    "adoption": "ADOPTION.json",
    "official": "S4_OFFICIAL.json",
    "validation": "S4_L3_VALIDATION.json",
    "development": "S4_L3_DEVELOPMENT.json",
    "freeze": "FREEZE.json",
    "d4": "S5_D4.json",
    "holdout": "S5_HOLDOUT.json",
    "timing": "S5_TIMING.json",
    "perturbations": "S5_PERTURBATIONS.json",
    "geometry": "GEOMETRY_AUDIT.json",
}
EXPECTED_ROWS = {"d4": 1540, "holdout": 152, "timing": 12, "perturbations": 136}
TRACE_NAMES = tuple(f"{version}_{architecture}.npz" for version in ("base", "v1", "v2")
                    for architecture in ("servo", "torque"))
TRACE_KEYS = {"t", "tau", "e_mm", "f_table", "r", "detect", "q", "q_d", "f_ext"}
OUTCOMES = ("detected", "missed", "alarm_before_collision", "indistinguishable")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required final input is absent: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def rows_of(document: dict[str, Any], name: str, expected: int | None = None) -> list[dict[str, Any]]:
    rows = document.get("results")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{name} must contain a results object-array")
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{name} must contain {expected} rows, found {len(rows)}")
    return rows


def validate_inputs(documents: dict[str, dict[str, Any]]) -> None:
    official = documents["official"]
    if official.get("official") is not True or official.get("source") != "unchanged evaluate.py":
        raise ValueError("S4_OFFICIAL must be the unchanged evaluate.py output")
    combinations = official.get("combinations")
    if not isinstance(combinations, dict) or len(combinations) != 8:
        raise ValueError("S4_OFFICIAL must contain exactly eight official combinations")
    for key, expected in EXPECTED_ROWS.items():
        rows_of(documents[key], key, expected)
    if "nominal_parity" not in documents["perturbations"]:
        raise ValueError("S5_PERTURBATIONS must retain nominal_parity")
    if documents["holdout"].get("metadata", {}).get("official") is True:
        raise ValueError("sealed holdout is supplemental, not official evaluate.py output")


def load_final_inputs(data_dir: Path = DATA) -> dict[str, dict[str, Any]]:
    documents = {key: load_json(data_dir / filename) for key, filename in REQUIRED_JSON.items()}
    validate_inputs(documents)
    return documents


def outcome_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count references, offsets and collisions without conflating their guards."""
    summary = {name: 0 for name in OUTCOMES}
    summary.update(no_collision_false_alarms=0, collision_extra_false_alarms=0,
                   misses=0, masked=0, collisions=0, references=0,
                   offset_episodes=0, offset_false_alarm_rises=0,
                   offset_tool_contacts=0, offset_tool_contact_misses=0)
    for row in rows:
        collision = row.get("collision")
        kind = row.get("kind")
        if kind == "offset" or (collision is None and row.get("pick_offset") is not None):
            summary["offset_episodes"] += 1
            summary["offset_false_alarm_rises"] += int(row.get("false_alarms", 0))
            contact = float(row.get("tool_contact_peak_n", 0.0)) > 1.0
            summary["offset_tool_contacts"] += int(contact)
            summary["offset_tool_contact_misses"] += int(
                contact and row.get("tool_contact_delay_ms") is None)
            continue
        if collision is None:
            summary["references"] += 1
            summary["no_collision_false_alarms"] += int(row.get("false_alarms", 0))
            continue
        summary["collisions"] += 1
        outcome = row.get("collision_outcome")
        if outcome not in OUTCOMES:
            raise ValueError(f"collision row lacks a recognized outcome: {outcome!r}")
        summary[outcome] += 1
        summary["collision_extra_false_alarms"] += int(row.get("false_alarms", 0))
    summary["misses"] = summary["missed"]
    summary["masked"] = summary["alarm_before_collision"] + summary["indistinguishable"]
    return summary


def grouped_counts(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = row.get("variant", row.get("version"))
        grouped[(str(label), str(row.get("architecture")))].append(row)
    return {f"{variant}/{architecture}": outcome_counts(group)
            for (variant, architecture), group in sorted(grouped.items())}


def timing_extremes(named_rows: Iterable[tuple[str, dict[str, Any]]] | Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Find the global measured maximum and retain every >=1 ms timing event."""
    values: list[tuple[float, float, str, dict[str, Any]]] = []
    events: list[dict[str, Any]] = []
    for item in named_rows:
        if isinstance(item, tuple):
            dataset, row = item
        else:
            dataset, row = "unspecified", item
        combined = row.get("timing", {}).get("combined", {})
        wall = combined.get("wall", combined)
        cpu = combined.get("thread_cpu", combined)
        if "max_us" not in wall:
            continue
        values.append((float(wall["max_us"]), float(cpu.get("max_us", wall["max_us"])), dataset, row))
        for event in row.get("timing", {}).get("outliers", []):
            saved = {"dataset": dataset,
                     "context": {key: row.get(key) for key in
                                 ("variant", "version", "architecture", "case_id", "mass", "kind")},
                     **event}
            events.append(saved)
    if not values:
        raise ValueError("timing rows contain no combined wall maxima")
    wall_max, row_cpu_max, dataset, row = max(values, key=lambda item: item[0])
    worst_event = max(events, key=lambda event: event.get("combined", {}).get("wall_ns", -1),
                      default=None)
    exact = (worst_event is not None and
             abs(float(worst_event["combined"]["wall_ns"]) / 1000.0 - wall_max) < 1e-6)
    if exact:
        combined_event = worst_event["combined"]
        cpu_detail = {
            "value_us": float(combined_event["cpu_ns"]) / 1000.0,
            "semantics": "thread CPU at the exact worst-wall cycle",
            "voluntary_context_switches": int(combined_event.get("voluntary_context_switches", 0)),
            "involuntary_context_switches": int(combined_event.get("involuntary_context_switches", 0)),
        }
    else:
        cpu_detail = {
            "value_us": row_cpu_max,
            "semantics": "row aggregate thread-CPU maximum; exact worst-wall cycle is below the outlier capture threshold",
            "voluntary_context_switches": None,
            "involuntary_context_switches": None,
        }
    return {
        "observed_wall_max_us": wall_max,
        "cpu_at_worst_wall": cpu_detail,
        "strict_below_1ms": wall_max < 1000.0,
        "retained_outlier_records": len(events),
        "outlier_events": events,
        "worst_dataset": dataset,
        "worst_context": {key: row.get(key) for key in
                          ("variant", "version", "architecture", "case_id", "mass", "kind")},
    }


def timing_group_summaries(named_rows: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Aggregate timing without mixing versions or architectures."""
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for dataset, row in named_rows:
        label = str(row.get("variant", row.get("version")))
        architecture = str(row.get("architecture"))
        combined = row.get("timing", {}).get("combined", {})
        if "max_us" not in combined.get("wall", combined):
            continue
        grouped[(label, architecture)].append((dataset, row))
    summaries = {}
    for (label, architecture), items in sorted(grouped.items()):
        walls = [row["timing"]["combined"].get("wall", row["timing"]["combined"])
                 for _, row in items]
        extreme = timing_extremes(items)
        summaries[f"{label}/{architecture}"] = {
            "version": label,
            "architecture": architecture,
            "measured_rows": len(items),
            "largest_row_wall_median_us": max(float(wall.get("median_us", 0.0)) for wall in walls),
            "largest_row_wall_p99_us": max(float(wall.get("p99_us", 0.0)) for wall in walls),
            "observed_wall_max_us": extreme["observed_wall_max_us"],
            "cpu_at_worst_wall": extreme["cpu_at_worst_wall"],
            "retained_outlier_records": extreme["retained_outlier_records"],
            "outlier_events": extreme["outlier_events"],
            "worst_dataset": extreme["worst_dataset"],
            "worst_context": extreme["worst_context"],
        }
    return summaries


def final_timing_goal(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Judge G5 from final detector rows only; other versions remain comparisons."""
    per_architecture = {}
    for architecture in ("servo", "torque"):
        key = next((candidate for candidate in (f"v2/{architecture}", f"L3/{architecture}")
                    if candidate in groups), None)
        if key is None:
            raise ValueError(f"timing evidence lacks final v2/L3 rows for {architecture}")
        summary = groups[key]
        per_architecture[architecture] = {
            "group": key,
            "observed_wall_max_us": summary["observed_wall_max_us"],
            "strict_below_1ms": summary["observed_wall_max_us"] < 1000.0,
            "cpu_at_worst_wall": summary["cpu_at_worst_wall"],
            "worst_dataset": summary["worst_dataset"],
            "worst_context": summary["worst_context"],
        }
    return {"per_architecture": per_architecture,
            "G5": all(item["strict_below_1ms"] for item in per_architecture.values())}


PHYSICAL_FIELDS = ("rms_mm", "max_mm", "saturation_pct", "diverged",
                   "torque_step_pick_nm", "torque_step_place_nm", "table_force_place_max_n",
                   "tool_contact_peak_n", "tool_contact_first_s")


def matched_physical_comparison(rows: Iterable[dict[str, Any]], *, references_only: bool) -> dict[str, Any]:
    """Compare v1 and final detector rows with identical physical case labels."""
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if references_only and row.get("kind") != "reference":
            continue
        label = str(row.get("variant", row.get("version")))
        normalized = "v1" if label == "v1" else "v2" if label in {"v2", "L3"} else None
        if normalized is None:
            continue
        key = (row.get("architecture"), row.get("case_id"), row.get("mass"),
               json.dumps(row.get("com"), sort_keys=True), row.get("kind"), row.get("phase"),
               row.get("direction"), row.get("force_n"), json.dumps(row.get("pick_offset"), sort_keys=True))
        pairs[key][normalized] = row
    compared = changed = 0
    max_abs_delta = {field: 0.0 for field in PHYSICAL_FIELDS if field != "diverged"}
    for pair in pairs.values():
        if set(pair) != {"v1", "v2"}:
            continue
        compared += 1
        row_changed = False
        for field in PHYSICAL_FIELDS:
            before, after = pair["v1"].get(field), pair["v2"].get(field)
            if before is None and after is None:
                continue
            if (before is None) != (after is None):
                row_changed = True
                continue
            if field == "diverged":
                row_changed |= before != after
            else:
                delta = abs(float(after) - float(before))
                max_abs_delta[field] = max(max_abs_delta[field], delta)
                row_changed |= delta > 1e-12
        changed += int(row_changed)
    return {"matched_pairs": compared, "changed_pairs": changed,
            "unchanged": compared > 0 and changed == 0, "max_abs_delta": max_abs_delta}


def g4_goal(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    details = {}
    rows = list(rows)
    for architecture in ("servo", "torque"):
        candidates = [row for row in rows if row.get("architecture") == architecture and
                      float(row.get("mass", -1)) == 8.0 and row.get("kind") == "reference"]
        base = next((row for row in candidates if row.get("variant", row.get("version")) == "v1"), None)
        final = next((row for row in candidates if row.get("variant", row.get("version")) in {"v2", "L3"}), None)
        if base is None or final is None:
            raise ValueError(f"timing data lacks matched 8 kg v1/final G4 rows for {architecture}")
        before, after = base["g4"], final["g4"]
        increase = float(after["table_mean_over_mg"]) - float(before["table_mean_over_mg"])
        checks = {
            "last20_mean_ge_0p5mg": float(after["table_mean_over_mg"]) >= .5,
            "last20_min_ge_0p25mg": float(after["table_min_over_mg"]) >= .25,
            "mean_increase_ge_0p25mg": increase >= .25,
            "place_peak_le_1p1mg": float(after["place_peak_over_mg"]) <= 1.1,
        }
        details[architecture] = {"v1": before, "v2": after,
                                 "mean_increase_over_mg": increase, "checks": checks,
                                 "pass": all(checks.values())}
    return {"per_architecture": details, "G4": all(value["pass"] for value in details.values())}


def official_delay(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def final_official_names(combinations: dict[str, Any]) -> dict[str, str]:
    result = {}
    for architecture in ("servo", "torque"):
        names = [name for name in combinations if name.startswith(architecture + "_")]
        both = [name for name in names if name.endswith("_v2_v2") or name.endswith("_final_final")]
        if len(both) != 1:
            raise ValueError(f"cannot identify final official combination for {architecture}")
        result[architecture] = both[0]
    return result


def official_goals(official: dict[str, Any]) -> dict[str, Any]:
    combinations = official["combinations"]
    final_names = final_official_names(combinations)
    per_architecture = {}
    for architecture, name in final_names.items():
        combo = combinations[name]
        rows = [row for row in combo["collisions"] if float(row["mass"]) == 8.0 and
                float(row["time"]) == 3.0]
        if len(rows) != 1:
            raise ValueError(f"missing unique 8 kg transport row for {name}")
        collision = rows[0]
        d15 = official_delay(collision.get("force15"))
        d10 = official_delay(collision.get("force10"))
        per_architecture[architecture] = {
            "combination": name,
            "G1_8kg_15N_le_68ms": d15 is not None and d15 <= 68.0,
            "delay_15N_ms": d15,
            "G2_8kg_10N_detected": d10 is not None,
            "delay_10N_ms": d10,
        }
    return {"per_architecture": per_architecture,
            "G1": all(v["G1_8kg_15N_le_68ms"] for v in per_architecture.values()),
            "G2": all(v["G2_8kg_10N_detected"] for v in per_architecture.values())}


def d4_goal(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    details = {}
    for architecture in ("servo", "torque"):
        base_key = next((k for k in counts if k.endswith("/" + architecture) and k.split("/")[0] == "v1"), None)
        final_key = next((k for k in counts if k.endswith("/" + architecture) and k.split("/")[0] in {"v2", "L3"}), None)
        if base_key is None or final_key is None:
            raise ValueError(f"D4 lacks v1/final rows for {architecture}")
        before, after = counts[base_key], counts[final_key]
        details[architecture] = {
            "v1_detected": before["detected"], "v2_detected": after["detected"],
            "v1_collision_extra_false_alarms": before["collision_extra_false_alarms"],
            "v2_collision_extra_false_alarms": after["collision_extra_false_alarms"],
            "v1_no_collision_false_alarms": before["no_collision_false_alarms"],
            "v2_no_collision_false_alarms": after["no_collision_false_alarms"],
            "pass": after["detected"] > before["detected"] and
                    after["collision_extra_false_alarms"] <= before["collision_extra_false_alarms"] and
                    after["no_collision_false_alarms"] <= before["no_collision_false_alarms"],
        }
    return {"per_architecture": details, "G3": all(v["pass"] for v in details.values())}


def baseline_cells(text: str) -> dict[float, list[str]]:
    """Small independent parser for the unchanged evaluator control table."""
    import re
    cells: dict[float, list[str]] = {}
    for line in text.splitlines():
        if not re.match(r"\s*\d+(?:\.\d+)?kg\s", line):
            continue
        columns = [part.strip() for part in line.split("|")]
        key = columns[0].split()
        if len(key) == 1:
            cells[float(key[0].removesuffix("kg"))] = columns[1:]
    if not cells:
        raise ValueError("unchanged evaluator table has no control rows")
    return cells


def evaluator_summary(text: str) -> dict[str, str]:
    """Extract the complete displayed 8 kg control and transport cells."""
    cells = baseline_cells(text)[8.0]
    if len(cells) < 8:
        raise ValueError("8 kg evaluator control row is incomplete")
    transport = None
    for line in text.splitlines():
        columns = [part.strip() for part in line.split("|")]
        key = columns[0].split() if columns else []
        try:
            matches_transport = (len(key) == 2 and float(key[0].removesuffix("kg")) == 8.0 and
                                 float(key[1].removesuffix("s")) == 3.0)
        except ValueError:
            matches_transport = False
        if matches_transport:
            transport = columns[1:]
            break
    if transport is None or len(transport) < 6:
        raise ValueError("8 kg transport evaluator row is absent")
    step = [part.strip() for part in cells[3].split("/")]
    false_alarms = "/".join(part.strip() for part in cells[6].removesuffix(" 회").split("/"))
    return {"rms_max": f"{cells[0]}/{cells[1]}", "pick_place": "/".join(step),
            "table_n": cells[4].removesuffix(" N"), "false_alarms": false_alarms,
            "transport_15n": transport[1], "transport_10n": transport[0],
            "minimum": transport[5]}


def structured_official_summary(combo: dict[str, Any]) -> dict[str, str]:
    control = combo["control"]["8.0"]
    transport = next(row for row in combo["collisions"]
                     if float(row["mass"]) == 8.0 and float(row["time"]) == 3.0)
    return {"rms_max": f"{control['rms_mm']:.2f}/{control['max_mm']:.2f}",
            "pick_place": f"{control['pick_nm']:.1f}/{control['place_nm']:.1f}",
            "table_n": f"{control['table_n']:g}",
            "false_alarms": "/".join(str(value) for value in control["false_alarms"]),
            "transport_15n": str(transport["force15"]),
            "transport_10n": str(transport["force10"]), "minimum": str(transport["minimum"])}


def build_assessment(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validation_rows = rows_of(documents["validation"], "validation")
    development_rows = rows_of(documents["development"], "development")
    d4_rows = rows_of(documents["d4"], "d4")
    holdout_rows = rows_of(documents["holdout"], "holdout")
    timing_rows = rows_of(documents["timing"], "timing")
    perturbation_rows = rows_of(documents["perturbations"], "perturbations")
    d4_counts = grouped_counts(d4_rows)
    holdout_counts = grouped_counts(holdout_rows)
    perturbation_counts = grouped_counts(perturbation_rows)
    all_timing = [(name, row) for name, rows in
                  (("d4", d4_rows), ("holdout", holdout_rows), ("timing", timing_rows),
                   ("perturbations", perturbation_rows)) for row in rows]
    timing = timing_extremes(all_timing)
    timing_groups = timing_group_summaries(all_timing)
    final_timing = final_timing_goal(timing_groups)
    official = official_goals(documents["official"])
    d4 = d4_goal(d4_counts)
    g4 = g4_goal(timing_rows)
    d4_physical = matched_physical_comparison(d4_rows, references_only=True)
    perturbation_physical = matched_physical_comparison(perturbation_rows, references_only=False)
    physical_unchanged = d4_physical["unchanged"] and perturbation_physical["unchanged"]
    perturbation_false = {}
    for architecture in ("servo", "torque"):
        before = perturbation_counts[f"v1/{architecture}"]["no_collision_false_alarms"]
        after = perturbation_counts[f"v2/{architecture}"]["no_collision_false_alarms"]
        perturbation_false[architecture] = {"v1": before, "v2": after, "increase": after - before}
    robustness_false_regression = any(item["increase"] > 0 for item in perturbation_false.values())
    learning_adopted = bool(documents["adoption"].get("learning", {}).get("adopt", False))
    final_variant = "L3" if learning_adopted else "v1"
    return {
        "status": "final evidence assessed; no placeholder claims",
        "provenance": {
            "official": "unchanged evaluate.py; consumed during debugging and not independent efficacy evidence",
            "final_claim_basis": "fresh non-official L3 studies and sealed holdout",
            "traces": "analysis-only internal values; never training inputs",
        },
        "official": official,
        "final_variant": final_variant,
        "supplemental_counts": {"fresh_validation": grouped_counts(validation_rows),
                                "fresh_development": grouped_counts(development_rows),
                                "d4": d4_counts, "sealed_holdout": holdout_counts,
                                "perturbations": perturbation_counts},
        "goals": {
            "G1": official["G1"], "G2": official["G2"], "G3": d4["G3"],
            "G4": g4["G4"],
            "G5": final_timing["G5"],
            "G6": False if physical_unchanged or robustness_false_regression else
                  "not established: changed metrics alone do not prove fewer adverse conditions",
            "G7": "not solved; identical instantaneous vertical wrench remains ambiguous",
        },
        "d4": d4,
        "g4": g4,
        "perturbation_summary": {
            "counts": perturbation_counts,
            "nominal_parity_checked_in_private_harness": True,
            "nominal_parity_values_published": False,
            "nominal_parity_boolean_results_published": True,
            "reference_false_rises_comparison": perturbation_false,
            "reference_false_alarm_regression": robustness_false_regression,
            "G6_reason": "No demonstrated reduction of inherited control failures; all analysis perturbation false rises retained.",
        },
        "physical_parity": {"d4_references": d4_physical,
                            "perturbations": perturbation_physical,
                            "unchanged": physical_unchanged},
        "timing": {"all_versions_global_comparison": timing,
                   "by_version_architecture": timing_groups,
                   "final_v2_goal": final_timing},
        "candidate_disposition": {
            "C": "rejected: G4 0/4 for each architecture; fixed 0.10 s return with 1.5 mm and 0.01 m/s bounds cannot cover the entire registered state box (not a claim that every tested trajectory violated a bound)",
            "L1": "rejected after tentative development gain because disclosed official false alarms regressed",
            "L2": "preserved as exploratory train-envelope evidence; not preferred",
            "L3": ("adopted exact original geometry-height gate repair; weights and thresholds unchanged"
                   if learning_adopted else
                   "not adopted because fresh L3 rule failed; frozen v1 retained"),
        },
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _summary_table(documents: dict[str, dict[str, Any]]) -> str:
    official = documents["official"]["combinations"]
    final = final_official_names(official)
    lines = ["| 구조 | 버전 | TCP RMS/max mm | torque step pick/place N m | table N | FA pick/place/other | 8 kg transport 15 N delay ms | 10 N delay ms | minimum |",
             "|---|---|---:|---:|---:|---:|---|---|---|"]
    for arch, label in (("servo", "A"), ("torque", "B")):
        for version, source in (("baseline", "base"), ("v1", "v1")):
            item = evaluator_summary((DATA / f"S0_{source}_{arch}.txt").read_text(encoding="utf-8"))
            lines.append(f"| {label} | {version} | {item['rms_max']} | {item['pick_place']} | "
                         f"{item['table_n']} | {item['false_alarms']} | {item['transport_15n']} | "
                         f"{item['transport_10n']} | {item['minimum']} |")
        item = structured_official_summary(official[final[arch]])
        lines.append(f"| {label} | v2 | {item['rms_max']} | {item['pick_place']} | {item['table_n']} | "
                     f"{item['false_alarms']} | {item['transport_15n']} | {item['transport_10n']} | "
                     f"{item['minimum']} |")
    return "\n".join(lines)


def render_markdown(documents: dict[str, dict[str, Any]], assessment: dict[str, Any]) -> dict[Path, str]:
    goals = assessment["goals"]
    counts = assessment["supplemental_counts"]
    final_variant = assessment["final_variant"]
    adoption_sentence = ("새 L3 검증이 등록 규칙을 통과해 정확한 기하 높이 gate 복구를 채택했다."
                         if final_variant == "L3" else
                         "새 L3 검증이 등록 규칙을 통과하지 못해 동결된 v1을 유지했다.")
    readme = f"""# ICPBL v2 결과

## 결론

v2는 **v1 제어기 + 정상-only ridge 잔차/대칭 EMA 감지 채널**이며, 기존 v1 경보와 OR한다. 제한된 연구용 채택이며 실기 배치를 권고하는 결과가 아니다. 제어기는 바꾸지 않았다. {adoption_sentence} L3의 가중치, 임계값, EMA와 지속시간은 바꾸지 않았다. 공식 세트는 디버깅에 사용되어 독립 효능 증거가 아니며 G1/G2는 공식 호환성 결과로만 공개한다. 최종 선택 근거는 새 비공식 L3 연구와 봉인 holdout에 한정한다.

{_summary_table(documents)}

- G1/G2/G3/G5: `{goals['G1']}` / `{goals['G2']}` / `{goals['G3']}` / `{goals['G5']}`. D4 B의 추가 오감지 1→3 때문에 G3는 미달이며 검출 수 증가만으로 성공이라 하지 않는다.
- G4: C 후보는 각 구조 0/4로 탈락했다. v2 제어 개선 주장은 없으며 검출기 변경으로 인한 추적 회귀도 없다.
- G6: `{goals['G6']}`. 제어 악화조건은 그대로이며 분석 섭동 무충돌 FA는 A {assessment['perturbation_summary']['reference_false_rises_comparison']['servo']['v1']}→{assessment['perturbation_summary']['reference_false_rises_comparison']['servo']['v2']}, B {assessment['perturbation_summary']['reference_false_rises_comparison']['torque']['v1']}→{assessment['perturbation_summary']['reference_false_rises_comparison']['torque']['v2']}로 별도 보고한다. detector-only 변경을 제어 개선으로 주장하지 않는다.
- G7: 같은 순간 수직 wrench의 지지/충돌 모호성은 해결하지 못했다.

## 영상과 재현

기존 `01`--`04` 영상은 v1이다. 새 `05`/`06`은 3.7 kg 운반 중 15 N의 baseline/v1/v2 3-way 비교이며 `v2_manifest`로 대응 관계를 공개한다. 내부 trace는 그래프 전용 분석값이며 학습에 사용하지 않았다.

## 실행과 제출 파일명 overlay

아래 명령은 키트 작업 사본의 `.venv`를 사용하고 외부 `PYTHONPATH`를 제거한다. 제출 파일은 `my_servo.py`, `my_torque.py`, `my_detector.py`이며 검증기는 이를 기본 이름 `servo_controller.py`, `torque_controller.py`, `collision_detector.py`로 복사한 격리 overlay에서 실행한다.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -c "import numpy, mujoco; print(numpy.__version__, mujoco.__version__)"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/validate_submission.py --prefix my --out results/v2_submission_check
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_servo --detector my_detector --out results/evaluation_v2_A.txt
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/evaluate_checked.py --ctrl my_torque --detector my_detector --out results/evaluation_v2_B.txt
```

깨끗한 data collection은 MuJoCo가 있는 공식 `.venv`에서 수행하고, fitting만 NumPy 전용 `.venv-train`에서 offline으로 수행한다. 공식/holdout을 다시 학습에 넣지 않는다. 같은 saved data/manifest, script와 seed로 byte-identical model JSON 재학습을 검증했다. 새 collection부터의 end-to-end byte 비교는 미실행이며 수치 동일성을 보장하지 않는다. 새 manifest의 host telemetry와 `data_sha256`도 전체 model JSON hash를 바꿀 수 있다. 아래는 별도 재현 환경의 준비 예시이며 이번 작업에서 공식 `.venv`의 패키지를 추가하지 않았다.

```bash
env -u PYTHONPATH python3.12 -m venv --without-pip .venv-train
env -u PYTHONPATH pip3 --python .venv-train/bin/python install numpy==2.5.4
: "${{V2_EVIDENCE:?set an external private evidence directory}}"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_collect.py --out "$V2_EVIDENCE/learning-data"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-train/bin/python tools/v2_train.py --data-dir "$V2_EVIDENCE/learning-data" --out "$V2_EVIDENCE/model-reproduced.json" --seed 369401
```

최종 보고서 생성은 최종 JSON과 여섯 private NPZ trace가 모두 있을 때만 수행한다.

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/v2_report.py --trace-dir "${{V2_EVIDENCE:?set private evidence root}}/traces"
```

## 연구 범위와 산출물

- [방법과 수식](docs/V2_METHODS.md), [결과와 누락 없는 집계](docs/V2_RESULTS.md), [음성 결과·시간 진단 자체 검토](docs/V2_REVIEW.md)
- [요구사항 감사](docs/REQUIREMENTS_AUDIT.md), [문헌과 채택 범위](docs/LITERATURE.md)
- [발표 구성](report/PRESENTATION_V2.md), [16쪽 PDF](results/presentation.pdf)
- [최종 기계판독 판정](results/v2_protocol/FINAL_ASSESSMENT.json), [자체 작성 그림](results/figures/)
- 실제 하드웨어, hard real-time, 확률적 일반화, safety stop을 주장하지 않는다. 내부 trace는 학습 입력이 아니다.
"""
    methods = f"""# V2 방법

## 등록된 최소 변경

### 원래 learner 가설

Learner는 gate 수정과 별개의 가설이다. 충돌 없는 정상 동작에서만 77개 basis `Phi(q_ref, qd_ref, qdd_ref, jerk, m, c)`를 만들고, v1 관측기의 허용된 수평 wrench `w_v1,xy`를 target으로 표준화·clip[-8,8] 후 ridge `beta=(Phi^T Phi+Lambda)^-1 Phi^T w_v1,xy`를 맞춘다. `Lambda=diag(0,0.02,...,0.02)`로 상수 intercept는 벌점을 주지 않는다. 입력은 현재/과거 공개 reference, 질량과 COM뿐이며 plant/log/oracle, case index와 미래값은 없다. train 조건으로 mean/scale/weight를 구하고 분리된 no-collision validation으로 threshold를 한 번 동결했다. runtime 점수는 `e_k=w_v1,xy-Phi_k beta`, `s_k=(1-0.1)s_(k-1)+0.1||e_k||_2`이며 4-cycle persistence와 8-cycle release를 쓴다. 동기는 정상 운반의 질량/COM/운동 의존 baseline을 제거해 약한 수평 외력을 드러내는 것이다.

### L3 gate 구현 복구

L3는 위 learner를 다시 학습한 것이 아니다. packet의 `q_prev`와 공개 nominal 점 Jacobian으로 상자 바닥 중심 높이 `z_b(q_prev)`를 계산한다. 게이트는 모든 질량에서

`active = grip AND min_i |z_b(q_prev)-h_table,i| > h_band`

이며, 기존 v1 경보와 논리 OR 한다. 원래 모델의 ridge 가중치, 표준화, `EMA alpha=0.1`, 4-cycle persistence, 8-cycle release, 높이 band와 임계값은 그대로다. 이 수정은 새 수치 검출 규칙이 아니라 ORIGINAL 기하 높이 계약의 구현 복구다.

공식 데이터는 L1 오류 디버깅에 이미 노출되었으므로 효능의 독립 검정으로 사용하지 않는다. L3 뒤 재학습·재보정이나 holdout 이후 tuning은 없다.

### Frame, power와 유한 기하 감사

점 `p`에서 `q`로 wrench 기준점을 옮기면 `f_q=f_p`, `tau_q=tau_p+(p-q)×f_p`이고, twist는 `v_p=v_q+omega×(p-q)`이므로 `f_p·v_p+tau_p·omega=f_q·v_q+tau_q·omega`다. 또한 `tau_joint·qdot = wrench·(J qdot)`인 virtual-power 항등을 coadjoint shift와 Jacobian 수치 검사로 확인한다. 이는 frame/sign 오류를 찾는 감사이지 관측성을 만들지 않는다. nonlinear bottom-centre 위치와 1차 Jacobian shift의 유한 오차도 등록 범위에서만 검사했으며, 전역 SE(3) 정확성·접촉모델 타당성·실기 안전성을 증명하지 않는다. 기하 원리는 `docs/LITERATURE.md`에 열거한 Park·Choi·Chung (2024) §§2.2, 6.1, 7.2와 Choi (2024) §§11.1, 11.4, 11.5의 frame/power 범위만 참고하며 원자료·그림을 재사용하지 않는다.

## 제어 후보의 자체 변형과 탄성·이산화 한계

C는 full SE(3) impedance나 논문의 ILC/QP를 재현하지 않는다. 공개 nominal endpoint의 6D 점 Jacobian으로 `J a=[0,0,1,0,0,0]^T`를 풀고, 한 개의 scalar 보정에 대해 `q_ref+=a*x`, `qd_ref+=a*v`, `qdd_ref+=a*accel`을 일관되게 더했다. 놓기 전 0.12 s의 5차 blend로 목표 지지를 0.85mg까지 올리고 `v_raw=(Fz_hat-Fz_target)/12000`을 10 ms filter 뒤 제한했다(변위1.5 mm,속도0.01 m/s,가속도0.5 m/s²). 기존 탄성 보상 inner loop,1 ms sampling과고정grip시각을 유지한다. `Fz_hat`은 허용된 JTS 관측기 추정값이며 실제 작업대 힘은 사후 G4 계산에만 썼다.

이 설계는 지연/잡음이 있는 추정 렌치를 쓰는 국소 외부 admittance다. 강체·연속시간 교재 안정성 결과를 탄성 관절·1 ms 구현의 보장으로 전용하지 않는다. 경보 시 추가 하강 억제가 가속도 smoothness보다 우선하고, release 뒤 고정0.10 s C2 복귀는 전체 독립 bound box를 모두 덮지 못한다. 이 예외를 숨겨 안전하게 bounded라고 단정하지 않으며, 실제 검증의 G4 실패와 함께 후보를 탈락시켰다.

## 후보 처분

- C bounded-normal admittance: 각 구조 G4 0/4. 고정 0.10 s 복귀와 1.5 mm/0.01 m/s 경계는 등록된 **전체 state box**를 모두 덮을 수 없어 무조정 탈락했다. 실제 시험 trajectory가 모두 경계를 위반했다는 뜻은 아니다.
- L1: development 검출 이득 뒤 공식 무충돌 경보 회귀로 탈락.
- L2: train bounding-box 탐색 결과로 보존하되 critic이 선호한 최종 규칙이 아님.
- L3: 정확한 기하 높이 gate bugfix. weight/threshold 변경 없음. 최종 선택은 `{final_variant}`이다.

## 평가와 한계

공식 표는 변경 없는 `evaluate.py` JSON만 사용한다. D4, 봉인 holdout, timing, perturbation은 명시적으로 별도 supplemental 데이터다. 모든 무충돌 false alarm rise, 충돌 중 추가 false alarm, miss, prior-alarm/indistinguishable masked 사례를 버리지 않고 센다. G5는 모든 source의 **최종 v2 A/B** 관측 wall 최대가 각각 1 ms 미만일 때만 통과한다. baseline/v1 및 전체 버전 최악은 비교값으로 별도 보존한다. 초과 시 outlier와 같은 cycle의 wall--thread CPU 차이를 보존하고 context switch 가능성은 추론으로만 한정한다.

같은 순간의 가능한 table-normal wrench는 현재 입력만으로 외력과 지지를 식별할 수 없다. 실제 하드웨어 안전성, 확률적 일반화, hard real-time, collision stop은 입증하지 않는다.

## 문헌 사용 범위

문헌 표기와 채택 범위는 `docs/LITERATURE.md`의 bibliography/sections만 따른다. Albu-Schaeffer et al. (2007)은 유연관절 제어 원리, van Steen et al. (2024)은 bounded reference 동기, Chi et al. (2022)은 residual 분해의 원리 참고일 뿐 원방법 재현이나 안전성 증명이 아니다.
"""
    result_lines = ["# V2 결과", "", "## 증거 분리", "",
                    "공식 8조합은 호환성 회귀표이며 디버깅에 소비되어 독립 효능 증거가 아니다. 최종 선택 주장은 fresh non-official L3 연구와 sealed holdout에서만 가져온다.", "",
                    "## 목표 판정", "", "| 목표 | 판정 |", "|---|---|"]
    result_lines += [f"| {key} | {value} |" for key, value in goals.items()]
    result_lines += ["", "## 누락 없는 supplemental 집계", "",
                     "| 데이터/그룹 | pure ref FA | offset episodes/FA/contact miss | collision extra FA | detected | miss | masked |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, groups in counts.items():
        for group, item in groups.items():
            result_lines.append(f"| {dataset}/{group} | {item['no_collision_false_alarms']} | "
                                f"{item['offset_episodes']}/{item['offset_false_alarm_rises']}/"
                                f"{item['offset_tool_contact_misses']} | "
                                f"{item['collision_extra_false_alarms']} | {item['detected']} | "
                                f"{item['misses']} | {item['masked']} |")
    result_lines += ["", "## 해석", "",
                     "검출기만 바뀌었으므로 v2 제어 개선은 0이다. 제어 추적 회귀가 없다는 범위와 검출 개선을 분리한다. D4와 holdout의 질량/COM/방향별 failure map은 유한 조건 결과이며 확률이나 실제 현장 성능이 아니다.",
                     "", "Timing은 D4/holdout/timing/perturbation의 모든 outlier를 포함한다. G5는 final v2 A/B만 판정하고 v1 및 전체 최악은 비교로 분리한다. 정확한 worst-wall cycle의 thread CPU/context switch를 함께 보여 주며 wall-only 초과를 context switch로 단정하지 않는다."]
    presentation = """# V2 발표 구성

1. Problem and decision boundary
2. Evidence audit and independence
3. Original learner: method and motivation
4. L3 is a gate bugfix, not relearning
5. Frame, power and finite geometry audit
6. Candidate failures
7. Official compatibility: A
8. Official compatibility: B
9. Fresh development: v1 versus final
10. D4 mass/COM failure map
11. Sealed holdout failure map
12. G4 measured support
13. Timing: exact worst-cycle evidence
14. G1-G7 strict status
15. Utility, scope and cost
16. Field applicability and unresolved ambiguity

PDF도 정확히 이 16장 순서이며 각 그림은 해당 evidence slide에만 배치한다.
"""
    return {ROOT / "README.md": readme, ROOT / "docs/V2_METHODS.md": methods,
            ROOT / "docs/V2_RESULTS.md": "\n".join(result_lines) + "\n",
            ROOT / "report/PRESENTATION_V2.md": presentation}


def _configure_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    candidates = list(Path("/usr/share/fonts").rglob("NotoSansCJK*Regular*.otf"))
    candidates += list(Path("/usr/share/fonts").rglob("NotoSansCJK*Regular*.ttc"))
    if candidates:
        font_manager.fontManager.addfont(str(candidates[0]))
        rcParams["font.family"] = font_manager.FontProperties(fname=str(candidates[0])).get_name()
    rcParams["axes.unicode_minus"] = False


def load_traces(trace_dir: Path) -> dict[str, dict[str, Any]]:
    import numpy as np
    traces = {}
    for name in TRACE_NAMES:
        path = trace_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required private trace is absent: {path}")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != TRACE_KEYS:
                raise ValueError(f"{name} keys must be exactly {sorted(TRACE_KEYS)}")
            traces[name.removesuffix(".npz")] = {key: data[key].copy() for key in data.files}
    return traces


def make_figures(documents: dict[str, dict[str, Any]], assessment: dict[str, Any],
                 traces: dict[str, dict[str, Any]]) -> dict[str, Path]:
    _configure_plots()
    import matplotlib.pyplot as plt
    import numpy as np
    out = ROOT / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    official = documents["official"]["combinations"]
    final = final_official_names(official)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, architecture in zip(axes, ("servo", "torque")):
        names = [n for n in official if n.startswith(architecture + "_")]
        base = next(n for n in names if n.endswith("_base_base"))
        v1_rms = float(baseline_cells(
            (DATA / f"S0_v1_{architecture}.txt").read_text(encoding="utf-8"))[8.0][0])
        values = [official[base]["control"]["8.0"]["rms_mm"], v1_rms,
                  official[final[architecture]]["control"]["8.0"]["rms_mm"]]
        ax.bar(["baseline", "v1", "v2"], values, color=["#7f8c8d", "#ef8354", "#087e8b"])
        ax.set_title(f"{architecture}: three versions (official compatibility)")
        ax.set_ylabel("8 kg RMS [mm]"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); paths["official"] = out / "v2_official.png"
    fig.savefig(paths["official"], dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, architecture in enumerate(("servo", "torque")):
        for version, color in (("base", "#7f8c8d"), ("v1", "#ef8354"), ("v2", "#087e8b")):
            trace = traces[f"{version}_{architecture}"]
            axes[0, col].plot(trace["t"], np.asarray(trace["e_mm"]), color=color, label=version)
            axes[1, col].plot(trace["t"], np.asarray(trace["f_table"]), color=color, label=version)
        axes[0, col].set_title(architecture); axes[0, col].set_ylabel("TCP error [mm]")
        axes[1, col].set_ylabel("table Fz [N]"); axes[1, col].set_xlabel("time [s]")
        for ax in axes[:, col]: ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Private analysis traces: original runner, 8 kg; not training data")
    fig.tight_layout(); paths["control"] = out / "v2_control_traces.png"
    fig.savefig(paths["control"], dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, architecture in enumerate(("servo", "torque")):
        for version, color in (("base", "#7f8c8d"), ("v1", "#ef8354"), ("v2", "#087e8b")):
            trace = traces[f"{version}_{architecture}"]
            tau = np.asarray(trace["tau"])
            residual = np.asarray(trace["r"])
            tau_value = np.linalg.norm(tau, axis=1) if tau.ndim > 1 else np.abs(tau)
            residual_value = np.linalg.norm(residual, axis=1) if residual.ndim > 1 else np.abs(residual)
            axes[0, col].plot(trace["t"], tau_value, color=color, label=version)
            axes[1, col].plot(trace["t"], residual_value, color=color, label=version)
        axes[0, col].set_title(architecture); axes[0, col].set_ylabel("||motor torque|| [N m]")
        axes[1, col].set_ylabel("||joint residual|| [N m]"); axes[1, col].set_xlabel("time [s]")
        for ax in axes[:, col]: ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Torque-step and residual analysis; private original-runner traces, never training inputs")
    fig.tight_layout(); paths["torque_residual"] = out / "v2_torque_residual.png"
    fig.savefig(paths["torque_residual"], dpi=170); plt.close(fig)

    colors = {"detected": "#49a078", "missed": "#e76f51",
              "alarm_before_collision": "#e9c46a", "indistinguishable": "#7b2cbf"}
    for dataset, filename in (("d4", "v2_d4_failure_map.png"),
                              ("holdout", "v2_holdout_failure_map.png")):
        study_rows = rows_of(documents[dataset], dataset)
        final_rows = [row for row in study_rows if row.get("collision") is not None and
                      row.get("variant") in {"v2", "L3"}]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, architecture in zip(axes, ("servo", "torque")):
            selected = [row for row in final_rows if row.get("architecture") == architecture]
            for outcome in OUTCOMES:
                subset = [row for row in selected if row.get("collision_outcome") == outcome]
                if not subset:
                    continue
                com_norm = [float(np.linalg.norm(row.get("com", [0, 0, 0]))) for row in subset]
                ax.scatter([row.get("mass", 0) for row in subset],
                           [row.get("force_n", 0) for row in subset],
                           s=[25 + 900 * value for value in com_norm], color=colors[outcome],
                           label=outcome, alpha=.75, edgecolors="none")
            ax.set_title(architecture); ax.set_xlabel("mass [kg]"); ax.grid(alpha=.2)
        axes[0].set_ylabel("collision force [N]"); axes[-1].legend(fontsize=7)
        fig.suptitle(("D4" if dataset == "d4" else "Sealed holdout") +
                     " final failure map; marker size encodes COM offset norm")
        fig.tight_layout()
        paths[dataset] = out / filename; fig.savefig(paths[dataset], dpi=170); plt.close(fig)

    events = assessment["timing"]["all_versions_global_comparison"]["outlier_events"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if events:
        for dataset in sorted({event["dataset"] for event in events}):
            subset = [event for event in events if event["dataset"] == dataset]
            ax.scatter([event["combined"]["wall_ns"] / 1000 for event in subset],
                       [event["combined"]["cpu_ns"] / 1000 for event in subset],
                       label=dataset, alpha=.7)
        limit = max(event["combined"]["wall_ns"] / 1000 for event in events) * 1.05
        ax.plot([0, limit], [0, limit], color="#607d8b", ls=":", label="wall = thread CPU")
    else:
        ax.text(.5, .5, "No retained >=1 ms event", transform=ax.transAxes, ha="center")
    ax.axvline(1000, color="#c1121f", ls="--", label="strict wall 1 ms")
    ax.set_xlabel("exact outlier combined wall [us]"); ax.set_ylabel("same-cycle thread CPU [us]")
    ax.set_title("All retained outlier events: D4 + holdout + timing + perturbations")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); paths["timing"] = out / "v2_timing.png"
    fig.savefig(paths["timing"], dpi=170); plt.close(fig)

    g = assessment["goals"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    states = [g[n] is True for n in names]
    ax.bar(names, [1] * len(names), color=["#49a078" if state else "#d1495b" for state in states])
    ax.set_ylim(0, 1.15); ax.set_yticks([]); ax.set_title("Strict goals: green=met, red=not met/not claimed")
    for i, state in enumerate(states): ax.text(i, .5, "PASS" if state else "NO", ha="center", color="white", weight="bold")
    fig.tight_layout(); paths["goals"] = out / "v2_goals.png"
    fig.savefig(paths["goals"], dpi=170); plt.close(fig)
    return paths


def official_deck_rows(documents: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    combinations = documents["official"]["combinations"]
    final = final_official_names(combinations)
    answer = {}
    for architecture, label in (("servo", "A"), ("torque", "B")):
        rows = []
        for version, source in (("baseline", "base"), ("v1", "v1")):
            item = evaluator_summary(
                (DATA / f"S0_{source}_{architecture}.txt").read_text(encoding="utf-8"))
            rows.append(f"{version}: TCP {item['rms_max']} mm; step {item['pick_place']} N m; "
                        f"table {item['table_n']} N; FA {item['false_alarms']}; "
                        f"transport 15 N {item['transport_15n']} ms, 10 N {item['transport_10n']}, min {item['minimum']}")
        item = structured_official_summary(combinations[final[architecture]])
        rows.append(f"v2: TCP {item['rms_max']} mm; step {item['pick_place']} N m; "
                    f"table {item['table_n']} N; FA {item['false_alarms']}; "
                    f"transport 15 N {item['transport_15n']} ms, 10 N {item['transport_10n']}, min {item['minimum']}")
        answer[label] = rows
    return answer


def _count_lines(groups: dict[str, dict[str, int]], title: str) -> list[str]:
    lines = [title]
    for group, item in sorted(groups.items()):
        lines.append(f"{group}: detected {item['detected']}, missed {item['misses']}, "
                     f"masked {item['masked']}, collision-extra FA {item['collision_extra_false_alarms']}, "
                     f"pure-ref FA {item['no_collision_false_alarms']}")
    return lines


def build_deck_slides(assessment: dict[str, Any], official_rows: dict[str, list[str]],
                      figures: dict[str, Path]) -> list[dict[str, Any]]:
    """Return exactly 16 explicitly mapped evidence slides."""
    final_variant = assessment["final_variant"]
    l3_result = ("L3 exact ORIGINAL geometry-height repair adopted" if final_variant == "L3"
                 else "Fresh L3 rule failed; frozen v1 retained")
    counts = assessment["supplemental_counts"]
    g4 = assessment["g4"]["per_architecture"]
    timing = assessment["timing"]
    final_timing = timing["final_v2_goal"]["per_architecture"]
    global_timing = timing["all_versions_global_comparison"]
    goals = assessment["goals"]
    g4_lines = ["Final 8 kg last-20 ms support, v1 -> v2:"]
    for architecture in ("servo", "torque"):
        item = g4[architecture]
        g4_lines.append(
            f"{architecture}: mean {item['v1']['table_mean_over_mg']:.3f}->{item['v2']['table_mean_over_mg']:.3f} mg; "
            f"min {item['v2']['table_min_over_mg']:.3f}; peak {item['v2']['place_peak_over_mg']:.3f}; "
            f"increase {item['mean_increase_over_mg']:.3f}; pass={item['pass']}")
    timing_lines = []
    for architecture in ("servo", "torque"):
        item = final_timing[architecture]; cpu = item["cpu_at_worst_wall"]
        timing_lines.append(
            f"v2 {architecture}: wall {item['observed_wall_max_us']:.1f} us; exact-cycle CPU "
            f"{cpu['value_us']:.1f} us; ctx v/i {cpu['voluntary_context_switches']}/"
            f"{cpu['involuntary_context_switches']}; pass={item['strict_below_1ms']}")
    cpu = global_timing["cpu_at_worst_wall"]
    timing_lines.append(f"All-version comparison worst: {global_timing['observed_wall_max_us']:.1f} us "
                        f"in {global_timing['worst_dataset']}; CPU {cpu['value_us']:.1f} us ({cpu['semantics']})")
    goal_lines = [f"{name}: {'PASS' if value is True else 'FAIL' if value is False else value}"
                  for name, value in goals.items()]
    slides = [
        {"title": "1. Problem and decision boundary", "lines": ["Separate weak transport collision evidence from normal payload/support wrench.", "A=servo, B=torque; final controller is unchanged from v1.", "Finite-condition utility, not safety certification."], "image": None},
        {"title": "2. Evidence audit and independence", "lines": ["Official evaluate.py was consumed while debugging L1: compatibility evidence only.", "Fresh development, D4 and sealed holdout support final efficacy claims.", "No post-holdout retuning; all misses, masked cases and false rises remain."], "image": None},
        {"title": "3. Original learner: method and motivation", "lines": ["Normal-operation-only 77-basis ridge predicts v1 horizontal wrench from causal reference, mass and COM inputs.", "Target=w_v1,xy; train mean/scale/weights on train only; threshold once on disjoint no-collision validation.", "e=w_v1,xy-Phi beta; EMA alpha=.1; 4-cycle persistence, 8-cycle release.", "Motivation: remove motion/load baseline to expose weak horizontal force."], "image": None},
        {"title": "4. L3 is a gate bugfix, not relearning", "lines": ["active=grip AND min |z_bottom(q_prev)-h_table| > unchanged band.", "Exact ORIGINAL geometry-height contract; v1 OR retained.", "Weights, threshold, EMA and persistence unchanged. " + l3_result], "image": None},
        {"title": "5. Frame, power and finite geometry audit", "lines": ["f_q=f_p; tau_q=tau_p+(p-q)x f_p; v_p=v_q+omega x(p-q).", "Power invariant W·V and joint virtual power W·Jqdot=tau·qdot were checked.", "Finite frame/Jacobian tests catch sign errors; they do not prove global geometry, observability or safety."], "image": None},
        {"title": "6. Candidate failures", "lines": ["C bounded-normal: G4 0/4 for each architecture.", "Fixed .10 s return plus 1.5 mm/.01 m/s bounds cannot cover the entire state box; not every tested trajectory is claimed to violate bounds.", "L1 official FA regression; L2 exploratory train box; " + l3_result], "image": None},
        {"title": "7. Official compatibility: A", "lines": official_rows["A"] + ["Post-debugging cells: not independent efficacy evidence."], "image": figures.get("official")},
        {"title": "8. Official compatibility: B", "lines": official_rows["B"] + ["Same unchanged evaluator and caveat."], "image": figures.get("official")},
        {"title": "9. Fresh development: v1 versus final", "lines": _count_lines(counts["fresh_development"], "All collision classifications"), "image": None},
        {"title": "10. D4 mass/COM failure map", "lines": _count_lines(counts["d4"], "385 cases per architecture/version"), "image": figures.get("d4")},
        {"title": "11. Sealed holdout failure map", "lines": _count_lines(counts["sealed_holdout"], "38 cases per group; opened after freeze"), "image": figures.get("holdout")},
        {"title": "12. G4 measured support", "lines": g4_lines + ["C remains rejected; final v2 claims no controller improvement."], "image": figures.get("control")},
        {"title": "13. Timing: exact worst-cycle evidence", "lines": timing_lines + ["G5 uses final v2 A/B only; global and v1 maxima are comparisons."], "image": figures.get("timing")},
        {"title": "14. G1-G7 strict status", "lines": goal_lines + ["G1/G2 are post-debugging official compatibility observations."], "image": figures.get("goals")},
        {"title": "15. Utility, scope and cost", "lines": ["Utility: causal residual learning can expose weak horizontal force when the normal-operation baseline generalizes.", "Scope: finite simulation development/D4/sealed holdout; no probability, hardware or hard-RT claim.", "Cost: NumPy embedded weights; detector compute added, controller behavior remains v1.", "Private traces are analysis-only and never training targets."], "image": None},
        {"title": "16. Field applicability and unresolved ambiguity", "lines": ["Same instantaneous feasible vertical wrench can be support or collision: G7 unresolved.", "Detector-only v2 does not improve inherited control saturation/friction/stiffness failures.", "Deployment needs sensing/process constraints, collision response and hardware validation."], "image": figures.get("torque_residual")},
    ]
    if len(slides) != 16:
        raise AssertionError("presentation contract requires exactly 16 slides")
    return slides


def make_pdf(figures: dict[str, Path], assessment: dict[str, Any],
             documents: dict[str, dict[str, Any]]) -> None:
    _configure_plots()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    slides = build_deck_slides(assessment, official_deck_rows(documents), figures)
    path = ROOT / "results/presentation.pdf"; path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for index, slide in enumerate(slides):
            fig = plt.figure(figsize=(13.33, 7.5), facecolor="#f7fafc")
            fig.text(.055, .91, slide["title"], fontsize=23, weight="bold", color="#17324d")
            image = slide["image"]; y = .80
            for bullet in slide["lines"]:
                for line in textwrap.wrap("• " + bullet, 62 if image is not None else 104):
                    fig.text(.060, y, line, fontsize=13.5, color="#263238"); y -= .047
                y -= .012
            if image is not None:
                ax = fig.add_axes([.51, .10, .47, .67]); ax.imshow(plt.imread(image)); ax.axis("off")
            fig.text(.055, .025, f"v2 study | {index+1}/{len(slides)} | official and supplemental evidence separated",
                     fontsize=9, color="#607d8b")
            pdf.savefig(fig); plt.close(fig)


def generate(trace_dir: Path, data_dir: Path = DATA) -> dict[str, Any]:
    documents = load_final_inputs(data_dir)
    traces = load_traces(trace_dir)
    assessment = build_assessment(documents)
    for path, content in render_markdown(documents, assessment).items():
        atomic_text(path, content)
    figures = make_figures(documents, assessment, traces)
    make_pdf(figures, assessment, documents)
    atomic_text(DATA / "FINAL_ASSESSMENT.json",
                json.dumps(assessment, ensure_ascii=False, indent=2) + "\n")
    return assessment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, type=Path,
                        help="private directory containing the six base/v1/v2 servo/torque NPZ traces")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate(args.trace_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
