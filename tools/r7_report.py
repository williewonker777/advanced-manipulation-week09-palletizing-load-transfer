"""Generate deterministic public R7 documents from hash-pinned result files.

This module is deliberately presentation-only.  It does not import the simulation
runner, create holdout conditions, or alter a controller/detector file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import textwrap


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.r7_json import read_json, write_json
D = ROOT / "results/v2_protocol"
CURATION = D / "R7_REPORT_CURATION.json"
OUTPUTS = [
    "README.md",
    "docs/R7_RESULTS.md",
    "report/PRESENTATION_R7.md",
    "results/presentation.pdf",
    "docs/INDEX.md",
]
REQUIRED = [
    "results/v2_protocol/R5_REPORT_CURATION.json",
    "results/v2_protocol/R5_REFERENCE_METRICS.json",
    "results/v2_protocol/R5_FINAL_ADOPTION.json",
    "results/v2_protocol/R5_OFFICIAL_REGRESSION.json",
    "results/v2_protocol/R5_PERTURB_RESULTS.json",
    "results/v2_protocol/S4_OFFICIAL.json",
    "results/v2_protocol/R5_VALIDATION_REVIEW.json",
    "results/v2_protocol/R5_DEVELOPMENT_REVIEW.json",
    "results/v2_protocol/R5_D4_RESULTS.json",
    "results/v2_protocol/PROTOCOL_R5.json",
    "results/v2_protocol/R7_ARCHIVE.json",
    "results/v2_protocol/R7_BUILD.json",
    "results/v2_protocol/R7_PREFREEZE_CONTRACT.json",
    "results/v2_protocol/R7_MASS_EXCLUSION.json",
    "results/v2_protocol/R7_PARITY_PLAN.json",
    "results/v2_protocol/R7_S1_RUNTIME_LOCK.json",
    "results/v2_protocol/R7_PARITY_RESULT.json",
    "results/v2_protocol/R7_BASE_parity_R5_A_servo.json",
    "results/v2_protocol/R7_BASE_parity_R7_servo.json",
    "results/v2_protocol/R7_BASE_parity_R1_torque.json",
    "results/v2_protocol/R7_BASE_parity_R7_torque.json",
    "results/v2_protocol/R7_SAVED_EQUIVALENCE.json",
    "results/v2_protocol/PROTOCOL_R7.json",
    "results/v2_protocol/R7_FREEZE.json",
    "results/v2_protocol/R7_HOLDOUT_PAIRING.json",
    "results/v2_protocol/R7_HOLDOUT_REVIEW.json",
    "results/v2_protocol/R7_OFFICIAL_REGRESSION.json",
    "results/v2_protocol/R7_FINAL_CONTRACT_REVIEW.json",
    "results/v2_protocol/R7_ADOPTION.json",
    "docs/R6_README.md",
    "results/presentation_R6.pdf",
    "docs/R7_METHODS.md",
    "tools/r7_report.py",
    "tools/test_r7_report.py",
    "tools/r4_report_links.py",
]
OPTIONAL = [
    "results/v2_protocol/R7_PREFREEZE_FINAL_CONTRACT.json",
    "artifacts/media/r7_manifest.json",
]
FAILURE_REQUIRED = [
    "results/v2_protocol/R7_FAILURE.json",
    "results/v2_protocol/R7_EXECUTION_STATUS.json",
    "results/v2_protocol/R7_ADOPTION.json",
]
FAILURE_OPTIONAL = [
    "results/v2_protocol/R7_PARITY_RESULT.json",
    "results/v2_protocol/R7_BASE_parity_R5_A_servo.json",
    "results/v2_protocol/R7_BASE_parity_R7_servo.json",
    "results/v2_protocol/R7_BASE_parity_R1_torque.json",
    "results/v2_protocol/R7_BASE_parity_R7_torque.json",
    "results/v2_protocol/PROTOCOL_R7.json",
    "results/v2_protocol/R7_FREEZE.json",
    "results/v2_protocol/R7_HOLDOUT_PAIRING.json",
    "results/v2_protocol/R7_HOLDOUT_REVIEW.json",
    "results/v2_protocol/R7_OFFICIAL_REGRESSION.json",
    "results/v2_protocol/R7_FINAL_CONTRACT_REVIEW.json",
]
HISTORICAL_MEDIA = [
    ("R1 A 서보", "05_v2_servo_transport_collision"),
    ("R1 B 토크", "06_v2_torque_transport_collision"),
    ("기준/v1/R1/R2 A", "07_r2_servo_comparison"),
    ("기준/v1/R1/R2 B", "08_r2_torque_comparison"),
]
FORBIDDEN = [
    re.compile(x, re.I)
    for x in (
        "/ho" + "me/",
        "TA" + "SK-",
        "Cl" + "aude",
        "Co" + "dex",
        "G" + "PT",
        "O" + "MX",
        "No" + "tion",
        "local " + "staging",
        "미" + "게시",
    )
]
MODES = {"servo": "A 서보", "torque": "B 토크"}
FORCES = ("10N", "15N", "40N", "100N")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path):
    return read_json(Path(path))


def write(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def safe(text: str, label: str) -> None:
    for pattern in FORBIDDEN:
        if pattern.search(text):
            raise ValueError(f"{label}: prohibited public wording {pattern.pattern}")


def build_curation(root: Path = ROOT):
    root = Path(root)
    sources = {}
    pending = []
    required = required_inputs(root)
    for name in required + OPTIONAL:
        path = root / name
        if path.is_file():
            sources[name] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
        elif name in required:
            pending.append(name)
    return {
        "schema": "R7_REPORT_CURATION.v1",
        "sources": sources,
        "pending_required": pending,
        "outputs": OUTPUTS,
        "official_strings_preserve_source_precision": True,
        "private_raw_or_acquisition_inputs": False,
        "created_utc": "2000-01-01T00:00:00+00:00",
    }


def validate_curation(root: Path, complete: bool = True):
    root = Path(root)
    document = read(root / "results/v2_protocol/R7_REPORT_CURATION.json")
    if document.get("schema") != "R7_REPORT_CURATION.v1":
        raise ValueError("bad R7 curation schema")
    if complete and document.get("pending_required"):
        raise ValueError("report inputs pending: " + str(document["pending_required"]))
    for name, metadata in document.get("sources", {}).items():
        path = root / name
        if (not path.is_file() or sha(path) != metadata.get("sha256")
                or path.stat().st_size != metadata.get("size_bytes")):
            raise ValueError("curated input changed: " + name)
    required = required_inputs(root)
    if complete and set(required) - set(document.get("sources", {})):
        raise ValueError("curation lacks required inputs")
    existing = {name for name in OPTIONAL if (root / name).is_file()}
    if complete and existing - set(document.get("sources", {})):
        raise ValueError("curation lacks existing optional input")
    return document


def required_inputs(root: Path):
    root = Path(root)
    if (root / "results/v2_protocol/R7_FAILURE.json").is_file():
        common = [name for name in REQUIRED if not name.startswith("results/v2_protocol/R7_")]
        common += [name for name in REQUIRED if name in {
            "results/v2_protocol/R7_ARCHIVE.json",
    "results/v2_protocol/R7_BUILD.json",
            "results/v2_protocol/R7_PREFREEZE_CONTRACT.json",
            "results/v2_protocol/R7_MASS_EXCLUSION.json",
            "results/v2_protocol/R7_PARITY_PLAN.json",
            "results/v2_protocol/R7_S1_RUNTIME_LOCK.json",
            "results/v2_protocol/R7_SAVED_EQUIVALENCE.json",
        }]
        common += [name for name in FAILURE_OPTIONAL if (root / name).is_file()]
        return list(dict.fromkeys(common + FAILURE_REQUIRED))
    return REQUIRED


def md_table(columns, rows):
    def cell(value):
        return str(value).replace("\n", "<br>").replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |"
              for row in rows]
    return "\n".join(lines)


def _optional(root, name):
    path = Path(root) / "results/v2_protocol" / name
    return read(path) if path.is_file() else None


def _actual_adoption(document):
    effective = document.get("effective_variants", document.get("selected_variants", {}))
    if set(effective) != {"servo", "torque"}:
        raise ValueError("R7 adoption lacks both architectures")
    if effective["torque"] != "R1":
        raise ValueError("R7 cannot promote the disqualified torque channel")
    if effective["servo"] not in {"R1", "SCALE_STATE"}:
        raise ValueError("unknown R7 servo adoption")
    expected = "R7" if effective["servo"] == "SCALE_STATE" else "R1"
    if document.get("final_assembly") != expected:
        raise ValueError("R7 final assembly differs from effective variants")
    return effective


def _holdout_rows(document):
    if document is None:
        return [{"방식": label, "파일": "미실행", "충돌 검출": "0조건",
                 "오감지": "0회 실행·미측정", "R1 계산 시간": "미실행",
                 "시험 계산 시간": "미실행", "판정": "미실행"}
                for label in MODES.values()]
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError("executed holdout lacks entries")
    rows = []
    for mode in MODES:
        matches = [row for row in entries if row.get("mode") == mode]
        if len(matches) != 1:
            raise ValueError("holdout must contain exactly one row per architecture")
        row = matches[0]
        metrics = row.get("metrics", {})
        required = ("R1_detected_collisions", "tested_detected_collisions",
                    "collision_count", "additional_detected", "lost_R1", "tool_losses",
                    "FA_increased_cases", "reference_FA", "combined_p99_worst_us",
                    "combined_max_worst_us", "cycles_over_1ms")
        if any(key not in metrics for key in required):
            raise ValueError("executed holdout row lacks measured values")
        details = row.get("details", [])
        if not isinstance(details, list) or len(details) != 38:
            raise ValueError("executed holdout must retain all 38 condition details")
        collisions = [item for item in details
                      if item.get("kind", item.get("case", {}).get("kind")) == "collision"]
        if len(collisions) != 30:
            raise ValueError("executed holdout must retain all 30 collision details")
        def fa_counts(event_key):
            groups = {"all": 0, "reference": 0, "offset": 0}
            for detail in details:
                kind = detail.get("kind", detail.get("case", {}).get("kind"))
                event = detail.get(event_key)
                if not isinstance(event, dict) or "false_alarms" not in event:
                    raise ValueError("holdout detail lacks measured event FA: " + event_key)
                value = int(event["false_alarms"])
                groups["all"] += value
                if kind in groups:
                    groups[kind] += value
            return groups
        r1_fa = fa_counts("R1")
        tested_fa = fa_counts("tested")
        if metrics["collision_count"] != 30:
            raise ValueError("holdout collision count differs")
        timing = row.get("timing", {})
        timing_text = {}
        for label in ("R1", "tested"):
            branch = timing.get(label, {})
            combined = branch.get("components", {}).get("combined", {})
            required_timing = ("p99_worst_us", "max_worst_us", "samples")
            if any(key not in combined for key in required_timing):
                raise ValueError("holdout lacks complete paired timing: " + label)
            outliers = branch.get("outliers", [])
            strict_cycles = sum(item.get("combined", {}).get("wall_ns", -1) > 1_000_000
                                for item in outliers)
            if (branch.get("all_outliers_retained") is not True
                    or branch.get("outlier_count") != len(outliers)
                    or branch.get("cycles_over_1ms") != strict_cycles
                    or any(item.get("combined", {}).get("wall_ns", -1) < 1_000_000
                           for item in outliers)):
                raise ValueError("holdout timing outliers are incomplete: " + label)
            timing_text[label] = (f"p99 {combined['p99_worst_us']:.3f} / 최대 "
                                  f"{combined['max_worst_us']:.3f} µs / 1 ms 초과 "
                                  f"{branch['cycles_over_1ms']}주기")
        if (metrics["combined_p99_worst_us"]
                != timing["tested"]["components"]["combined"]["p99_worst_us"]
                or metrics["combined_max_worst_us"]
                != timing["tested"]["components"]["combined"]["max_worst_us"]
                or metrics["cycles_over_1ms"] != timing["tested"]["cycles_over_1ms"]):
            raise ValueError("holdout tested timing summary differs")
        rows.append({
            "방식": MODES[mode],
            "파일": ("R7 파일(토크 R1 분기)" if mode == "torque"
                    and row.get("tested_variant") == "R1"
                    else str(row.get("tested_variant", row.get("variant", "R7 파일")))),
            "충돌 검출": (f"{metrics['R1_detected_collisions']}→"
                           f"{metrics['tested_detected_collisions']}/30, "
                           f"추가 +{metrics['additional_detected']}, "
                           f"손실 {metrics['lost_R1']}, 도구 손실 {metrics['tool_losses']}"),
            "오감지": (f"전체 {r1_fa['all']}→{tested_fa['all']}; "
                        f"reference {r1_fa['reference']}→{tested_fa['reference']}; "
                        f"offset {r1_fa['offset']}→{tested_fa['offset']}; "
                        f"증가 조건 {metrics['FA_increased_cases']}"),
            "R1 계산 시간": timing_text["R1"],
            "시험 계산 시간": timing_text["tested"],
            "판정": "통과" if str(row.get("status")).upper() == "PASS" else "탈락",
        })
    return rows


def _official_payload(document, mode):
    if document is None:
        return None
    records = document.get("records", {})
    row = records.get(mode)
    if not isinstance(row, dict):
        raise ValueError("executed official result lacks per-mode raw record")
    payload = row.get("control_and_collisions", row.get("official_output"))
    if not isinstance(payload, dict):
        raise ValueError("official record lacks exact evaluator payload")
    return payload


def _control_rows(document):
    if document is None:
        return [{"방식": label, "질량": "미실행", "TCP RMS/최대": "미실행",
                 "포화": "미실행", "집기/놓기 계단": "미실행", "오감지": "미실행",
                 "상태": "미실행"} for label in MODES.values()]
    rows = []
    for mode in MODES:
        control = _official_payload(document, mode).get("control", {})
        if set(control) != {"0.0", "4.0", "8.0"}:
            raise ValueError("official control payload must contain 0/4/8 kg")
        for mass in ("0.0", "4.0", "8.0"):
            cells = control[mass]
            if not isinstance(cells, list) or len(cells) != 10:
                raise ValueError("official control cells must preserve evaluator columns")
            rows.append({"방식": MODES[mode], "질량": mass + " kg",
                         "TCP RMS/최대": cells[0] + " / " + cells[1] + " mm",
                         "포화": cells[2] + " %",
                         "집기/놓기 계단": cells[3] + " / " + cells[4] + " N·m",
                         "오감지": " / ".join(cells[6:9]), "상태": cells[9]})
    return rows


def _collision_rows(document):
    if document is None:
        return [{"방식": label, "질량/시각": "미실행", "10 N": "미실행",
                 "15 N": "미실행", "40 N": "미실행", "100 N": "미실행",
                 "충돌 세트 FA": "미실행", "최소 검출력": "미실행"}
                for label in MODES.values()]
    rows = []
    for mode in MODES:
        collisions = _official_payload(document, mode).get("collisions", [])
        if len(collisions) != 6:
            raise ValueError("official collision payload must contain six rows per architecture")
        seen = set()
        for row in collisions:
            cells = row.get("cells")
            if not isinstance(cells, list) or len(cells) != 6:
                raise ValueError("collision row lacks 10/15/40/100 N, FA, or minimum")
            key = (float(row["mass"]), float(row["time"]))
            seen.add(key)
            rows.append({"방식": MODES[mode],
                         "질량/시각": f"{row['mass']:g} kg / {row['time']:g} s",
                         "10 N": cells[0], "15 N": cells[1], "40 N": cells[2],
                         "100 N": cells[3], "충돌 세트 FA": cells[4],
                         "최소 검출력": cells[5]})
        expected = {(mass, time) for mass in (0.0, 8.0) for time in (1.42, 3.0, 3.86)}
        if seen != expected:
            raise ValueError("official collision mass/time grid differs")
    torque = [row for row in rows
              if row["방식"] == "B 토크" and row["질량/시각"] == "8 kg / 3 s"]
    if len(torque) != 1 or torque[0]["충돌 세트 FA"] != "1회":
        raise ValueError("B 8 kg 3.0 s collision-set false alarm must be explicit")
    return rows


def _preserved_r1_official(root):
    """Combine exact historical R1 control tokens with the full R1 collision grid."""
    root = Path(root)
    historical = read(root / "results/v2_protocol/R5_OFFICIAL_REGRESSION.json")
    source = read(root / "results/v2_protocol/S4_OFFICIAL.json")
    records = {}
    for mode in MODES:
        control = historical["records"][mode]["control_and_collisions"]["control"]
        rows = source["combinations"][mode + "_v2_v2"]["collisions"]
        collisions = [{"mass": row["mass"], "time": row["time"],
                       "cells": [row["force10"], row["force15"], row["force40"],
                                 row["force100"], row["false_max"], row["minimum"]]}
                      for row in rows]
        records[mode] = {"control_and_collisions": {"control": control,
                                                     "collisions": collisions}}
    return {"schema": "R7_PRESERVED_R1_OFFICIAL_VIEW.v1", "records": records,
            "scope": "historical R1 values; shown only when R7 official is unavailable"}


def _parity_timing_rows(document):
    rows = []
    timing = document.get("timing")
    if not isinstance(timing, list) or len(timing) != 4:
        raise ValueError("S1 raw review must retain four timing groups")
    for row in timing:
        required = ("mode", "variant", "episodes", "combined_p99_worst_us",
                    "combined_max_worst_us", "cycles_over_1ms",
                    "detector_p99_worst_us", "detector_max_worst_us", "outliers")
        if any(key not in row for key in required) or row["episodes"] != 2:
            raise ValueError("S1 timing group is incomplete")
        outliers = row["outliers"]
        if (not isinstance(outliers, list) or row.get("outlier_count") != len(outliers)
                or row.get("all_outliers_retained") is not True):
            raise ValueError("S1 timing outliers are missing")
        rows.append({"방식": MODES[row["mode"]], "비교 파일": str(row["variant"]),
                     "에피소드": row["episodes"],
                     "감지 p99/최대 [µs]": (f"{row['detector_p99_worst_us']:.3f} / "
                                            f"{row['detector_max_worst_us']:.3f}"),
                     "합산 p99/최대 [µs]": (f"{row['combined_p99_worst_us']:.3f} / "
                                            f"{row['combined_max_worst_us']:.3f}"),
                     "1 ms 초과": row["cycles_over_1ms"],
                     "보존 outlier": len(outliers)})
    return rows


def _equivalence_text(saved, parity, parity_plan):
    streams = saved.get("streams")
    aggregate = saved.get("aggregate")
    if (saved.get("stream_count") != 990 or saved.get("total_samples") != 5_940_000
            or saved.get("all_990_streams_bit_exact") is not True
            or not isinstance(streams, list) or len(streams) != 990
            or not isinstance(aggregate, list) or len(aggregate) != 8):
        raise ValueError("saved equivalence must cover all 990 streams")
    if (sum(row.get("streams", -1) for row in aggregate) != 990
            or sum(row.get("samples", -1) for row in aggregate) != 5_940_000
            or any(row.get("passed_streams") != row.get("streams")
                   or row.get("all_bit_comparisons_passed") is not True
                   for row in aggregate)):
        raise ValueError("saved equivalence aggregate did not pass")
    ids = [row.get("stream_id") for row in streams]
    if (len(set(ids)) != 990
            or any(row.get("passed") is not True
                   or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_raw_sha256", "")))
                   or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("proof_sha256", "")))
                   or (row.get("offline_decisions_exact") is not True
                       if row.get("mode") == "servo"
                       else row.get("offline_decisions_exact") is not None)
                   for row in streams)):
        raise ValueError("saved equivalence lacks all per-stream witnesses")
    if parity_plan.get("episodes") != 8:
        raise ValueError("fresh parity plan does not register eight episodes")
    rows = parity.get("rows")
    if (parity.get("passed") is not True or parity.get("pair_count") != 4
            or parity.get("episodes_run") != 8 or not isinstance(rows, list) or len(rows) != 4):
        raise ValueError("fresh full-loop parity must pass eight episodes")
    return ("저장 자료 990개(방식별 495개)의 실제 감지기 출력과 새 조건 8회의 물리 8배열을 "
            "dtype·shape·바이트 단위로 비교해 모두 같았다.")


def _historical_context(root):
    # Imported lazily so synthetic unit tests can exercise the R7 adapters alone.
    from tools import r5_report
    return r5_report.context(root)


def _t_rows(historical, adopted):
    from tools import r5_report
    old = r5_report.t_rows(historical)
    rows = []
    d4 = historical["docs"]["R5_D4_RESULTS"]
    state = next(row for row in d4["entries"]
                 if row["variant"] == "SCALE_STATE" and row["mode"] == "servo")
    collision_rows = state["kind_counts"]["collision"]
    r7_servo = (f"A {state['candidate_detected']}/"
                f"{collision_rows - state['candidate_detected']}/0")
    for index, row in enumerate(old):
        r7 = row["R1"]
        if index == 2:
            r7 += " (과거 전체 34조건 R1 보존·새 실행 아님)"
        if index == 5 and adopted:
            torque = row["R1"].split(";", 1)[1].strip()
            r7 = r7_servo + "; " + torque
        rows.append({"항목": row["항목"], "기준": row["기준"], "v1": row["v1"],
                     "R1": row["R1"], "R7": r7})
    return rows


def _perturb_fa_counts(document, adopted):
    counts = {}
    for mode in MODES:
        matches = [row for row in document["entries"]
                   if row.get("mode") == mode and row.get("variant") == "SCALE_STATE"]
        if len(matches) != 1:
            raise ValueError("R5 perturbation source lacks SCALE_STATE row")
        row = matches[0]
        references = [item for item in row["details"] if item["case"]["kind"] == "reference"]
        if len(references) != 28:
            raise ValueError("R5 perturbation reference inventory differs")
        r1_reference = sum(item["R1"]["false_alarms"] for item in references)
        candidate_reference = sum(item["candidate"]["false_alarms"] for item in references)
        use_candidate = mode == "servo" and adopted
        counts[mode] = {"r1_reference": r1_reference,
                        "final_reference": candidate_reference if use_candidate else r1_reference,
                        "r1_all": row["R1_FA"],
                        "final_all": row["candidate_FA"] if use_candidate else row["R1_FA"],
                        "state_reference": candidate_reference,
                        "state_all": row["candidate_FA"]}
    return counts


def _inherited_rows(c):
    docs = c["historical"]["docs"]
    rows = []
    for label, name in (("validation", "R5_VALIDATION_REVIEW"),
                        ("development", "R5_DEVELOPMENT_REVIEW")):
        values = {}
        for mode in MODES:
            state = next(row for row in docs[name]["entries"]
                         if row["variant"] == "SCALE_STATE" and row["mode"] == mode)
            metrics = state["metrics"]
            tested = (metrics["candidate_detected_collisions"]
                      if mode == "servo"
                      else metrics["R1_detected_collisions"])
            values[mode] = (f"{metrics['R1_detected_collisions']}→{tested}/"
                            f"{metrics['collision_rows']}")
        rows.append({"단계": label, "A 서보": values["servo"],
                     "B 토크": values["torque"], "성격": "저장 자료 비트 동등성으로 승계"})
    values = {}
    for mode in MODES:
        state = next(row for row in docs["R5_D4_RESULTS"]["entries"]
                     if row["variant"] == "SCALE_STATE" and row["mode"] == mode)
        tested = state["candidate_detected"] if mode == "servo" else state["R1_detected"]
        event_key = "candidate" if mode == "servo" else "R1"
        reference_fa = sum(item[event_key]["false_alarms"] for item in state["details"]
                           if item["case"]["kind"] == "reference")
        values[mode] = f"{state['R1_detected']}→{tested}/180; reference FA {reference_fa}"
    rows.append({"단계": "D4", "A 서보": values["servo"], "B 토크": values["torque"],
                 "성격": "노출 분석 replay·새 독립 holdout 아님"})
    values = {}
    counts = _perturb_fa_counts(docs["R5_PERTURB_RESULTS"], True)
    for mode in MODES:
        row = counts[mode]
        values[mode] = (f"28 reference FA {row['r1_reference']}→{row['final_reference']}; "
                        f"전체 34 FA {row['r1_all']}→{row['final_all']}")
    rows.append({"단계": "섭동", "A 서보": values["servo"], "B 토크": values["torque"],
                 "성격": "노출 분석 replay·새 독립 holdout 아님"})
    return rows


def _media_block(root, adopted):
    columns = [item[0] for item in HISTORICAL_MEDIA]
    embeds = [f"[![{label}](artifacts/media/{stem}.gif)](artifacts/media/{stem}.mp4)"
              for label, stem in HISTORICAL_MEDIA]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|",
             "| " + " | ".join(embeds) + " |"]
    manifest = Path(root) / "artifacts/media/r7_manifest.json"
    if adopted and not manifest.is_file():
        raise ValueError("adopted R7 report requires the two-pair media manifest")
    if adopted:
        data = read(manifest)
        cases = data.get("cases", [])
        if data.get("schema") != "r7.media-manifest.v1" or len(cases) != 2:
            raise ValueError("adopted R7 media manifest must contain two cases")
        lines += ["", "| R7 양성 비교 | R7 음성 비교 |", "|---|---|"]
        rendered = []
        for index, case in enumerate(cases):
            files = {Path(item.get("file", "")).suffix: item.get("file")
                     for item in case.get("artifacts", [])}
            gif, mp4 = files.get(".gif"), files.get(".mp4")
            if not gif or not mp4 or not (Path(root) / gif).is_file() or not (Path(root) / mp4).is_file():
                raise ValueError("R7 media manifest points to a missing artifact")
            label = "R7 양성 비교" if index == 0 else "R7 음성 비교"
            rendered.append(f"[![{label}]({gif})]({mp4})")
        lines.append("| " + " | ".join(rendered) + " |")
        lines += ["", "09/10은 R5 저장 validation 장면을 비트 동등성 근거로 다시 쓴 설명용 비교다. 새 holdout 효능 영상이 아니다."]
    return "\n".join(lines)


def _negative_context(root, docs):
    failure = docs["R7_FAILURE"]
    adoption = docs["R7_ADOPTION"]
    effective = _actual_adoption(adoption)
    if failure.get("schema") != "R7_FAILURE.v1" or failure.get("status") != "FAIL":
        raise ValueError("R7 failure record differs")
    if effective != {"servo": "R1", "torque": "R1"}:
        raise ValueError("R7 failure must select R1")
    historical = _historical_context(root)
    official = docs.get("R7_OFFICIAL_REGRESSION")
    presented = official if official is not None else _preserved_r1_official(root)
    holdout = docs.get("R7_HOLDOUT_REVIEW")
    parity = docs.get("R7_PARITY_RESULT")
    result = {"root": root, "docs": docs, "adoption": adoption, "effective": effective,
              "adopted": False, "holdout": holdout, "official": official,
              "official_fresh": official is not None, "historical": historical,
              "equivalence": failure.get("summary", "실패 기록에 따라 R1 fallback을 선택했다."),
              "holdout_rows": _holdout_rows(holdout),
              "control_rows": _control_rows(presented),
              "collision_rows": _collision_rows(presented),
              "t_rows": _t_rows(historical, False),
              "contract": docs["R7_PREFREEZE_CONTRACT"],
              "prefinal_contract": docs.get("R7_PREFREEZE_FINAL_CONTRACT"),
              "final_contract": docs.get("R7_FINAL_CONTRACT_REVIEW"),
              "negative": True,
              "parity_timing_rows": (_parity_timing_from_base(docs)
                  if all(f"R7_BASE_parity_{variant}_{mode}" in docs
                         for mode, variant in (("servo", "R5_A"), ("servo", "R7"),
                                               ("torque", "R1"), ("torque", "R7"))) else []),
              "failure_stage": failure.get("failed_stage", "기록된 단계")}
    result["inherited_rows"] = _inherited_rows(result)
    return result


def _validate_prefinal_contract(document):
    if document is None:
        return None
    suite = document.get("full_suite", {})
    submission = document.get("default_contract", document.get("default_submission_contract", {}))
    cases = submission.get("cases", []) if isinstance(submission, dict) else []
    if (str(document.get("status")).upper() != "PASS"
            or document.get("errors") != 0 or document.get("failures") != 0
            or suite.get("status") != "PASS" or suite.get("errors") != 0
            or suite.get("failures") != 0
            or not isinstance(suite.get("tests_run"), int) or suite["tests_run"] < 1
            or not isinstance(submission, dict) or submission.get("status") != "PASS"
            or submission.get("default_names_exact") is not True
            or not isinstance(cases, list) or len(cases) != 8
            or any(row.get("status") != "PASS" for row in cases)):
        raise ValueError("final pre-freeze contract did not pass")
    return document


def context(root: Path = ROOT):
    root = Path(root)
    validate_curation(root)
    names = list(dict.fromkeys(required_inputs(root) + OPTIONAL))
    docs = {Path(name).stem: read(root / name) for name in names
            if name.endswith(".json") and (root / name).is_file()}
    if "R7_FAILURE" in docs:
        return _negative_context(root, docs)
    adoption = docs["R7_ADOPTION"]
    effective = _actual_adoption(adoption)
    adopted = effective["servo"] != "R1"
    protocol = docs["PROTOCOL_R7"]
    inherited = read(root / "results/v2_protocol/PROTOCOL_R5.json")
    for key in ("detector_gates", "timing", "relative_R1_interpretation",
                "control_gates_inherited", "official"):
        if protocol.get(key) != inherited.get(key):
            raise ValueError("R7 changed inherited gate object: " + key)
    build = docs["R7_BUILD"]
    if (build.get("assembly_logic_changes") != 0
            or build.get("model_weight_or_threshold_updates") != 0
            or build.get("model_count") != 8):
        raise ValueError("R7 build changed frozen scientific content")
    contract = docs["R7_PREFREEZE_CONTRACT"]
    if (contract.get("status") != "PASS" or contract["full_suite"].get("tests_run", 0) < 1
            or len(contract.get("interface", {}).get("cases", [])) != 8):
        raise ValueError("R7 pre-freeze contract proof differs")
    parity = docs["R7_PARITY_RESULT"]
    saved = docs["R7_SAVED_EQUIVALENCE"]
    equivalence = _equivalence_text(saved, parity, docs["R7_PARITY_PLAN"])
    holdout = docs.get("R7_HOLDOUT_REVIEW")
    pairing = docs["R7_HOLDOUT_PAIRING"]
    if pairing.get("episodes_run") != 152 or pairing.get("pair_count") != 76:
        raise ValueError("R7 holdout pairing does not cover 152 episodes")
    official = docs.get("R7_OFFICIAL_REGRESSION")
    if official is not None:
        episodes = official.get("episodes_started", {})
        if (episodes.get("total") != 58 or episodes.get("required_total") != 58
                or episodes.get("per_mode") != {"servo": 29, "torque": 29}):
            raise ValueError("R7 official batch does not cover 58 episodes")
        if "R3_delay_ms" in json.dumps(official) or "R3_false_alarms" in json.dumps(official):
            raise ValueError("R7 official serialization reused historical misnamed fields")
        if not all(token in json.dumps(official)
                   for token in ("tested_delay_ms", "tested_false_alarms")):
            raise ValueError("R7 official result lacks correctly named tested fields")
        architectures = official.get("architectures", {})
        torque_identity = architectures.get("torque", {}).get("exact_R1_identity", {})
        if (torque_identity.get("status") != "PASS"
                or torque_identity.get("checked_cells") != 66
                or torque_identity.get("raw_tokens_preserved") is not True
                or torque_identity.get("errors") != []):
            raise ValueError("official R1 identity proof differs: torque")
        servo_identity = architectures.get("servo", {}).get("exact_R1_identity", {})
        if servo_identity.get("status") == "PASS":
            if (servo_identity.get("checked_cells") != 66
                    or servo_identity.get("raw_tokens_preserved") is not True
                    or servo_identity.get("errors") != []):
                raise ValueError("official R1 identity proof differs: servo")
        elif (servo_identity.get("status") != "NOT_APPLICABLE"
              or servo_identity.get("checked_cells") != 0
              or servo_identity.get("raw_tokens_preserved") is not True
              or servo_identity.get("errors") != []):
            raise ValueError("official candidate identity proof differs: servo")
        servo_det = architectures.get("servo", {}).get("exact_R5_candidate_determinism", {})
        det_status = servo_det.get("status")
        if det_status == "NOT_APPLICABLE":
            if adopted or servo_det.get("checked_cells") != 0 or servo_det.get("errors") != []:
                raise ValueError("R7 fallback determinism proof differs")
        elif (det_status not in {"PASS", "FAIL"}
              or servo_det.get("checked_cells") != 54
              or servo_det.get("timing_compared") is not False
              or not isinstance(servo_det.get("errors"), list)
              or (det_status == "PASS") != (servo_det.get("errors") == [])
              or (adopted and det_status != "PASS")):
            raise ValueError("R7 candidate official determinism proof differs")
        if ((servo_identity.get("status") == "PASS") != (det_status == "NOT_APPLICABLE")):
            raise ValueError("R7 official assembly identity and determinism scopes conflict")
    final_contract = docs["R7_FINAL_CONTRACT_REVIEW"]
    if str(final_contract.get("status")).upper() != "PASS":
        if adopted:
            raise ValueError("failed final contract cannot adopt R7")
    historical = _historical_context(root)
    prefinal = _validate_prefinal_contract(docs.get("R7_PREFREEZE_FINAL_CONTRACT"))
    result = {"root": root, "docs": docs, "adoption": adoption, "effective": effective,
            "adopted": adopted, "holdout": holdout, "official": official, "official_fresh": official is not None,
            "historical": historical, "equivalence": equivalence,
            "holdout_rows": _holdout_rows(holdout),
            "control_rows": _control_rows(official),
            "collision_rows": _collision_rows(official),
            "t_rows": _t_rows(historical, adopted),
            "contract": contract, "prefinal_contract": prefinal,
            "final_contract": final_contract, "negative": False,
            "parity_timing_rows": _parity_timing_from_base(docs)}
    result["inherited_rows"] = _inherited_rows(result)
    return result


def _conclusion(c):
    if c.get("negative"):
        return (f"{c.get('failure_stage', '실행 단계')} 실패로 A 서보와 B 토크 모두 R1을 유지했다. "
                "실패 뒤 후보를 고치거나 다시 실행하지 않았다.")
    if c["adopted"]:
        return ("A 서보에는 기록 결함을 고친 검증 경로로 확인한 SCALE_STATE 감지 채널을 "
                "채택했고, B 토크는 R1을 유지했다.")
    return "R7 후보는 관문을 모두 통과하지 못해 A 서보와 B 토크 모두 R1을 유지했다."


def _official_miss_text(c):
    counts = {}
    for mode in MODES.values():
        selected = [row for row in c["collision_rows"] if row["방식"] == mode]
        pick = next((row for row in selected if row["질량/시각"] == "8 kg / 1.42 s"), None)
        place = next((row for row in selected if row["질량/시각"] == "8 kg / 3.86 s"), None)
        if pick is None or place is None:
            counts[mode] = None
        else:
            counts[mode] = sum(value == "놓침" for value in
                               (pick["10 N"], pick["15 N"], place["10 N"]))
    if any(value is None for value in counts.values()):
        return "공식 8 kg 집기 10·15 N과 놓기 10 N은 실행되지 않아 미검출 잔존 여부를 판단할 수 없다."
    return ("공식 8 kg의 집기 1.42 s 10·15 N과 놓기 3.86 s 10 N에서 "
            f"현재 미검출은 A {counts['A 서보']}건, B {counts['B 토크']}건이다. "
            "R1은 두 방식 각 3건, 총 6건이었다.")


def _unmet(c):
    old = read(c["root"] / "results/v2_protocol/R5_PERTURB_RESULTS.json")
    counts = _perturb_fa_counts(old, c["adopted"])
    if c["holdout"] is None:
        timing = ("새 holdout 계산 시간은 미실행이다. R1에서 관측한 합산 최대는 "
                  "A 11.703 ms, B 14.163 ms이며 최대 1 ms를 보장으로 주장하지 않는다.")
    else:
        maximum = max(row["metrics"]["combined_max_worst_us"] for row in c["holdout"]["entries"])
        timing = (f"새 holdout 합산 최대 {maximum:.3f} µs다. R1 관측 최대 A 11.703 ms, "
                  "B 14.163 ms와 함께 모든 1 ms 초과 주기를 보존했다.")
    return [
        ("G4 기준은 마지막 20 ms 평균 ≥50 %mg, 최소 ≥25 %mg, v1 대비 +25 %p, "
         "놓기 힘 ≤1.1 mg다. 현재 A는 평균 0.2098, 최소 0.0000, 증가 0, peak 1.0045; "
         "B는 평균 11.1507, 최소 0.3802, 증가 0, peak 1.0140이다."),
        timing,
        _official_miss_text(c),
        (f"무충돌 섭동 28 reference FA는 A {counts['servo']['r1_reference']}→{counts['servo']['final_reference']}, "
         f"B {counts['torque']['r1_reference']}→{counts['torque']['final_reference']}다."),
        "D4 R1 A 서보 포화는 2/189다. 후보 채택 여부와 별개로 이 한계를 공개한다.",
        "유한 시뮬레이션 결과이며 하드웨어 안전 보장이나 충돌 정지 인증이 아니다.",
    ]


def _prefreeze_contract_text(c):
    initial = c["contract"]["full_suite"]["tests_run"]
    final = c["prefinal_contract"]
    if final is None:
        return f"사전 전체 {initial}개와 기본 이름 8개는 통과했다. 최종 계약 검사는 아직 기록되지 않았다."
    count = final.get("full_suite", {}).get("tests_run", final.get("tests_run"))
    if not isinstance(count, int):
        raise ValueError("final pre-freeze contract lacks test count")
    return f"사전 전체 {initial}개와 최종 전체 {count}개, 기본 이름 8개 검사가 통과했다."


def _parity_timing_from_base(docs):
    rows = []
    groups = (("servo", "R5_A"), ("servo", "R7"),
              ("torque", "R1"), ("torque", "R7"))
    for mode, variant in groups:
        document = docs.get(f"R7_BASE_parity_{variant}_{mode}")
        if not isinstance(document, dict) or len(document.get("rows", [])) != 2:
            raise ValueError("R7 parity timing source must retain two episodes per group")
        summaries = [row.get("timing_summary", {}) for row in document["rows"]]
        def values(component, field):
            return [summary.get(component, {}).get("wall", {}).get(field)
                    for summary in summaries]
        detector_p99 = values("detector", "p99_us")
        detector_max = values("detector", "max_us")
        combined_p99 = values("combined", "p99_us")
        combined_max = values("combined", "max_us")
        if any(not isinstance(value, (int, float)) for values_ in
               (detector_p99, detector_max, combined_p99, combined_max)
               for value in values_):
            raise ValueError("R7 parity timing summary is incomplete")
        outliers = [item for summary in summaries for item in summary.get("outliers", [])]
        over = sum(item.get("combined", {}).get("wall_ns", -1) > 1_000_000
                   for item in outliers)
        if any(item.get("combined", {}).get("wall_ns", -1) < 1_000_000
               for item in outliers):
            raise ValueError("R7 parity outlier inventory differs")
        rows.append({"방식": MODES[mode], "비교 파일": variant, "에피소드": 2,
                     "감지 p99/최대 [µs]": f"{max(detector_p99):.3f} / {max(detector_max):.3f}",
                     "합산 p99/최대 [µs]": f"{max(combined_p99):.3f} / {max(combined_max):.3f}",
                     "1 ms 초과": over, "보존 outlier": len(outliers)})
    return rows


def _source_label(c):
    return "R7 새 공식 1 batch" if c.get("official_fresh") else "과거 R1 보존값"


def _timing_table(c):
    return md_table(["방식", "비교 파일", "에피소드", "감지 p99/최대 [µs]",
                     "합산 p99/최대 [µs]", "1 ms 초과", "보존 outlier"],
                    c["parity_timing_rows"])


def _requirements_block(prefix):
    return ("가이드 33개 항목은 24개 충족, 4개 부분 충족, 5개 판정 불가다. "
            "런타임 입력·제출 형식·무결성 8개 검사는 모두 통과했다. "
            "최종 감지 파일이 바뀌어도 허용 입력과 제어기를 바꾸지 않아 이 판정은 유지된다. "
            f"상세는 [요구사항 점검]({prefix}REQUIREMENTS_AUDIT.md)에 있다.")


def _history_block():
    return """- README: [R1](docs/R1_README.md), [R2](docs/R2_README.md), [R3](docs/R3_README.md), [R4](docs/R4_README.md), [R5](docs/R5_README.md), [R6](docs/R6_README.md)
- 결과: [R1](docs/V2_RESULTS.md), [R2](docs/R2_RESULTS.md), [R3](docs/R3_RESULTS.md), [R4](docs/R4_RESULTS.md), [R5](docs/R5_RESULTS.md), [R6](docs/R6_RESULTS.md), [R7](docs/R7_RESULTS.md)
- 발표: [R5 PDF](results/presentation_R5.pdf), [R6 PDF](results/presentation_R6.pdf), [현재 PDF](results/presentation.pdf)
- 전체 문서 목록: [문서 색인](docs/INDEX.md)"""


def render_readme(c):
    official = _source_label(c)
    holdout = md_table(["방식", "파일", "충돌 검출", "오감지", "R1 계산 시간",
                        "시험 계산 시간", "판정"], c["holdout_rows"])
    control = md_table(["방식", "질량", "TCP RMS/최대", "포화", "집기/놓기 계단",
                        "오감지", "상태"], c["control_rows"])
    collisions = md_table(["방식", "질량/시각", "10 N", "15 N", "40 N", "100 N",
                           "충돌 세트 FA", "최소 검출력"], c["collision_rows"])
    unmet = "\n".join("- " + item for item in _unmet(c))
    return f"""# Week 9 Palletizing Load Transfer — R7 기록 수정 재검증

**최종 제출: {_conclusion(c)}**

기준=초기 제공본, v1=첫 개선본, R1=고정 비교 제어·감지기, R7=후보 바이트와 판정 규칙은 그대로 두고 결과 기록 경로만 고친 재검증본이다. A=서보, B=토크, FA=오감지다.

{_media_block(c['root'], c['adopted'])}

## 결론과 근거

{_conclusion(c)} NumPy scalar만 기본 JSON형으로 엄격하게 바꾸고 배열·NaN·무한대·비문자 key는 거부했다. 후보·모델 263개 파라미터·임계값·정규화·R1 제어기·관문 숫자는 바꾸지 않았다. {_prefreeze_contract_text(c)} {c['equivalence']} 상세는 [R7 결과](docs/R7_RESULTS.md), [R7 방법](docs/R7_METHODS.md), [발표 PDF](results/presentation.pdf)에 있다.

## 기준/v1/R1/R7 — T1~T6

{md_table(['항목','기준','v1','R1','R7'], c['t_rows'])}

T1/T2/T4/T5 물리값은 감지 채널 변경과 무관해 R1과 같다. T3은 과거 전체 34조건 보존값이다. T6의 저장 평가값은 990개 신호 스트림 동등성으로 승계하며 새 효능 측정으로 세지 않는다.

## 저장 자료로 승계한 단계

{md_table(['단계','A 서보','B 토크','성격'], c['inherited_rows'])}

## 새 holdout 1 batch

{holdout}

실행된 값만 적었다. 실행 전 실패 경로에서는 `미실행`으로 표시한다.

## 계산 시간

{_timing_table(c)}

위 표는 사전 시험과 별개인 실제 full-loop 대조 8회다. 새 holdout의 R1·시험 p99, 최대, 1 ms 초과 주기 수는 holdout 표에 모두 적었다. p99 800 µs는 선택 관문이고 최대 1 ms는 목표다.

## {official} — 0/4/8 kg

{control}

## {official} — 전체 10/15/40/100 N 충돌표

{collisions}

B 토크 8 kg, 3.0 s의 충돌 세트 FA 1회를 포함했다. `R5_OFFICIAL_REGRESSION.json`의 `R3_delay_ms`와 `R3_false_alarms`는 이름과 달리 R5 후보 값을 담는다. 과거 JSON은 바꾸지 않았고 R7은 `tested_delay_ms`와 `tested_false_alarms`를 쓴다.

## 요구사항 점검

{_requirements_block("docs/")}

## 평가기 점검과 R1 놓침

원본 평가기는 조건·집계·출력에서 판정을 뒤집는 결함이 확인되지 않았다. R1의 공식 8 kg 놓침 6건은 두 방식에서 집기 1.42 s의 10·15 N, 놓기 3.86 s의 10 N이며, 연속성·그립·정상 분기·임계 조건의 결과다. [평가기 점검과 미검출 원인](docs/R4_README.md#평가기-점검과-미검출-원인)을 함께 본다.

## 미달과 범위

{unmet}

## 이전 라운드와 문서

{_history_block()}
"""


def artifact_links(c):
    links = []
    for name, label in (("R7_PARITY_RESULT.json", "full-loop 대조"),
                        ("PROTOCOL_R7.json", "protocol"), ("R7_FREEZE.json", "동결"),
                        ("R7_HOLDOUT_REVIEW.json", "holdout"),
                        ("R7_OFFICIAL_REGRESSION.json", "공식"),
                        ("R7_ADOPTION.json", "판정")):
        if (c["root"] / "results/v2_protocol" / name).is_file():
            links.append(f"[{label}](../results/v2_protocol/{name})")
    return ", ".join(links) + "이다."


def render_results(c):
    official = _source_label(c)
    return f"""# R7 결과

## 판정

{_conclusion(c)} 재설계는 0회다. S2 실제 대조를 시작한 뒤에는 후보·기록 도구·임계값·관문을 고치거나 실행을 반복하지 않았다.

## 기록 결함과 사전 시험

R6 중단 원인은 비교 결과의 `numpy.bool_`를 표준 JSON encoder가 거부한 것이다. R7은 NumPy bool·정수·실수 scalar만 기본형으로 바꾸는 단일 기록기를 사용한다. 배열, NaN, 무한대, 그밖의 객체, 비문자 key는 오류로 멈춘다. 합성 seed를 쓴 별도 사본에서 대조부터 문서 생성까지 전 구간 기록을 사전 시험했다. 후보로 공식 조건을 미리 실행하지 않았다.

## 동등성과 full-loop 대조

{c['equivalence']}

비교 대상은 `q`, `q_d`, `tau`, `e_mm`, `f_table`, `f_tool`, `qpos`, `r`, packet, 경보, 점수, EMA다. 허용오차 없이 dtype·shape·C-order 바이트를 비교했다.

## 실제 full-loop 대조 8회 계산 시간

{_timing_table(c)}

## 저장 자료로 승계한 단계

{md_table(['단계','A 서보','B 토크','성격'], c['inherited_rows'])}

## Holdout

{md_table(['방식','파일','충돌 검출','오감지','R1 계산 시간','시험 계산 시간','판정'], c['holdout_rows'])}

## 공식 회귀 — {official}

{md_table(['방식','질량','TCP RMS/최대','포화','집기/놓기 계단','오감지','상태'], c['control_rows'])}

### 전체 충돌 행

{md_table(['방식','질량/시각','10 N','15 N','40 N','100 N','충돌 세트 FA','최소 검출력'], c['collision_rows'])}

R7 서보 후보를 실제 공식 조건에서 시험한 경우 지연, FA, TCP RMS·최대, 포화, 집기·놓기 계단의 54개 지정 셀을 R5 후보 기록과 정확히 비교했다. 계산 시간은 이 결정론 비교에서 제외했다. R1 fallback이면 후보 결정론 검사는 적용하지 않는다.

## 요구사항·평가기 점검

{_requirements_block("")} 원본 평가기 탓으로 판정이 바뀐 항목은 없다. 공식 8 kg의 R1 놓침 6건 원인은 [이전 감사 요약](../docs/R4_README.md#평가기-점검과-미검출-원인)에 있다.

## 기준/v1/R1/R7

{md_table(['항목','기준','v1','R1','R7'], c['t_rows'])}

## 미달

""" + "\n".join("- " + item for item in _unmet(c)) + f"""

## 공개 재현 범위

공개 파일에는 조립기, 모델, 검증 집계, 문서 생성기와 검사 도구가 들어간다. 원시 시계열과 취득 드라이버는 포함하지 않는다. 근거 파일은 {artifact_links(c)}
"""


def render_notes(c):
    return f"""# R7 발표 요약

## 1. 결론

{_conclusion(c)}

## 2. 수정 범위

후보·모델·임계값·정규화·제어·관문은 그대로다. NumPy scalar 결과를 엄격하게 기록하는 새 R7 경로만 만들었다.

## 3. 사전 시험과 full-loop 대조

합성 seed 별도 사본에서 대조→protocol→동결→holdout→공식 기록→판정→문서를 끝까지 사전 시험했다. 후보로 공식 조건을 선행 실행하지 않았다. {c['equivalence']}

## 4. 계산 시간

{_timing_table(c)}

## 5. Holdout

{md_table(['방식','파일','충돌 검출','오감지','R1 계산 시간','시험 계산 시간','판정'], c['holdout_rows'])}

## 6. 공식 0/4/8 kg — {_source_label(c)}

{md_table(['방식','질량','TCP RMS/최대','포화','집기/놓기 계단','오감지','상태'], c['control_rows'])}

## 7. 공식 10/15/40/100 N

{md_table(['방식','질량/시각','10 N','15 N','40 N','100 N','충돌 세트 FA','최소 검출력'], c['collision_rows'])}

## 8. 요구사항과 남은 한계

{_requirements_block("../docs/")}

""" + "\n".join("- " + item for item in _unmet(c)) + "\n"


def _slides(c):
    return [
        {"title": "R7 최종 판정", "bullets": [_conclusion(c), "후보·모델·임계·관문 불변", "재설계 0회"]},
        {"title": "기록 결함 수정", "bullets": ["NumPy scalar만 기본 JSON형으로 변환", "배열·NaN·무한대·비문자 key 거부", "R6 기록 파일은 보존"]},
        {"title": "전 구간 사전 시험", "bullets": ["별도 사본·합성 seed", "실제 Plant와 설치 NumPy", "후보 공식 조건 선행 실행 없음"]},
        {"title": "저장·full-loop 동등성", "bullets": [c["equivalence"], "허용오차 없이 dtype·shape·바이트 비교"]},
        {"title": "실제 대조 계산 시간", "table": (["방식","비교 파일","에피소드","감지 p99/최대 [µs]","합산 p99/최대 [µs]","1 ms 초과","보존 outlier"], c["parity_timing_rows"])},
        {"title": "새 holdout", "table": (["방식","파일","충돌 검출","오감지","R1 계산 시간","시험 계산 시간","판정"], c["holdout_rows"])},
        {"title": f"{_source_label(c)} — 0/4/8 kg", "table": (["방식","질량","TCP RMS/최대","포화","집기/놓기 계단","오감지","상태"], c["control_rows"])},
        {"title": "공식 충돌 A 서보", "table": (["질량/시각","10 N","15 N","40 N","100 N","충돌 세트 FA","최소 검출력"], [{k:v for k,v in row.items() if k != "방식"} for row in c["collision_rows"] if row["방식"] == "A 서보"])},
        {"title": "공식 충돌 B 토크", "table": (["질량/시각","10 N","15 N","40 N","100 N","충돌 세트 FA","최소 검출력"], [{k:v for k,v in row.items() if k != "방식"} for row in c["collision_rows"] if row["방식"] == "B 토크"])},
        {"title": "T1–T3", "table": (["항목","기준","v1","R1","R7"], c["t_rows"][:3])},
        {"title": "T4–T6", "table": (["항목","기준","v1","R1","R7"], c["t_rows"][3:])},
        {"title": "요구사항", "bullets": ["가이드 33항목: 24 충족·4 부분·5 판정 불가", "입력·형식·무결성 8 통과", "평가기 결함으로 바뀐 판정 없음"]},
        {"title": "미달", "bullets": _unmet(c)},
        {"title": "최종 제출", "bullets": [_conclusion(c), "상세: docs/R7_RESULTS.md", "문서 색인: docs/INDEX.md"]},
    ]


def render_index(root):
    root = Path(root)
    entries = []
    for directory, pattern in ((root / "docs", "*.md"), (root / "report", "*.md"),
                               (root / "results", "*.pdf")):
        if directory.is_dir():
            for path in sorted(directory.glob(pattern)):
                if path == root / "docs/INDEX.md":
                    continue
                relative = path.relative_to(root).as_posix()
                target = path.name if path.parent == root / "docs" else "../" + relative
                entries.append((relative, target))
    lines = ["# 문서 색인", "", "[README로 돌아가기](../README.md)", "",
             "아래 목록은 공개 결과 문서와 발표 자료를 한곳에서 찾기 위한 색인이다.", ""]
    lines += [f"- [{name}]({target})" for name, target in entries]
    return "\n".join(lines) + "\n"


def _configure_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    candidates = ["Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "DejaVu Sans"]
    names = {font.name for font in fm.fontManager.ttflist}
    font = next((name for name in candidates if name in names), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})


def table_row_heights(cells, columns):
    """Reserve vertical space for wrapped text instead of crossing cell edges."""
    base = min(.15, .70 / max(1, len(cells) + 1))
    line_height = .024 if columns >= 7 else .027
    heights = [base] + [max(base, line_height * max(
        (str(cell).count("\n") + 1 for cell in row), default=1) + .025)
        for row in cells]
    total = sum(heights)
    return [height * min(1., .85 / total) for height in heights]


def make_pdf(path: Path, c):
    _configure_plot()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    metadata = {"Title": "R7 assembly fix and revalidation", "Creator": "r7_report.py",
                "CreationDate": datetime(2000, 1, 1, tzinfo=timezone.utc),
                "ModDate": datetime(2000, 1, 1, tzinfo=timezone.utc)}
    pages = _slides(c)
    with PdfPages(path, metadata=metadata) as pdf:
        for index, page in enumerate(pages, 1):
            figure = plt.figure(figsize=(13.333, 7.5), facecolor="#f6f8fb")
            figure.text(.04, .92, page["title"], fontsize=23, weight="bold", color="#183153")
            if "bullets" in page:
                y = .82
                for bullet in page["bullets"]:
                    for line in textwrap.wrap("• " + str(bullet), 72, break_long_words=False):
                        figure.text(.055, y, line, fontsize=14, color="#263238")
                        y -= .05
                    y -= .018
            else:
                columns, rows = page["table"]
                axis = figure.add_axes([.015, .08, .97, .78])
                axis.axis("off")
                width = max(8, 62 // len(columns))
                cells = [["\n".join(textwrap.wrap(str(row.get(column, "")), width,
                                                    break_long_words=False))
                          for column in columns] for row in rows]
                table = axis.table(cellText=cells, colLabels=columns, loc="center", cellLoc="left")
                table.auto_set_font_size(False)
                table.set_fontsize(8.5 if len(columns) >= 7 else 10)
                heights = table_row_heights(cells, len(columns))
                for (row, _), cell in table.get_celld().items():
                    cell.set_height(heights[row])
                    cell.set_edgecolor("#b0bec5")
                    cell.set_facecolor("#dbe7f3" if row == 0 else "#ffffff")
            figure.text(.04, .025, f"R7 검증 | {index}/{len(pages)}", fontsize=9, color="#607d8b")
            pdf.savefig(figure)
            plt.close(figure)


def render_all(c):
    return render_readme(c), render_results(c), render_notes(c)


def generate(root: Path = ROOT):
    root = Path(root)
    c = context(root)
    for name, text in zip(OUTPUTS[:3], render_all(c)):
        safe(text, name)
        write(root / name, text)
    output = root / OUTPUTS[3]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        make_pdf(temporary, c)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    index = render_index(root)
    safe(index, "docs/INDEX.md")
    write(root / "docs/INDEX.md", index)
    return c


def pdf_text(path: Path):
    with tempfile.NamedTemporaryFile(suffix=".txt") as stream:
        subprocess.run(["pdftotext", str(path), stream.name], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return Path(stream.name).read_text(encoding="utf-8")


def check_links(root: Path):
    from tools.r4_report_links import check_public_links
    result = check_public_links(root)
    if not result["passed"]:
        raise ValueError("broken public links: " + str(result["missing"]))
    return result


def check_one(root: Path):
    root = Path(root)
    c = context(root)
    expected = render_all(c)
    for name, text in zip(OUTPUTS[:3], expected):
        path = root / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise ValueError("stale report output: " + name)
        safe(text, name)
    expected_index = render_index(root)
    if (root / "docs/INDEX.md").read_text(encoding="utf-8") != expected_index:
        raise ValueError("stale report output: docs/INDEX.md")
    safe(expected_index, "docs/INDEX.md")
    check_links(root)
    pdf = root / OUTPUTS[3]
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF"):
        raise ValueError("missing R7 PDF")
    extracted = pdf_text(pdf)
    for token in ("R7 최종 판정", "새 holdout", "공식 충돌", "요구사항", "미달"):
        if token not in extracted:
            raise ValueError("PDF text missing " + token)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as stream:
        make_pdf(Path(stream.name), c)
        if Path(stream.name).read_bytes() != pdf.read_bytes():
            raise ValueError("R7 PDF is stale or nondeterministic")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--build-curation", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--projection", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.build_curation:
        path = root / "results/v2_protocol/R7_REPORT_CURATION.json"
        if path.exists():
            from tools.r7_json import replace_json
            replace_json(path, build_curation(root))
        else:
            write_json(path, build_curation(root))
        print(path)
        return
    if args.check:
        check_one(root)
        for projection in args.projection:
            check_one(projection.resolve())
        print("R7 report check passed")
        return
    generate(root)
    print("R7 report generated")


if __name__ == "__main__":
    main()
