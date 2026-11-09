"""Generate deterministic public R6 documents from hash-pinned result files.

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
D = ROOT / "results/v2_protocol"
CURATION = D / "R6_REPORT_CURATION.json"
OUTPUTS = [
    "README.md",
    "docs/R6_RESULTS.md",
    "report/PRESENTATION_R6.md",
    "results/presentation.pdf",
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
    "results/v2_protocol/R6_BUILD.json",
    "results/v2_protocol/R6_PREFREEZE_CONTRACT.json",
    "results/v2_protocol/R6_MASS_EXCLUSION.json",
    "results/v2_protocol/R6_PARITY_PLAN.json",
    "results/v2_protocol/R6_PARITY_RESULT.json",
    "results/v2_protocol/R6_SAVED_EQUIVALENCE.json",
    "results/v2_protocol/PROTOCOL_R6.json",
    "results/v2_protocol/R6_FREEZE.json",
    "results/v2_protocol/R6_HOLDOUT_PAIRING.json",
    "results/v2_protocol/R6_ADOPTION.json",
    "docs/R5_README.md",
    "docs/R6_METHODS.md",
    "tools/r6_report.py",
    "tools/test_r6_report.py",
    "tools/r4_report_links.py",
]
OPTIONAL = [
    "results/v2_protocol/R6_PREFREEZE_FINAL_CONTRACT.json",
    "results/v2_protocol/R6_FINAL_CONTRACT_REVIEW.json",
    "results/v2_protocol/R6_HOLDOUT_REVIEW.json",
    "results/v2_protocol/R6_OFFICIAL_REGRESSION.json",
    "artifacts/media/r6_manifest.json",
]
NEGATIVE_REQUIRED = [
    "results/v2_protocol/R6_S1_FAILURE.json",
    "results/v2_protocol/R6_PARITY_RAW_REVIEW.json",
    "results/v2_protocol/R6_EXECUTION_STATUS.json",
    "results/v2_protocol/R6_STOP_SNAPSHOT.json",
]
POSITIVE_ONLY = {
    "results/v2_protocol/R6_PARITY_RESULT.json",
    "results/v2_protocol/R6_SAVED_EQUIVALENCE.json",
    "results/v2_protocol/PROTOCOL_R6.json",
    "results/v2_protocol/R6_FREEZE.json",
    "results/v2_protocol/R6_HOLDOUT_PAIRING.json",
}
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        "schema": "R6_REPORT_CURATION.v1",
        "sources": sources,
        "pending_required": pending,
        "outputs": OUTPUTS,
        "official_strings_preserve_source_precision": True,
        "private_raw_or_acquisition_inputs": False,
        "created_utc": "2000-01-01T00:00:00+00:00",
    }


def validate_curation(root: Path, complete: bool = True):
    root = Path(root)
    document = read(root / "results/v2_protocol/R6_REPORT_CURATION.json")
    if document.get("schema") != "R6_REPORT_CURATION.v1":
        raise ValueError("bad R6 curation schema")
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
    negative = (root / "results/v2_protocol/R6_S1_FAILURE.json").is_file()
    base = [name for name in REQUIRED if not negative or name not in POSITIVE_ONLY]
    return base + (NEGATIVE_REQUIRED if negative else [])


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
        raise ValueError("R6 adoption lacks both architectures")
    if effective["torque"] != "R1":
        raise ValueError("R6 cannot promote the disqualified torque channel")
    if effective["servo"] not in {"R1", "SCALE_STATE", "A_STATE", "R6"}:
        raise ValueError("unknown R6 servo adoption")
    expected_assembly = "R6" if effective["servo"] != "R1" else "R1"
    if document.get("final_assembly") != expected_assembly:
        raise ValueError("R6 final assembly differs from effective variants")
    return effective


def _holdout_rows(document):
    if document is None:
        return [{"방식": label, "파일": "미실행", "충돌 검출": "0조건",
                 "오감지": "0회 실행·미측정", "R1 계산 시간": "미실행",
                 "시험 계산 시간": "미실행", "판정": "S1 조기 종료"}
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
            "파일": ("R6 파일(토크 R1 분기)" if mode == "torque"
                    and row.get("tested_variant") == "R1"
                    else str(row.get("tested_variant", row.get("variant", "R6 파일")))),
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
    return {"schema": "R6_PRESERVED_R1_OFFICIAL_VIEW.v1", "records": records,
            "scope": "historical R1 values; R6 official episodes executed: 0"}


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
    # Imported lazily so synthetic unit tests can exercise the R6 adapters alone.
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
    r6_servo = (f"A {state['candidate_detected']}/"
                f"{collision_rows - state['candidate_detected']}/0")
    for index, row in enumerate(old):
        r6 = row["R1"]
        if index == 2:
            r6 += " (과거 전체 34조건 R1 보존·새 실행 아님)"
        if index == 5 and adopted:
            torque = row["R1"].split(";", 1)[1].strip()
            r6 = r6_servo + "; " + torque
        rows.append({"항목": row["항목"], "기준": row["기준"], "v1": row["v1"],
                     "R1": row["R1"], "R6": r6})
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
    manifest = Path(root) / "artifacts/media/r6_manifest.json"
    if adopted and manifest.is_file():
        data = read(manifest)
        pairs = data.get("pairs", [])
        if len(pairs) != 2:
            raise ValueError("adopted R6 media manifest must contain two pairs")
        lines += ["", "| R6 양성 비교 | R6 음성 비교 |", "|---|---|"]
        rendered = []
        for pair in pairs:
            gif, mp4 = pair.get("gif"), pair.get("mp4")
            if not gif or not mp4 or not (Path(root) / gif).is_file() or not (Path(root) / mp4).is_file():
                raise ValueError("R6 media manifest points to a missing artifact")
            rendered.append(f"[![{pair.get('label', 'R6 비교')}]({gif})]({mp4})")
        lines.append("| " + " | ".join(rendered) + " |")
        lines += ["", "09/10은 R5 저장 validation 장면을 비트 동등성 근거로 다시 쓴 설명용 비교다. 새 holdout 효능 영상이 아니다."]
    return "\n".join(lines)


def _negative_context(root, docs):
    failure = docs["R6_S1_FAILURE"]
    raw = docs["R6_PARITY_RAW_REVIEW"]
    execution = docs["R6_EXECUTION_STATUS"]
    adoption = docs["R6_ADOPTION"]
    effective = _actual_adoption(adoption)
    if (failure.get("schema") != "R6_S1_FAILURE.v1" or failure.get("status") != "FAIL"
            or failure.get("failed_stage") != "S1_PARITY_RESULT_SERIALIZATION"):
        raise ValueError("R6 S1 failure record differs")
    rows = raw.get("rows")
    def all_true(value):
        return isinstance(value, dict) and value and all(item is True for item in value.values())
    if (raw.get("schema") != "R6_PARITY_RAW_REVIEW.v1"
            or raw.get("physical_episodes_completed") != 8 or raw.get("pairs") != 4
            or raw.get("all_raw_bits_equal") is not True
            or not isinstance(rows, list) or len(rows) != 4
            or any(row.get("passed") is not True
                   or not all_true(row.get("physical_arrays_bit_equal"))
                   or not all_true(row.get("packet_arrays_bit_equal"))
                   or row.get("final_detection_bit_equal") is not True
                   or not all_true(row.get("monitor_bit_equal"))
                   for row in rows)):
        raise ValueError("R6 raw parity review differs")
    if (effective != {"servo": "R1", "torque": "R1"}
            or adoption.get("final_assembly") != "R1"
            or adoption.get("sealed_stages_skipped") is not True):
        raise ValueError("S1 failure must select the prefrozen R1 submission")
    serialized = json.dumps(execution, ensure_ascii=False)
    if ("PROTOCOL_R6" in serialized or "R6_HOLDOUT" in serialized
            or "R6_OFFICIAL" in serialized):
        # Referencing a planned filename is allowed; existence/count validation is below.
        pass
    actual = execution.get("actual", execution.get("actual_counts", {}))
    if not isinstance(actual, dict):
        raise ValueError("R6 execution status lacks actual counts")
    for key in ("holdout_episodes", "official_episodes", "holdout_batches",
                "official_batches", "holdout_design_RNG_calls",
                "completed_trial_reexecutions", "GPU_jobs", "remote_operations"):
        if actual.get(key) != 0:
            raise ValueError("sealed stage unexpectedly executed: " + key)
    if (actual.get("saved_streams") != 990 or actual.get("parity_episodes") != 8
            or actual.get("valid_parity_base_documents") != 4):
        raise ValueError("R6 S1 completed-count record differs")
    historical = _historical_context(root)
    official = _preserved_r1_official(root)
    contract = docs["R6_PREFREEZE_CONTRACT"]
    final_contract = docs.get("R6_FINAL_CONTRACT_REVIEW")
    if final_contract is not None and str(final_contract.get("status")).upper() != "PASS":
        raise ValueError("post-S1-failure publication contract did not pass")
    result = {"root": root, "docs": docs, "adoption": adoption, "effective": effective,
              "adopted": False, "holdout": None, "official": None,
              "historical": historical,
              "equivalence": ("새 full-loop 8회는 모두 완료됐고 사후 raw 검토에서 물리·packet·결정 비트가 같았다. "
                              "그러나 결과 직렬화가 실패했으므로 이 사후 검토는 실패한 S1 transaction을 수리하거나 통과로 바꾸지 않는다."),
              "holdout_rows": _holdout_rows(None),
              "control_rows": _control_rows(official),
              "collision_rows": _collision_rows(official),
              "t_rows": _t_rows(historical, False),
              "contract": contract, "prefinal_contract": docs.get("R6_PREFREEZE_FINAL_CONTRACT"),
              "final_contract": final_contract, "negative": True,
              "parity_timing_rows": _parity_timing_rows(raw)}
    result["inherited_rows"] = _inherited_rows(result)
    return result


def context(root: Path = ROOT):
    root = Path(root)
    validate_curation(root)
    names = list(dict.fromkeys(required_inputs(root) + OPTIONAL))
    docs = {Path(name).stem: read(root / name) for name in names
            if name.endswith(".json") and (root / name).is_file()}
    if "R6_S1_FAILURE" in docs:
        return _negative_context(root, docs)
    adoption = docs["R6_ADOPTION"]
    effective = _actual_adoption(adoption)
    adopted = effective["servo"] != "R1"
    protocol = docs["PROTOCOL_R6"]
    inherited = read(root / "results/v2_protocol/PROTOCOL_R5.json")
    for key in ("detector_gates", "timing", "relative_R1_interpretation",
                "control_gates_inherited", "official"):
        if protocol.get(key) != inherited.get(key):
            raise ValueError("R6 changed inherited gate object: " + key)
    build = docs["R6_BUILD"]
    diff = build.get("diff_proof", {})
    if diff.get("changed_line_count", 1) != 1:
        raise ValueError("R6 build is not the registered one-line assembly repair")
    contract = docs["R6_PREFREEZE_CONTRACT"]
    if (contract.get("status") != "PASS" or contract["full_suite"].get("tests_run") != 534
            or len(contract.get("interface", {}).get("cases", [])) != 8):
        raise ValueError("R6 pre-freeze contract proof differs")
    parity = docs["R6_PARITY_RESULT"]
    saved = docs["R6_SAVED_EQUIVALENCE"]
    equivalence = _equivalence_text(saved, parity, docs["R6_PARITY_PLAN"])
    holdout = docs.get("R6_HOLDOUT_REVIEW")
    pairing = docs["R6_HOLDOUT_PAIRING"]
    if pairing.get("episodes_run") != 152 or pairing.get("pair_count") != 76:
        raise ValueError("R6 holdout pairing does not cover 152 episodes")
    official = docs.get("R6_OFFICIAL_REGRESSION")
    if official is not None:
        episodes = official.get("episodes_started", {})
        if (episodes.get("total") != 58 or episodes.get("required_total") != 58
                or episodes.get("per_mode") != {"servo": 29, "torque": 29}):
            raise ValueError("R6 official batch does not cover 58 episodes")
        if "R3_delay_ms" in json.dumps(official) or "R3_false_alarms" in json.dumps(official):
            raise ValueError("R6 official serialization reused historical misnamed fields")
        if not all(token in json.dumps(official)
                   for token in ("tested_delay_ms", "tested_false_alarms")):
            raise ValueError("R6 official result lacks correctly named tested fields")
        for mode in ("torque",) if adopted else ("servo", "torque"):
            identity = official.get("architectures", {}).get(mode, {}).get("exact_R1_identity", {})
            if (identity.get("status") != "PASS" or identity.get("checked_cells") != 66
                    or identity.get("raw_tokens_preserved") is not True
                    or identity.get("errors") != []):
                raise ValueError("official R1 identity proof differs: " + mode)
    final_contract = docs["R6_FINAL_CONTRACT_REVIEW"]
    if str(final_contract.get("status")).upper() != "PASS":
        if adopted:
            raise ValueError("failed final contract cannot adopt R6")
    historical = _historical_context(root)
    prefinal = docs.get("R6_PREFREEZE_FINAL_CONTRACT")
    if prefinal is not None and (str(prefinal.get("status")).upper() != "PASS"
                                 or prefinal.get("errors") != 0
                                 or prefinal.get("failures") != 0
                                 or prefinal.get("default_contract", {}).get("status") != "PASS"):
        raise ValueError("final pre-freeze contract did not pass")
    result = {"root": root, "docs": docs, "adoption": adoption, "effective": effective,
            "adopted": adopted, "holdout": holdout, "official": official,
            "historical": historical, "equivalence": equivalence,
            "holdout_rows": _holdout_rows(holdout),
            "control_rows": _control_rows(official),
            "collision_rows": _collision_rows(official),
            "t_rows": _t_rows(historical, adopted),
            "contract": contract, "prefinal_contract": prefinal,
            "final_contract": final_contract, "negative": False,
            "parity_timing_rows": []}
    result["inherited_rows"] = _inherited_rows(result)
    return result


def _conclusion(c):
    if c.get("negative"):
        return ("S1 결과 직렬화 실패로 R6 후보를 채택하지 않고 A 서보와 B 토크 모두 R1을 유지했다. "
                "holdout과 공식 평가는 열지 않았다.")
    if c["adopted"]:
        return ("A 서보에는 조립 결함을 고친 SCALE_STATE 감지 채널을 채택했고, "
                "B 토크는 R1을 유지했다.")
    return "R6 후보는 채택하지 않았고 A 서보와 B 토크 모두 R1을 유지했다."


def _unmet(c):
    old = read(c["root"] / "results/v2_protocol/R5_PERTURB_RESULTS.json")
    counts = _perturb_fa_counts(old, c["adopted"])
    if c["holdout"] is None:
        timing = "새 holdout 계산 시간은 미실행이다. 최대 1 ms를 보장으로 주장하지 않는다."
    else:
        maximum = max(row["metrics"]["combined_max_worst_us"] for row in c["holdout"]["entries"])
        timing = (f"새 holdout 합산 최대 {maximum:.3f} µs로 1 ms 목표를 넘었다. 모든 초과 주기와 최대값을 보존했다."
                  if maximum > 1000 else
                  f"새 holdout 합산 최대 {maximum:.3f} µs는 1 ms 안이지만 하드 실시간 보장은 아니다.")
    torque_d4 = next(row for row in c["historical"]["docs"]["R5_D4_RESULTS"]["entries"]
                     if row["mode"] == "torque" and row["variant"] == "SCALE_STATE")
    t = c["t_rows"]
    return [
        timing,
        (f"무충돌 섭동 28 reference FA는 A {counts['servo']['r1_reference']}→{counts['servo']['final_reference']}, "
         f"B {counts['torque']['r1_reference']}→{counts['torque']['final_reference']}다."),
        (f"전체 34조건 FA는 A {counts['servo']['r1_all']}→{counts['servo']['final_all']}, "
         f"B {counts['torque']['r1_all']}→{counts['torque']['final_all']}다. 28 reference 범위와 섞지 않는다."),
        (f"탈락한 B STATE 진단값은 D4 추가 +{torque_d4['additional_detected']}로 기준 6에 못 미쳤고, "
         f"28 reference FA가 {counts['torque']['r1_reference']}→{counts['torque']['state_reference']}로 늘었다. R6 B에는 승계하지 않았다."),
        f"놓기 전 평균 지지율은 R1에서 {t[0]['R1']}로 50 %mg 기준에 못 미친다.",
        f"D4 R1 포화/놓기 peak는 {t[3]['R1']}로 A 포화 0회와 peak 1.1 mg 기준을 만족하지 못한다.",
        "유한 시뮬레이션 결과이며 하드웨어 안전 보장이나 충돌 정지 인증이 아니다.",
    ]


def _prefreeze_contract_text(c):
    initial = c["contract"]["full_suite"]["tests_run"]
    final = c["prefinal_contract"]
    if final is None:
        return f"초기 전체 {initial}개와 기본 이름 8개는 통과했다. 새 도구를 포함한 최종 동결 전 전체 검사는 미실행이다."
    count = final.get("full_suite", {}).get("tests_run", final.get("tests_run"))
    if not isinstance(count, int):
        raise ValueError("final pre-freeze contract lacks test count")
    return f"초기 전체 {initial}개와 최종 동결 전 전체 {count}개, 기본 이름 8개 검사가 통과했다."


def render_readme(c):
    holdout = md_table(["방식", "파일", "충돌 검출", "오감지", "R1 계산 시간", "시험 계산 시간", "판정"],
                       c["holdout_rows"])
    control = md_table(["방식", "질량", "TCP RMS/최대", "포화", "집기/놓기 계단", "오감지", "상태"],
                       c["control_rows"])
    collisions = md_table(["방식", "질량/시각", "10 N", "15 N", "40 N", "100 N",
                           "충돌 세트 FA", "최소 검출력"], c["collision_rows"])
    timing = (md_table(["방식", "비교 파일", "에피소드", "감지 p99/최대 [µs]",
                        "합산 p99/최대 [µs]", "1 ms 초과", "보존 outlier"],
                       c["parity_timing_rows"])
              if c["parity_timing_rows"] else "미실행")
    return f"""# Week 9 Palletizing Load Transfer — R6 조립 수정 재검증

**최종 결론: {_conclusion(c)}**

기준=초기 제공본, v1=첫 개선본, R1=고정 비교 제어·감지기, R6=R5 서보 학습 채널의 조립 결함 한 줄만 고친 재검증본이다. A=서보, B=토크, FA=오감지다.

{_media_block(c['root'], c['adopted'])}

## 결론

{_conclusion(c)} 결함은 한 원소 모드 집합이 문자열로 만들어진 것이며, 이를 `('servo',)` 튜플로 고쳤다. 모델 263개 파라미터, 임계값, R1 제어기와 모든 관문 숫자는 바꾸지 않았다. {_prefreeze_contract_text(c)} {c['equivalence']} 상세 근거는 [R6 결과](docs/R6_RESULTS.md), 방법은 [R6 방법](docs/R6_METHODS.md), 발표 자료는 [PDF](results/presentation.pdf)에 있다.

## 기준/v1/R1/R6 — T1~T6

{md_table(['항목','기준','v1','R1','R6'], c['t_rows'])}

감지기 조립만 고쳤으므로 T1/T2/T4/T5 물리값은 R1과 같다. T3은 과거 전체 34조건 보존값이며 새 실행이 아니다. T6을 포함한 저장 평가 수치는 990개 신호 스트림의 비트 동등성으로 승계했으며 새 효능 측정으로 세지 않는다.

## 시험 후보로 승계한 validation·development·D4·섭동

{md_table(['단계','A 서보','B 토크','성격'], c['inherited_rows'])}

이 표는 S1에서 저장 자료에서 비트 동등성을 확인한 A STATE 시험 후보의 과거 노출 평가값이다. 이후 holdout·공식·최종 계약이 실패하면 최종 제출은 R1이며, 이 양성 결과가 veto를 뒤집지 않는다.

## 새 holdout

{holdout}

표의 수치는 새 봉인 batch에서 실제 실행된 값만 쓴다. 실행되지 않은 단계는 추정값이나 자리표시자 대신 `미실행`으로 표시한다.

## S1 full-loop 8회 계산 시간

{timing}

이 표는 S1 parity 4쌍·8회에서 실제 측정한 값이다. holdout이나 공식 평가 시간으로 해석하지 않는다. 모든 outlier를 보존했다.

## 보존 R1 공식값 — 0/4/8 kg

{control}

## 보존 R1 공식 충돌 — 전체 10/15/40/100 N

{collisions}

B 토크 8 kg, 3.0 s 행의 충돌 세트 FA 1회도 표에 포함했다. 위 표는 과거 R1 보존값이며 R6에서 공식 평가를 새로 실행한 값이 아니다. R6 새 공식 실행은 0회다. `R5_OFFICIAL_REGRESSION.json`의 `R3_delay_ms`와 `R3_false_alarms`는 실제로 R5 후보 측정값을 담은 과거 호환 필드다. 원자료는 보존했다. R6 serializer는 `tested_delay_ms`와 `tested_false_alarms`라는 정확한 이름으로 설계·검사했지만, 이번 조기 종료에서는 공식 결과 JSON을 만들지 않았다.

## 미달과 범위

""" + "\n".join("- " + item for item in _unmet(c)) + "\n"


def artifact_links(c):
    common = ("[저장 동등성](../results/v2_protocol/R6_SAVED_EQUIVALENCE.json), "
              "[판정](../results/v2_protocol/R6_ADOPTION.json)")
    if c.get("negative"):
        return ("정식 protocol·동결·holdout·공식 결과는 만들지 않았다. 실제 기록은 "
                "[실패](../results/v2_protocol/R6_S1_FAILURE.json), "
                "[사후 배열 검토](../results/v2_protocol/R6_PARITY_RAW_REVIEW.json), "
                "[중단 시 소스 잠금](../results/v2_protocol/R6_STOP_SNAPSHOT.json), "
                + common + "이다.")
    return ("[protocol](../results/v2_protocol/PROTOCOL_R6.json), "
            "[동결](../results/v2_protocol/R6_FREEZE.json), " + common + "이다.")


def render_results(c):
    parity = c["docs"].get("R6_PARITY_RESULT")
    saved = c["docs"].get("R6_SAVED_EQUIVALENCE")
    official_status = "미실행" if c["official"] is None else str(c["official"].get("status"))
    holdout_status = "미실행" if c["holdout"] is None else "1 batch 완료"
    evidence_status = ("S1 transaction 실패·사후 raw 8회 비트 검토만 통과"
                       if c["negative"] else "990/990 비트 일치")
    parity_status = ("직렬화 실패" if c["negative"]
                     else ("통과" if parity["passed"] else "실패"))
    timing = (md_table(["방식", "비교 파일", "에피소드", "감지 p99/최대 [µs]",
                        "합산 p99/최대 [µs]", "1 ms 초과", "보존 outlier"],
                       c["parity_timing_rows"])
              if c["parity_timing_rows"] else "미실행")
    return f"""# R6 결과

## 판정

{_conclusion(c)} 결함 수정 외 재설계는 0회다. 실제 S1 실패 뒤 검증 드라이버·후보·모델·임계값·관문을 고치거나 실험을 재실행하지 않았다.

## 결함과 검사

- 원인: 한 원소 모드 집합이 문자열로 생성돼 제어기 모드가 없는 `reset()`에서 예외가 발생했다.
- 수정: 문자열 한 줄을 한 원소 튜플로 바꿨다. 모델, 임계값, 제어기에는 차이가 없다.
- 알려진 결함 재현 검사는 수정 전 실패했고 수정 후 통과했다.
- {_prefreeze_contract_text(c)}
- 첫 전체 검사 1회는 검증 사본의 저장소 메타데이터가 없어 환경 무효로 분류했다. 533개는 통과했고, 감지기 행동을 보지 못한 저장소 조회 검사 1개만 오류였다. 소스 바이트를 그대로 둔 유효 환경에서 같은 534개를 통과했다.

## 동등성

{c['equivalence']}

- 저장·S1 상태: {evidence_status}
- 새 full-loop 결과 transaction: {parity_status}
- 비교 배열: `q`, `q_d`, `tau`, `e_mm`, `f_table`, `f_tool`, `qpos`, `r`
- 허용오차를 쓰지 않았으며 dtype·shape·C-order 바이트를 모두 비교했다.

실패 원인은 토크 monitor 비교 flag가 `numpy.bool_`인 상태에서 표준 JSON encoder로 직렬화된 것이다. 이 알려진 도구 결함은 후보·driver·sealed 도구를 고치지 않고 그대로 실패로 처리했다. 사후 raw 검토는 원시 8회가 완료됐다는 사실만 확인하며 실패 transaction을 복구하지 않는다.

## S1 parity 계산 시간

{timing}

## 시험 후보로 승계한 과거 단계

{md_table(['단계','A 서보','B 토크','성격'], c['inherited_rows'])}

A 열은 저장 자료에서 비트 동등성을 확인한 시험 후보 값이다. 최종 판정이 R1 fallback이면 양성 노출 결과를 보존하되 최종 제출 성능으로 쓰지 않는다.

## Holdout

상태: **{holdout_status}**

{md_table(['방식','파일','충돌 검출','오감지','R1 계산 시간','시험 계산 시간','판정'], c['holdout_rows'])}

## 공식 회귀

상태: **R6 새 실행 0회**. 아래는 과거 R1 보존값이다.

{md_table(['방식','질량','TCP RMS/최대','포화','집기/놓기 계단','오감지','상태'], c['control_rows'])}

### 충돌 전 행

{md_table(['방식','질량/시각','10 N','15 N','40 N','100 N','충돌 세트 FA','최소 검출력'], c['collision_rows'])}

과거 R5 공식 JSON의 `R3_delay_ms`·`R3_false_alarms`는 R5 후보 값을 가리키는 잘못된 호환 이름이다. 원자료는 고치지 않았다. R6 serializer는 `tested_delay_ms`·`tested_false_alarms`를 쓰도록 설계·검사했지만 공식 단계가 열리지 않아 결과 JSON은 없다. 원 요구사항과 평가기 점검·미검출 원인은 [이전 감사 요약](../docs/R4_README.md#평가기-점검과-미검출-원인)에 보존했다.

## 기준/v1/R1/R6

{md_table(['항목','기준','v1','R1','R6'], c['t_rows'])}

## 미달

""" + "\n".join("- " + item for item in _unmet(c)) + """

## 공개 재현 범위

공개 파일에는 조립기, 모델, 검증 집계, 문서 생성기와 검사 도구가 들어간다. 비공개 원시 시계열과 취득 드라이버는 포함하지 않는다. """ + artifact_links(c) + """
"""


def render_notes(c):
    return f"""# R6 발표 요약

## 1. 질문

R5 서보 후보의 수치 성능이 아니라 조립 결함만 고쳤을 때 제출 호환성, 저장 자료 동등성, 새 holdout과 공식 회귀를 모두 통과하는가?

## 2. 한 줄 수정

한 원소 문자열을 `('servo',)` 튜플로 만들었다. 모델 263개 파라미터와 임계값, 제어기, 관문은 그대로다.

## 3. 결함 재현과 계약

수정 전 알려진 예외를 회귀 검사로 재현했다. 수정 후 신규 5개 검사, 전체 534개 검사, 기본 제출 이름 8개 인터페이스 검사가 통과했다.

## 4. 저장 자료와 새 물리 동등성

{c['equivalence']}

S1 결과 직렬화가 실패했으므로 사후 raw 검토의 비트 일치는 후보를 되살리는 통과 판정이 아니다. 알려진 직렬화 결함은 고치지 않았고 R1 fallback을 선택했다.

## 4-1. S1 parity 계산 시간

{md_table(['방식','비교 파일','에피소드','감지 p99/최대 [µs]','합산 p99/최대 [µs]','1 ms 초과','보존 outlier'], c['parity_timing_rows']) if c['parity_timing_rows'] else '미실행'}

## 5. Holdout

{md_table(['방식','파일','충돌 검출','오감지','R1 계산 시간','시험 계산 시간','판정'], c['holdout_rows'])}

## 6. 보존 R1 공식 0/4/8 kg

{md_table(['방식','질량','TCP RMS/최대','포화','집기/놓기 계단','오감지','상태'], c['control_rows'])}

## 7. 보존 R1 공식 10/15/40/100 N

{md_table(['방식','질량/시각','10 N','15 N','40 N','100 N','충돌 세트 FA','최소 검출력'], c['collision_rows'])}

## 8. 결론

{_conclusion(c)} 재설계는 0회이며 실패 뒤 수리나 재실행을 허용하지 않았다.

## 9. 제한

""" + "\n".join("- " + item for item in _unmet(c)) + "\n"


def _slides(c):
    return [
        {"title": "R6 결론", "bullets": [_conclusion(c), "조립 결함 한 줄만 수정", "재설계 0회"]},
        {"title": "결함 원인", "bullets": ["한 원소 모드 집합이 문자열로 생성", "제어기 모드가 없는 reset에서 예외", "모델·임계·제어기와 무관"]},
        {"title": "수정과 회귀 검사", "bullets": ["문자열을 한 원소 튜플로 변경", "수정 전 red, 수정 후 신규 5개 통과", (("초기 534개·최종 사전 " + str(c["prefinal_contract"]["tests_run"]) + "개, 기본 이름 8개 계약 통과") if c.get("prefinal_contract") else "전체 534개와 기본 이름 8개 계약 통과")]},
        {"title": "환경 무효 실행의 처리", "bullets": ["첫 사본에는 저장소 메타데이터가 없음", "행동 검사가 아닌 저장소 조회 1개만 오류", "소스 바이트 불변을 확인하고 같은 전체 검사를 유효하게 통과"]},
        {"title": "저장 자료 동등성", "bullets": [c["equivalence"], "validation·development·D4·섭동 결과를 승계", "새 효능 측정으로 세지 않음"]},
        {"title": "시험 후보로 승계한 과거 단계", "table": (["단계","A 서보","B 토크","성격"], c["inherited_rows"])},
        {"title": "새 full-loop 동등성", "bullets": ["새 질량·시드 8회", "물리 8배열의 dtype·shape·바이트 일치", "허용오차 없음"]},
        {"title": "S1 결과 직렬화 실패", "bullets": ["물리 8회 완료 뒤 결과 JSON 직렬화에서 실패", "numpy bool 값을 표준 encoder가 거부", "도구를 고치거나 재실행하지 않고 R1 선택", "사후 raw 검토는 실패 판정을 바꾸지 않음"]},
        {"title": "Protocol과 동결 — 미실행" if c.get("negative") else "Protocol과 동결", "bullets": (["S1 실패로 정식 protocol·동결은 미실행", "holdout 조건 생성·공식 실행 각각 0회", "사전 준비한 R1 fallback과 S1 소스 잠금만 보존"] if c.get("negative") else ["holdout 조건 생성 전에 protocol 커밋", "후보와 R1 fallback을 함께 동결", "실패 시 추가 수정 없이 R1"])},
        {"title": "Holdout", "table": (["방식","파일","충돌 검출","오감지","R1 계산 시간","시험 계산 시간","판정"], c["holdout_rows"])},
        {"title": "보존 R1 공식 0/4/8 kg", "table": (["방식","질량","TCP RMS/최대","포화","집기/놓기 계단","오감지","상태"], c["control_rows"])},
        {"title": "보존 R1 공식 10/15/40/100 N — A", "table": (["질량/시각","10 N","15 N","40 N","100 N","충돌 세트 FA","최소 검출력"], [{k:v for k,v in row.items() if k != "방식"} for row in c["collision_rows"] if row["방식"] == "A 서보"])},
        {"title": "보존 R1 공식 충돌 B 토크", "table": (["질량/시각","10 N","15 N","40 N","100 N","충돌 세트 FA","최소 검출력"], [{k:v for k,v in row.items() if k != "방식"} for row in c["collision_rows"] if row["방식"] == "B 토크"])},
        {"title": "T1–T3", "table": (["항목","기준","v1","R1","R6"], c["t_rows"][:3])},
        {"title": "T4–T6", "table": (["항목","기준","v1","R1","R6"], c["t_rows"][3:])},
        ({"title": "S1 계산 시간 — 각 2조건·holdout 아님", "table": (["방식","비교 파일","에피소드","감지 p99/최대 [µs]","합산 p99/최대 [µs]","1 ms 초과","보존 outlier"], c["parity_timing_rows"])} if c.get("negative") else {"title": "계산 시간", "bullets": [row["방식"] + " R1: " + row["R1 계산 시간"] + "; 시험: " + row["시험 계산 시간"] for row in c["holdout_rows"]] + ["p99 800 µs 관문", "최대 1 ms는 목표이며 초과값을 유지"]}),
        {"title": "과거 필드 이름", "bullets": ["R5의 R3_delay_ms와 R3_false_alarms는 R5 후보 값", "과거 JSON은 보존", ("R6 serializer 설계는 tested_delay_ms·tested_false_alarms; 이번 공식 JSON은 미생성" if c.get("negative") else "R6은 tested_delay_ms와 tested_false_alarms 사용")]},
        {"title": "미달", "bullets": _unmet(c)},
        {"title": "재현 범위", "bullets": ["공개: 조립기·모델·집계·문서·검사", "비공개: 원시 시계열·취득 드라이버", "원본 키트·sim·model 변경 없음"]},
        {"title": "최종 판정", "bullets": [_conclusion(c), "실패 뒤 재설계·재실행 없음", "상세: docs/R6_RESULTS.md"]},
    ]


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
    metadata = {"Title": "R6 assembly fix and revalidation", "Creator": "r6_report.py",
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
            figure.text(.04, .025, f"R6 검증 | {index}/{len(pages)}", fontsize=9, color="#607d8b")
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
    check_links(root)
    pdf = root / OUTPUTS[3]
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF"):
        raise ValueError("missing R6 PDF")
    extracted = pdf_text(pdf)
    for token in ("R6 결론", "Holdout", "공식 10/15/40/100 N", "미달", "최종 판정"):
        if token not in extracted:
            raise ValueError("PDF text missing " + token)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as stream:
        make_pdf(Path(stream.name), c)
        if Path(stream.name).read_bytes() != pdf.read_bytes():
            raise ValueError("R6 PDF is stale or nondeterministic")
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
        path = root / "results/v2_protocol/R6_REPORT_CURATION.json"
        write(path, json.dumps(build_curation(root), ensure_ascii=False, indent=2) + "\n")
        print(path)
        return
    if args.check:
        check_one(root)
        for projection in args.projection:
            check_one(projection.resolve())
        print("R6 report check passed")
        return
    generate(root)
    print("R6 report generated")


if __name__ == "__main__":
    main()
