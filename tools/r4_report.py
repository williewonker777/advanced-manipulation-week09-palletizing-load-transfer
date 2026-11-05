#!/usr/bin/env python3
"""Build the deterministic R4 public report from frozen evidence only.

This module imports no simulator, controller, study, or holdout builder.  It
does not construct random conditions.  Every displayed measurement is copied
or aggregated from a hash-bound JSON artifact.
"""
from __future__ import annotations

from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import argparse
import copy
import hashlib
import json
import math
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = ROOT / "results/v2_protocol"
MANIFEST = D / "R4_REPORT_MANIFEST.json"
OUTPUTS = (ROOT / "README.md", ROOT / "docs/R4_RESULTS.md",
           ROOT / "report/PRESENTATION_R4.md", ROOT / "results/presentation.pdf")
MODES = ("servo", "torque")
VARIANTS = ("W40_ACTIVE", "CUSUM_ACTIVE_K05", "CUSUM_ACTIVE_K10",
            "CUSUM_PHASE_K05")
STAGES = ("VALIDATION", "DEVELOPMENT", "D4", "PERTURB")
PROVENANCE = {"fresh", "reused_exact_source", "literal_copy", "link_path_only"}
ARTIFACTS = {
    "protocol": "results/v2_protocol/PROTOCOL_R4.json",
    "holdout_reseal": "results/v2_protocol/PROTOCOL_R4_HOLDOUT_RESEAL.json",
    "literature": "docs/R4_LITERATURE.md",
    "cause_analysis": "results/v2_protocol/R4_CAUSE_ANALYSIS.json",
    "methods": "docs/R4_METHODS.md",
    "calibration": "results/v2_protocol/R4_CALIBRATION.json",
    "equivalence": "results/v2_protocol/R4_EQUIVALENCE.json",
    "validation_review": "results/v2_protocol/R4_VALIDATION_REVIEW.json",
    "development_review": "results/v2_protocol/R4_DEVELOPMENT_REVIEW.json",
    "d4_review": "results/v2_protocol/R4_D4_REVIEW.json",
    "perturb_review": "results/v2_protocol/R4_PERTURB_REVIEW.json",
    "redesign": "results/v2_protocol/R4_REDESIGN_DECISION.json",
    "selection": "results/v2_protocol/R4_PREFREEZE_DECISION.json",
    "freeze": "results/v2_protocol/R4_FREEZE.json",
    "holdout": "results/v2_protocol/R4_HOLDOUT.json",
    "holdout_review": "results/v2_protocol/R4_HOLDOUT_REVIEW.json",
    "official": "results/v2_protocol/R4_OFFICIAL_REGRESSION.json",
    "integrity": "results/v2_protocol/R4_REPORT_INTEGRITY.json",
    "r3_metrics": "results/v2_protocol/R3_REPORT_CURATION.json",
    "r3_readme": "docs/R3_README.md",
    "r3_presentation": "results/presentation_R3.pdf",
    "document_link_repairs": "results/v2_protocol/R4_REPORT_LINK_REPAIRS.json",
    "requirements": "docs/REQUIREMENTS_AUDIT.md",
}
for _mode in MODES:
    ARTIFACTS.update({
        "perturb_w40_" + _mode: f"results/v2_protocol/R4_PERTURB_W40_ACTIVE_{_mode}.json",
        "perturb_r1_" + _mode: f"results/v2_protocol/R4_BASE_perturb_{_mode}.json",
        "timing_w40_" + _mode: f"results/v2_protocol/R4_REVALIDATION_W40_ACTIVE_{_mode}.json",
        "timing_r1_" + _mode: f"results/v2_protocol/R4_BASE_revalidation_{_mode}.json",
    })
REVISION_ARTIFACTS = (
    "results/v2_protocol/PROTOCOL_R4_REDESIGN_W40.json",
    "results/v2_protocol/R4_REDESIGN_DECISION_V2.json",
    "results/v2_protocol/R4_REDESIGN_BUILD.json",
    "results/v2_protocol/R4_REDESIGN_NORMAL_AUDIT.json",
    "results/v2_protocol/R4_REDESIGN_EQUIVALENCE.json",
    "results/v2_protocol/R4_REVALIDATION_REVIEW.json",
)
PUBLIC_FORBIDDEN = (
    re.compile("\x2fhome\x2f", re.I), re.compile(r"TA" + r"SK-\d+", re.I),
    re.compile(r"\b(?:Cl" + r"aude|Co" + r"dex|G" + r"PT|O" + r"MX)\b", re.I),
    re.compile(r"\b(?:staging|push)\b|미게시|게시\s*상태", re.I),
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError("required R4 report input is absent: " + str(path)) from exc
    if not isinstance(value, dict):
        raise ValueError("R4 report input must be a JSON object: " + str(path))
    return value


def artifact_entry(root: Path, relative: str, provenance: str = "fresh") -> dict:
    if provenance not in PROVENANCE:
        raise ValueError("invalid artifact provenance")
    relative_path = Path(relative)
    path = (root / relative_path).resolve()
    if relative_path.is_absolute() or ".." in relative_path.parts \
            or (path != root.resolve() and root.resolve() not in path.parents):
        raise ValueError("artifact escapes report root: " + relative)
    if not path.is_file():
        raise FileNotFoundError("missing report artifact: " + relative)
    return {"path": relative_path.as_posix(), "sha256": sha(path),
            "provenance": provenance}


def _artifact_entry_shape(entry: object, label: str) -> bool:
    mandatory = {"path", "sha256", "provenance"}
    optional = {"prepublication_work_sha256"}
    if not isinstance(entry, dict) or not mandatory <= set(entry) \
            or set(entry) - mandatory - optional:
        return False
    previous = entry.get("prepublication_work_sha256")
    if previous is not None and (not isinstance(previous, str)
                                 or re.fullmatch(r"[0-9a-fA-F]{64}", previous) is None):
        raise ValueError("malformed prepublication artifact hash: " + label)
    return True


def verify_artifacts(manifest: dict, root: Path = ROOT) -> dict:
    artifacts = manifest.get("artifacts")
    revision_registered = (Path(root) / REVISION_ARTIFACTS[0]).is_file()
    expected_keys = set(ARTIFACTS) | ({"registered_revision"} if revision_registered else set())
    if not isinstance(artifacts, dict) or set(artifacts) != expected_keys:
        raise ValueError("R4 manifest artifact inventory is incomplete")
    loaded = {}
    for key, expected_path in ARTIFACTS.items():
        entry = artifacts[key]
        if not _artifact_entry_shape(entry, key):
            raise ValueError("malformed artifact entry: " + key)
        if entry["path"] != expected_path or entry["provenance"] not in PROVENANCE:
            raise ValueError("artifact path/provenance differs: " + key)
        path = root / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("artifact hash mismatch: " + entry["path"])
        if path.suffix == ".json":
            loaded[key] = load(path)
        else:
            loaded[key] = path
    if revision_registered:
        entries = artifacts["registered_revision"]
        if not isinstance(entries, list) or len(entries) != len(REVISION_ARTIFACTS):
            raise ValueError("registered R4 revision artifact group is incomplete")
        revision = {}
        for entry, expected_path in zip(entries, REVISION_ARTIFACTS):
            if (not _artifact_entry_shape(entry, expected_path)
                    or entry["path"] != expected_path or entry["provenance"] != "fresh"):
                raise ValueError("registered R4 revision artifact differs")
            path = Path(root) / expected_path
            if sha(path) != entry["sha256"]:
                raise ValueError("artifact hash mismatch: " + expected_path)
            revision[Path(expected_path).name] = load(path)
        loaded["registered_revision"] = revision
    return loaded


def _finite(value, label):
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(float(value)):
        raise ValueError(label + " must be finite")
    return float(value)


def stage_map(document: dict, stage: str) -> dict[tuple[str, str], dict]:
    stage_label = "VALIDATION" if stage == "REVALIDATION" else stage
    allowed_schemas = {"R4_STAGE_REVIEW.v1"}
    if stage == "REVALIDATION":
        allowed_schemas.add("R4_REVALIDATION_REVIEW.v1")
    if document.get("schema") not in allowed_schemas \
            or document.get("stage") != stage.lower() \
            or document.get("diagnostics_cannot_revive") is not True:
        raise ValueError(stage + " review header is invalid")
    rows = document.get("results")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError(stage + " review must retain all eight mode/configuration rows")
    mapped = {}
    expected_cases = {"VALIDATION": 38, "DEVELOPMENT": 38,
                      "D4": 385, "PERTURB": 34}[stage_label]
    for row in rows:
        key = (row.get("architecture"), row.get("variant"))
        if key in mapped or key[0] not in MODES or key[1] not in VARIANTS:
            raise ValueError(stage + " review has duplicate/foreign candidate")
        review = row.get("review")
        required = ("passed", "additional_detected", "lost_R1", "R1_detected",
                    "candidate_detected", "reference_FA_R1", "reference_FA_candidate",
                    "outside_event_FA_R1", "outside_event_FA_candidate", "phases",
                    "failures", "every_case_retained")
        if not isinstance(review, dict) or any(name not in review for name in required):
            raise ValueError(stage + " candidate review is incomplete")
        for name in required[1:9]:
            _finite(review[name], stage + "." + name)
        if int(review["every_case_retained"]) != expected_cases:
            raise ValueError(stage + " review did not retain every registered case")
        if not isinstance(review["phases"], dict) or not isinstance(review["failures"], list):
            raise ValueError(stage + " phase/failure detail is malformed")
        for name in ("timing_p99_us", "timing_max_us"):
            _finite(row.get(name), stage + "." + name)
        if stage_label == "VALIDATION" and row.get("full_loop_parity_pass") is not True:
            raise ValueError(stage + " full-loop physical/decision parity failed")
        mapped[key] = row
    if set(mapped) != {(mode, variant) for mode in MODES for variant in VARIANTS}:
        raise ValueError(stage + " review candidate matrix differs")
    return mapped


def _first_failure(rows: dict[str, dict]) -> str:
    """Report the first irreversible gate, never a later green diagnostic row."""
    labels = {"VALIDATION": "사전 검증", "DEVELOPMENT": "개발 검증",
              "D4": "D4", "PERTURB": "분석 섭동"}
    for stage in STAGES:
        row = rows[stage]
        if not row.get("adoption_eligible", False):
            reason = _failure_text(row)
            if reason == "통과":
                continue
            return labels[stage] + ": " + reason
    return ""


def _failure_text(row: dict) -> str:
    failures = row["review"]["failures"]
    reasons = []
    for failure in failures:
        for reason in failure.get("reasons", []):
            labels = {
                "physical_metric_changed": "물리 비트/지표",
                "extra_outside_event_false_alarm": "오감지 증가",
                "tool_contact_lost": "의도 접촉 손실",
                "R1_detection_lost_or_masked": "R1 검출 손실/선행 경보",
                "insufficient_added_detections": "추가 검출 기준 미달",
            }
            if reason not in labels:
                raise ValueError("untranslated failure reason: " + str(reason))
            value = labels[reason]
            if value not in reasons:
                reasons.append(value)
    if not row.get("normal_audit_pass", False): reasons.append("무충돌 보정 검사 실패")
    if not row.get("full_loop_parity_pass", False): reasons.append("제어·물리값 불일치")
    if not row.get("timing_pass", False): reasons.append("p99 계산시간")
    return ", ".join(reasons) if reasons else "통과"


def _phase_text(row: dict) -> str:
    phases = row["review"]["phases"]
    order = ("pick", "transport", "place")
    cells = []
    for phase in order:
        value = phases.get(phase)
        if value:
            label = {"pick": "집기", "transport": "운반", "place": "놓기"}[phase]
            cells.append(f"{label} {value['R1_detected']}→{value['candidate_detected']}/{value['cases']}")
    return "; ".join(cells) if cells else "해당 없음"


def _official_context(official: dict) -> dict:
    if official.get("schema") != "R4_OFFICIAL_REGRESSION.v1" \
            or set(official.get("architectures", {})) != set(MODES):
        raise ValueError("official regression is incomplete")
    records = official.get("results")
    if not isinstance(records, dict) or set(records) != set(MODES):
        raise ValueError("official literal results are incomplete")
    rows = []
    controls, collisions = {}, {}
    for mode in MODES:
        payload = records[mode].get("control_and_collisions", {})
        control = payload.get("control")
        events = payload.get("collisions")
        if not isinstance(control, dict) or set(control) != {"0.0", "4.0", "8.0"}:
            raise ValueError("official control rows differ for " + mode)
        if not isinstance(events, list) or not events:
            raise ValueError("official collision rows are missing for " + mode)
        # Preserve evaluator-origin strings exactly.  Never parse/reformat cells.
        for mass in ("0.0", "4.0", "8.0"):
            if not isinstance(control[mass], list) or any(not isinstance(x, str) for x in control[mass]):
                raise ValueError("official control precision cells must remain strings")
        controls[mode] = copy.deepcopy(control)
        collisions[mode] = copy.deepcopy(events)
        for event in events:
            if float(event["mass"]) == 8.0:
                rows.append({"방식": "A 서보" if mode == "servo" else "B 토크",
                             "시각 [s]": str(event["time"]),
                             "10/15/40/100 N·FA·최소": " / ".join(event["cells"])})
    return {"control": controls, "collisions": collisions, "eight_kg": rows}


def _replace_fa(text: str, value: int) -> str:
    # The R3 T3 strings each contain one literal 'FA <integer>'.
    replaced, count = re.subn(r"FA\s+\d+", "FA " + str(int(value)), text)
    if count != 1:
        raise ValueError("could not replace preserved T3 FA cell")
    return replaced


def _diagnostic_findings(loaded: dict) -> list[dict]:
    """Aggregate already measured paired rows; never execute a detector."""
    summaries = []
    for mode, label in (("servo", "A 서보"), ("torque", "B 토크")):
        candidates = loaded["perturb_w40_" + mode]["results"]
        baselines = loaded["perturb_r1_" + mode]["results"]
        if len(candidates) != 34 or len(baselines) != 34:
            raise ValueError("perturbation source must retain all 34 rows")
        baseline = {row["case_id"]: row for row in baselines}
        if set(baseline) != set(range(34)) or {r["case_id"] for r in candidates} != set(range(34)):
            raise ValueError("perturbation case IDs differ")
        increased, noise = [], []
        for row in candidates:
            old = baseline[row["case_id"]]
            for key in ("mass", "com", "kind", "perturb", "perturb_family"):
                if row[key] != old[key]:
                    raise ValueError("perturbation paired condition differs: " + key)
            if row["R1_false_alarms"] != old["false_alarms"]:
                raise ValueError("perturbation R1 alarm source differs")
            if row["false_alarms"] > old["false_alarms"]:
                increased.append(row)
            if row["perturb_family"] == "new_noise":
                noise.append(row)
        if {r["case_id"] for r in noise} != set(range(25, 34)) \
                or {r["seed"] for r in noise} != {371401, 371402, 371403}:
            raise ValueError("fresh noise rows differ from registered nine conditions")
        if any(r["perturb_family"] not in {"stiffness", "corner"}
               or r["stiffness_scale"] not in {0.8, 1.2} for r in increased):
            raise ValueError("additional false alarms are not confined to stiffness changes")
        summaries.append({
            "방식": label, "증가 행": len(increased),
            "강성 단독": sum(r["perturb_family"] == "stiffness" for r in increased),
            "강성·마찰 조합": sum(r["perturb_family"] == "corner" for r in increased),
            "새 잡음 증가/조건": f"{sum(r['false_alarms'] > r['R1_false_alarms'] for r in noise)}/{len(noise)}",
        })
    return summaries


def _timing_summary(loaded: dict) -> list[dict]:
    summaries = []
    for mode, label in (("servo", "A 서보"), ("torque", "B 토크")):
        for source, name in (("r1", "R1"), ("w40", "W40")):
            rows = loaded[f"timing_{source}_{mode}"]["results"]
            if len(rows) != 38 or {row["case_id"] for row in rows} != set(range(38)):
                raise ValueError("timing requires the same 38 revalidation conditions")
            wall = [row["timing"]["combined"]["wall"] for row in rows]
            if any(t["samples"] != 6000 for t in wall):
                raise ValueError("timing sample count differs")
            summaries.append({
                "방식·소스": label + "/" + name,
                "최악 행 p99 [µs]": f"{max(t['p99_us'] for t in wall):.3f}",
                "관측 최대 [ms]": str((Decimal(str(max(t['max_us'] for t in wall))) / 1000)
                                      .quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)),
                "1 ms 초과 주기": sum(o["combined"]["wall_ns"] > 1_000_000
                                  for row in rows for o in row["timing"]["outliers"]),
            })
    return summaries


def _requirements_summary(path: Path) -> dict:
    counts = {"충족": 0, "부분": 0, "판정 불가": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"\| R\d{2} \|", line):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            counts[cells[5]] += 1
    if sum(counts.values()) != 33:
        raise ValueError("preserved requirements audit must retain all 33 guide items")
    return counts


def _r3_r1_timing(path: Path) -> dict:
    """Quote preserved display cells, not a new aggregation of past samples."""
    labels = {"서보 A/R1": "servo", "토크 B/R1": "torque"}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] not in labels:
            continue
        if len(cells) != 4 or any(re.fullmatch(r"\d+\.\d{3}", value) is None
                                  for value in cells[1:3]) or not cells[3].isdigit():
            raise ValueError("preserved R3 R1 timing table differs")
        mode = labels[cells[0]]
        if mode in rows:
            raise ValueError("duplicate preserved R3 R1 timing row")
        rows[mode] = dict(p99_us=cells[1], max_ms=cells[2], outlier_count=cells[3])
    if set(rows) != set(MODES):
        raise ValueError("preserved R3 R1 timing rows are missing")
    return rows


def _metric_rows(r3: dict, final: dict, perturb: dict, d4: dict) -> list[dict]:
    metrics = copy.deepcopy(r3.get("metrics"))
    if not isinstance(metrics, list) or [row.get("id") for row in metrics] != \
            ["T1", "T2", "T3", "T4", "T5", "T6"]:
        raise ValueError("preserved T1–T6 source is malformed")
    pmap, dmap = stage_map(perturb, "PERTURB"), stage_map(d4, "D4")
    for row in metrics:
        row["r4"] = row.pop("r3")
        if row["id"] == "T3":
            if any(final[mode] != "R1" for mode in MODES):
                row["r4"] = "미실행: 새 잡음 조건의 R4 수치는 별도 표"
            row["condition"] = "R2와 같은 전체 34조건 보존값(충돌 외 FA 포함); R4 새 섭동은 별도 표"
            row["unit"] = "mm / N / 상승 횟수"
        if row["id"] == "T6":
            cells = []
            for mode, label in (("servo", "A"), ("torque", "B")):
                chosen = final[mode]
                if chosen == "R1":
                    source = row["r1"].split("; ")[0 if mode == "servo" else 1]
                    cells.append(source)
                else:
                    review = dmap[mode, chosen]["review"]
                    detected = int(review["candidate_detected"])
                    cells.append(f"{label} {detected}/{180-detected}/0")
            row["r4"] = "; ".join(cells)
    korean = {
        "T1": ("8 kg, 같은 COM, 놓기 직전 20 ms", "%mg (A/B)"),
        "T2": ("8 kg, 같은 COM, 놓기 토크 계단", "N·m (A/B)"),
        "T4": ("방식별 무충돌 질량·COM 189조건", "포화 조건 수 / mg"),
        "T5": ("8 kg, 같은 COM, 무충돌", "mm (A/B)"),
        "T6": ("방식별 충돌 180조건; 선행 경보를 별도 집계", "검출/놓침/선행 경보 조건 수"),
    }
    for row in metrics:
        if row["id"] in korean:
            row["condition"], row["unit"] = korean[row["id"]]
        if row["id"] in {"T1", "T2", "T4", "T5"}:
            row["condition"] += "; 보존값이며 R4는 최종 소스·물리 비간섭 동일성"
    return metrics


def curate(loaded: dict) -> dict:
    protocol, cal = loaded["protocol"], loaded["calibration"]
    if protocol.get("schema") != "R4_PROTOCOL.v1" or len(protocol.get("candidate_grid", [])) != 4:
        raise ValueError("R4 protocol is malformed")
    reseal = loaded["holdout_reseal"]
    if (reseal.get("schema") != "R4_HOLDOUT_RESEAL.v1"
            or reseal.get("invalidated_seed") != 371501
            or reseal.get("replacement_seed") != 371599
            or reseal.get("all_other_protocol_gates_and_design_ranges_unchanged") is not True):
        raise ValueError("holdout reseal record is malformed")
    if cal.get("schema") != "R4_CALIBRATION.v1" or cal.get("audit_passed") is not True \
            or len(cal.get("audit", [])) != 128:
        raise ValueError("normal-only calibration/audit is incomplete")
    revision = loaded.get("registered_revision")
    if revision:
        revision_protocol = revision["PROTOCOL_R4_REDESIGN_W40.json"]
        if (revision_protocol.get("schema") != "R4_REDESIGN_W40.v1"
                or revision_protocol.get("variant") != "W40_ACTIVE"
                or revision_protocol.get("redesign_number") != 1
                or revision_protocol.get("original_candidates_literal_store_free") is not False
                or revision_protocol.get("revalidation", {}).get("all_numeric_gates_unchanged") is not True
                or revision_protocol.get("revalidation", {}).get("rows_per_mode") != 38
                or revision_protocol.get("revalidation", {}).get("full_loop") is not True
                or revision_protocol.get("redesigns_used", {}).get("W40") != 1
                or revision_protocol.get("first_holdout_generated") is not False
                or revision_protocol.get("official_runs") != 0):
            raise ValueError("registered W40 redesign protocol is malformed")
        eq = revision["R4_REDESIGN_EQUIVALENCE.json"]
        validation_document = revision["R4_REVALIDATION_REVIEW.json"]
        validation_stage = "REVALIDATION"
        redesign = revision["R4_REDESIGN_DECISION_V2.json"]
        if (redesign.get("schema") != "R4_REDESIGN_DECISION.v2"
                or redesign.get("redesigns_used") != 1
                or not any(row.get("family") == "W40" and row.get("used") is True
                           for row in redesign.get("decisions", []))):
            raise ValueError("causal W40 redesign decision is incomplete")
    else:
        revision_protocol = None
        eq = loaded["equivalence"]
        validation_document = loaded["validation_review"]
        validation_stage = "VALIDATION"
        redesign = loaded["redesign"]
    if not str(eq.get("schema", "")).startswith("R4_") or eq.get("passed") is not True:
        raise ValueError("R4 equivalence proof failed")
    stages = {"VALIDATION": stage_map(validation_document, validation_stage)}
    stages.update({stage: stage_map(loaded[stage.lower() + "_review"], stage)
                   for stage in STAGES if stage != "VALIDATION"})
    original_validation = stage_map(loaded["validation_review"], "VALIDATION")
    selection = loaded["selection"]
    final = selection.get("chosen")
    if selection.get("schema") != "R4_PREFREEZE_DECISION.v1" or set(final or {}) != set(MODES):
        raise ValueError("R4 selection is incomplete")
    official = _official_context(loaded["official"])
    effective = {mode: loaded["official"]["architectures"][mode].get("effective_selection")
                 for mode in MODES}
    if any(value not in set(VARIANTS) | {"R1"} for value in effective.values()):
        raise ValueError("official effective selection is noncanonical")
    if effective != final:
        # An immutable official failure may only fall back to R1.
        if any(effective[m] not in (final[m], "R1") for m in MODES):
            raise ValueError("official result promoted an unselected candidate")
        final = effective

    candidates = []
    perturb_whole = []
    for mode in MODES:
        for variant in VARIANTS:
            rows = {stage: stages[stage][mode, variant] for stage in STAGES}
            audit = [r for r in cal["audit"] if r["architecture"] == mode and r["variant"] == variant]
            literal_compliant = revision is not None and variant == "W40_ACTIVE"
            failure = _first_failure(rows)
            if not literal_compliant:
                failure = "공유 상태 저장 금지 위반" + ("; " + failure if failure else "")
            candidates.append({
                "방식": "A 서보" if mode == "servo" else "B 토크", "후보": variant,
                "정상 추가 FA": f"표본 {sum(r['extra_true_samples'] for r in audit)}, 상승 {sum(r['extra_rises'] for r in audit)}",
                "validation": f"+{rows['VALIDATION']['review']['additional_detected']}, 손실 {rows['VALIDATION']['review']['lost_R1']}",
                "development": f"+{rows['DEVELOPMENT']['review']['additional_detected']}, 손실 {rows['DEVELOPMENT']['review']['lost_R1']}",
                "D4": f"+{rows['D4']['review']['additional_detected']}, 손실 {rows['D4']['review']['lost_R1']}",
                "섭동 무충돌 FA [28조건]": f"{rows['PERTURB']['review']['reference_FA_R1']}→{rows['PERTURB']['review']['reference_FA_candidate']}",
                "p99/최대 [µs]": f"{rows['VALIDATION']['timing_p99_us']:.3f}/{rows['VALIDATION']['timing_max_us']:.3f}",
                "판정": "채택" if final[mode] == variant else "탈락",
                "실패 관문": failure or "없음",
            })
            perturb_whole.append({
                "방식": "A 서보" if mode == "servo" else "B 토크",
                "후보": variant,
                "전체 34조건 충돌 외 FA":
                    f"{rows['PERTURB']['review']['outside_event_FA_R1']}→{rows['PERTURB']['review']['outside_event_FA_candidate']}",
            })
    phase_rows = []
    for mode in MODES:
        for variant in VARIANTS:
            phase_rows.append({"방식": "A 서보" if mode == "servo" else "B 토크",
                               "후보": variant,
                               "development R1→후보": _phase_text(stages["DEVELOPMENT"][mode, variant]),
                               "D4 R1→후보": _phase_text(stages["D4"][mode, variant])})

    if not str(redesign.get("schema", "")).startswith("R4_REDESIGN_DECISION") \
            or redesign.get("redesigns_used") not in (0, 1, 2, 3):
        raise ValueError("redesign decision is incomplete")
    redesign_reasons = {
        "W40": "기존 모델의 임시값을 같은 값으로 다시 저장하는 위치를 찾았다. 두 방식에서 추가 검출과 정상 오감지 보존을 보인 W40만 한 번 고쳐 중복 저장을 없앴다. 가중치·통계량·임계값은 바꾸지 않았다.",
        "CUSUM_ACTIVE": "같은 저장 문제를 고치지 않아 채택 대상에서 제외했다. 토크의 0.5 설정은 오감지·추가 검출 관문도 실패했다. 두 방식 모두 사전 수치 관문을 넘긴 더 단순한 W40에 수정을 집중했다.",
        "CUSUM_PHASE": "같은 저장 문제를 고치지 않았고 정상·충돌 외 오감지 관문도 실패했다. 더 단순한 W40만 고쳤으며, 이 설정의 후속 수치로 탈락을 취소하지 않았다.",
    }
    redesign_rows = [{"family": r["family"], "사용": "예" if r["used"] else "아니오",
                      "당시 사유": redesign_reasons[r["family"]]}
                     for r in redesign.get("decisions", [])]
    if revision:
        redesign_rows.insert(0, {"family": "초기 자동 기록", "사용": "불완전",
            "당시 사유": "처음 자동 판정은 중복 저장 문제를 발견하지 못해 재설계 0회로 기록했다. 그 기록은 그대로 두고, 문제를 찾은 시점의 새 결정 기록으로 바로잡았다."})
    holdout_review = loaded["holdout_review"]
    if holdout_review.get("schema") != "R4_HOLDOUT_REVIEW.v1" \
            or holdout_review.get("second_holdout_permitted") is not False:
        raise ValueError("holdout review is incomplete")
    holdout_rows = loaded["holdout"].get("results", [])
    holdout_summary = []
    for mode, label in (("servo", "A 서보"), ("torque", "B 토크")):
        rows = [row for row in holdout_rows if row.get("architecture") == mode]
        collisions = [row for row in rows if row.get("kind") == "collision"]
        references = [row for row in rows if row.get("kind") == "reference"]
        review = holdout_review.get("architectures", {}).get(mode, {})
        if (len(rows) != 38 or len(collisions) != 30 or len(references) != 4
                or review.get("frozen_selection") != "R1"
                or review.get("review") is not None
                or review.get("efficacy_claim") is not False):
            raise ValueError("holdout is not an R1-only preservation batch")
        holdout_summary.append({
            "방식": label,
            "충돌 검출": f"{sum(row.get('collision_outcome') == 'detected' for row in collisions)}/30",
            "정상 4조건 FA": sum(int(row.get("false_alarms", 0)) for row in references),
            "전체 충돌 외 FA": sum(int(row.get("false_alarms", 0)) for row in rows),
            "용도": "R1 보존 확인; W40 미실행",
        })
    integrity = loaded["integrity"]
    if integrity.get("passed") is not True:
        raise ValueError("R4 report integrity is not passing")
    metrics = _metric_rows(loaded["r3_metrics"], final,
                           loaded["perturb_review"], loaded["d4_review"])
    w40_d4 = {mode: stages["D4"][mode, "W40_ACTIVE"]["review"]["additional_detected"]
              for mode in MODES}
    w40_perturb = {mode: stages["PERTURB"][mode, "W40_ACTIVE"]["review"]
                   for mode in MODES}
    if final == {"servo": "R1", "torque": "R1"}:
        conclusion = ("새 채택 0개, 최종 3파일은 R1과 바이트가 같다. 수정 W40은 D4에서 "
                      f"A +{w40_d4['servo']}, B +{w40_d4['torque']} 검출했지만 무충돌 섭동 FA가 "
                      f"A {w40_perturb['servo']['reference_FA_R1']}→{w40_perturb['servo']['reference_FA_candidate']}, "
                      f"B {w40_perturb['torque']['reference_FA_R1']}→{w40_perturb['torque']['reference_FA_candidate']}로 "
                      "늘어 탈락했다. 모든 후보는 동결 전에 탈락했고 새 holdout은 R1 보존만 확인했다.")
    else:
        conclusion = ("A 서보는 " + final["servo"] + ", B 토크는 " + final["torque"] +
                      "를 최종 방식으로 유지한다.")
    cause = loaded["cause_analysis"]
    if len(cause["score_families"]) != 7:
        raise ValueError("diagnostic score family count differs")
    cause_summary = {}
    for mode in MODES:
        values = cause["modes"][mode]
        cause_summary.update({
            "gate_hidden_" + mode: values["learned_gate"]["misses_gate_off_entire_physical_pulse"],
            "mean40_any_" + mode: values["global_headroom"]["xy_mean40_active"]["miss_extra_any"],
            "mean40_four_" + mode: values["global_headroom"]["xy_mean40_active"]["miss_extra_four"],
            "all_scores_below_" + mode: values["miss_union"]["all_families_below_global_reference_maximum"],
        })
    return {
        "schema": "R4_REPORT_CURATION.v1", "final": final, "conclusion": conclusion,
        "candidates": candidates, "perturb_whole": perturb_whole,
        "phases": phase_rows, "metrics": metrics,
        "holdout_summary": holdout_summary,
        "redesign": redesign_rows, "official": official,
        "perturb_findings": _diagnostic_findings(loaded),
        "timing": _timing_summary(loaded),
        "r3_r1_timing": _r3_r1_timing(loaded["r3_readme"]),
        "requirements": _requirements_summary(loaded["requirements"]),
        "causal_repair": ({
            "applied": True,
            "original_w40_gains": {mode: original_validation[mode, "W40_ACTIVE"]["review"]["additional_detected"] for mode in MODES},
            "revalidation_w40_gains": {mode: stages["VALIDATION"][mode, "W40_ACTIVE"]["review"]["additional_detected"] for mode in MODES},
            "revalidation_w40_max_us": {mode: stages["VALIDATION"][mode, "W40_ACTIVE"]["timing_max_us"] for mode in MODES},
            "original_literal_store_free": False,
            "repaired_family": "W40",
            "other_families_adoption_eligible": False,
            "parameters_changed": False,
            "redesign_build_bound": True,
            "redesign_normal_audit_bound": True,
            "original_online_rows": 304,
            "repair_online_rows": 76,
        } if revision else {"applied": False}),
        "normal_audit": {"rows": len(cal["audit"]),
                         "extra_samples": sum(r["extra_true_samples"] for r in cal["audit"]),
                         "extra_rises": sum(r["extra_rises"] for r in cal["audit"])},
        "equivalence": {"passed": eq["passed"], "rows": len(eq.get("results", [])),
                        "fresh_full_loop_simulations": eq.get("fresh_full_loop_simulations"),
                        "fresh_w40_rows": eq.get("fresh_full_loop_W40_rows"),
                        "fresh_representative_rows": eq.get("fresh_representative_rows"),
                        "reused_rows": eq.get("unchanged_source_reused_rows")},
        "reseal": {"invalidated_seed": reseal["invalidated_seed"],
                   "replacement_seed": reseal["replacement_seed"],
                   "pre_generated_designs": 2, "actual_episodes_before_reseal": 0,
                   "revalidation_exclusion_masses":
                       revision_protocol.get("new_holdout_exclusion_masses", [])
                       if revision_protocol else []},
        "literature": {"opened": 10, "full_text": 7, "abstract_bibliography": 2,
                       "publisher_metadata": 1},
        "cause": cause_summary,
        "limitations": [
            "D4는 이미 후보 설계에 사용한 기존 관측 자료의 재생이며 독립 효능 자료가 아니다.",
            "새 잡음 9조건을 포함한 섭동은 유한 분석이며 실기 안전이나 확률 보장이 아니다.",
            "p99 800 µs는 선택 관문이고 최대 1 ms는 관측 목표다. 모든 초과 표본을 보존했다.",
            "서로 독립이 아닌 정상 표본의 최대값으로 보정했으므로 정해진 확률로 오감지를 막는다고 보장하지 않는다.",
            "감지기 비간섭 검사는 유한 조건이며 전역 안정성 보장이 아니다.",
        ],
    }


def build_manifest(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    provenance = {"r3_metrics": "literal_copy", "r3_readme": "link_path_only",
                  "r3_presentation": "reused_exact_source"}
    artifacts = {key: artifact_entry(root, relative, provenance.get(key, "fresh"))
                 for key, relative in ARTIFACTS.items()}
    if (root / REVISION_ARTIFACTS[0]).is_file():
        artifacts["registered_revision"] = [artifact_entry(root, relative)
                                                for relative in REVISION_ARTIFACTS]
    holder = {"schema": "R4_REPORT_MANIFEST.v1", "status": "frozen",
              "artifacts": artifacts}
    loaded = verify_artifacts(holder, root)
    holder["curation"] = curate(loaded)
    validate_manifest(holder, root)
    return holder


def validate_manifest(manifest: dict, root: Path = ROOT) -> dict:
    if manifest.get("schema") != "R4_REPORT_MANIFEST.v1" or manifest.get("status") != "frozen":
        raise ValueError("unexpected R4 report manifest schema/status")
    loaded = verify_artifacts(manifest, Path(root))
    expected = curate(loaded)
    if manifest.get("curation") != expected:
        raise ValueError("manifest curation differs from frozen evidence")
    from tools.r4_report_links import verify_archive_link_proof
    verify_archive_link_proof(root, loaded["document_link_repairs"])
    return {"manifest": manifest, "artifacts": loaded, "curation": expected,
            "root": Path(root).resolve()}


def _cell(value) -> str:
    if isinstance(value, bool): return "예" if value else "아니오"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(columns, rows) -> str:
    headings = {"family": "계열", "validation": "사전 검증", "development": "개발",
                "development R1→후보": "개발 R1→후보"}
    return "\n".join(["| " + " | ".join(headings.get(c, c) for c in columns) + " |",
                      "|" + "|".join("---" for _ in columns) + "|"] +
                     ["| " + " | ".join(_cell(row.get(c, "")) for c in columns) + " |"
                      for row in rows])


def metric_table(rows) -> str:
    display = [{"항목": r["id"] + ": " + r["name"], "조건·단위": r["condition"] + " / " + r["unit"],
                "기준": r["baseline"], "v1": r["v1"], "R1": r["r1"], "R4": r["r4"]}
               for r in rows]
    return table(["항목", "조건·단위", "기준", "v1", "R1", "R4"], display)


GLOSSARY = (
    "W40은 현재까지의 40 ms 후행 평균으로 짧은 오차를 모아 보는 감지 방법이다.",
    "CUSUM은 정상 수준보다 큰 오차를 시간에 따라 누적하는 변화 감지 방법이다.",
    "D4는 질량·무게중심·외력을 넓힌 기존 개발 평가로, 방식당 385조건 중 충돌은 180조건이다.",
    "holdout은 후보 선정에 쓰지 않고 마지막에 한 번 확인하는 별도 평가 자료다.",
    "동결은 별도 평가 전에 코드·보정값·최종 선택을 고정하는 절차다.",
)
NOT_ATTEMPTED = (
    "토크 추적 feedforward는 구현하지 않았고 효과는 모른다. 감지기의 추가 검출 여지를 먼저 시험해 2순위로 남겼다.",
    "공식 8 kg 미검출 3조건에 W40을 실행하지 않았다. 집기 10·15 N과 놓기 10 N은 최종 R1의 미검출이며 W40의 개선 여부는 모른다.",
)


def official_control_rows(c: dict) -> list[dict]:
    rows = []
    for mode, label in (("servo", "A 서보"), ("torque", "B 토크")):
        for mass in ("0.0", "4.0", "8.0"):
            cells = c["official"]["control"][mode][mass]
            if len(cells) != 10:
                raise ValueError("official control must retain ten literal evaluator cells")
            rows.append({"방식": label, "질량 [kg]": mass,
                         "TCP RMS/최대 [mm]": cells[0] + "/" + cells[1],
                         "놓기 계단 [N·m]": cells[4], "포화 [%]": cells[2],
                         "FA 집기/놓기/기타": "/".join(cells[6:9]), "상태": cells[9]})
    return rows


def _requirements_note(c) -> str:
    r = c["requirements"]
    return (f"기존 가이드 33항목의 {r['충족']}충족/{r['부분']}부분/{r['판정 불가']}판정 불가와 "
            "입력·형식·무결성 8통과 판정은 최종 3파일이 R1과 같아 유지한다. "
            "이는 모든 성능 목표의 충족이나 새 후보의 성공을 뜻하지 않는다. "
            "판정 불가에는 선택하지 않은 권고 항목과 비공개 평가도 포함된다.")


def _timing_note(c) -> str:
    a, b = (c["r3_r1_timing"][mode] for mode in MODES)
    return ("같은 새 사전 재검증 38조건/방식의 제어기+감지기 합산 벽시계 시간이다. "
            "각 소스는 228,000주기이며, p99는 주기 시간의 99백분위값 중 가장 나쁜 조건의 값이다. "
            "p99 800 µs 관문은 통과했지만 A W40·B R1의 관측 최대 1 ms 목표는 미달이다. "
            "직렬 실행 순서와 OS 부하의 영향이 있어 순수 알고리즘 속도 비나 시간 상한 보장으로 쓰지 않는다.\n\n"
            f"**보존된 R3 표:** 서보 R1은 p99 {a['p99_us']} µs, 최대 {a['max_ms']} ms, "
            f"1 ms 초과 {a['outlier_count']}주기다. 토크 R1은 p99 {b['p99_us']} µs, "
            f"최대 {b['max_ms']} ms, 1 ms 초과 {b['outlier_count']}주기다. "
            "**R1은 최대 1 ms 목표 미달이다.** R3 표의 원래 표시값을 그대로 인용했으며 재계측하지 않았다. "
            "위 38조건 표와 평가 범위가 다르고 이 유한 표로 과거 미달이 해소됐다고 주장하지 않는다.")


def _incident(c) -> str:
    masses = c.get("reseal", {}).get("revalidation_exclusion_masses", [])
    exclusion = (" 재검증에서 쓴 질량 " + ", ".join(str(value) for value in masses) +
                 " kg은 동결 전에 새 별도 평가의 제외 목록에 추가했다.") \
                if masses else ""
    return ("검사 코드의 실수로 원래 별도 평가 시드 371501의 조건을 동결 전에 메모리에서 "
            "두 번 만들었다. 조건을 출력·저장하거나 시뮬레이션을 실행하지는 않았지만 "
            "사전 생성 금지 위반으로 기록했다. 후보 충돌 결과를 보기 전에 그 시드를 폐기하고 "
            "새 시드 371599로 바꿨다. 관문과 조건 범위는 그대로다. 실제 새 holdout은 "
            "코드와 선택을 동결하고 실행 시작을 한 번만 기록한 뒤 1회 진행했다." + exclusion)


def _repair(c) -> str:
    repair = c.get("causal_repair", {})
    if not repair.get("applied"):
        return "등록된 재설계는 없다."
    old, new = repair["original_w40_gains"], repair["revalidation_w40_gains"]
    return ("처음 만든 감지기는 기존 모델의 임시값을 같은 값으로 다시 저장했다. "
            "물리 결과는 같았지만 공유 상태에 쓰지 말라는 규칙을 어겼다. 당시 W40은 "
            f"추가 검출 A {old['servo']}/B {old['torque']}건과 정상 오감지 보존을 보여, "
            "이 방법만 재설계 1회를 써 중복 저장을 없앴다. 가중치·통계량·임계값은 그대로다. "
            f"새 조건의 재검증에서 추가 검출은 A {new['servo']}/B {new['torque']}건이었다. "
            "CUSUM 3설정은 고치지 않았고 계속 탈락이다.")


def _audit_summary() -> str:
    return """## 평가기 점검과 미검출 원인

R4를 끝낸 뒤 평가기 자체에 문제가 없는지 점검하고, 공식 8 kg 미검출 3조건의 원인을 측정했다. **확인한 범위에서 R1 유지와 R4 탈락을 뒤집을 근거는 없었고, 공식 수치를 틀리게 만든 버그도 찾지 못했다. 미검출은 평가기 오류가 아니라 R1 감지기의 세 분기가 모두 조건을 못 채운 결과다.** 제어·감지 코드, 임계값, 공식 표는 바꾸지 않았다.

### 평가기 점검

공식 평가 코드와 우리 분석 도구를 읽고, 기대값을 먼저 적어 둔 합성 입력으로 판정 함수를 시험했다. 공식 실행이 제출 파일을 평가했다는 것은 파일 해시와 출력으로 확인했다(8 kg A RMS 1.17 mm, 키트 기준 제어기 6.14 mm).

| 찾은 것 | 크기 | 영향 |
|---|---|---|
| RMS·최대: 적분 후 상태를 한 주기 전 기준과 비교 | 시각을 맞추면 R1 RMS가 0.29~0.45 mm 낮아짐 (R3 검증 자료, 비공식 1.27~7.17 kg) | TCP 1.05 비교 16개의 판정 변화 0. 검증 기록상 TCP 지표만으로 탈락한 R2·R3 후보 없음. 의도 여부는 가이드에 없어 공식 수치는 그대로 둠 |
| 음수 지연: 툴 접촉과 인공 충돌이 한 실행에 있으면 이른 경보가 충돌 검출로 계산됨 | 합성 입력 −200 ms | 공식 평가는 위치 오차를 넘기지 않아 발생하지 않음. 점검한 저장 자료에는 초기 진단 1건(−155 ms)뿐이고 당시 구분 불가로 처리함 |
| 공식 "계산 us": 제어기 중앙값만 측정 | 감지기와 최악값 제외 | 1 ms 판단에는 아래 [합산 시간 표](#계산-시간과-미달)를 쓴다 |
| R4 판정 도구: 출처 ID와 섭동 값을 짝 검사에 넣지 않음 | 잘못 묶여도 통과 가능 | 현재 34행은 모두 일치, 앞으로의 방어 공백이며 고치지 않았다 |

- 공식 잡음 시드는 실행마다 같다. R3와 R4 공식 결과가 같은 것은 결정성을 보여 줄 뿐 독립 반복이 아니다.
- FA 숫자는 표마다 집계가 다르다. 공식 충돌 표는 힘별 실행 중 최대, R4 섭동 표는 합계, 채택 관문은 조건별 비증가다.
- W40 탈락은 점검한 20 ms 경보 병합에서도 유지됐다. 20 ms 안에 다시 켜진 경보를 하나로 합쳐도 A 6→9, B 9→14(R1 9)로 탈락이다.
- 하지 않은 것: `evaluate.py` 전체를 더미 감지기로 돌리지 않고 판정 함수만 호출했다. 실행 순서 교환, R1~R4 전체 재채점, 실기 확인도 하지 않았다.

### 공식 8 kg 미검출 원인 측정

최종 R1을 공식과 같은 8 kg 조건에서 8회(방식별 무충돌, 1.42 s 10·15 N, 3.86 s 10 N) 돌려 신호를 기록했다. 공식 평가가 아닌 사후 진단이다. 판정 규칙과 가설을 실행 전에 고정했고 재실행은 0회다. 결과는 공식 표의 자릿수와 같았고, 기록이 결과를 바꾸지 않는다는 것은 저장된 비공식 6행과의 비트 일치로 확인했다. 원시 신호는 공개본에 넣지 않았으므로 아래 표는 공개본만으로 재현할 수 없다.

R1 경보는 세 분기 중 하나만 켜져도 울린다. 주 분기는 점수가 1을 넘은 채 4표본 연속이어야 한다. 빠른 분기는 그립이 꺼진 구간에서만 쓰고 점수가 1을 넘은 채 8표본 연속이어야 한다. 학습 분기는 그립 중 작업대 높이 ±35 mm 밖에서만 켜지고 오차 평균(EMA)이 임계를 넘은 채 4표본 연속이어야 한다. 펄스는 외력이 들어가는 120 ms, 평가창은 공식 판정이 검출을 인정하는 충돌 뒤 320 ms다.

| 조건 (A/B) | 주 분기: 펄스 중 점수 (무충돌) | 빠른 분기: 평가창 점수 (무충돌) | 학습 분기 |
|---|---|---|---|
| 집기 10 N | 0.80/0.74 (0.80/0.71) | 그립 중 꺼짐 | 펄스 내내 꺼짐, 켜진 뒤 임계 미달 |
| 집기 15 N | 1.09/1.06 (0.80/0.71), 연속 1표본 | 그립 중 꺼짐 | 펄스 내내 꺼짐, 켜진 뒤 임계 미달 |
| 놓기 10 N | 0.99/0.96 (0.55/0.51) | 1.45/1.55 (1.86/1.70), 연속 1표본 | 평가창 내내 꺼짐 |

- 힘은 표기대로 들어갔고(10/15 N, 1.0/1.5 N·s) 8회 모두 경보가 켜지지 않아 판독 오류도 아니다. 펄스 동안 작업대가 무충돌보다 더 받는 수평 반력의 충격량은 주입 충격량의 집기 약 0.2~0.4 %, 놓기 약 7~12 %다. 이는 부호 있는 추가 반력비이지 에너지 흡수율은 아니다. 추정 수평 힘에는 주입력의 1.17~1.39배 차이가 나타난다. 이 차이는 무충돌 실행과 비교한 값이라 감지기가 실시간으로 쓸 수는 없다.
- 놓기 10 N의 빠른 분기 임계 초과 자체는 충돌의 독립적인 증거가 아니다. 같은 창의 무충돌 최대가 더 크다.
- 임계를 낮추거나 게이트를 그냥 여는 수정만으로 해결됐다고 할 수 없다. 집기 평가창에서 무충돌 주 분기 점수가 이미 0.99/1.05까지 오른다. B는 1을 넘어도 연속 조건을 못 채워 경보가 나지 않는다. 학습 분기를 평가창 내내 켜면 6건 모두 경보가 나지만 무충돌도 0~1 ms 차이로 경보를 내어 구분할 수 없고, 전 구간에서 켜면 무충돌 FA가 0회에서 3회로 는다.
- 다른 감지기로 구분할 수 있는지는 판정 불가다. 잡음 시드 1개와 R4의 일곱 진단 점수만 봤다.
- 이 신호를 임계값, 학습, 후보 선정에 쓰지 않았다. 앞으로 이 신호를 보고 만든 방법은 공식 조건을 독립 검증으로 쓸 수 없다. W40은 이 조건에서 돌리지 않았다."""


def render_readme(context: dict) -> str:
    c = context["curation"]
    final = c["final"]
    media = """[![07 서보 A · 기준/v1/R1/R2 네 열 비교](artifacts/media/07_r2_servo_comparison.gif)](artifacts/media/07_r2_servo_comparison.mp4)
[![08 토크 B · 기준/v1/R1/R2 네 열 비교](artifacts/media/08_r2_torque_comparison.gif)](artifacts/media/08_r2_torque_comparison.mp4)

[![05 서보 A · 기준/v1/R1 비교](artifacts/media/05_v2_servo_transport_collision.gif)](artifacts/media/05_v2_servo_transport_collision.mp4)
[![06 토크 B · 기준/v1/R1 비교](artifacts/media/06_v2_torque_transport_collision.gif)](artifacts/media/06_v2_torque_transport_collision.mp4)"""
    adoption = any(value != "R1" for value in final.values())
    media_note = ("R4 채택 방식의 새 비교 영상은 아래 기존 영상 뒤에 이어 제공한다."
                  if adoption else "R4는 새 방법을 채택하지 않아 비교 영상을 다시 만들지 않았다.")
    if adoption:
        adopted_media = "\n\n[![09 R4 서보 A 비교](artifacts/media/09_r4_servo_comparison.gif)](artifacts/media/09_r4_servo_comparison.mp4)\n[![10 R4 토크 B 비교](artifacts/media/10_r4_torque_comparison.gif)](artifacts/media/10_r4_torque_comparison.mp4)"
    else:
        adopted_media = ""
    return f"""# 팔레타이징 하중 이양 · 순차 충돌 감지 R4

A=서보, B=토크, FA=오감지다. 기준=배포 키트, v1=탄성·하중 이양 제어와 기하 감지,
R1=v1 제어와 정상-only ridge/EMA 감지, R4=고정 정상 보정 순차 감지의 최종 판정이다.

**{c['conclusion']}**

[상세 결과](docs/R4_RESULTS.md) · [방법과 가정](docs/R4_METHODS.md) · [문헌 조사](docs/R4_LITERATURE.md) · [요구사항 점검](docs/REQUIREMENTS_AUDIT.md) · [재현 절차](docs/R4_REPRODUCE.md) · [발표 PDF](results/presentation.pdf) · [R3 보존본](docs/R3_README.md) · [평가기 점검](#평가기-점검과-미검출-원인)

## 용어

{chr(10).join('- ' + text for text in GLOSSARY)}

## 영상으로 먼저 보기

{media_note} 아래 07/08은 R2 당시 기준/v1/R1/R2 네 열이고, 05/06은 기준/v1/R1 세 열이다.

{media}{adopted_media}

## T1–T6: 기준/v1/R1/R4

기준/v1/R1은 보존된 비공식 분석 집계다(공식 evaluator가 아님). T3의 네 열은 R2와 같은 34조건 보존값이며,
R4의 새 섭동을 네 버전 모두 다시 실행한 독립 효능 표가 아니다.
추가 소수 자릿수는 원래 분석 원자료의 정밀도이며 공식 출력의 자릿수를 늘린 것이 아니다.

{metric_table(c['metrics'])}

## 후보 판정

{table(['방식','후보','정상 추가 FA','validation','development','D4','섭동 무충돌 FA [28조건]','p99/최대 [µs]','판정','실패 관문'], c['candidates'])}

전체 34조건의 충돌 외 FA는 아래처럼 별도로 보존한다. W40의 전체값은 A 6→9,
B 10→18이며, 후보 판정의 주 관문인 무충돌 28조건 값과 합치지 않았다.

{table(['방식','후보','전체 34조건 충돌 외 FA'], c['perturb_whole'])}

## 새 holdout: R1 보존 전용

모든 후보가 동결 전에 탈락했으므로 W40 holdout은 실행하지 않았다.

{table(['방식','충돌 검출','정상 4조건 FA','전체 충돌 외 FA','용도'], c['holdout_summary'])}

## 구현에서 발견한 문제와 수정

{_repair(c)}

[공유 임시값과 재설계의 세부 설명](docs/R4_METHODS.md#비간섭과-안정성의-범위)을 따로 적었다.

## 별도 평가 조건을 미리 만든 사고와 조치

{_incident(c)}

## 마지막 공식 회귀: 최종 R1

공식 결과는 R1 보존을 확인한 마지막 1회 회귀 검사다. 후보 선정이나 임계값 보정에 쓰지 않았다. 이 회귀 뒤에 같은 조건에서 R1 신호만 따로 측정했다. 공식 평가가 아니며 아래 표의 값을 바꾸지 않는다([측정 내용](#공식-8-kg-미검출-원인-측정)).

### 공식 0/4/8 kg 무충돌

{table(['방식','질량 [kg]','TCP RMS/최대 [mm]','놓기 계단 [N·m]','포화 [%]','FA 집기/놓기/기타','상태'], official_control_rows(c))}

### 공식 8 kg 충돌

{table(['방식','시각 [s]','10/15/40/100 N·FA·최소'], c['official']['eight_kg'])}

앞 네 값은 각 힘의 검출 지연 [ms], 뒤는 충돌 외 FA [회]와 최소 검출력 [N]이다.
공식 문자열은 evaluator의 원래 자릿수를 그대로 쓴다. **B 8 kg 3.0 s 충돌 세트의 추가 오감지 1회는 R1부터 동일하게 남아 있다.**
[공식 원자료](results/v2_protocol/R4_OFFICIAL_REGRESSION.json)를 함께 확인한다.

{_audit_summary()}

## 계산 시간과 미달

{table(['방식·소스','최악 행 p99 [µs]','관측 최대 [ms]','1 ms 초과 주기'], c['timing'])}

{_timing_note(c)} [보존된 R3 시간 표](docs/R3_README.md#계산-시간과-미달)를 함께 본다.

- G4 지속 지지(평균 ≥50 %mg, 최소 ≥25 %mg, v1 대비 +25 %p, 놓기 힘 ≤1.1 mg)는 완화하지 않았고 미달이다.
- 공식 8 kg 집기 10·15 N과 놓기 10 N 미검출([원인 측정](#공식-8-kg-미검출-원인-측정)), 보존 무충돌 격자의 서보 포화 2/189조건, R1 섭동 FA A 6/B 9가 남는다.
- 수정 W40도 섭동 FA 증가로 탈락했다. 새 방법의 전체 우위·실기 안전·엄격 최대 1 ms를 주장하지 않는다.

## 요구사항 점검

{_requirements_note(c)} [가이드 요구와 자체 목표의 구분](docs/REQUIREMENTS_AUDIT.md)을 보존했다.

## 이번에 시도하지 않은 것

{chr(10).join('- ' + text for text in NOT_ATTEMPTED)}

문헌 10편을 직접 확인해 순차 누적과 고정 정상 보정을 골랐다. 온라인 ACI/AgACI,
RL, 리그룹 재시도는 이번 후보에서 제외했다. 자세한 확인 범위와
논문에서 바꾼 점은 방법·문헌 문서에 구분했다.
"""


def render_results(context: dict) -> str:
    c = context["curation"]
    official = c["official"]
    lines = ["# R4 결과", "", "A=서보, B=토크, FA=오감지다.", "",
             "## 결론", "", c["conclusion"], "",
             "## 후보 8개: 모든 행 보존", "",
             table(["방식","후보","정상 추가 FA","validation","development","D4","섭동 무충돌 FA [28조건]","p99/최대 [µs]","판정","실패 관문"], c["candidates"]), "",
             "후보는 방식별로 독립 판정했다. 앞 단계에서 탈락한 설정은 뒤 단계의 진단 수치로 되살리지 않았다.", "",
             "### 전체 34조건 충돌 외 FA", "",
             "주 관문인 무충돌 28조건과 합치지 않은 별도 집계다. W40은 A 6→9, B 10→18이다.", "",
             table(["방식","후보","전체 34조건 충돌 외 FA"], c["perturb_whole"]), "",
             "### W40 오감지가 늘어난 조건", "",
             "오감지가 늘어난 행은 전부 강성을 ±20 % 바꾼 조건이다. 강성만 바꾼 조건과 강성·마찰을 함께 바꾼 조합을 나눴다. 새 잡음 9조건에서는 증가가 없었다. 기존 잡음 9조건과 새 잡음 9조건은 서로 다른 자료다.", "",
             table(["방식","증가 행","강성 단독","강성·마찰 조합","새 잡음 증가/조건"], c["perturb_findings"]), "",
             "새 잡음은 결합 자료의 25~33행, 시드 371401~371403이다. 이는 원인별 분포를 보여 줄 뿐 강성 변경이 유일한 물리 원인이라는 증명은 아니다. [서보 원자료](../results/v2_protocol/R4_PERTURB_W40_ACTIVE_servo.json)와 [토크 원자료](../results/v2_protocol/R4_PERTURB_W40_ACTIVE_torque.json)의 모든 행을 보존했다.", "",
             "## 구간별 검출", "",
             table(["방식","후보","development R1→후보","D4 R1→후보"], c["phases"]), "",
             "## T1–T6", "", "기준/v1/R1은 보존된 비공식 분석 집계이며 공식 evaluator 결과가 아니다. T3는 R2와 같은 전체 34조건 충돌 외 FA 보존값이며 새 R4 조건의 네 버전 독립 재실행이 아니다. 추가 소수 자릿수는 원래 분석 원자료의 정밀도다.", "", metric_table(c["metrics"]), "",
             "## 원인 분석과 경험적 상한", "",
             f"R1 D4 놓침 중 높이 조건 때문에 학습 분기가 외력 전체에서 꺼진 경우는 A {c['cause']['gate_hidden_servo']}개, B {c['cause']['gate_hidden_torque']}개였다. 이는 그 분기가 쓰이지 못한 이유이지, 전체 감지기 실패의 단독 원인이라는 증명은 아니다.", "",
             f"놓침 69건 중 A {c['cause']['all_scores_below_servo']}건, B {c['cause']['all_scores_below_torque']}건은 7개 진단 점수 모두의 평가창 최댓값이 각각 정상 189조건의 전역 최댓값 이하였다. 진단 코드의 한 부동소수 간격 높은 경계를 넘지 않은 경우를 원자료와도 대조했다. 경계 설정과 성능 확인에 같은 자료를 쓴 비독립 진단이므로, 일반화 한계나 모든 허용 신호의 원리적 구분 불가능성을 증명하지 않는다.", "",
             f"같은 개발 자료의 정상 최대 경계에서 40 ms 평균은 추가 {c['cause']['mean40_any_servo']}/{c['cause']['mean40_any_torque']}개(A/B), 4연속 조건은 {c['cause']['mean40_four_servo']}/{c['cause']['mean40_four_torque']}개를 분리했다. 같은 자료를 쓴 진단 상한이라 독립 효능이나 일반화 보장이 아니다.", "",
             "## 문헌과 후보", "",
             f"기존 목록과 겹치지 않는 {c['literature']['opened']}편을 확인했다. 원문 {c['literature']['full_text']}편, 기관 초록·서지 {c['literature']['abstract_bibliography']}편, 출판 서지 {c['literature']['publisher_metadata']}편이다. W40과 고정 CUSUM을 구현했고 온라인 적응 보정·RL·리그룹 재시도는 제외했다.", "",
             "## 정상 보정·비간섭", "",
             f"무충돌 검사 {c['normal_audit']['rows']}행에서 추가 경보 표본 {c['normal_audit']['extra_samples']}개, 경보 시작 {c['normal_audit']['extra_rises']}개였다. 기존 R1 전체 경보를 0으로 가정한 것이 아니라 새 분기의 추가 경보만 분리해 검사했다.", "",
             f"감지기 변경 전후 제어·물리값 일치 기록은 {c['equivalence']['rows']}행이며 결과는 {'통과' if c['equivalence']['passed'] else '미달'}다. 새 W40 대표 {c['equivalence'].get('fresh_representative_rows')}행과 변경 없는 CUSUM의 기존 {c['equivalence'].get('reused_rows')}행을 합쳤다. 16행 전부가 새 실행은 아니다. W40은 별도로 새 {c['equivalence'].get('fresh_w40_rows')}조건의 전체 시뮬레이션에서 물리 배열과 감지 판정을 대조했고, 최초 후보 304행 기록도 보존했다. CUSUM은 수치가 같아도 공유 임시값을 다시 저장해 쓰기 금지 규칙을 어겼으므로 계속 탈락이다.", "",
             "## 재설계", "", table(["family","사용","당시 사유"], c["redesign"]), "",
             "위 표는 당시 사유의 한글 요약이다. [당시 결정 원문](../results/v2_protocol/R4_REDESIGN_DECISION_V2.json)은 바꾸지 않았다.", "",
             "### 공유 임시값을 다시 저장한 문제와 수정", "", _repair(c), "",
             "계산 순서·공유 임시값·행 재사용의 [세부 설명](R4_METHODS.md#비간섭과-안정성의-범위)을 별도로 둔다.", "",
             "## 별도 평가 조건을 미리 만든 사고와 조치", "", _incident(c), "",
             "### 새 holdout은 R1 보존 전용", "",
             "모든 후보가 동결 전에 탈락해 W40 holdout은 실행하지 않았다.", "",
             table(["방식","충돌 검출","정상 4조건 FA","전체 충돌 외 FA","용도"], c["holdout_summary"]), "",
             "## 마지막 공식 8 kg 회귀", "",
             "공식 evaluator 문자열은 원래 자릿수를 그대로 보존했다. 공식 결과는 마지막 회귀 확인이며 독립 효능이나 임계값 선정 자료가 아니다.", "",
             table(["방식","시각 [s]","10/15/40/100 N·FA·최소"], official["eight_kg"]), "",
             "각 힘의 검출 지연은 ms, FA는 회, 최소 검출력은 N이다. 문자열 자릿수는 원본 그대로다.", "",
             "B 8 kg 3.0 s 충돌 세트의 추가 오감지 1회는 R1부터 동일하게 남아 있다.", "",
             "### 공식 0/4/8 kg 원문 셀", ""]
    for mode in MODES:
        lines += ["#### " + ("A 서보" if mode == "servo" else "B 토크"), "",
                  table(["질량","원문 evaluator 셀"],
                        [{"질량": mass + " kg", "원문 evaluator 셀": " / ".join(official["control"][mode][mass])}
                         for mass in ("0.0", "4.0", "8.0")]), ""]
    lines += ["## 계산 시간과 미달", "",
              table(["방식·소스","최악 행 p99 [µs]","관측 최대 [ms]","1 ms 초과 주기"], c["timing"]), "",
              _timing_note(c), "",
              "[보존된 R3 시간 표](R3_README.md#계산-시간과-미달)를 함께 본다.", "",
              "G4 지속 지지, 약한 집기·놓기 충돌 감지, 보존 격자의 서보 포화, 섭동 오감지의 미달은 남는다. 최종 3파일은 R1이며 새 후보의 추적 개선이나 실기 안전을 주장하지 않는다.", "",
              "## 요구사항 점검", "", _requirements_note(c), "",
              "[가이드 요구와 자체 목표의 구분](REQUIREMENTS_AUDIT.md)은 기존 기록을 보존한다.", "",
              "## 이번에 시도하지 않은 것", ""] + ["- " + text for text in NOT_ATTEMPTED]
    lines += ["", "## 한계", ""] + ["- " + x for x in c["limitations"]]
    lines += ["", "## 과거 README 링크 경로 정정", "",
              "R1/R2/R3 보존 README의 상대 경로 9/31/20개, 합계 60개만 고쳤다. 링크 경로 외 본문·라벨·수치의 차이는 0이다. 과거 보존 해시·검증 기록은 고치지 않았으며, [URL만 역복원한 대조 증거](../results/v2_protocol/R4_REPORT_LINK_REPAIRS.json)로 이번 문서 변경을 따로 검증한다."]
    lines += ["", "## 원자료", "",
              "- [사전 계획](../results/v2_protocol/PROTOCOL_R4.json), [새 평가 시드 추가 계획](../results/v2_protocol/PROTOCOL_R4_HOLDOUT_RESEAL.json)",
              "- [방법과 가정](R4_METHODS.md), [문헌 조사](R4_LITERATURE.md), [구현 전 원인 분석](R4_CAUSE_ANALYSIS.md)",
              "- [수정 뒤 사전 검증 판정](../results/v2_protocol/R4_REVALIDATION_REVIEW.json), [W40 재검증 A](../results/v2_protocol/R4_REVALIDATION_W40_ACTIVE_servo.json), [W40 재검증 B](../results/v2_protocol/R4_REVALIDATION_W40_ACTIVE_torque.json)",
              "- [W40 재설계 등록](../results/v2_protocol/PROTOCOL_R4_REDESIGN_W40.json), [정정 결정](../results/v2_protocol/R4_REDESIGN_DECISION_V2.json), [제어·물리값 일치 기록](../results/v2_protocol/R4_REDESIGN_EQUIVALENCE.json)",
              "- [최초 사전 검증 기록](../results/v2_protocol/R4_VALIDATION_REVIEW.json)",
              "- [개발 검증 판정](../results/v2_protocol/R4_DEVELOPMENT_REVIEW.json)",
              "- [D4 판정](../results/v2_protocol/R4_D4_REVIEW.json), [섭동 판정](../results/v2_protocol/R4_PERTURB_REVIEW.json)",
              "- [holdout 판정](../results/v2_protocol/R4_HOLDOUT_REVIEW.json), [공식 회귀](../results/v2_protocol/R4_OFFICIAL_REGRESSION.json)", ""]
    return "\n".join(lines)


PDF_VARIANT_LABELS = {
    "W40_ACTIVE": "W40",
    "CUSUM_ACTIVE_K05": "활성\nCUSUM0.5",
    "CUSUM_ACTIVE_K10": "활성\nCUSUM1",
    "CUSUM_PHASE_K05": "구간\nCUSUM0.5",
}


def _pdf_failure_label(value: str) -> str:
    store = "공유 상태 저장 금지 위반" in value
    false_alarm = "오감지 증가" in value
    gain = "추가 검출 기준 미달" in value
    if store and false_alarm and gain:
        return "공유값 저장\nFA/검출 부족"
    if store and false_alarm:
        return "공유값 저장\nFA"
    if store:
        return "공유값 저장"
    if false_alarm:
        return "섭동 FA"
    return value


def _pdf_candidate_rows(rows: list[dict]) -> list[dict]:
    compact = []
    for row in rows:
        p99, maximum = (float(value) for value in row["p99/최대 [µs]"].split("/"))
        compact.append({
            "방식": row["방식"],
            "후보": PDF_VARIANT_LABELS[row["후보"]],
            "사전 검증": row["validation"],
            "개발": row["development"],
            "D4": row["D4"],
            "섭동 FA [28]": row["섭동 무충돌 FA [28조건]"],
            "p99/최대 [µs]": f"{p99:.0f}/{maximum:.0f}",
            "판정": row["판정"],
            "실패 관문": _pdf_failure_label(row["실패 관문"]),
        })
    return compact


def _pdf_phase_cell(value: str) -> str:
    pattern = (r"(?:pick|집기) (\d+→\d+)/(\d+); (?:transport|운반) (\d+→\d+)/(\d+); "
               r"(?:place|놓기) (\d+→\d+)/(\d+)")
    match = re.fullmatch(pattern, value)
    if not match or len({match.group(2), match.group(4), match.group(6)}) != 1:
        raise ValueError("phase cell cannot be compacted without loss")
    return (f"집기 {match.group(1)} · 운반 {match.group(3)}\n"
            f"놓기 {match.group(5)} (각 {match.group(2)})")


def _pdf_phase_rows(rows: list[dict]) -> list[dict]:
    return [{"방식": row["방식"], "후보": PDF_VARIANT_LABELS[row["후보"]],
             "개발 R1→후보": _pdf_phase_cell(row["development R1→후보"]),
             "D4 R1→후보": _pdf_phase_cell(row["D4 R1→후보"])}
            for row in rows]


def _pdf_metric_value(value: str) -> str:
    """Keep each architecture's numeric pair intact at the explicit separator."""
    return str(value).replace("; B ", ";\nB ").replace(" / B ", "\nB ")


def slides(context: dict) -> list[dict]:
    c = context["curation"]
    metric_units = {"T1": "[%mg]", "T2": "[N·m]", "T3": "[mm/N/회]",
                    "T4": "[건/mg]", "T5": "[mm]", "T6": "[건]"}
    metric_rows = [{"항목": r["id"] + " " + metric_units[r["id"]] + ": " + r["name"], "조건": r["condition"],
                    "기준": _pdf_metric_value(r["baseline"]), "v1": _pdf_metric_value(r["v1"]),
                    "R1": _pdf_metric_value(r["r1"]), "R4": _pdf_metric_value(r["r4"])}
                   for r in c["metrics"]]
    official = c["official"]
    holdout = {row["방식"]: row for row in c["holdout_summary"]}
    timing = {row["방식·소스"]: row["관측 최대 [ms]"] for row in c["timing"]}
    return [
        {"title": "R4 결론과 용어", "bullets": [c["conclusion"], "A=서보, B=토크, FA=오감지"] + list(GLOSSARY)},
        {"title": "문헌·원인·경험적 상한", "bullets": ["새 문헌 10편: 원문 7, 기관 초록·서지 2, 출판 서지 1", "높이 조건으로 학습 분기가 외력 전체에서 꺼진 놓침 A 43/B 46", "40 ms 평균의 같은 자료 추가 분리 A 17/B 15; 4연속 A 13/B 14", f"놓침 69건 중 7점수 모두 정상 최대 이하: A {c['cause']['all_scores_below_servo']}/B {c['cause']['all_scores_below_torque']}", "같은 자료를 쓴 진단이며 독립 효능·일반화 보장이 아니다.", NOT_ATTEMPTED[0]]},
        {"title": "후보 8개 판정", "table": (["방식","후보","사전 검증","개발","D4","섭동 FA [28]","p99/최대 [µs]","판정","실패 관문"], _pdf_candidate_rows(c["candidates"])), "font": 10.5,
         "caption": "약칭: W40=W40_ACTIVE; 활성 CUSUM0.5/1=CUSUM_ACTIVE_K05/K10; 구간 CUSUM0.5=CUSUM_PHASE_K05.\n공유값 저장: 기존 제어·모델의 공유 임시값을 다시 저장한 규칙 위반(값이 같아도 금지).\n전체 34조건 충돌 외 FA: W40 A 6→9, B 10→18.", "caption_y": .09},
        {"title": "T1–T3", "table": (["항목","조건","기준","v1","R1","R4"], metric_rows[:3]), "font": 10.5,
         "caption": "보존된 비공식 분석 집계이며 공식 evaluator 결과가 아니다. 추가 자릿수는 원 분석 원자료 정밀도다."},
        {"title": "T4–T6", "table": (["항목","조건","기준","v1","R1","R4"], metric_rows[3:]), "font": 10.5,
         "caption": "보존된 비공식 분석 집계이며 공식 evaluator 결과가 아니다. 추가 자릿수는 원 분석 원자료 정밀도다."},
        {"title": "구간별 검출", "table": (["방식","후보","개발 R1→후보","D4 R1→후보"], _pdf_phase_rows(c["phases"])), "font": 10.5},
        {"title": "구현 문제와 한 번의 수정", "bullets": [
            "최초 감지기는 모델의 임시값을 같은 값으로 다시 저장해 쓰기 금지 규칙을 어겼다.",
            "W40만 재설계 1회를 써 중복 저장을 없앴다. 가중치·통계량·임계값은 그대로다.",
            "새 76조건에서 제어 궤적·물리값·감지 판정의 일치를 확인했다.",
            "CUSUM 3설정은 고치지 않았고, 이후 수치가 좋아도 채택하지 않았다."]},
        {"title": "개발 검증·D4·섭동", "bullets": ["D4는 이미 설계에 사용한 기존 385조건/방식의 관측 재생", "무충돌 28조건: W40 FA A 6→9, B 9→17로 양쪽 탈락", "증가 A 3행/B 6행은 강성 ±20 % 조건(강성·마찰 조합 포함)", "새 잡음 9조건은 증가 0; 전체 34조건 충돌 외 FA A 6→9/B 10→18", "앞 단계에서 탈락한 후보는 뒤 수치로 되살리지 않았다."]},
        {"title": "무충돌 보정·물리값 일치·시간", "bullets": [f"무충돌 검사 {c['normal_audit']['rows']}행: 추가 경보 표본 {c['normal_audit']['extra_samples']}, 시작 {c['normal_audit']['extra_rises']}", f"물리값 일치 16행: 새 W40 대표 {c['equivalence'].get('fresh_representative_rows')}행 + 기존 CUSUM {c['equivalence'].get('reused_rows')}행 재사용", f"W40 새 전체 시뮬레이션 {c['equivalence'].get('fresh_w40_rows')}조건과 최초 304행 기록 보존", "같은 38조건의 p99는 모두 800 µs 미만; 엄격 최대 1 ms 목표는 미달", f"A W40 최대 {timing['A 서보/W40']} ms, B R1 최대 {timing['B 토크/R1']} ms와 모든 초과 표본을 보존"]},
        {"title": "평가 조건 생성 사고와 R1 확인", "bullets": [_incident(c), f"R1 holdout: A {holdout['A 서보']['충돌 검출']}, 정상 FA {holdout['A 서보']['정상 4조건 FA']}, 전체 충돌 외 FA {holdout['A 서보']['전체 충돌 외 FA']}", f"R1 holdout: B {holdout['B 토크']['충돌 검출']}, 정상 FA {holdout['B 토크']['정상 4조건 FA']}, 전체 충돌 외 FA {holdout['B 토크']['전체 충돌 외 FA']}", "모든 후보는 동결 전에 탈락; W40 holdout 미실행"]},
        {"title": "공식 8 kg과 한계", "table": (["방식","시각 [s]","10/15/40/100 N·FA·최소"], official["eight_kg"]), "font": 10.5,
         "caption": "지연 [ms] / FA [회] / 최소 [N] 순서다. 원래 평가 문자열 자릿수를 보존했다.\n공식 8 kg 미검출 3조건에 W40을 실행하지 않았다. G4·약한 충돌·최대 1 ms 미달은 남는다.", "caption_y": .075},
    ]


def render_notes(context: dict) -> str:
    lines = ["# R4 발표 자료 원고", "", "PDF와 같은 순서다.", ""]
    for index, slide in enumerate(slides(context), 1):
        lines += [f"## {index}. {slide['title']}", ""]
        if "table" in slide:
            columns, rows = slide["table"]
            lines += [table(columns, rows), ""]
            if slide.get("caption"): lines += [slide["caption"], ""]
        else:
            lines += ["- " + value for value in slide["bullets"]] + [""]
    return "\n".join(lines)


def _configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "DejaVu Sans") if name in names), "DejaVu Sans")
    matplotlib.rcParams.update({"font.family": font, "axes.unicode_minus": False})


def _wrap(value: str, width: int) -> str:
    lines = []
    for source_line in str(value).splitlines() or [""]:
        text = source_line
        if re.fullmatch(r"[AB]\s+\d+(?:\.\d+)?/\d+(?:\.\d+)?;?", text):
            lines.append(text)
            continue
        while len(text) > width:
            cut = text.rfind(" ", 0, width + 1)
            cut = cut if cut > width // 3 else width
            lines.append(text[:cut]); text = text[cut:].lstrip()
        lines.append(text)
    return "\n".join(lines)


def _draw_table(fig, columns, rows, font_size):
    ax = fig.add_axes([.025, .08, .95, .78]); ax.axis("off")
    width = max(9, int(82 / len(columns)))
    data = [[_wrap(row.get(column, ""), width) for column in columns] for row in rows]
    headings = [_wrap(column, width) for column in columns]
    tab = ax.table(cellText=data, colLabels=headings, loc="center", cellLoc="left", colLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(font_size)
    height = .78 / max(1, len(rows) + 1)
    for (row, _), cell in tab.get_celld().items():
        cell.set_height(min(.16, height)); cell.set_edgecolor("#b0bec5")
        if row == 0: cell.set_facecolor("#dbe7f3"); cell.set_text_props(weight="bold")


def make_pdf(path: Path, context: dict):
    _configure_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pages = slides(context)
    with PdfPages(path, metadata={"Title": "R4 sequential detector results", "Creator": "r4_report.py"}) as pdf:
        for index, slide in enumerate(pages, 1):
            fig = plt.figure(figsize=(13.333, 7.5), facecolor="#f6f8fb")
            fig.text(.04, .92, slide["title"], fontsize=23, weight="bold", color="#183153")
            if "table" in slide:
                _draw_table(fig, *slide["table"], slide.get("font", 11.0))
                if slide.get("caption"):
                    fig.text(.04, slide.get("caption_y", .055),
                             _wrap(slide["caption"], 105), fontsize=10.5)
            else:
                y = .82
                for bullet in slide["bullets"]:
                    for line in _wrap("• " + bullet, 72).splitlines():
                        if y < .08: raise ValueError("R4 PDF bullet slide overflow")
                        fig.text(.055, y, line, fontsize=14, color="#263238"); y -= .045
                    y -= .018
            fig.text(.04, .025, f"R4 검증 | {index}/{len(pages)}", fontsize=9, color="#607d8b")
            pdf.savefig(fig); plt.close(fig)


def privacy_check(text: str, label: str):
    found = [pattern.pattern for pattern in PUBLIC_FORBIDDEN if pattern.search(text)]
    if found: raise ValueError(label + " contains prohibited publication wording: " + str(found))


def pdf_pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as out:
        out.write(text); temporary = Path(out.name)
    temporary.replace(path)


def generate(manifest_path: Path = MANIFEST, root: Path = ROOT):
    root = Path(root).resolve()
    if not manifest_path.is_file():
        document = build_manifest(root)
        write_text(manifest_path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    context = validate_manifest(load(manifest_path), root)
    readme, results, notes = render_readme(context), render_results(context), render_notes(context)
    for name, text in (("README", readme), ("R4_RESULTS", results), ("PRESENTATION_R4", notes)):
        privacy_check(text, name)
    outputs = (root / "README.md", root / "docs/R4_RESULTS.md",
               root / "report/PRESENTATION_R4.md", root / "results/presentation.pdf")
    outputs[3].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", dir=outputs[3].parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        make_pdf(temporary, context)
        write_text(outputs[0], readme); write_text(outputs[1], results); write_text(outputs[2], notes)
        temporary.replace(outputs[3])
    finally:
        temporary.unlink(missing_ok=True)
    return context


def check_current(manifest_path: Path = MANIFEST, root: Path = ROOT):
    root = Path(root).resolve()
    context = validate_manifest(load(manifest_path), root)
    expected = (render_readme(context), render_results(context), render_notes(context))
    paths = (root / "README.md", root / "docs/R4_RESULTS.md", root / "report/PRESENTATION_R4.md")
    for path, text in zip(paths, expected):
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise ValueError("generated R4 document is absent or stale: " + str(path))
        privacy_check(text, path.name)
    from tools.r4_report_links import check_public_links
    links = check_public_links(root)
    if not links["passed"]:
        raise ValueError("broken public relative file links: " + str(links["missing"]))
    pdf = root / "results/presentation.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-") \
            or pdf_pages(pdf) != len(slides(context)):
        raise ValueError("R4 presentation PDF is absent, invalid, or stale in page count")
    # Regenerate to a temporary file and compare extracted text, which ignores
    # PDF metadata/object IDs while proving every page's displayed content.
    with tempfile.TemporaryDirectory() as folder:
        candidate = Path(folder) / "candidate.pdf"
        make_pdf(candidate, context)
        import subprocess
        pdftotext = "pdftotext"
        def extract(path):
            return subprocess.check_output([pdftotext, str(path), "-"], text=True).split()
        if extract(candidate) != extract(pdf):
            raise ValueError("R4 presentation PDF text differs from deterministic regeneration")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.build_manifest:
        write_text(args.manifest, json.dumps(build_manifest(args.root), ensure_ascii=False, indent=2) + "\n")
    elif args.check:
        check_current(args.manifest, args.root)
    else:
        generate(args.manifest, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
