#!/usr/bin/env python3
"""Deterministic manifest-driven R3 public report and presentation generator.

This postprocessor imports no simulator or controller.  It never derives a
missing number: every displayed value must be curated in a frozen manifest and
every evidence artifact is hash checked before rendering.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import re
import tempfile
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results/v2_protocol/R3_REPORT_MANIFEST.json"
DEFAULT_OUTPUTS = (
    ROOT / "docs/R3_RESULTS.md",
    ROOT / "report/PRESENTATION_R3.md",
    ROOT / "results/presentation.pdf",
)
CANDIDATES = (
    "A_handoff_short", "A_handoff_long", "B_phase", "B_mass", "B_mass_phase",
)
ARCHITECTURES = ("servo", "torque")
PROVENANCE = {"fresh", "reused_exact_source", "literal_copy"}
DECISIONS = {"adopt", "reject"}
REQUIRED_ARTIFACTS = {
    "protocol", "r2_report_manifest", "r2_presentation", "validation_review", "challengers",
    "forced_stability", "actual_stability", "redesign_decision",
    "development_review", "d4_review", "perturbation_review", "freeze",
    "holdout_review", "official_review", "timing_review", "final_selection",
    "integrity",
    "r1_stability", "cause_analysis",
    "d4_servo_r1", "d4_servo_candidate", "d4_torque_r1", "d4_torque_candidate",
    "perturb_servo_r1", "perturb_servo_candidate", "perturb_torque_r1", "perturb_torque_candidate",
}
PRIVATE_PATTERNS = (
    re.compile(r"notion", re.IGNORECASE), re.compile(r"\x2fhome\x2f", re.IGNORECASE),
    re.compile(r"TASK-\d+", re.IGNORECASE), re.compile(r"private", re.IGNORECASE),
    re.compile(r"미게시|스테이징|staging|\bpush\b|원격\s*(?:작업|횟수)", re.IGNORECASE),
)
TABLE_FONT_PT = 11.0
CANDIDATE_TABLE_FONT_PT = 10.5


def manifest_contract():
    """Return the required top-level and row schemas for manifest builders."""
    return {
        "schema": "R3_REPORT_MANIFEST.v1", "status": "frozen",
        "required_top_level": [
            "schema", "status", "artifacts", "candidates",
            "per_mass_validation", "metrics", "final_architectures",
            "redesign", "lie_group_usage", "normal_false_alarms", "timing",
            "exposure", "stage_outcomes", "verification", "limitations",
            "conclusion", "document_details",
        ],
        "candidate_ids": list(CANDIDATES),
        "artifact_keys": sorted(REQUIRED_ARTIFACTS),
        "artifact_entry": ["path", "sha256", "provenance"],
        "candidate_columns": [
            "id", "method", "architecture", "validation_cost",
            "validation_gain", "forced_stability", "actual_stability",
            "timing", "decision", "reason",
        ],
        "per_mass_columns": [
            "candidate", "architecture", "mass_kg", "r1_rms_mm",
            "diagnostic_rms_mm", "r1_max_mm", "diagnostic_max_mm",
            "r1_place_step_nm", "diagnostic_place_step_nm",
            "validation_status", "final_r3_alias",
        ],
        "metric_columns": [
            "id", "name", "condition", "unit", "baseline", "v1", "r1",
            "r3", "status",
        ],
    }


def _require_keys(value, keys, context):
    if not isinstance(value, dict):
        raise ValueError(context + " must be an object")
    missing = sorted(set(keys) - value.keys())
    if missing:
        raise ValueError(f"{context} lacks required keys: {missing}")


def _load(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required report input is absent: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _inside_root(root, relative):
    path = (Path(root) / str(relative)).resolve()
    root = Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes repository: {relative}")
    return path


def artifact_entry(path, provenance="fresh", root=ROOT):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError("artifact path must be inside repository root")
    if provenance not in PROVENANCE:
        raise ValueError("invalid artifact provenance")
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provenance": provenance}


def build_manifest(curation, artifact_specs, root=ROOT):
    """Build and validate a manifest from curated fields and artifact paths.

    ``artifact_specs`` maps every required evidence key to either a path or a
    ``(path, provenance)`` pair.  It reads evidence only to hash/validate it.
    """
    manifest = copy.deepcopy(dict(curation))
    if "artifacts" in manifest:
        raise ValueError("curation must not pre-populate artifacts")
    artifacts = {}
    for key, spec in artifact_specs.items():
        if isinstance(spec, (list, tuple)):
            path, provenance = spec
        else:
            path, provenance = spec, "fresh"
        artifacts[key] = artifact_entry(path, provenance, root)
    manifest["artifacts"] = artifacts
    validate_manifest(manifest, root)
    return manifest


def _validate_artifacts(manifest, root):
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError("artifact keys must exactly match the R3 report contract")
    loaded = {}
    for key, entry in artifacts.items():
        _require_keys(entry, ("path", "sha256", "provenance"), f"artifact {key}")
        if entry["provenance"] not in PROVENANCE:
            raise ValueError(f"artifact {key} has invalid provenance")
        path = _inside_root(root, entry["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"artifact hash mismatch: {entry['path']}")
        if key == "r2_presentation":
            if (entry["provenance"] != "reused_exact_source"
                    or entry["path"] != "results/presentation_R2.pdf"
                    or not path.read_bytes().startswith(b"%PDF-")):
                raise ValueError("r2_presentation must be a reused exact-source PDF")
            loaded[key] = {"pdf_path": path, "pages": _pdf_pages(path),
                           "sha256": entry["sha256"]}
        else:
            loaded[key] = _load(path)
    return loaded


def _scalar_rows(rows, required, context):
    if not isinstance(rows, list) or not rows:
        raise ValueError(context + " must be a non-empty array")
    for index, row in enumerate(rows):
        _require_keys(row, required, f"{context}[{index}]")
        for key in required:
            if isinstance(row[key], (dict, list, tuple)):
                raise ValueError(f"{context}[{index}].{key} must be a readable scalar")
    return rows


def _validate_candidates(rows):
    required = manifest_contract()["candidate_columns"]
    _scalar_rows(rows, required, "candidates")
    if tuple(row["id"] for row in rows) != CANDIDATES:
        raise ValueError("candidates must contain all five registered settings in order")
    for row in rows:
        expected = "servo" if str(row["id"]).startswith("A_") else "torque"
        if row["architecture"] != expected or row["decision"] not in DECISIONS:
            raise ValueError(f"candidate {row['id']} has invalid architecture/decision")
        if not str(row["reason"]).strip():
            raise ValueError(f"candidate {row['id']} needs an explicit reason")


def _number(value, context):
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(float(value)):
        raise ValueError(context + " must be a finite number")
    return float(value)


def _validate_per_mass(rows, protocol, final_by_architecture):
    required = manifest_contract()["per_mass_columns"]
    _scalar_rows(rows, required, "per_mass_validation")
    expected_masses = sorted(float(row["mass"]) for row in protocol["validation"])
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate]
        if sorted(float(row["mass_kg"]) for row in selected) != expected_masses:
            raise ValueError(f"candidate {candidate} must retain all four validation masses")
        architecture = "servo" if candidate.startswith("A_") else "torque"
        if any(row["architecture"] != architecture for row in selected):
            raise ValueError(f"candidate {candidate} has mixed architecture rows")
        for row in selected:
            for key in ("mass_kg", "r1_rms_mm", "diagnostic_rms_mm", "r1_max_mm",
                        "diagnostic_max_mm", "r1_place_step_nm",
                        "diagnostic_place_step_nm"):
                _number(row[key], f"{candidate}.{key}")
        if final_by_architecture[architecture]["decision"] == "reject":
            if any(row["final_r3_alias"] != "R1" for row in selected):
                raise ValueError("rejected architecture per-mass rows must distinguish final R3=R1")
    if len(rows) != len(CANDIDATES) * len(expected_masses):
        raise ValueError("per_mass_validation has duplicate or foreign candidate rows")


def _validate_metrics(rows, r2_manifest, final_by_architecture):
    required = manifest_contract()["metric_columns"]
    _scalar_rows(rows, required, "metrics")
    if [row["id"] for row in rows] != [f"T{i}" for i in range(1, 7)]:
        raise ValueError("metrics must be ordered T1 through T6")
    r2 = {row["id"]: row for row in r2_manifest.get("metrics", ())}
    if set(r2) != {f"T{i}" for i in range(1, 7)}:
        raise ValueError("R2 report manifest lacks the literal T1-T6 source")
    for row in rows:
        source = r2[row["id"]]
        for key in ("name", "condition", "unit", "baseline", "v1", "r1"):
            if row[key] != source[key]:
                raise ValueError(f"{row['id']} {key} must be a literal R2 manifest copy")
    if all(item["final_alias"] == "R1" for item in final_by_architecture.values()):
        if any(row["r3"] != row["r1"] for row in rows):
            raise ValueError("all-R1 final alias requires every R3 metric to equal R1")


def _validate_final(rows):
    required = ("architecture", "decision", "selected", "final_alias",
                "my_sha256", "r1_sha256", "reason")
    _scalar_rows(rows, required, "final_architectures")
    if [row["architecture"] for row in rows] != list(ARCHITECTURES):
        raise ValueError("final_architectures must independently order servo then torque")
    result = {}
    for row in rows:
        if row["decision"] not in DECISIONS:
            raise ValueError("invalid architecture decision")
        if row["final_alias"] == "R1":
            if row["decision"] != "reject" or row["my_sha256"] != row["r1_sha256"]:
                raise ValueError("R1 alias requires rejection and exact my/R1 SHA equality")
        elif row["decision"] != "adopt":
            raise ValueError("non-R1 final source requires adoption")
        result[row["architecture"]] = row
    return result


def _validate_redesign(rows):
    required = ("architecture", "used", "reason", "recorded_before_development",
                "arbitrary_parameter_search_performed")
    _scalar_rows(rows, required, "redesign")
    if [row["architecture"] for row in rows] != list(ARCHITECTURES):
        raise ValueError("redesign must cover servo and torque")
    for row in rows:
        if (row["used"] != 0 or row["recorded_before_development"] is not True
                or row["arbitrary_parameter_search_performed"] is not False
                or not str(row["reason"]).strip()):
            raise ValueError("unused redesign needs a prior reason and no arbitrary search")


def _validate_lie(rows):
    required = ("method", "section", "equation", "elastic_1ms_zoh_delay_adaptation",
                "comparison", "decision")
    _scalar_rows(rows, required, "lie_group_usage")
    for row in rows:
        adaptation = str(row["elastic_1ms_zoh_delay_adaptation"]).lower()
        if not all(token in adaptation for token in ("elastic", "1 ms", "zoh", "delay")):
            raise ValueError("each Lie row must state elastic, 1 ms, ZOH, and delay adaptation")
        if row["decision"] not in DECISIONS:
            raise ValueError("invalid Lie-group decision")


def validate_manifest(manifest, root=ROOT):
    required = manifest_contract()["required_top_level"]
    _require_keys(manifest, required, "manifest")
    if manifest["schema"] != "R3_REPORT_MANIFEST.v1" or manifest["status"] != "frozen":
        raise ValueError("report requires frozen R3_REPORT_MANIFEST.v1")
    loaded = _validate_artifacts(manifest, root)
    protocol = loaded["protocol"]
    if protocol.get("schema") != "R3_PROTOCOL.v1":
        raise ValueError("protocol artifact is not R3_PROTOCOL.v1")
    final = _validate_final(manifest["final_architectures"])
    _validate_candidates(manifest["candidates"])
    candidate_by_id = {row["id"]: row for row in manifest["candidates"]}
    for architecture, row in final.items():
        architecture_candidates = [item for item in manifest["candidates"]
                                   if item["architecture"] == architecture]
        if row["decision"] == "adopt":
            if (row["selected"] not in candidate_by_id
                    or candidate_by_id[row["selected"]]["decision"] != "adopt"
                    or sum(item["decision"] == "adopt"
                           for item in architecture_candidates) != 1):
                raise ValueError("adopted architecture needs exactly one matching adopted candidate")
        elif any(item["decision"] != "reject" for item in architecture_candidates):
            raise ValueError("rejected architecture cannot contain an adopted candidate")
    _validate_per_mass(manifest["per_mass_validation"], protocol, final)
    _validate_metrics(manifest["metrics"], loaded["r2_report_manifest"], final)
    _validate_redesign(manifest["redesign"])
    _validate_lie(manifest["lie_group_usage"])
    details = manifest['document_details']
    _require_keys(details, ('stability', 'costs', 'positive_mass_perturbation',
                           'stability_notes', 'redesign_context', 'cost_notes', 'causes'),
                  'document_details')
    if [row['candidate'] for row in details['stability']] != list(CANDIDATES):
        raise ValueError('stability comparison must retain all five candidates')
    normal = _scalar_rows(manifest["normal_false_alarms"],
                          ("stage", "architecture", "count", "rise_vs_r1", "status"),
                          "normal_false_alarms")
    for row in normal:
        if (not isinstance(row["count"], int) or row["count"] < 0
                or not isinstance(row["rise_vs_r1"], int)):
            raise ValueError("normal false-alarm counts must be recorded integers")
        expected = "pass" if row["count"] == 0 and row["rise_vs_r1"] <= 0 else "fail"
        if str(row["status"]).lower() != expected:
            raise ValueError("normal false-alarm status must match the recorded result")
    timing = _scalar_rows(manifest["timing"],
                          ("stage", "architecture", "p99_us", "max_us",
                           "outlier_count", "outliers_retained_raw",
                           "p99_gate_pass", "max_1000_goal_met", "status"),
                          "timing")
    if any(row["outliers_retained_raw"] is not True for row in timing):
        raise ValueError("every timing row must retain all raw outliers")
    for row in timing:
        p99 = _number(row["p99_us"], "timing.p99_us")
        maximum = _number(row["max_us"], "timing.max_us")
        if (row["p99_gate_pass"] is not (p99 < 800.0)
                or row["max_1000_goal_met"] is not (maximum <= 1000.0)
                or not isinstance(row["outlier_count"], int)
                or row["outlier_count"] < 0):
            raise ValueError("timing gate/goal flags must exactly match retained values")
    _require_keys(manifest["exposure"], ("holdout", "official"), "exposure")
    holdout, official = manifest["exposure"]["holdout"], manifest["exposure"]["official"]
    _require_keys(holdout, ("opened_after_freeze", "batches", "no_tuning_after"),
                  "exposure.holdout")
    _require_keys(official, ("executed_last", "no_tuning_after"), "exposure.official")
    if (holdout["opened_after_freeze"] is not True or holdout["batches"] != 1
            or holdout["no_tuning_after"] is not True
            or official["executed_last"] is not True
            or official["no_tuning_after"] is not True):
        raise ValueError("holdout/official ordering and no-tuning constraints must hold")
    _require_keys(manifest["verification"], ("executed", "not_executed"), "verification")
    if not all(isinstance(manifest["verification"][key], list)
               for key in ("executed", "not_executed")):
        raise ValueError("verification executed/not_executed must be arrays")
    if not manifest["verification"]["executed"] or not manifest["verification"]["not_executed"]:
        raise ValueError("verification must state both executed and not-executed scope")
    if not isinstance(manifest["stage_outcomes"], list) or not manifest["stage_outcomes"]:
        raise ValueError("stage_outcomes must be non-empty")
    if not isinstance(manifest["limitations"], list) or not manifest["limitations"]:
        raise ValueError("limitations must be non-empty")
    for name, rows in (("stage_outcomes", manifest["stage_outcomes"]),
                       ("limitations", manifest["limitations"]),
                       ("verification.executed", manifest["verification"]["executed"]),
                       ("verification.not_executed", manifest["verification"]["not_executed"])):
        if any(not isinstance(item, (str, int, float, bool)) for item in rows):
            raise ValueError(name + " must contain readable scalar rows")
    if not isinstance(manifest["conclusion"], str) or not manifest["conclusion"].strip():
        raise ValueError("conclusion is required")
    return {"manifest": manifest, "artifacts": loaded,
            "final_by_architecture": final}


def _cell(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _md_table(columns, rows):
    header = "| " + " | ".join(columns) + " |"
    line = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_cell(row.get(column, "")) for column in columns) + " |"
            for row in rows]
    return "\n".join([header, line] + body)


def _prose(value):
    """Presentation spelling only; source values and identifiers stay intact."""
    parts = re.split(r'(\]\([^\n)]+\)|`[^`]*`)', _cell(value))
    return ''.join(part if i % 2 else _prose_text(part) for i, part in enumerate(parts))


def _prose_text(text):
    for before, after in (
        ('one-cycle delay', '한 주기 지연'), ('one-주기 delay', '한 주기 지연'),
        ('offsetFA', '편향 조건 오감지'), ('offset FA', '편향 조건 오감지'),
        ('same-case', '동일 조건'), ('forced', '강제점 검사'), ('actual', '실제 후보점 검사'),
        ('servo', '서보 A'), ('torque', '토크 B'), ('reject', '탈락'),
        ('PASS', '통과'), ('FAIL', '실패'), ('radius+FD', 'radius·FD 동시 통과'),
        ('validation', '사전 검증'), ('development', '개발 검증'),
        ('peak', '최대값'), ('sat', '포화'), ('masked', '기존 경보·구분 불가'),
        ('row', '행'), ('cycle', '주기'), ('report', '보고서'),
        ('byte', '바이트'), ('reference', '기준'), ('case', '조건'), ('score', '효능 점수'),
        ('step', '계단'), ('최대목표', '최대 목표'), ('RMS비', 'RMS 비'),
        ('1.13565', '1.136'), ('1.25525', '1.255'), ('1.05842', '1.058'),
        ('1.47268', '1.473'), ('.95145', '0.951'), ('.833553', '0.834'),
    ):
        if before.startswith('.'):
            text = re.sub(r'(?<!\d)' + re.escape(before), after, text)
        else:
            text = re.sub(r'(?<![A-Za-z_])' + re.escape(before) + r'(?![A-Za-z_])', after, text) if before.isalpha() else text.replace(before, after)
    text = re.sub(r'(?<![\w.])\.(\d+)', r'0.\1', text)
    text = re.sub(r'([가-힣])(?=[A-Za-z0-9])', r'\1 ', text)
    text = re.sub(r'(?<![A-Za-z_])([AB])(?=\d)', r'\1 ', text)
    text = re.sub(r'(?<![A-Za-z_])FA\s*(?=\d)', '오감지 ', text)
    text = re.sub(r'(?<![A-Za-z_])(pRNE|overlay)(?=\d)', r'\1 ', text)
    text = re.sub(r'(?<=\d)(kg|ms|us|Nm|GiB|mm|rad/s|s|N)(?=\d|[^A-Za-z_]|$)', r' \1', text)
    text = re.sub(r'(kg|ms|us|Nm|mm|N)(?=\d)', r'\1 ', text)
    text = re.sub(r'(?<=\d)(episode)\b', r' \1', text)
    text = text.replace('A0/B0회', 'A 0회, B 0회')
    return text


COLUMN_LABELS = {
    'architecture': '방식', 'decision': '판정', 'selected': '탈락 진단 후보',
    'final_alias': '최종 제출', 'reason': '근거', 'id': '후보', 'method': '방법',
    'validation_cost': '사전 검증 비용', 'validation_gain': '사전 검증 추적비',
    'forced_stability': '강제점 국소 검사', 'actual_stability': '실제 후보점 국소 검사',
    'timing': '계산 p99/최대 [us]', 'candidate': '후보', 'mass_kg': '질량 [kg]',
    'r1_rms_mm': 'R1 RMS [mm]', 'diagnostic_rms_mm': '후보 RMS [mm]',
    'r1_max_mm': 'R1 최대 [mm]', 'diagnostic_max_mm': '후보 최대 [mm]',
    'r1_place_step_nm': 'R1 놓기 계단 [N·m]', 'diagnostic_place_step_nm': '후보 놓기 계단 [N·m]',
    'validation_status': '판정', 'final_r3_alias': '최종 R3', 'used': '사용 횟수',
    'recorded_before_development': '개발 검증 전 기록',
    'arbitrary_parameter_search_performed': '임의 파라미터 탐색',
    'section': '서지·절', 'equation': '사용 식',
    'elastic_1ms_zoh_delay_adaptation': '탄성·1 ms ZOH·지연 변경', 'comparison': '측정 효과·한계',
    'stage': '구분', 'count': '오감지 수', 'rise_vs_r1': 'R1 대비 증가', 'status': '상태',
    'p99_us': '최대 행 p99 [us]', 'max_us': '관측 최대 [us]',
    'outlier_count': '1 ms 초과 주기', 'outliers_retained_raw': '이상치 전수 보존',
    'p99_gate_pass': 'p99 관문 통과', 'max_1000_goal_met': '최대 1 ms 목표',
    'forced': '강제점 통과', 'actual': '실제 후보점 통과', 'r1': '같은 강제점 R1',
    'exceeded': '1.05 초과 행', 'total': '무충돌 전체 행', 'metrics': '초과 지표별 행',
    'masses': '초과 질량별 행', 'mass_metrics': '질량·지표별 행', 'worst_ratio': '최악 비',
    'rows': '평균한 행 수', 'rms_ratio': '평균 RMS 비', 'max_ratio': '평균 최대 TCP 비',
}


def _display_rows(columns, rows):
    projected = []
    for row in rows:
        output = {}
        for key in columns:
            value = row.get(key, '')
            if key == 'status' and (row.get('stage', '').endswith(' R1') or row.get('stage') == 'R1'):
                value = '기준'
            if isinstance(value, bool):
                value = '예' if value else '아니오'
            elif isinstance(value, float):
                value = f'{value:g}' if key == 'mass_kg' else f'{value:.3f}'
            output[_prose(COLUMN_LABELS.get(key, key))] = _prose(value)
        projected.append(output)
    return projected


def _display_table(columns, rows):
    return _md_table([_prose(COLUMN_LABELS.get(k, k)) for k in columns], _display_rows(columns, rows))


def _metric_rows(rows):
    conditions = {
        'T1': '8 kg, 동일 COM, 놓기 직전 20 ms 평균 [%mg, A/B]',
        'T2': '8 kg, 동일 COM, 놓기 토크 계단 [N·m, A/B]',
        'T3': 'R2와 같은 34조건의 보존값: 방식별 무충돌 28+외력 6 [mm/N/오감지]',
        'T4': '방식별 무충돌 189조건 [포화 조건 수/놓기 힘 mg]',
        'T5': '8 kg, 동일 COM, 무충돌 [RMS/최대 mm, A/B]',
        'T6': '방식별 충돌 180조건, 기존 경보·구분 불가 별도 [검출/놓침/구분 불가]',
    }
    return [{'항목': _prose(f"{r['id']} {r['name']}"), '조건·단위': conditions[r['id']],
             '기준': r['baseline'], 'v1': r['v1'], 'R1': r['r1'], 'R3': r['r3']} for r in rows]


def _official_eight_kg(context):
    """Readable cells from the last evaluator, not inferred from the R1 alias."""
    evidence = context["artifacts"]["official_review"]
    records = evidence.get("results")
    if not isinstance(records, dict):
        return []  # A failed wrapper may have no measured evaluator cells.
    answer = []
    for mode in ARCHITECTURES:
        payload = records[mode]["control_and_collisions"]
        cells = payload["control"]["8.0"]
        collision = next(row for row in payload["collisions"]
                         if float(row["mass"]) == 8.0 and float(row["time"]) == 3.0)
        answer.append({"방식": mode, "RMS / 최대 mm": f"{cells[0]} / {cells[1]}",
                       "놓기 step Nm": cells[4],
                       "운반 15N / 10N ms": f"{collision['cells'][1]} / {collision['cells'][0]}",
                       "0/4/8kg 무충돌 FA": '/'.join(str(sum(int(v) for v in payload['control'][mass][6:9]))
                                                        for mass in ('0.0', '4.0', '8.0')),
                       "8kg 포화 %": cells[2],
                       "판정": evidence["architectures"][mode]["status"]})
    return answer


def render_results(context):
    m = context['manifest']; d = m['document_details']
    table = _display_table
    lines = ['# R3 결과', '', 'A=서보, B=토크, FA=오감지다. 강제점은 R1 관측·물리점에 후보를 재생한 검사, 실제 후보점은 후보 자체 궤적의 검사다. radius는 이산 폐루프 고유값의 최대 절댓값, FD는 유한차분 수렴 검사다.', '',
             '## 최종 방식별 판정', '',
             table(['architecture', 'decision', 'selected', 'final_alias', 'reason'], m['final_architectures']), '',
             '탈락 진단 후보는 최종 제출이 아니다. 두 방식 모두 독립적으로 판정해 최종 R1을 유지한다.', '',
             '## 등록 후보 5개', '', table(manifest_contract()['candidate_columns'], m['candidates']), '']
    lines += ['후보 표의 근거는 사전 검증 시점 기록이다. R1과의 국소 비교 및 최종 탈락을 뒷받침하는 D4·섭동 근거는 아래에 함께 제시한다.', '']
    for mode, title in [('servo', '서보 A 사전 검증'), ('torque', '토크 B 사전 검증')]:
        rows = [r for r in m['per_mass_validation'] if r['architecture'] == mode]
        lines += [f'## {title}: R1 → 미채택 진단 후보', '',
                  table(manifest_contract()['per_mass_columns'], rows), '']
    lines += ['## T1–T6', '',
              '기준/v1/R1은 보존된 동일 조건 수치이고 최종 R3=R1은 같은 소스다. 특히 T3는 R2와 같은 34조건의 보존값이며 새 R3 조건에서 네 버전을 재실행한 효능 표가 아니다. 새 검증·D4·섭동은 아래에서 분리한다.', '',
              table(['항목', '조건·단위', '기준', 'v1', 'R1', 'R3'], _metric_rows(m['metrics'])), '',
              '## 국소 안정성: 같은 점의 R1 비교', '',
              table(['candidate', 'forced', 'actual', 'r1'], d['stability']), '',
              '표의 분자는 radius와 FD를 동시에 통과한 점 수다. R1 열은 같은 강제점 격자만 비교하며 후보 실제점에 대응하는 R1 재실행은 아니다.', '']
    lines += ['- ' + _prose(x) for x in d['stability_notes']]
    lines += ['', '## D4·분석 섭동의 비용 초과 분포', '',
              table(['stage', 'architecture', 'exceeded', 'total', 'metrics', 'masses', 'worst_ratio'], d['costs']), '',
              '### 질량·지표별 분포', '',
              table(['stage', 'architecture', 'mass_metrics'], d['costs']), '']
    lines += ['- ' + _prose(x) for x in d['cost_notes']]
    lines += ['', '### 양의 질량 무충돌 섭동의 참고 평균', '',
              table(['architecture', 'rows', 'rms_ratio', 'max_ratio'], d['positive_mass_perturbation']), '',
              '**효능 관문이 아닌 참고값이다.** 평균 개선으로 같은 조건의 비용·오감지 관문을 대체하지 않는다.', '',
              '## 재설계 사용 여부', '',
              table(['architecture', 'used', 'reason', 'recorded_before_development',
                     'arbitrary_parameter_search_performed'], m['redesign']), '']
    lines += ['- ' + _prose(x) for x in d['redesign_context']]
    lines += ['', '## Lie group 적용', '',
              table(['method', 'section', 'equation', 'elastic_1ms_zoh_delay_adaptation',
                     'comparison', 'decision'], m['lie_group_usage']), '',
              '## 정상 오감지', '',
              table(['stage', 'architecture', 'count', 'rise_vs_r1', 'status'], m['normal_false_alarms']), '',
              'R1 행은 비교 기준이다. 기준의 오감지를 통과로 바꾼 것이 아니며 수치는 그대로 보존했다.', '',
              '## 계산 시간과 모든 이상치', '',
              table(['stage', 'architecture', 'p99_us', 'max_us', 'outlier_count',
                     'outliers_retained_raw', 'p99_gate_pass', 'max_1000_goal_met', 'status'], m['timing']), '',
              '모든 이상치를 보존했다. p99<800 us는 관문, 최대 1 ms는 미달을 포함해 기록하는 목표다. R1에는 추가 진단·holdout이 있어 전체 행 수가 다르므로 서로 다른 전수 집계를 같은 조건의 속도 비로 해석하지 않는다.', '',
              '## 단계별 결과', '']
    lines += ['- ' + _prose(x) for x in m['stage_outcomes']]
    official = _official_eight_kg(context)
    if official:
        lines += ['', '## 마지막 공식 8 kg 회귀', '',
                  '최종 R1 바이트를 원본 evaluator로 재실행한 값이다. 진단 후보의 공식 효능이 아니며 0/4/8 kg 전체 회귀와 기존 미검출은 원자료에 보존했다.', '',
                  table(list(official[0]), official)]
    lines += ['', '## 검증 범위', '', '### 실행', '']
    lines += ['- ' + _prose(x) for x in m['verification']['executed']]
    lines += ['', '### 실행하지 않음', '']
    lines += ['- ' + _prose(x) for x in m['verification']['not_executed']]
    lines += ['', '## 한계', ''] + ['- ' + _prose(x) for x in m['limitations']]
    lines += ['', '## 결론', '', _prose(m['conclusion']), '', '## 원자료와 자체 유도', '',
              '- [R3 방법·가정](R3_METHODS.md), [구현 전 원인 기록](R3_CAUSE_ANALYSIS.md)',
              '- [충돌 신호 분석](../results/v2_protocol/R3_SIGNAL_ANALYSIS.json), [pRNE 수치 검사](../results/v2_protocol/R3_PRNE_MATH.json)',
              '- [동일 관측 계산비용](../results/v2_protocol/R3_REPLAY_TIMING.json)', '',
              '| 증거 | 원자료 경로 | 출처 |', '|---|---|---|']
    for key, entry in m['artifacts'].items():
        provenance = {'fresh': 'R3 기록', 'reused_exact_source': '동일 원본 보존', 'literal_copy': '보존값 인용'}[entry['provenance']]
        lines.append(f"| {key} | [{entry['path']}](../{entry['path']}) | {provenance} |")
    lines.append('')
    return '\n'.join(lines)


def _slides(context):
    m = context["manifest"]
    d = m['document_details']
    a_candidates = [row for row in m["candidates"] if row["architecture"] == "servo"]
    b_candidates = [row for row in m["candidates"] if row["architecture"] == "torque"]
    final = {row["architecture"]: row for row in m["final_architectures"]}

    def candidate_rows(rows):
        return [{"설정": row["id"],
                 "비용 / 효능": f"{row['validation_cost']} / {row['validation_gain']}",
                 "강제 / 실제 국소": f"{row['forced_stability']} / {row['actual_stability']}",
                 "계산 p99 / max": row["timing"],
                 "판정": f"{row['decision']}: {row['reason']}"} for row in rows]

    def mass_rows(architecture):
        selected = final[architecture]["selected"]
        rows = [row for row in m["per_mass_validation"]
                if row["architecture"] == architecture and row["candidate"] == selected]
        return [{"kg": row["mass_kg"],
                 "RMS mm R1 → 후보": f"{row['r1_rms_mm']:.3f} → {row['diagnostic_rms_mm']:.3f}",
                 "최대 mm R1 → 후보": f"{row['r1_max_mm']:.3f} → {row['diagnostic_max_mm']:.3f}",
                 "놓기 step Nm R1 → 후보":
                     f"{row['r1_place_step_nm']:.3f} → {row['diagnostic_place_step_nm']:.3f}",
                 "최종 R3": row["final_r3_alias"]} for row in rows]

    compact_candidate_columns = ["설정", "비용 / 효능", "강제 / 실제 국소",
                                 "계산 p99 / max", "판정"]
    compact_mass_columns = ["kg", "RMS mm R1 → 후보", "최대 mm R1 → 후보",
                            "놓기 step Nm R1 → 후보", "최종 R3"]
    compact_metric_columns = ["항목", "조건·단위", "기준", "v1", "R1", "R3"]
    lie_rows = [{"방법·절": f"{r['method']}\n{r['section']}", '사용 식': r['equation'],
                 '탄성·1 ms ZOH·지연 변경': r['elastic_1ms_zoh_delay_adaptation'],
                 '효과와 분리 한계': r['comparison'], '판정': r['decision']}
                for r in m['lie_group_usage']]
    slides = [
        {"title": "R3 질문과 불변 관문", "bullets": [
            "A=서보, B=토크, FA=오감지. 기준=배포 키트, v1=탄성·하중전이/기하 감지, R1=정상-only ridge/EMA, R3=이번 최종 판정",
            "서보 DOB 대역 4.5와 토크 P-IDC rate 4.5/damping 1을 원인 분석 뒤 새 조건에서 검증했다.",
            "비용은 모든 동일조건 R1의1.05배 이하, p99<800us. 최대1ms는 목표이며 outlier를 보존한다.",
            "T1–T6은 보존 동일조건 비교와 최종 소스 alias다. 새 R3 split 네 버전의 독립 재실행 표가 아니다."]},
        {"title": "구현 전 원인 분석", "bullets": d['causes']},
        {"title": "A 후보 판정", "table": (compact_candidate_columns,
                                             candidate_rows(a_candidates)),
         "candidate": True},
        {"title": "B 후보 판정", "table": (compact_candidate_columns,
                                             candidate_rows(b_candidates)),
         "candidate": True},
        {"title": "A 질량별 R1 → 진단 후보", "table": (compact_mass_columns,
                                                       mass_rows("servo"))},
        {"title": "B 질량별 R1 → 진단 후보", "table": (compact_mass_columns,
                                                       mass_rows("torque"))},
        {"title": "아키텍처 독립 최종 판정", "table": (
            ["방식", "판정", "탈락 진단 후보", "최종 제출", "근거"],
            [{"방식": row['architecture'], "판정": row['decision'],
              "탈락 진단 후보": row['selected'], "최종 제출": row['final_alias'],
              "근거": row['reason']} for row in m["final_architectures"]])},
        {"title": "T1–T3", "table": (compact_metric_columns,
                                      _metric_rows(m["metrics"][:3]))},
        {"title": "T4–T6", "table": (compact_metric_columns,
                                      _metric_rows(m["metrics"][3:]))},
        {"title": "국소 안정성: R1과 같은 강제점 비교", "table": (
            ['candidate', 'forced', 'actual', 'r1'], d['stability']),
         'caption': '분자는 radius·FD 동시 통과 수. R1 열은 강제점에만 대응한다. 임계값은 같고 R3에서 검사점이 늘었다. 이 관문 단독으로는 후보와 R1을 가를 수 없다. 규칙·판정은 유지하고 D4·섭동 근거를 함께 본다.'},
        {"title": "정상 오감지", "table": (["stage", "architecture", "count", "rise_vs_r1", "status"], m["normal_false_alarms"])},
        {"title": "계산 시간과 모든 outlier", "table": (
            ["소스", "방식", "최대 row p99 us", "관측 최대 ms", ">1ms cycle", "p99 / 최대목표"],
            [{"소스": row["stage"], "방식": row["architecture"],
              "최대 row p99 us": f"{row['p99_us']:.2f}",
              "관측 최대 ms": f"{row['max_us'] / 1000:.3f}",
              ">1ms cycle": row["outlier_count"],
              "p99 / 최대목표": f"{'기준' if row['stage'] == 'R1' else row['status']} / {'PASS' if row['max_1000_goal_met'] else '미달'}"}
             for row in m["timing"]])},
        {"title": "development·D4·섭동", "bullets": m["stage_outcomes"][:6]},
        {"title": "재설계 0회", "bullets": [
            f"{row['architecture']}: 사용 {row['used']}회. {row['reason']}"
            for row in m["redesign"]] + d['redesign_context']},
        {"title": "Lie group 적용", "table": (
            ['방법·절', '사용 식', '탄성·1 ms ZOH·지연 변경', '효과와 분리 한계', '판정'], lie_rows)},
        {"title": "holdout·공식·no-tuning", "bullets": [
            "holdout: freeze 뒤 1 batch, 이후 tuning 없음",
            "official: 마지막 실행, 이후 tuning 없음",
            *m["stage_outcomes"][-2:],
            *m["limitations"][:3]]},
        {"title": "실행 / 실행하지 않음", "bullets": [
            *["실행: " + _cell(x) for x in m["verification"]["executed"]],
            *["미실행: " + _cell(x) for x in m["verification"]["not_executed"]]]},
        {"title": "결론", "bullets": [m["conclusion"]]},
        {"title": "추가 검사점과 R1: 관문의 해석 한계", "bullets": d['stability_notes']},
        {"title": "D4·섭동 비용 초과의 분포", "table": (
            ['stage', 'architecture', 'exceeded', 'total', 'metrics', 'masses', 'worst_ratio'], d['costs'])},
        {"title": "양의 질량 섭동: 참고 평균과 오감지", "table": (
            ['architecture', 'rows', 'rms_ratio', 'max_ratio'], d['positive_mass_perturbation']),
         'caption': '양의 질량 무충돌 22행의 조건별 비 산술평균: 효능 관문이 아닌 참고값. B 8 kg 강성 1.2배 오감지 2→32회, 강성 1.2배·마찰 0.8배 복합 조건은 2→28회. 평균 추적 개선으로 비용·오감지 실패를 대체하지 않는다.'},
    ]
    for slide in slides:
        if 'table' in slide:
            columns, rows = slide['table']
            slide['table'] = ([_prose(COLUMN_LABELS.get(k, k)) for k in columns], _display_rows(columns, rows))
        else:
            slide['bullets'] = [_prose(x) for x in slide['bullets']]
    return slides


def render_presentation_notes(context):
    lines = ["# R3 발표 자료 원고", "",
             "PDF는 아래 슬라이드를 같은 순서로 생성한다.", ""]
    for index, slide in enumerate(_slides(context), 1):
        lines += [f"## {index}. {slide['title']}", ""]
        if "table" in slide:
            columns, rows = slide["table"]
            lines += [_md_table(columns, rows), ""]
            if slide.get('caption'):
                lines += [_prose(slide['caption']), '']
        else:
            lines += [f"- {_cell(item)}" for item in slide["bullets"]]
            lines.append("")
    return "\n".join(lines)


def _configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    candidates = ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "DejaVu Sans")
    names = {item.name for item in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in names), "DejaVu Sans")
    matplotlib.rcParams.update({"font.family": selected, "axes.unicode_minus": False})


def _wrap_display(value, width):
    """Conservative CJK-aware wrapping without dropping characters."""
    lines, line, used = [], '', 0.0
    for char in str(value):
        if char == '\n':
            lines.append(line); line, used = '', 0.0
            continue
        advance = 1.0 if unicodedata.east_asian_width(char) in ('W', 'F') else .6
        if line and used + advance > width:
            # Prefer a word boundary so a number fitting the column does not
            # split across lines. Retain the space: wrapping is lossless.
            cut = line.rfind(' ')
            if cut > len(line) * .4:
                lines.append(line[:cut + 1]); line = line[cut + 1:]
                used = sum(1.0 if unicodedata.east_asian_width(c) in ('W', 'F')
                           else .6 for c in line)
            else:
                lines.append(line); line, used = '', 0.0
        line += char
        used += advance
    lines.append(line)
    return '\n'.join(lines)


def _draw_table(fig, columns, rows, font_size):
    ax = fig.add_axes([.035, .08, .93, .77]); ax.axis("off")
    width_pt = fig.get_figwidth() * 72 * .93 / len(columns)
    width = (width_pt - 10) / font_size
    wrapped = [[_wrap_display(_cell(row.get(column, "")), width) for column in columns]
               for row in rows]
    headings = [_wrap_display(column, width) for column in columns]
    table = ax.table(cellText=wrapped, colLabels=headings, cellLoc="left",
                     colLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(font_size)
    height_pt = fig.get_figheight() * 72 * .77
    heights = [(max(cell.count('\n') + 1 for cell in row) * font_size * 1.3 + 9)
               / height_pt for row in [headings] + wrapped]
    if sum(heights) > .97:
        raise ValueError('PDF table requires a shorter projection or a split slide')
    for (row, _), cell in table.get_celld().items():
        cell.set_height(heights[row])
        cell.set_edgecolor("#b0bec5")
        if row == 0:
            cell.set_facecolor("#dbe7f3"); cell.set_text_props(weight="bold")


def make_pdf(path, context):
    _configure_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    slides = _slides(context)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path, metadata={"Title": "R3 results", "Creator": "r3_report.py"}) as pdf:
        for index, slide in enumerate(slides, 1):
            fig = plt.figure(figsize=(13.333, 7.5), facecolor="#f6f8fb")
            fig.text(.045, .92, slide["title"], fontsize=23, weight="bold", color="#183153")
            if "table" in slide:
                columns, rows = slide["table"]
                _draw_table(fig, columns, rows,
                            CANDIDATE_TABLE_FONT_PT if slide.get("candidate") else TABLE_FONT_PT)
                if slide.get('caption'):
                    fig.text(.045, .09, _wrap_display(_prose(slide['caption']), 90),
                             fontsize=11, color='#263238')
            else:
                y = .82
                for bullet in slide["bullets"]:
                    for line in _wrap_display("• " + _cell(bullet), 61).splitlines():
                        if y < .075:
                            raise ValueError('PDF bullet slide overflows; shorten presentation text')
                        fig.text(.06, y, line, fontsize=14, color="#263238")
                        y -= .037
                    y -= .014
            fig.text(.045, .025, f"R3 검증 | {index}/{len(slides)}",
                     fontsize=9, color="#607d8b")
            pdf.savefig(fig); plt.close(fig)


def _pdf_pages(path):
    pages = len(re.findall(rb"/Type\s*/Page\b", Path(path).read_bytes()))
    if pages <= 0:
        raise ValueError(f"could not read PDF page count: {path}")
    return pages


def _privacy_check(text, label):
    found = [pattern.pattern for pattern in PRIVATE_PATTERNS if pattern.search(text)]
    if found:
        raise ValueError(f"{label} contains prohibited publication/private wording: {found}")


def _atomic_text(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name + ".", delete=False) as stream:
        stream.write(content); temporary = Path(stream.name)
    temporary.replace(path)


def check_current(context, outputs=DEFAULT_OUTPUTS):
    results = render_results(context)
    notes = render_presentation_notes(context)
    for path, expected in zip(outputs[:2], (results, notes)):
        if not Path(path).is_file() or Path(path).read_text(encoding="utf-8") != expected:
            raise ValueError(f"generated document is absent or stale: {path}")
    pdf = Path(outputs[2])
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("generated presentation PDF is absent or invalid")
    if _pdf_pages(pdf) != len(_slides(context)):
        raise ValueError("current presentation PDF page count differs from the R3 slide specifications")
    archive = context["artifacts"]["r2_presentation"]
    if hashlib.sha256(archive["pdf_path"].read_bytes()).hexdigest() != archive["sha256"]:
        raise ValueError("archived R2 presentation changed after manifest validation")
    return True


def generate(manifest_path=DEFAULT_MANIFEST, outputs=DEFAULT_OUTPUTS, root=ROOT):
    context = validate_manifest(_load(manifest_path), root)
    results, notes = render_results(context), render_presentation_notes(context)
    _privacy_check(results, "R3_RESULTS.md"); _privacy_check(notes, "PRESENTATION_R3.md")
    pdf_target = Path(outputs[2]); pdf_target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", dir=pdf_target.parent,
                                     delete=False) as stream:
        r3_pdf = Path(stream.name)
    try:
        make_pdf(r3_pdf, context)
        _atomic_text(outputs[0], results); _atomic_text(outputs[1], notes)
        r3_pdf.replace(pdf_target)
        archive = context["artifacts"]["r2_presentation"]
        if hashlib.sha256(archive["pdf_path"].read_bytes()).hexdigest() != archive["sha256"]:
            raise RuntimeError("archived R2 presentation changed during R3 generation")
    finally:
        r3_pdf.unlink(missing_ok=True)
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true",
                        help="validate hashes and require current generated MD/notes/PDF")
    args = parser.parse_args(argv)
    if args.check:
        context = validate_manifest(_load(args.manifest))
        check_current(context)
    else:
        generate(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
