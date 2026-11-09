"""Render the restored R7 decision without altering frozen scientific records."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import tempfile
import textwrap
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import r7_report as R
from tools.r7_json import read_json, replace_json, write_json


D = "results/v2_protocol/"
CURATION = D + "R7_RESTORE_REPORT_CURATION.json"
RESTORATION = D + "R7_RESTORATION.json"
VALIDATION = D + "R7_RESTORATION_VALIDATION.json"
RESTORATION_MEDIA = D + "R7_RESTORATION_MEDIA.json"
MEASUREMENTS = [
    D + "R7_BASE_holdout_R1_servo.json",
    D + "R7_BASE_holdout_R1_torque.json",
    D + "R7_BASE_holdout_R7_servo.json",
    D + "R7_BASE_holdout_R7_torque.json",
    D + "R7_HOLDOUT_DESIGN.json",
    D + "R7_HOLDOUT_OPENED.json",
    D + "R7_HOLDOUT_FINALIZED.json",
    D + "R7_HOLDOUT_PAIRING.json",
    D + "R7_HOLDOUT_REVIEW.json",
    D + "R7_OFFICIAL_OPENED.json",
    D + "R7_OFFICIAL_REGRESSION.json",
    D + "R7_FINAL_CONTRACT_REVIEW.json",
]
SCIENTIFIC_RECORDS = MEASUREMENTS + [D + "PROTOCOL_R7.json", D + "R7_FREEZE.json"]
RESTORATION_REQUIRED = [
    RESTORATION,
    D + "R7_VETO_ADOPTION.json",
    D + "R7_FAILURE.json",
    D + "R7_EXECUTION_STATUS.json",
    D + "R7_REPORT_CURATION.json",
    D + "R7_S4_ADOPTION.json",
    "tools/r7_restore_report.py",
    "tools/test_r7_restore_report.py",
]
REQUIRED = list(dict.fromkeys(R.REQUIRED + SCIENTIFIC_RECORDS + RESTORATION_REQUIRED))
OPTIONAL = list(dict.fromkeys(R.OPTIONAL + [VALIDATION, RESTORATION_MEDIA,
                                           D + "R7_S4_MEDIA_MANIFEST.json"]))
OUTPUTS = list(R.OUTPUTS)

SERVO_SHA = "e26c203aabc476fc51a1464877a3435e484bb8fc33244777e59968f00969c643"
SERVO_CONTROL_SHA = "40706968fb72ed8d7a864e44e828787ae8db0365c2927e5166e4a5b2d83bba8b"
TORQUE_CONTROL_SHA = "8f66f8f03cde65fcc6d8ef2f11643691cd8fe2e96dad6eaeaa17a0c7aa3db371"
USER_QUOTE = "서보 후보 채택 복원"


def _sha(path: Path) -> str:
    return R.sha(path)


def _pin(root: Path, relative: str) -> dict:
    path = root / relative
    return {"sha256": _sha(path), "size_bytes": path.stat().st_size}


def build_curation(root: Path = ROOT) -> dict:
    root = Path(root)
    sources = {}
    pending = []
    for relative in REQUIRED + OPTIONAL:
        path = root / relative
        if path.is_file():
            sources[relative] = _pin(root, relative)
        elif relative in REQUIRED:
            pending.append(relative)
    return {
        "schema": "R7_RESTORE_REPORT_CURATION.v1",
        "sources": sources,
        "pending_required": pending,
        "outputs": OUTPUTS,
        "official_strings_preserve_source_precision": True,
        "private_raw_or_acquisition_inputs": False,
        "created_utc": "2000-01-01T00:00:00+00:00",
    }


def validate_curation(root: Path, complete: bool = True) -> dict:
    root = Path(root)
    document = read_json(root / CURATION)
    if document.get("schema") != "R7_RESTORE_REPORT_CURATION.v1":
        raise ValueError("bad restored R7 curation schema")
    if complete and document.get("pending_required"):
        raise ValueError("restored report inputs pending: " + str(document["pending_required"]))
    for relative, metadata in document.get("sources", {}).items():
        path = root / relative
        if (not path.is_file() or _sha(path) != metadata.get("sha256")
                or path.stat().st_size != metadata.get("size_bytes")):
            raise ValueError("curated restoration input changed: " + relative)
    if complete and set(REQUIRED) - set(document.get("sources", {})):
        raise ValueError("restored curation lacks required inputs")
    existing = {name for name in OPTIONAL if (root / name).is_file()}
    if complete and existing - set(document.get("sources", {})):
        raise ValueError("restored curation lacks existing optional input")
    return document


def _verify_reference(root: Path, item: dict, relative: str, label: str) -> None:
    if item != {"file": relative, "sha256": _sha(root / relative)}:
        raise ValueError(label + " reference differs")


def validate_restoration(root: Path, restoration: dict, adoption: dict) -> None:
    if restoration.get("schema") != "R7_ADOPTION_RESTORATION.v1":
        raise ValueError("bad R7 restoration schema")
    if restoration.get("user_decision") != {"quote": USER_QUOTE}:
        raise ValueError("restoration decision quote differs")
    if restoration.get("final_decision") != {
            "servo": "SCALE_STATE", "torque": "R1", "final_assembly": "R7"}:
        raise ValueError("restored final decision differs")
    if restoration.get("basis") != {
            "protocol_fallback_excludes_document_checks": True,
            "execution_order_excludes_document_checks": True,
            "no_candidate_data_gate_change": True,
            "no_simulation_holdout_official_rerun": True}:
        raise ValueError("restoration basis differs")
    sources = restoration.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
            "s4_adoption", "veto_adoption", "failure"}:
        raise ValueError("restoration source inventory differs")
    source_specs = (
        ("s4_adoption", D + "R7_S4_ADOPTION.json", "S4 adoption"),
        ("veto_adoption", D + "R7_VETO_ADOPTION.json", "veto adoption"),
        ("failure", D + "R7_FAILURE.json", "historical failure"),
    )
    for key, relative, label in source_specs:
        _verify_reference(root, sources.get(key), relative, label)
    for key, relative, label in (
        ("historical_veto_execution_status", D + "R7_EXECUTION_STATUS.json", "execution status"),
        ("historical_veto_report_curation", D + "R7_REPORT_CURATION.json", "historical curation"),
    ):
        _verify_reference(root, restoration.get(key), relative, label)
    _verify_reference(root, restoration.get("candidate"),
                      "experiments/r7_final/A_STATE.py", "servo candidate")
    expected_files = {
        "my_detector.py": SERVO_SHA,
        "my_servo.py": SERVO_CONTROL_SHA,
        "my_torque.py": TORQUE_CONTROL_SHA,
    }
    if restoration.get("final_files") != expected_files:
        raise ValueError("restored final file hashes differ")
    for relative, digest in expected_files.items():
        if not (root / relative).is_file() or _sha(root / relative) != digest:
            raise ValueError("restored submission file changed: " + relative)
    scientific = restoration.get("scientific_record_sha256")
    if not isinstance(scientific, dict) or set(scientific) != set(SCIENTIFIC_RECORDS):
        raise ValueError("restoration scientific record inventory differs")
    for relative in SCIENTIFIC_RECORDS:
        if scientific[relative] != _sha(root / relative):
            raise ValueError("restoration scientific record changed: " + relative)
    models = restoration.get("preserved_models")
    if not isinstance(models, dict) or len(models) != 8:
        raise ValueError("restoration must preserve all eight models")
    for relative, digest in models.items():
        if not isinstance(relative, str) or not (root / relative).is_file() or _sha(root / relative) != digest:
            raise ValueError("restoration model record changed: " + str(relative))
    zero_actions = ("training_runs", "new_physical_simulations", "new_holdout_batches",
                    "new_official_batches", "threshold_updates", "remote_operations")
    if any(type(restoration.get(key)) is not int or restoration[key] != 0
           for key in zero_actions):
        raise ValueError("restoration zero-action counters differ")
    if restoration.get("current_validation_record") != VALIDATION:
        raise ValueError("restoration validation path differs")

    s4 = read_json(root / (D + "R7_S4_ADOPTION.json"))
    for key, value in s4.items():
        if adoption.get(key) != value:
            raise ValueError("canonical adoption differs from S4 core: " + key)
    if adoption.get("effective_variants") != {"servo": "SCALE_STATE", "torque": "R1"}:
        raise ValueError("restored architecture decision differs")
    if adoption.get("final_assembly") != "R7" or adoption.get("final_files") != expected_files:
        raise ValueError("restored assembly differs")
    _verify_reference(root, adoption.get("restoration_decision"), RESTORATION,
                      "canonical restoration")
    for key, source_key in (("source_S4_adoption", "s4_adoption"),
                            ("source_veto_adoption", "veto_adoption"),
                            ("source_failure", "failure")):
        if adoption.get(key) != sources.get(source_key):
            raise ValueError("canonical adoption provenance differs: " + key)
    if (adoption.get("historical_publication_veto") is not True
            or adoption.get("restored_by_explicit_user_decision") is not True
            or adoption.get("user_decision") != {"quote": USER_QUOTE}
            or adoption.get("no_new_physical_evaluation") is not True
            or adoption.get("current_validation_record") != VALIDATION):
        raise ValueError("canonical restoration flags differ")


def validate_media_restoration(root: Path, restoration: dict, adoption: dict) -> None:
    historical_relative = D + "R7_S4_MEDIA_MANIFEST.json"
    _verify_reference(root, restoration.get("historical_media_manifest"),
                      historical_relative, "historical media manifest")
    expected_paths = {
        "artifacts/media/09_r7_servo_positive_comparison.gif",
        "artifacts/media/09_r7_servo_positive_comparison.mp4",
        "artifacts/media/10_r7_servo_negative_comparison.gif",
        "artifacts/media/10_r7_servo_negative_comparison.mp4",
    }
    restored = restoration.get("restored_media_original_sha256")
    if not isinstance(restored, dict) or set(restored) != expected_paths:
        raise ValueError("restored media inventory differs")
    for relative, digest in restored.items():
        if not (root / relative).is_file() or _sha(root / relative) != digest:
            raise ValueError("restored media bytes changed: " + relative)

    historical = read_json(root / historical_relative)
    current_path = root / "artifacts/media/r7_manifest.json"
    current = read_json(current_path)
    adoption_sha = _sha(root / (D + "R7_ADOPTION.json"))
    adoption_binding = {"file": D + "R7_ADOPTION.json", "sha256": adoption_sha,
                        "effective_variants": {"servo": "SCALE_STATE", "torque": "R1"}}
    historical_binding = dict(adoption_binding)
    historical_binding["sha256"] = restoration["sources"]["s4_adoption"]["sha256"]
    if current.get("adoption") != adoption_binding:
        raise ValueError("current media manifest is not bound to canonical adoption")
    if historical.get("adoption") != historical_binding:
        raise ValueError("historical media manifest is not bound to S4 adoption")
    historical_semantic = dict(historical)
    current_semantic = dict(current)
    historical_semantic.pop("adoption", None)
    current_semantic.pop("adoption", None)
    if current_semantic != historical_semantic:
        raise ValueError("restored media manifest changed beyond adoption binding")
    manifest_artifacts = {
        item.get("file"): item.get("sha256")
        for case in current.get("cases", []) for item in case.get("artifacts", [])
    }
    if manifest_artifacts != restored:
        raise ValueError("restored media manifest artifact witness differs")

    media_record_path = root / RESTORATION_MEDIA
    if media_record_path.is_file():
        media_record = read_json(media_record_path)
        if (media_record.get("schema") != "R7_RESTORATION_MEDIA.v1"
                or media_record.get("status") != "PASS"):
            raise ValueError("bad restoration media schema")
        if media_record.get("restoration_sha256") != _sha(root / RESTORATION):
            raise ValueError("restoration media decision hash differs")
        if media_record.get("historical_manifest") != {
                "file": historical_relative, "sha256": _sha(root / historical_relative)}:
            raise ValueError("restoration media historical hash differs")
        if media_record.get("current_manifest") != {
                "file": "artifacts/media/r7_manifest.json", "sha256": _sha(current_path)}:
            raise ValueError("restoration media current hash differs")
        if media_record.get("adoption_sha256") != adoption_sha:
            raise ValueError("restoration media adoption hash differs")
        if media_record.get("original_media_sha256") != restored:
            raise ValueError("restoration media byte witness differs")
        if media_record.get("only_manifest_adoption_sha_rebound") is not True:
            raise ValueError("restoration media rebind scope differs")
        zero_actions = ("new_rendered_media", "new_physical_simulations",
                        "new_holdout_batches", "new_official_batches", "remote_operations")
        if any(type(media_record.get(key)) is not int or media_record[key] != 0
               for key in zero_actions):
            raise ValueError("restoration media zero-action counters differ")


def validate_restoration_validation(root: Path, document: dict) -> dict:
    if document.get("schema") != "R7_RESTORATION_VALIDATION.v1" or document.get("status") != "PASS":
        raise ValueError("bad restoration validation record")
    if (document.get("restoration_sha256") != _sha(root / RESTORATION)
            or document.get("adoption_sha256") != _sha(root / (D + "R7_ADOPTION.json"))
            or document.get("detector_sha256") != SERVO_SHA):
        raise ValueError("restoration validation provenance differs")
    for key in ("new_physical_simulations", "new_holdout_batches", "new_official_batches"):
        if type(document.get(key)) is not int or document[key] != 0:
            raise ValueError("restoration validation execution counter differs")
    for label in ("work", "public"):
        record = document.get(label)
        if not isinstance(record, dict):
            raise ValueError("restoration validation lacks " + label)
        suite = record.get("full_suite", {})
        contract = record.get("default_submission_contract", {})
        cases = contract.get("cases", [])
        assembly = record.get("tested_assembly", {})
        if (suite.get("status") != "PASS" or type(suite.get("tests_run")) is not int
                or suite["tests_run"] < 1 or suite.get("errors") != 0
                or suite.get("failures") != 0 or contract.get("status") != "PASS"
                or not isinstance(cases, list) or len(cases) != 8
                or any(case.get("status") != "PASS" for case in cases)
                or assembly != {"path": "my_detector.py", "sha256": SERVO_SHA}):
            raise ValueError("restoration validation failed for " + label)
    return document


def _positive_context(root: Path) -> dict:
    root = Path(root)
    validate_curation(root)
    docs = {Path(name).stem: read_json(root / name) for name in REQUIRED + OPTIONAL
            if name.endswith(".json") and (root / name).is_file()}
    adoption = docs["R7_ADOPTION"]
    validate_restoration(root, docs["R7_RESTORATION"], adoption)
    validate_media_restoration(root, docs["R7_RESTORATION"], adoption)
    effective = R._actual_adoption(adoption)
    if effective != {"servo": "SCALE_STATE", "torque": "R1"}:
        raise ValueError("restoration must select servo candidate and torque R1")

    protocol = docs["PROTOCOL_R7"]
    inherited = read_json(root / (D + "PROTOCOL_R5.json"))
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
    equivalence = R._equivalence_text(saved, parity, docs["R7_PARITY_PLAN"])
    pairing = docs["R7_HOLDOUT_PAIRING"]
    if pairing.get("episodes_run") != 152 or pairing.get("pair_count") != 76:
        raise ValueError("R7 holdout pairing does not cover 152 episodes")
    official = docs["R7_OFFICIAL_REGRESSION"]
    episodes = official.get("episodes_started", {})
    if (episodes.get("total") != 58 or episodes.get("required_total") != 58
            or episodes.get("per_mode") != {"servo": 29, "torque": 29}):
        raise ValueError("R7 official batch does not cover 58 episodes")
    serialized = json.dumps(official)
    if "R3_delay_ms" in serialized or "R3_false_alarms" in serialized:
        raise ValueError("R7 official serialization reused historical fields")
    if not all(token in serialized for token in ("tested_delay_ms", "tested_false_alarms")):
        raise ValueError("R7 official result lacks tested fields")
    architectures = official.get("architectures", {})
    torque_identity = architectures.get("torque", {}).get("exact_R1_identity", {})
    if (torque_identity.get("status") != "PASS" or torque_identity.get("checked_cells") != 66
            or torque_identity.get("raw_tokens_preserved") is not True
            or torque_identity.get("errors") != []):
        raise ValueError("official torque R1 identity proof differs")
    servo_identity = architectures.get("servo", {}).get("exact_R1_identity", {})
    servo_det = architectures.get("servo", {}).get("exact_R5_candidate_determinism", {})
    if (servo_identity.get("status") != "NOT_APPLICABLE"
            or servo_identity.get("checked_cells") != 0
            or servo_identity.get("raw_tokens_preserved") is not True
            or servo_identity.get("errors") != []
            or servo_det.get("status") != "PASS"
            or servo_det.get("checked_cells") != 54
            or servo_det.get("timing_compared") is not False
            or servo_det.get("errors") != []):
        raise ValueError("official servo candidate determinism proof differs")
    final_contract = docs["R7_FINAL_CONTRACT_REVIEW"]
    if str(final_contract.get("status")).upper() != "PASS":
        raise ValueError("failed final contract cannot restore R7")
    prefinal = R._validate_prefinal_contract(docs.get("R7_PREFREEZE_FINAL_CONTRACT"))
    historical = R._historical_context(root)
    result = {
        "root": root, "docs": docs, "adoption": adoption, "effective": effective,
        "adopted": True, "holdout": docs["R7_HOLDOUT_REVIEW"], "official": official,
        "official_fresh": True, "historical": historical, "equivalence": equivalence,
        "holdout_rows": R._holdout_rows(docs["R7_HOLDOUT_REVIEW"]),
        "control_rows": R._control_rows(official),
        "collision_rows": R._collision_rows(official),
        "t_rows": R._t_rows(historical, True), "contract": contract,
        "prefinal_contract": prefinal, "final_contract": final_contract,
        "negative": False, "parity_timing_rows": R._parity_timing_from_base(docs),
    }
    if "R7_RESTORATION_VALIDATION" in docs:
        result["restoration_validation"] = validate_restoration_validation(
            root, docs["R7_RESTORATION_VALIDATION"])
    result["inherited_rows"] = R._inherited_rows(result)
    return result


def _history_note() -> str:
    return ("S5 문서 연결 검사 뒤 R1으로 되돌린 기록은 보존했다. 이후 명시적 사용자 결정 "
            f"‘{USER_QUOTE}’에 따라 A 서보 후보를 최종 제출에 복원했다. "
            "protocol의 되돌림 조건과 실행 순서에는 문서 연결 검사가 포함되지 않는다. "
            "후보·자료·관문 숫자는 바꾸지 않았고 시뮬레이션·holdout·공식 평가는 다시 실행하지 않았다.")


def _validation_note(c: dict) -> str:
    document = c.get("restoration_validation")
    if document is None:
        return ""
    work = document["work"]["full_suite"]["tests_run"]
    public = document["public"]["full_suite"]["tests_run"]
    return (f"복원한 최종 조립으로 작업 사본 {work}개와 공개형 사본 {public}개 전체 테스트, "
            "각 기본 제출 계약 8개를 새로 통과했다. 기존 677개 기록은 S4 후보 계약의 보존값이며 "
            "이번 테스트 수와 합치지 않는다.")


def _comparison_rows(c: dict) -> list[dict]:
    current = {(row["방식"], row["질량"]): row for row in c["control_rows"]}
    old = {(row["방식"], row["질량"]): row
           for row in R._control_rows(R._preserved_r1_official(c["root"]))}
    rows = []
    for key in ((mode, f"{mass:.1f} kg") for mode in R.MODES.values() for mass in (0.0, 4.0, 8.0)):
        rows.append({"방식": key[0], "질량": key[1],
                     "R1 TCP RMS/최대": old[key]["TCP RMS/최대"],
                     "R7 TCP RMS/최대": current[key]["TCP RMS/최대"],
                     "R1 집기/놓기": old[key]["집기/놓기 계단"],
                     "R7 집기/놓기": current[key]["집기/놓기 계단"]})
    return rows


def _comparison_table(c: dict) -> str:
    return R.md_table(["방식", "질량", "R1 TCP RMS/최대", "R7 TCP RMS/최대",
                       "R1 집기/놓기", "R7 집기/놓기"], _comparison_rows(c))


def _collision_comparison_rows(c: dict) -> list[dict]:
    current = {(row["방식"], row["질량/시각"]): row for row in c["collision_rows"]}
    old = {(row["방식"], row["질량/시각"]): row
           for row in R._collision_rows(R._preserved_r1_official(c["root"]))}
    rows = []
    forces = ("10 N", "15 N", "40 N", "100 N")
    for mode in R.MODES.values():
        key = (mode, "8 kg / 3 s")
        rows.append({"방식": mode,
                     "R1 10/15/40/100 N": " / ".join(old[key][force] for force in forces),
                     "R7 10/15/40/100 N": " / ".join(current[key][force] for force in forces),
                     "R1/R7 충돌 세트 FA": (old[key]["충돌 세트 FA"] + " / "
                                             + current[key]["충돌 세트 FA"])})
    return rows


def _comparison_block(c: dict) -> str:
    control = _comparison_table(c)
    collision = R.md_table(["방식", "R1 10/15/40/100 N", "R7 10/15/40/100 N",
                            "R1/R7 충돌 세트 FA"], _collision_comparison_rows(c))
    return control + "\n\n8 kg, 3.0 s 운반 충돌의 R1/R7 비교다.\n\n" + collision


def _insert(text: str, marker: str, value: str) -> str:
    if marker not in text:
        raise ValueError("report insertion marker missing: " + marker)
    return text.replace(marker, value + "\n\n" + marker, 1)


def render_readme(c: dict) -> str:
    text = R.render_readme(c).replace("R7 기록 수정 재검증", "R7 서보 후보 채택 복원", 1)
    final = f"**최종 제출: {R._conclusion(c)}**\n\n"
    if final not in text:
        raise ValueError("README final decision marker missing")
    note = "## 판정 복원 경위\n\n" + _history_note()
    validation = _validation_note(c)
    if validation:
        note += " " + validation
    text = text.replace(final, final + note + "\n\n", 1)
    return _insert(text, "## 요구사항 점검",
                   "## 공식 R1과 R7 실측 비교\n\n" + _comparison_block(c))


def render_results(c: dict) -> str:
    text = R.render_results(c)
    note = "## 판정 복원 경위\n\n" + _history_note()
    if _validation_note(c):
        note += " " + _validation_note(c)
    text = _insert(text, "## 기록 결함과 사전 시험", note)
    return _insert(text, "## 요구사항·평가기 점검",
                   "## 공식 R1과 R7 실측 비교\n\n" + _comparison_block(c))


def render_notes(c: dict) -> str:
    text = R.render_notes(c)
    note = "### 판정 복원 경위\n\n" + _history_note()
    if _validation_note(c):
        note += " " + _validation_note(c)
    text = _insert(text, "## 2. 수정 범위", note)
    return _insert(text, "## 8. 요구사항과 남은 한계",
                   "## 8. 공식 R1과 R7 실측 비교\n\n" + _comparison_block(c)).replace(
                       "## 8. 요구사항과 남은 한계", "## 9. 요구사항과 남은 한계", 1)


def _slides(c: dict) -> list[dict]:
    pages = list(R._slides(c))
    history = [_history_note()]
    if _validation_note(c):
        history.append(_validation_note(c))
    pages.insert(1, {"title": "판정 복원 경위", "bullets": history})
    pages.insert(9, {"title": "공식 R1과 R7 실측 비교",
                     "table": (["방식", "질량", "R1 TCP RMS/최대", "R7 TCP RMS/최대",
                                "R1 집기/놓기", "R7 집기/놓기"], _comparison_rows(c))})
    pages.insert(10, {"title": "8 kg 운반 충돌 R1/R7 비교",
                      "table": (["방식", "R1 10/15/40/100 N", "R7 10/15/40/100 N",
                                 "R1/R7 충돌 세트 FA"], _collision_comparison_rows(c))})
    return pages


def make_pdf(path: Path, c: dict) -> None:
    R._configure_plot()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    metadata = {"Title": "R7 restored servo adoption", "Creator": "r7_restore_report.py",
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
                heights = R.table_row_heights(cells, len(columns))
                for (row, _), cell in table.get_celld().items():
                    cell.set_height(heights[row])
                    cell.set_edgecolor("#b0bec5")
                    cell.set_facecolor("#dbe7f3" if row == 0 else "#ffffff")
            figure.text(.04, .025, f"R7 검증 | {index}/{len(pages)}", fontsize=9, color="#607d8b")
            pdf.savefig(figure)
            plt.close(figure)


def render_all(c: dict) -> tuple[str, str, str]:
    return render_readme(c), render_results(c), render_notes(c)


def generate(root: Path = ROOT) -> dict:
    root = Path(root)
    c = _positive_context(root)
    for name, text in zip(OUTPUTS[:3], render_all(c)):
        R.safe(text, name)
        R.write(root / name, text)
    output = root / OUTPUTS[3]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        make_pdf(temporary, c)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    index = R.render_index(root)
    R.safe(index, "docs/INDEX.md")
    R.write(root / "docs/INDEX.md", index)
    return c


def check_one(root: Path) -> bool:
    root = Path(root)
    c = _positive_context(root)
    for name, expected in zip(OUTPUTS[:3], render_all(c)):
        path = root / name
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ValueError("stale restored report output: " + name)
        R.safe(expected, name)
    index = R.render_index(root)
    if (root / "docs/INDEX.md").read_text(encoding="utf-8") != index:
        raise ValueError("stale restored report output: docs/INDEX.md")
    R.check_links(root)
    pdf = root / OUTPUTS[3]
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF"):
        raise ValueError("missing restored R7 PDF")
    extracted = R.pdf_text(pdf)
    for token in ("R7 최종 판정", "판정 복원 경위", "공식 R1과 R7 실측 비교", "새 holdout", "미달"):
        if token not in extracted:
            raise ValueError("PDF text missing " + token)
    with tempfile.NamedTemporaryFile(suffix=".pdf") as stream:
        make_pdf(Path(stream.name), c)
        if Path(stream.name).read_bytes() != pdf.read_bytes():
            raise ValueError("restored R7 PDF is stale or nondeterministic")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--build-curation", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--projection", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.build_curation:
        path = root / CURATION
        document = build_curation(root)
        replace_json(path, document) if path.exists() else write_json(path, document)
        print(path)
        return
    if args.check:
        check_one(root)
        for projection in args.projection:
            check_one(projection.resolve())
        print("restored R7 report check passed")
        return
    generate(root)
    print("restored R7 report generated")


if __name__ == "__main__":
    main()
