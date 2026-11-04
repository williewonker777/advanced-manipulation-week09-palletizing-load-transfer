"""Generate the frozen R2 Korean report and 16-page presentation.

This is a postprocessor only.  It never imports the simulator or controller.
Final generation requires an explicit manifest so that absent R2 evidence cannot
be silently replaced with R1 results or a guessed value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import textwrap
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "results" / "v2_protocol" / "R2_REPORT_MANIFEST.json"
OUTPUTS = (
    ROOT / "docs" / "R2_RESULTS.md",
    ROOT / "report" / "PRESENTATION_R2.md",
    ROOT / "results" / "presentation.pdf",
)
ARTIFACT_KEYS = {
    "p0", "protocol", "validation_review", "prne_math", "prne_normal",
    "stability_full", "stability_compact", "r1_stability", "allocation_comparison",
    "common_reference", "control_detector_parity", "development", "development_review", "d4",
    "d4_reused", "lie_comparison", "environment_integrity",
    "perturbations", "holdout", "holdout_opened", "freeze", "adoption", "official",
    "official_started", "preholdout_review", "holdout_review",
    "timing_review",
}
PROVENANCE = {"fresh", "reused_exact_source"}
DECISIONS = {"adopt", "reject", "retain_r1"}
PRIVATE_PATTERNS = (
    re.compile("notion" + r"\.com", re.IGNORECASE),
    re.compile("/" + "home/", re.IGNORECASE),
    re.compile(r"TASK" + "-" + r"\d{3,}", re.IGNORECASE),
    re.compile("notes" + "_" + "digest", re.IGNORECASE),
)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required R2 input is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require_keys(value: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - value.keys())
    if missing:
        raise ValueError(f"{context} lacks required keys: {', '.join(missing)}")


def _inside_root(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT != path and ROOT not in path.parents:
        raise ValueError(f"artifact path escapes repository: {relative}")
    return path


def _artifact_entries(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.artifacts must be an object")
    missing = sorted(ARTIFACT_KEYS - artifacts.keys())
    extra = sorted(artifacts.keys() - ARTIFACT_KEYS)
    if missing or extra:
        raise ValueError(f"artifact key mismatch; missing={missing}, extra={extra}")
    answer: dict[str, list[dict[str, Any]]] = {}
    for key, raw in artifacts.items():
        entries = raw if isinstance(raw, list) else [raw]
        if not entries or any(not isinstance(item, dict) for item in entries):
            raise ValueError(f"artifact {key} must contain one or more objects")
        answer[key] = entries
        for index, item in enumerate(entries):
            require_keys(item, ("path", "provenance", "sha256"), f"artifact {key}[{index}]")
            if item["provenance"] not in PROVENANCE:
                raise ValueError(f"artifact {key}[{index}] has invalid provenance")
            path = _inside_root(str(item["path"]))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"artifact hash mismatch: {item['path']}")
            load_object(path)
    return answer


def _validate_public_documents(manifest: dict[str, Any]) -> list[dict[str, str]]:
    documents = manifest.get("method_documents")
    if not isinstance(documents, list) or len(documents) != 3:
        raise ValueError("method_documents must identify the two R2 derivations and literature")
    expected = {"docs/R2_METHODS.md", "docs/R2_ROBUST_DERIVATION.md", "docs/LITERATURE.md"}
    paths = set()
    for index, item in enumerate(documents):
        if not isinstance(item, dict):
            raise ValueError(f"method_documents[{index}] must be an object")
        require_keys(item, ("path", "sha256"), f"method_documents[{index}]")
        path = _inside_root(str(item["path"]))
        paths.add(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"public method document hash mismatch: {item['path']}")
        _privacy_check(path.read_text(encoding="utf-8"), item["path"])
    if paths != expected:
        raise ValueError(f"method_documents mismatch: expected {sorted(expected)}")
    literature = manifest.get("literature_updates")
    if not isinstance(literature, list) or not literature:
        raise ValueError("literature_updates must list verified primary links")
    for index, item in enumerate(literature):
        if not isinstance(item, dict):
            raise ValueError(f"literature_updates[{index}] must be an object")
        require_keys(item, ("title", "url", "verified"), f"literature_updates[{index}]")
        if item["verified"] is not True or not str(item["url"]).startswith("https://"):
            raise ValueError("literature updates require a directly verified HTTPS primary link")
    return documents


def _one_document(entries: dict[str, list[dict[str, Any]]], key: str) -> dict[str, Any]:
    if len(entries[key]) != 1:
        raise ValueError(f"artifact {key} must identify exactly one JSON document")
    return load_object(_inside_root(str(entries[key][0]["path"])))


def _validate_metrics(metrics: Any) -> list[dict[str, Any]]:
    if not isinstance(metrics, list) or len(metrics) != 6:
        raise ValueError("metrics must contain exactly T1 through T6")
    expected = [f"T{i}" for i in range(1, 7)]
    if [row.get("id") for row in metrics] != expected:
        raise ValueError("metrics must be ordered T1, T2, ..., T6")
    required = ("id", "name", "unit", "condition", "baseline", "v1", "r1",
                "r2", "r2_status", "source_keys")
    for row in metrics:
        require_keys(row, required, f"metric {row.get('id', '?')}")
        if not isinstance(row["source_keys"], list) or not row["source_keys"]:
            raise ValueError(f"metric {row['id']} must cite source_keys")
        unknown = set(row["source_keys"]) - ARTIFACT_KEYS
        if unknown:
            raise ValueError(f"metric {row['id']} cites unknown artifacts: {sorted(unknown)}")
        if row["r2_status"] not in {"met", "not_met", "not_applicable"}:
            raise ValueError(f"metric {row['id']} has invalid r2_status")
    return metrics


def _validate_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_decisions must be a non-empty array")
    families = set()
    for row in candidates:
        require_keys(row, ("id", "family", "method", "architecture", "decision",
                           "failed_gates", "false_rises", "g4_pass_cases",
                           "worst_max_ratio", "worst_rms_ratio",
                           "worst_place_step_ratio", "combined_p99_us",
                           "combined_max_us", "local_radius_pass", "source_keys"),
                     f"candidate {row.get('id', '?')}")
        families.add(row["family"])
        if row["decision"] not in DECISIONS:
            raise ValueError(f"candidate {row['id']} has invalid decision")
        if not isinstance(row["failed_gates"], list) or not row["failed_gates"]:
            raise ValueError(f"candidate {row['id']} must retain its gate results")
        if not isinstance(row["source_keys"], list) or not row["source_keys"]:
            raise ValueError(f"candidate {row['id']} must cite source keys")
        unknown = set(row["source_keys"]) - ARTIFACT_KEYS
        if unknown:
            raise ValueError(f"candidate {row['id']} cites unknown artifacts: {sorted(unknown)}")
    missing = {f"C{i}" for i in range(1, 7)} - families
    if missing:
        raise ValueError(f"candidate decisions omit registered families: {sorted(missing)}")
    return candidates


def _validate_lie_usage(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("lie_group_usage must be a non-empty array")
    required = ("method", "section", "formula", "plant_adaptation",
                "euclidean_comparison", "decision", "source_keys")
    for row in rows:
        require_keys(row, required, f"Lie-group row {row.get('method', '?')}")
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Lie-group row {row['method']} has invalid decision")
        if not isinstance(row["source_keys"], list) or not row["source_keys"]:
            raise ValueError(f"Lie-group row {row['method']} must cite source keys")
        unknown = set(row["source_keys"]) - ARTIFACT_KEYS
        if unknown:
            raise ValueError(f"Lie-group row {row['method']} cites unknown artifacts: {sorted(unknown)}")
        adaptation = str(row["plant_adaptation"]).lower()
        for token in ("elastic", "1 ms", "zoh", "delay"):
            if token not in adaptation:
                raise ValueError(f"Lie-group row {row['method']} must state {token} adaptation")
    return rows


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    require_keys(manifest, ("schema", "status", "remote_operations",
                            "artifacts", "metrics", "candidate_decisions",
                            "lie_group_usage", "misses_and_outliers", "verification",
                            "method_documents", "literature_updates", "changed_paths",
                            "limitations"),
                 "manifest")
    if manifest["schema"] != "R2_REPORT_MANIFEST.v1" or manifest["status"] != "frozen":
        raise ValueError("final report requires a frozen R2_REPORT_MANIFEST.v1")
    if manifest["remote_operations"] != 0:
        raise ValueError("remote_operations must be exactly 0")
    entries = _artifact_entries(manifest)
    method_documents = _validate_public_documents(manifest)
    metrics = _validate_metrics(manifest["metrics"])
    candidates = _validate_candidates(manifest["candidate_decisions"])
    lie_usage = _validate_lie_usage(manifest["lie_group_usage"])
    for field in ("misses_and_outliers", "verification"):
        if not isinstance(manifest[field], dict):
            raise ValueError(f"{field} must be an object")
    require_keys(manifest["misses_and_outliers"], ("summary", "unmet", "outliers_retained"),
                 "misses_and_outliers")
    require_keys(manifest["verification"], ("executed", "not_executed", "resources"),
                 "verification")
    for key in ("summary", "unmet", "outliers_retained"):
        if not isinstance(manifest["misses_and_outliers"][key], list):
            raise ValueError(f"misses_and_outliers.{key} must be an array")
    if not isinstance(manifest["changed_paths"], list) or not manifest["changed_paths"]:
        raise ValueError("changed_paths must be a non-empty array")
    if not isinstance(manifest["limitations"], list) or not manifest["limitations"]:
        raise ValueError("limitations must be a non-empty array")

    validation = _one_document(entries, "validation_review")
    require_keys(validation, ("official_used_for_selection", "holdout_opened", "results"),
                 "R2 validation review")
    if validation["official_used_for_selection"] is not False:
        raise ValueError("official evaluation must not be used for candidate selection")
    if not isinstance(validation["results"], list) or not validation["results"]:
        raise ValueError("R2 validation review contains no candidate rows")
    development_review = _one_document(entries, "development_review")
    require_keys(development_review, ("official_used_for_selection", "holdout_opened",
                                      "no_threshold_or_gain_tuning_after_collision",
                                      "observer_results", "control_eligible"),
                 "R2 development review")
    if development_review["official_used_for_selection"] is not False:
        raise ValueError("development review must precede official exposure")
    if development_review["holdout_opened"] is not False:
        raise ValueError("development review must precede holdout opening")
    if development_review["no_threshold_or_gain_tuning_after_collision"] is not True:
        raise ValueError("development review must prove no collision-driven threshold tuning")

    common_reference = _one_document(entries, "common_reference")
    require_keys(common_reference, ("official", "provenance", "R2_source_alias", "results"),
                 "common physical reference")
    if (entries["common_reference"][0]["provenance"] != "reused_exact_source"
            or common_reference["official"] is not False
            or len(common_reference["results"]) != 6):
        raise ValueError("common reference must be the six non-official base/v1/R1 rows")
    control_parity = _one_document(entries, "control_detector_parity")
    require_keys(control_parity, ("official", "pairs", "all_physical_pairs_exact", "comparison",
                          "keys", "results"), "control/detector parity")
    if (control_parity["official"] is not False or control_parity["pairs"] != 156
            or len(control_parity["results"]) != 156
            or control_parity["all_physical_pairs_exact"] is not True):
        raise ValueError("control/detector parity must retain all 156 exact physical pairs")
    lie_comparison = _one_document(entries, "lie_comparison")
    if (lie_comparison.get("official") is not False
            or lie_comparison.get("selection_unchanged") is not True
            or not isinstance(lie_comparison.get("results"), list)
            or len(lie_comparison["results"]) != 4):
        raise ValueError("Lie comparison must retain four non-official matched rows")
    environment = _one_document(entries, "environment_integrity")
    if (environment.get("original_entire_tree_unchanged") is not True
            or environment.get("official_venv_packages_unchanged") is not True
            or environment.get("packages_added") != [] or environment.get("packages_removed") != []):
        raise ValueError("original tree and official environment must remain unchanged")

    protocol = _one_document(entries, "protocol")
    if protocol.get("remote_operations_allowed") != 0:
        raise ValueError("PROTOCOL_R2 must prohibit remote operations")
    redesign = protocol.get("redesign")
    if (not isinstance(redesign, dict) or redesign.get("max_per_family") != 1
            or redesign.get("post_official_allowed") is not False):
        raise ValueError("PROTOCOL_R2 redesign rule must remain one per family and post-official forbidden")
    adoption = _one_document(entries, "adoption")
    require_keys(adoption, ("decision", "holdout_opened", "official_used_for_selection",
                            "post_holdout_or_official_tuning_allowed", "redesigns_used"),
                 "R2 adoption")
    if adoption["holdout_opened"] is not False or adoption["official_used_for_selection"] is not False:
        raise ValueError("adoption must precede holdout and official exposure")
    if adoption["post_holdout_or_official_tuning_allowed"] is not False:
        raise ValueError("post-holdout or post-official tuning must remain prohibited")
    if adoption["redesigns_used"] != 0:
        raise ValueError("frozen adoption must retain zero redesigns used")
    freeze = _one_document(entries, "freeze")
    require_keys(freeze, ("holdout_opened", "official_used_for_selection",
                          "post_holdout_and_official_tuning_allowed"), "R2 freeze")
    if any((freeze["holdout_opened"], freeze["official_used_for_selection"],
            freeze["post_holdout_and_official_tuning_allowed"])):
        raise ValueError("freeze record must precede exposure and prohibit later tuning")
    d4 = _one_document(entries, "d4")
    if not isinstance(d4.get("results"), list) or len(d4["results"]) != 770:
        raise ValueError("fresh R2 base D4 must retain all 770 registered rows")
    d4_reused = _one_document(entries, "d4_reused")
    if (entries["d4_reused"][0]["provenance"] != "reused_exact_source"
            or not isinstance(d4_reused.get("results"), list)
            or len(d4_reused["results"]) != 1540):
        raise ValueError("R1/v1 D4 comparison must cite all 1540 exact-source rows")
    perturbations = _one_document(entries, "perturbations")
    perturbation_parity = perturbations.get("metadata", {}).get("nominal_parity")
    if (not isinstance(perturbations.get("results"), list)
            or len(perturbations["results"]) != 272
            or not isinstance(perturbation_parity, list) or len(perturbation_parity) != 8
            or any(row.get("passed") is not True for row in perturbation_parity)):
        raise ValueError("R2 perturbations must retain 272 rows and eight passing parity checks")
    holdout = _one_document(entries, "holdout")
    if (not isinstance(holdout.get("results"), list) or len(holdout["results"]) != 152
            or holdout.get("metadata", {}).get("one_batch") is not True):
        raise ValueError("sealed holdout must retain one batch of 152 rows")
    holdout_opened = _one_document(entries, "holdout_opened")
    if holdout_opened.get("batch_attempt") != 1:
        raise ValueError("holdout opening marker must record exactly one batch attempt")
    preholdout = _one_document(entries, "preholdout_review")
    if (preholdout.get("holdout_opened") is not False
            or preholdout.get("official_exposed") is not False
            or preholdout.get("source_alias_R2_R1") is not True):
        raise ValueError("pre-holdout review must precede exposure and retain R1 alias")
    holdout_review = _one_document(entries, "holdout_review")
    if (holdout_review.get("one_batch") is not True
            or holdout_review.get("selection_unchanged") is not True
            or holdout_review.get("final_R2_alias") != "R1"
            or holdout_review.get("no_tuning_after_holdout") is not True):
        raise ValueError("holdout review must preserve the frozen R1 alias without retuning")
    official = _one_document(entries, "official")
    require_keys(official, ("official", "results", "regression_errors",
                            "R1_compatibility_pass", "holdout_sha256",
                            "tuning_after_exposure"), "official regression")
    if (official["official"] is not True or official["regression_errors"] != []
            or official["R1_compatibility_pass"] is not True
            or official["tuning_after_exposure"] is not False
            or set(official["results"]) != {"servo", "torque"}):
        raise ValueError("official regression must be the final unchanged R1-compatible A/B check")
    official_started = _one_document(entries, "official_started")
    if official_started.get("tuning_or_repair_after_official") is not False:
        raise ValueError("official-start marker must prohibit repair after exposure")
    timing = _one_document(entries, "timing_review")
    if (timing.get("official") is not False or timing.get("strict_goal_met") is not False
            or timing.get("global_GC_settings_changed") is not False
            or not isinstance(timing.get("results"), list) or len(timing["results"]) != 2
            or sum(int(row["rows"]) for row in timing["results"]) != 204
            or sum(int(row["cycles"]) for row in timing["results"]) != 1_224_000):
        raise ValueError("timing review must retain both architectures and all paired cycles")

    stability = _one_document(entries, "stability_compact")
    require_keys(stability, ("summary", "all_cases", "case_columns", "uncertainty"),
                 "compact stability ledger")
    summary = stability["summary"]
    require_keys(summary, ("total_cases", "strict_stable_cases", "protocol_radius_pass_cases"),
                 "stability summary")
    if summary["total_cases"] != len(stability["all_cases"]):
        raise ValueError("compact stability ledger does not retain every case")
    if summary["total_cases"] != 300:
        raise ValueError("final compact stability ledger must retain all 300 registered cases")
    full_stability = _one_document(entries, "stability_full")
    require_keys(full_stability, ("analyses", "summary", "scope_warning"),
                 "full stability record")
    if len(full_stability["analyses"]) != 300:
        raise ValueError("full stability record must retain all 300 pole analyses")

    prne = _one_document(entries, "prne_math")
    require_keys(prne, ("summary", "qualification", "cases"), "pRNE math evidence")
    if len(prne["cases"]) != 9:
        raise ValueError("pRNE math evidence must retain all nine registered cases")
    r1_stability = _one_document(entries, "r1_stability")
    require_keys(r1_stability, ("summary", "analyses", "global_scope_warning"),
                 "R1 stability evidence")
    if len(r1_stability["analyses"]) != 12:
        raise ValueError("R1 stability evidence must retain all 12 registered cases")
    allocation = _one_document(entries, "allocation_comparison")
    require_keys(allocation, ("rows", "global_GC_settings_changed", "plant_episodes", "warning"),
                 "allocation comparison")
    if allocation["global_GC_settings_changed"] is not False or allocation["plant_episodes"] != 0:
        raise ValueError("allocation comparison must remain a pure helper benchmark")
    if not isinstance(allocation["rows"], list) or len(allocation["rows"]) != 2:
        raise ValueError("allocation comparison must retain both matched methods")
    for row in allocation["rows"]:
        require_keys(row, ("method", "thread_median_us", "thread_p99_us",
                           "allocation_peak_bytes", "answer"), "allocation row")
    left, right = (row["answer"] for row in allocation["rows"])
    if len(left) != len(right) or max(abs(float(a) - float(b)) for a, b in zip(left, right)) > 1e-12:
        raise ValueError("allocation helpers must have numerically identical outputs")
    return {
        "manifest": manifest,
        "entries": entries,
        "metrics": metrics,
        "candidates": candidates,
        "lie_usage": lie_usage,
        "validation": validation,
        "development_review": development_review,
        "common_reference": common_reference,
        "parity": control_parity,
        "lie_comparison": lie_comparison,
        "environment": environment,
        "protocol": protocol,
        "adoption": adoption,
        "freeze": freeze,
        "d4": d4,
        "d4_reused": d4_reused,
        "perturbations": perturbations,
        "holdout": holdout,
        "holdout_opened": holdout_opened,
        "preholdout_review": preholdout,
        "holdout_review": holdout_review,
        "official": official,
        "official_started": official_started,
        "timing": timing,
        "stability": stability,
        "prne": prne,
        "r1_stability": r1_stability,
        "allocation": allocation,
        "method_documents": method_documents,
    }


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "통과" if value else "미달"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|")


def _source_table(entries: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["| 증거 묶음 | 파일 | 출처 |", "|---|---|---|"]
    for key in sorted(entries):
        for item in entries[key]:
            lines.append(f"| {key} | `{item['path']}` | {item['provenance']} |")
    return "\n".join(lines)


def _metric_table(metrics: list[dict[str, Any]]) -> str:
    lines = ["| 항목 | 동일 조건 | 단위 | 기준 | v1 | R1 | R2 | 판정 | 출처 |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in metrics:
        lines.append("| " + " | ".join((
            f"{row['id']} {row['name']}", _cell(row["condition"]), _cell(row["unit"]),
            _cell(row["baseline"]), _cell(row["v1"]), _cell(row["r1"]),
            _cell(row["r2"]), _cell(row["r2_status"]),
            ", ".join(row["source_keys"]),
        )) + " |")
    return "\n".join(lines)


def _candidate_table(rows: list[dict[str, Any]]) -> str:
    def scalar(value: Any, digits: int = 3) -> str:
        if value is None or value == "N/A":
            return "N/A"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return _cell(value)

    lines = ["| 후보 | 방식 | 실패 관문 | 오감지 | G4 통과 수 | TCP 최대/RMS 최악 비 | 놓기 계단 최악 비 | p99/최대 계산시간 [us] | 국소 radius 통과 수 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        gates = ", ".join(str(item) for item in row["failed_gates"])
        tcp = f"{scalar(row['worst_max_ratio'])}/{scalar(row['worst_rms_ratio'])}"
        if str(row["architecture"]).startswith("observer"):
            timing = "; ".join(f"{part}/N/A" for part in
                               str(row["combined_p99_us"]).split("; "))
        else:
            timing = f"{scalar(row['combined_p99_us'])}/{scalar(row['combined_max_us'])}"
        lines.append(f"| `{row['method']}` | {_cell(row['architecture'])} | {_cell(gates)} | "
                     f"{scalar(row['false_rises'])} | {scalar(row['g4_pass_cases'])} | {tcp} | "
                     f"{scalar(row['worst_place_step_ratio'])} | {timing} | "
                     f"{_cell(row['local_radius_pass'])} |")
    return "\n".join(lines)


def _redesign_section(context: dict[str, Any]) -> str:
    rule = context["protocol"]["redesign"]
    used = context["adoption"]["redesigns_used"]
    cost_only = {row["id"] for row in context["candidates"]
                 if row["failed_gates"] == ["cost"]}
    lines = [
        f"- protocol 허용량: 가족별 최대 {rule['max_per_family']}회; 실제 사용 {used}회.",
        "- 공식 회귀 이후 재설계 금지. holdout·공식 결과를 본 뒤 후보를 고치거나 재승격하지 않았다.",
    ]
    named = []
    for candidate in ("C3_DOB_w4.5/servo", "C4_P-IDC_w4.5_d1/torque"):
        if candidate in cost_only:
            named.append(f"`{candidate}`")
    if named:
        lines.append(f"- {', '.join(named)}: cost 관문만 실패한 상태에서 재설계 미사용; 사유 기록 없음.")
    else:
        lines.append("- 재설계 미사용 사유 기록 없음.")
    return "\n".join(lines)


def _lie_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| 방법 | 참조 절 | 직접 쓴 식 | 탄성 플랜트 적용 | 유클리드 대비 효과 | 판정 |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {_cell(row['method'])} | {_cell(row['section'])} | "
                     f"{_cell(row['formula'])} | {_cell(row['plant_adaptation'])} | "
                     f"{_cell(row['euclidean_comparison'])} | {row['decision']} |")
    return "\n".join(lines)


def render_results(context: dict[str, Any]) -> str:
    manifest = context["manifest"]
    stability = context["stability"]["summary"]
    r1 = context["r1_stability"]["summary"]
    math_summary = context["prne"]["summary"]
    allocation = context["allocation"]
    timing = context["timing"]
    misses = manifest["misses_and_outliers"]
    verification = manifest["verification"]
    return f"""# R2 결과: 제어·Lie group 2차 검증

## 결론

{manifest.get('conclusion', '동결된 채택 판정은 아래 후보 표와 수치표를 따른다.')}

공식 평가는 마지막 회귀 확인에만 사용했다. 후보 선정·임계값 조정에는 사용하지 않았다.

## T1–T6 동일 조건 비교

{_metric_table(context['metrics'])}

## P0 측정

{chr(10).join('- ' + str(x) for x in manifest.get('p0_findings', []))}

## 후보별 판정

{_candidate_table(context['candidates'])}

표의 {len(context['candidates'])}개 판정 행은 모두 최종 reject다. C5 정상 validation 6행은
제어 gate를 통과했지만 새 채택이 아니며, observer 효능 판정은 별도 행에서
reject로 기록했다. 모든 등록 후보와 불리한 결과를 유지했다. 검증 단계에서 버린 후보를 holdout이나
공식 평가 결과를 본 뒤 되살리지 않았다.
[원자료: `R2_ADOPTION.json`](../results/v2_protocol/R2_ADOPTION.json)

### 재설계 사용 여부

{_redesign_section(context)}

## Lie group 사용 내역

{_lie_table(context['lie_usage'])}

교재의 강체 안정성 결과를 그대로 이 탄성 관절 플랜트의 보장으로 사용하지 않았다.
각 제어 후보는 탄성 관절, 1 ms ZOH, 한 주기 지연을 포함한 유한 국소 분석으로만
판정했다. 구성 의존 PD gain `M(q)`, load blending, clipping도 강체 정리의 전제를
바꾸므로 수동 플랜트 보장을 주장하지 않는다.

## 명시적 한계

{chr(10).join('- ' + str(x) for x in manifest['limitations'])}

## 수학·안정성 검증

- pRNE 유한 표본: `{json.dumps(math_summary, ensure_ascii=False, sort_keys=True)}`
- R2 후보 안정성: 전체 {stability['total_cases']}건, protocol-radius 통과
  {stability['protocol_radius_pass_cases']}건, 엄격 안정 {stability['strict_stable_cases']}건.
- R1 비교 안정성: `{json.dumps(r1, ensure_ascii=False, sort_keys=True)}`
- compact ledger는 all_cases {len(context['stability']['all_cases'])}건을 모두 유지한다.
- 위 결과는 유한 국소·고정 접촉 모드 검증이며 비선형/하이브리드 플랜트의 전역
  안정성 또는 안전 보장이 아니다.

### pRNE 계산·할당 진단

{chr(10).join('- ' + row['method'] + ': median ' + f"{row['thread_median_us']:.1f}" +
              ' us, p99 ' + f"{row['thread_p99_us']:.1f}" + ' us, peak ' +
              str(row['allocation_peak_bytes']) + ' B' for row in allocation['rows'])}

이 비교는 같은 pRNE 수학 출력을 만드는 두 helper의 순수 microbenchmark다. R1
observer 대비 속도 향상이나 제어기 채택 근거로 사용하지 않았고, plant episode는
0회이며 전역 GC 설정도 바꾸지 않았다.

### 최종 R1/R2 alias timing

{chr(10).join('- ' + row['architecture'] + ': rows ' + str(row['rows']) +
              ', cycles ' + str(row['cycles']) + ', largest row p99 ' +
              f"{row['largest_row_p99_us']:.3f}" + ' us, maximum wall/CPU ' +
              f"{row['maximum_wall_us']:.3f}/{row['same_cycle_thread_CPU_us']:.3f}" +
              ' us, overruns ' + str(row['overrun_cycles']) for row in timing['results'])}

1 ms strict goal은 미달이다. 같은 cycle CPU가 wall과 거의 같고 context switch는
최악 cycle에서 0/0이지만, 이는 interpreter/allocation/GC 작업과 양립한다는 뜻일
뿐 원인을 유일하게 확정하지 않는다. 과거 R1 GC 상관은 역사적 보조 증거로만 둔다.

## 미달·outlier

```json
{json.dumps(misses, ensure_ascii=False, indent=2, sort_keys=True)}
```

## 동결·holdout·공식 회귀

### Holdout
{chr(10).join('- ' + str(x) for x in manifest.get('holdout_findings', []))}

### 공식 회귀
{chr(10).join('- ' + str(x) for x in manifest.get('official_findings', []))}

## 실행 및 미실행 검증

### 실행
{chr(10).join('- ' + str(x) for x in verification['executed'])}

### 미실행
{chr(10).join('- ' + str(x) for x in verification['not_executed'])}

### 자원·재현 조건
{chr(10).join('- ' + str(x) for x in verification['resources'])}

## 변경 경로

{chr(10).join('- `' + str(x) + '`' for x in manifest['changed_paths'])}

## 증거 출처

`fresh`는 R2에서 새로 실행한 증거이고, `reused_exact_source`는 명시된 원본 결과를
그대로 재사용한 비교 증거다. 재사용 행을 독립 재실행으로 주장하지 않는다.

{_source_table(context['entries'])}

### 공개 유도·문헌

{chr(10).join('- `' + item['path'] + '`' for item in context['method_documents'])}

{chr(10).join('- [' + item['title'] + '](' + item['url'] + ')' for item in manifest['literature_updates'])}
"""


def _slide_specs(context: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = context["manifest"]
    stability = context["stability"]["summary"]
    timing_rows = {row["architecture"]: row for row in context["timing"]["results"]}
    rejected = sum(row["decision"] == "reject" for row in context["candidates"])
    adopted = sum(row["decision"] == "adopt" for row in context["candidates"])
    family_methods: dict[str, set[str]] = {}
    for row in context["candidates"]:
        family_methods.setdefault(str(row["family"]), set()).add(str(row["method"]))
    family_lines = [f"{family}: {len(methods)}개 변형, 모두 reject"
                    for family, methods in sorted(family_methods.items())]
    control_count = context["adoption"].get("control_configurations", "등록된")
    slides = [
        ("R2 질문과 고정 규칙", ["R1을 비교 기준으로 고정", "G4 기준 불변", "holdout 동결 후 1회", "공식 평가는 마지막 회귀 확인만"]),
        ("증거 흐름과 독립성", ["P0 → protocol → C1–C6 → 동결 → holdout → 공식", "fresh와 exact-source 재사용을 분리", "불리한 결과와 outlier를 모두 유지"]),
        ("P0 물리량 측정", [str(x) for x in manifest.get("p0_findings", ["P0 파생 경계는 protocol에 사전 등록"])]),
        ("후보군 C1–C6", [f"제어 구성 {control_count}개와 C5 observer ablation 3개",
                           f"architecture별 판정 행: 채택 {adopted}, 탈락 {rejected}",
                           "후보별 gate·효과·비용을 함께 기록"]),
        ("탄성 관절 안정성 모델", ["실제 탄성 관절 국소 FD", "1 ms ZOH", "명시적 한 주기 지연", "유한 고정모드 결과만 주장"]),
        ("안정성 결과", [f"전체 {stability['total_cases']}건", f"protocol-radius {stability['protocol_radius_pass_cases']}건", f"strict {stability['strict_stable_cases']}건"]),
        ("pRNE 수학·계산 검증", [f"표본 {len(context['prne']['cases'])}건", "Mdot = C + C^T 유한차분 확인", "reverse VJP는 같은 출력에서 helper 비용만 감소", "R1 observer 대비 고유 속도 향상 주장은 하지 않음"]),
        ("정상 검증과 final timing", ["제어 변경마다 무충돌 정상 데이터 확인",
                                     f"A p99/max {timing_rows['servo']['largest_row_p99_us']:.1f}/{timing_rows['servo']['maximum_wall_us']:.1f} us; overrun {timing_rows['servo']['overrun_cycles']}",
                                     f"B p99/max {timing_rows['torque']['largest_row_p99_us']:.1f}/{timing_rows['torque']['maximum_wall_us']:.1f} us; overrun {timing_rows['torque']['overrun_cycles']}",
                                     "strict 1 ms 미달; 원인 단정 없음"]),
        ("T1–T3 동일 조건", ["기준/v1/R1/R2 네 열 표"]),
        ("T4–T6 동일 조건", ["기준/v1/R1/R2 네 열 표"]),
        ("후보 판정", family_lines),
        ("Lie group 적용", [f"{r['method']}: {r['decision']}" for r in context['lie_usage']]),
        ("강건성·holdout", [str(x) for x in manifest.get("holdout_findings", ["동결 뒤 새 seed holdout을 한 번만 개봉"])]),
        ("공식 회귀 확인", [str(x) for x in manifest.get("official_findings", ["공식 결과는 선택에 사용하지 않음"])]),
        ("미달·한계·outlier", ([str(x) for x in manifest["misses_and_outliers"].get("summary", [])]
                              + [str(x) for x in manifest["limitations"][:3]])),
        ("최종 결론", [manifest.get("conclusion", "동결된 판정을 유지"),
                       "새 후보 채택 0개; 최종 R2=R1 exact-source alias",
                       "재설계 가족별 1회 허용·사용 0회; 미사용 사유 기록 없음",
                       "P0/G4와 T1–T6 추가 개선 목표는 미달",
                       "실기·hidden·전역 안정성은 검증하지 않음"]),
    ]
    answer = [{"title": title, "lines": lines} for title, lines in slides]
    if len(answer) != 16:
        raise AssertionError("R2 deck must contain exactly 16 slides")
    return answer


def render_presentation_notes(context: dict[str, Any]) -> str:
    sections = ["# R2 발표 자료 원고", "", "PDF는 아래 16개 슬라이드를 같은 순서로 생성한다."]
    for index, slide in enumerate(_slide_specs(context), 1):
        sections += ["", f"## {index}. {slide['title']}"]
        if index in {9, 10}:
            subset = context["metrics"][:3] if index == 9 else context["metrics"][3:]
            columns, rows = _metric_table_data(subset)
            sections += ["| " + " | ".join(columns) + " |",
                         "|" + "|".join("---" for _ in columns) + "|"]
            sections += ["| " + " | ".join(cell.replace("\n", "<br>") for cell in row) + " |"
                         for row in rows]
        else:
            sections += [f"- {line}" for line in slide["lines"]]
    return "\n".join(sections) + "\n"


def _metric_table_data(metrics: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    columns = ["항목", "조건·단위", "기준", "v1", "R1", "R2"]
    rows = []
    for row in metrics:
        condition = f"{row['condition']}\n[{row['unit']}]"
        if row["id"] == "T3":
            condition = f"{row['condition']}; 34=28 normal+6 external\n[{row['unit']}]"
        rows.append([f"{row['id']} {row['name']}", condition,
                     _cell(row["baseline"]), _cell(row["v1"]),
                     _cell(row["r1"]), _cell(row["r2"])])
    return columns, rows


def _draw_metric_table(fig: Any, metrics: list[dict[str, Any]]) -> None:
    columns, rows = _metric_table_data(metrics)
    ax = fig.add_axes([.025, .105, .95, .73])
    ax.axis("off")
    wrap_widths = [12, 24, 19, 19, 19, 19]
    wrapped = [["\n".join(textwrap.wrap(str(cell), width=wrap_widths[column],
                                         break_long_words=True,
                                         break_on_hyphens=False))
                for column, cell in enumerate(row)] for row in rows]
    table = ax.table(cellText=wrapped, colLabels=columns, cellLoc="left", colLoc="center",
                     colWidths=[.18, .20, .155, .155, .155, .155], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 4.1)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#b0bec5")
        if row == 0:
            cell.set_facecolor("#dbe7f3")
            cell.set_text_props(weight="bold", color="#183153")


def _configure_plotting() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    candidates = ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "DejaVu Sans")
    names = {item.name for item in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in names), "DejaVu Sans")
    matplotlib.rcParams.update({"font.family": selected, "axes.unicode_minus": False})


def _metric_plot(ax: Any, metrics: list[dict[str, Any]]) -> None:
    numeric = []
    for row in metrics:
        if (isinstance(row["r1"], (int, float)) and isinstance(row["r2"], (int, float))
                and float(row["r1"]) != 0.0):
            numeric.append(row)
    if not numeric:
        ax.text(.5, .5, "수치형 T1–T6 없음", ha="center", va="center")
        ax.axis("off")
        return
    x = list(range(len(numeric)))
    ax.bar(x, [float(r["r2"]) / float(r["r1"]) for r in numeric], width=.55,
           color="#457b9d", label="R2 / R1")
    ax.axhline(1.0, color="#d1495b", ls="--", label="R1")
    ax.set_xticks(x, [r["id"] for r in numeric]); ax.legend(); ax.grid(axis="y", alpha=.2)


def _stability_plot(ax: Any, context: dict[str, Any]) -> None:
    summary = context["stability"]["summary"]
    total = summary["total_cases"]
    vals = [summary["strict_stable_cases"], summary["protocol_radius_pass_cases"], total]
    ax.bar(["strict", "protocol", "전체"], vals, color=["#2a9d8f", "#457b9d", "#adb5bd"])
    ax.set_ylim(0, max(vals) * 1.12); ax.grid(axis="y", alpha=.2)


def make_pdf(path: Path, context: dict[str, Any]) -> None:
    _configure_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    slides = _slide_specs(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for index, slide in enumerate(slides, 1):
            fig = plt.figure(figsize=(13.333, 7.5), facecolor="#f6f8fb")
            fig.text(.055, .91, slide["title"], fontsize=23, weight="bold", color="#183153")
            metric_table = index in {9, 10}
            use_plot = index in {6, 11}
            width = 55 if use_plot else 82
            y = .81
            if metric_table:
                subset = context["metrics"][:3] if index == 9 else context["metrics"][3:]
                _draw_metric_table(fig, subset)
            else:
                for bullet in slide["lines"]:
                    for line in textwrap.wrap("• " + str(bullet), width=width):
                        fig.text(.06, y, line, fontsize=14, color="#263238")
                        y -= .052
                    y -= .015
            if use_plot:
                ax = fig.add_axes([.54, .16, .41, .60])
                if index == 6:
                    _stability_plot(ax, context)
                else:
                    families: dict[str, set[str]] = {}
                    for row in context["candidates"]:
                        families.setdefault(str(row["family"]), set()).add(str(row["method"]))
                    labels = sorted(families)
                    values = [len(families[label]) for label in labels]
                    ax.bar(labels, values, color="#d1495b")
                    ax.set_ylabel("검증한 변형 수")
                    ax.set_title("가족별 변형: 모두 엄격 gate 탈락")
                    ax.grid(axis="y", alpha=.2)
            fig.text(.055, .025, f"R2 제어·Lie group 검증 | {index}/16",
                     fontsize=9, color="#607d8b")
            pdf.savefig(fig); plt.close(fig)


def _privacy_check(text: str, label: str) -> None:
    found = [pattern.pattern for pattern in PRIVATE_PATTERNS if pattern.search(text)]
    if found:
        raise ValueError(f"{label} contains prohibited private references: {found}")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content); temporary = Path(handle.name)
    temporary.replace(path)


def generate(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    context = validate_manifest(load_object(manifest_path))
    results = render_results(context)
    notes = render_presentation_notes(context)
    _privacy_check(results, "R2_RESULTS.md")
    _privacy_check(notes, "PRESENTATION_R2.md")
    pdf_target = OUTPUTS[2]
    pdf_target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", dir=pdf_target.parent, delete=False) as handle:
        temp_pdf = Path(handle.name)
    try:
        make_pdf(temp_pdf, context)
        _atomic_text(OUTPUTS[0], results)
        _atomic_text(OUTPUTS[1], notes)
        temp_pdf.replace(pdf_target)
    finally:
        temp_pdf.unlink(missing_ok=True)
    return context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="frozen R2_REPORT_MANIFEST.v1 JSON")
    parser.add_argument("--check", action="store_true",
                        help="validate all final inputs without writing public outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_object(args.manifest)
    validate_manifest(manifest)
    if not args.check:
        generate(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
