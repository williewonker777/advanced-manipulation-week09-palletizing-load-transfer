"""Build the frozen R2 report manifest from actual evidence and curation.

The adapter is intentionally separate from the report renderer.  It reads no
private traces and runs no simulation.  Missing final evidence, stale hashes,
post-exposure tuning, incomplete row counts, or a non-R1 final alias stop the
build before a manifest is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

try:
    from tools import r2_report
except ModuleNotFoundError:  # Direct `python tools/r2_report_manifest.py` execution.
    import r2_report  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "v2_protocol"
DEFAULT_CURATION = DATA / "R2_REPORT_CURATION.json"
DEFAULT_OUTPUT = DATA / "R2_REPORT_MANIFEST.json"

SINGLE_ARTIFACTS = {
    "protocol": "PROTOCOL_R2.json",
    "validation_review": "R2_VALIDATION_REVIEW.json",
    "prne_math": "R2_PRNE_MATH.json",
    "prne_normal": "R2_PRNE_NORMAL.json",
    "stability_full": "R2_CANDIDATE_STABILITY.json",
    "stability_compact": "R2_CANDIDATE_STABILITY_COMPACT.json",
    "r1_stability": "R2_R1_STABILITY.json",
    "allocation_comparison": "R2_ALLOCATION_COMPARISON.json",
    "common_reference": "R2_COMMON_REFERENCE.json",
    "control_detector_parity": "R2_CONTROL_DETECTOR_PARITY.json",
    "lie_comparison": "R2_LIE_COMPARISON.json",
    "environment_integrity": "R2_ENVIRONMENT_INTEGRITY.json",
    "development_review": "R2_DEVELOPMENT_REVIEW.json",
    "d4": "R2_D4_base.json",
    "d4_reused": "S5_D4.json",
    "perturbations": "R2_PERTURBATIONS.json",
    "holdout": "R2_HOLDOUT.json",
    "holdout_opened": "R2_HOLDOUT_OPENED.json",
    "preholdout_review": "R2_PREHOLDOUT_REVIEW.json",
    "holdout_review": "R2_HOLDOUT_REVIEW.json",
    "freeze": "R2_FREEZE.json",
    "adoption": "R2_ADOPTION.json",
    "official": "R2_OFFICIAL_REGRESSION.json",
    "official_started": "R2_OFFICIAL_STARTED.json",
    "timing_review": "R2_TIMING_REVIEW.json",
}
DEVELOPMENT_FILES = (
    "R2_DEVELOPMENT_R1.json",
    "R2_DEVELOPMENT_C5_prne_only.json",
    "R2_DEVELOPMENT_C5_prne_plus_ridge.json",
    "R2_DEVELOPMENT_C5_ridge_only_matched.json",
)
CURATED_KEYS = (
    "lie_group_usage", "misses_and_outliers", "verification",
    "changed_paths", "limitations", "literature_updates", "conclusion",
    "p0_findings", "holdout_findings", "official_findings",
)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required final R2 evidence is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _entry(path: Path, provenance: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"required final R2 evidence is absent: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "provenance": provenance,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _rows(path: Path, expected: int) -> list[dict[str, Any]]:
    document = load_object(path)
    rows = document.get("results")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} must contain a results object-array")
    if len(rows) != expected:
        raise ValueError(f"{path.name}: expected {expected} rows, found {len(rows)}")
    return rows


def _collect_artifacts() -> dict[str, Any]:
    p0_paths = sorted(DATA.glob("R2_P0*.json"))
    if len(p0_paths) < 7:
        raise ValueError(f"expected at least seven P0 JSON records, found {len(p0_paths)}")
    artifacts: dict[str, Any] = {"p0": [_entry(path, "fresh") for path in p0_paths]}
    for key, name in SINGLE_ARTIFACTS.items():
        provenance = ("reused_exact_source"
                      if key in {"d4_reused", "common_reference"} else "fresh")
        artifacts[key] = _entry(DATA / name, provenance)
    artifacts["development"] = [_entry(DATA / name, "fresh") for name in DEVELOPMENT_FILES]
    return artifacts


def _gate_lines(gates: dict[str, Any]) -> list[str]:
    if not isinstance(gates, dict) or not gates:
        raise ValueError("candidate row lacks gate results")
    return [f"{name}={'pass' if value is True else 'fail'}" for name, value in gates.items()]


def _candidate_rows(adoption: dict[str, Any]) -> list[dict[str, Any]]:
    if adoption.get("decision") != "retain_r1" or adoption.get("provisional_R2_equals_R1") is not True:
        raise ValueError("this R2 manifest adapter requires the frozen retain-R1 decision")
    if adoption.get("control_configurations") != 25:
        raise ValueError("R2 adoption must retain all 25 registered control configurations")
    rows = adoption.get("control_decisions")
    observers = adoption.get("observer_decisions")
    if not isinstance(rows, list) or not isinstance(observers, list):
        raise ValueError("R2 adoption lacks control or observer decisions")
    candidates = []
    for row in rows:
        family = row.get("family")
        if family not in {f"C{i}" for i in range(1, 7)}:
            raise ValueError(f"unknown candidate family: {family!r}")
        worst = row.get("worst_ratio") or {}
        local = row.get("local_stability") or {}
        failed = [name for name, passed in row.get("gates", {}).items() if passed is not True]
        candidates.append({
            "id": f"{row.get('id')}/{row.get('architecture')}",
            "family": family,
            "method": str(row.get("id")),
            "architecture": str(row.get("architecture")),
            "decision": "reject",
            "failed_gates": (failed or
                             (["없음(validation; observer 판정 별도)"]
                              if family == "C5" else ["none"])),
            "false_rises": row.get("false_rises"),
            "g4_pass_cases": f"{row.get('G4_pass_cases')}/{row.get('episodes')}",
            "worst_max_ratio": worst.get("max_mm"),
            "worst_rms_ratio": worst.get("rms_mm"),
            "worst_place_step_ratio": worst.get("torque_step_place_nm"),
            "combined_p99_us": row.get("combined_p99_us"),
            "combined_max_us": row.get("combined_max_us"),
            "local_radius_pass": (f"{local.get('protocol_pass_cases')}/{local.get('cases')}"
                                  if local else "N/A"),
            "source_keys": ["validation_review", "adoption", "stability_compact"],
        })
    for row in observers:
        development = row.get("development") or []
        failed = [name for name, passed in row.get("gates", {}).items() if passed is not True]
        by_arch = {str(item.get("architecture")): item for item in development}
        false_rises = "/".join(str(by_arch.get(arch, {}).get("false_rises", "N/A"))
                               for arch in ("servo", "torque"))
        p99 = "; ".join(
            f"{label} {float(by_arch[arch]['p99_us']):.3f}"
            if arch in by_arch and by_arch[arch].get("p99_us") is not None
            else f"{label} N/A"
            for label, arch in (("A", "servo"), ("B", "torque"))
        )
        candidates.append({
            "id": f"{row.get('id')}/observer",
            "family": "C5",
            "method": str(row.get("id")),
            "architecture": "observer (A/B)",
            "decision": "reject",
            "failed_gates": failed or ["none"],
            "false_rises": false_rises,
            "g4_pass_cases": "N/A",
            "worst_max_ratio": "N/A",
            # The observer review reports joint-residual RMS, not TCP RMS.
            "worst_rms_ratio": "N/A",
            "worst_place_step_ratio": "N/A",
            "combined_p99_us": p99,
            "combined_max_us": "N/A",
            "local_radius_pass": "N/A",
            "source_keys": ["development", "development_review", "adoption"],
        })
    if Counter(row["decision"] for row in candidates) != {"reject": len(candidates)}:
        raise AssertionError("all registered R2 challengers must remain rejected")
    return candidates


def _ab(a: float, b: float, digits: int = 4) -> str:
    return f"A {a:.{digits}f} / B {b:.{digits}f}"


def _common_by_variant() -> dict[str, dict[str, dict[str, Any]]]:
    rows = _rows(DATA / "R2_COMMON_REFERENCE.json", 6)
    answer: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        answer.setdefault(str(row["variant"]), {})[str(row["architecture"])] = row
    if set(answer) != {"baseline", "v1", "R1"}:
        raise ValueError("common reference must contain baseline, v1 and R1")
    return answer


def _metric_rows() -> list[dict[str, Any]]:
    common = _common_by_variant()

    def pair(variant: str, getter: Any, digits: int = 4) -> str:
        return _ab(float(getter(common[variant]["servo"])),
                   float(getter(common[variant]["torque"])), digits)

    metrics = [
        {"id": "T1", "name": "놓기 전 마지막 20 ms 평균 지지율",
         "unit": "% of mg (A/B)", "condition": "8 kg, 동일 COM, release 직전 20 ms",
         "baseline": pair("baseline", lambda row: 100 * row["g4"]["table_mean_over_mg"]),
         "v1": pair("v1", lambda row: 100 * row["g4"]["table_mean_over_mg"]),
         "r1": pair("R1", lambda row: 100 * row["g4"]["table_mean_over_mg"]),
         "r2": pair("R1", lambda row: 100 * row["g4"]["table_mean_over_mg"]),
         "r2_status": "not_met", "source_keys": ["common_reference", "adoption"]},
        {"id": "T2", "name": "놓기 토크 계단",
         "unit": "N m (A/B)", "condition": "8 kg, 동일 COM, place step",
         "baseline": pair("baseline", lambda row: row["place_step_Nm"], 3),
         "v1": pair("v1", lambda row: row["place_step_Nm"], 3),
         "r1": pair("R1", lambda row: row["place_step_Nm"], 3),
         "r2": pair("R1", lambda row: row["place_step_Nm"], 3),
         "r2_status": "not_met", "source_keys": ["common_reference", "adoption"]},
    ]

    perturb = _rows(DATA / "R2_PERTURBATIONS.json", 272)
    by_perturb: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in perturb:
        by_perturb.setdefault((str(row["variant"]), str(row["architecture"])), []).append(row)

    def perturb_cell(variant: str) -> str:
        cells = []
        for label, architecture in (("A", "servo"), ("B", "torque")):
            rows = by_perturb[(variant, architecture)]
            cells.append(f"{label} {max(float(r['max_mm']) for r in rows):.3f} mm / "
                         f"{max(float(r['table_force_place_max_n']) for r in rows):.3f} N / "
                         f"FA {sum(int(r['false_alarms']) for r in rows)}")
        return "; ".join(cells)

    metrics.append(
        {"id": "T3", "name": "분석 섭동 최악 TCP/놓기힘/오감지",
         "unit": "mm / N / rises", "condition": "fresh 34 perturbation rows per architecture",
         "baseline": perturb_cell("base"), "v1": perturb_cell("v1"),
         "r1": perturb_cell("R1"), "r2": perturb_cell("R1"),
         "r2_status": "not_met", "source_keys": ["perturbations", "adoption"]})

    fresh_d4 = _rows(DATA / "R2_D4_base.json", 770)
    reused_d4 = _rows(DATA / "S5_D4.json", 1540)

    def reference_cell(rows: list[dict[str, Any]], variant: str) -> str:
        cells = []
        for label, architecture in (("A", "servo"), ("B", "torque")):
            selected = [r for r in rows if r.get("variant") == variant
                        and r["architecture"] == architecture
                        and r.get("kind") == "reference"]
            saturated = sum(float(r["saturation_pct"]) > 0 for r in selected)
            peak = max(float(r["g4"]["place_peak_over_mg"]) for r in selected
                       if r["g4"]["place_peak_over_mg"] is not None)
            cells.append(f"{label} sat {saturated}/189, peak {peak:.4f} mg")
        return "; ".join(cells)

    metrics.append(
        {"id": "T4", "name": "질량·COM 격자 포화/놓기 peak",
         "unit": "rows / mg", "condition": "189 no-collision reference rows per architecture",
         "baseline": reference_cell(fresh_d4, "base"),
         "v1": reference_cell(reused_d4, "v1"),
         "r1": reference_cell(reused_d4, "v2"),
         "r2": reference_cell(reused_d4, "v2"),
         "r2_status": "not_met", "source_keys": ["d4", "d4_reused", "adoption"]})

    def tracking(variant: str) -> str:
        return (f"A {common[variant]['servo']['RMS_mm']:.3f}/"
                f"{common[variant]['servo']['max_mm']:.3f}; "
                f"B {common[variant]['torque']['RMS_mm']:.3f}/"
                f"{common[variant]['torque']['max_mm']:.3f}")

    metrics.append(
        {"id": "T5", "name": "TCP 추적 RMS/최대",
         "unit": "mm (A/B)", "condition": "8 kg, 동일 COM, no collision",
         "baseline": tracking("baseline"), "v1": tracking("v1"),
         "r1": tracking("R1"), "r2": tracking("R1"),
         "r2_status": "not_met", "source_keys": ["common_reference", "adoption"]})

    def collision_cell(rows: list[dict[str, Any]], variant: str) -> str:
        cells = []
        for label, architecture in (("A", "servo"), ("B", "torque")):
            selected = [r for r in rows if r.get("variant") == variant
                        and r["architecture"] == architecture and r.get("kind") == "collision"]
            outcomes = Counter(str(r.get("collision_outcome")) for r in selected)
            masked = outcomes["alarm_before_collision"] + outcomes["indistinguishable"]
            cells.append(f"{label} {outcomes['detected']}/{outcomes['missed']}/{masked}")
        return "; ".join(cells)

    metrics.append(
        {"id": "T6", "name": "D4 감지/놓침/masked",
         "unit": "episodes", "condition": "180 collision rows per architecture; masked kept separate",
         "baseline": collision_cell(fresh_d4, "base"),
         "v1": collision_cell(reused_d4, "v1"),
         "r1": collision_cell(reused_d4, "v2"),
         "r2": collision_cell(reused_d4, "v2"),
         "r2_status": "not_met",
         "source_keys": ["d4", "d4_reused", "development", "official", "adoption"]})
    return metrics


def _validate_actual_data() -> dict[str, Any]:
    development = {name: _rows(DATA / name, 52) for name in DEVELOPMENT_FILES}
    d4 = _rows(DATA / "R2_D4_base.json", 770)
    reused_d4 = _rows(DATA / "S5_D4.json", 1540)
    perturbations = load_object(DATA / "R2_PERTURBATIONS.json")
    perturb_rows = perturbations.get("results")
    parity_rows = perturbations.get("metadata", {}).get("nominal_parity")
    if not isinstance(perturb_rows, list) or len(perturb_rows) != 272:
        raise ValueError("R2 perturbations must retain all 272 registered rows")
    if (not isinstance(parity_rows, list) or len(parity_rows) != 8
            or any(row.get("passed") is not True for row in parity_rows)):
        raise ValueError("R2 perturbations must retain all eight passing variant/architecture parity rows")
    holdout = load_object(DATA / "R2_HOLDOUT.json")
    holdout_rows = holdout.get("results")
    if not isinstance(holdout_rows, list) or len(holdout_rows) != 152:
        raise ValueError("R2 holdout must retain all 152 sealed rows")
    opened = load_object(DATA / "R2_HOLDOUT_OPENED.json")
    if opened.get("batch_attempt") != 1 or holdout.get("metadata", {}).get("one_batch") is not True:
        raise ValueError("R2 holdout must be opened in exactly one sealed batch")
    marker_digest = hashlib.sha256((DATA / "R2_HOLDOUT_OPENED.json").read_bytes()).hexdigest()
    if holdout.get("metadata", {}).get("opened_marker_sha256") != marker_digest:
        raise ValueError("holdout opening marker hash mismatch")
    adoption = load_object(DATA / "R2_ADOPTION.json")
    freeze = load_object(DATA / "R2_FREEZE.json")
    if adoption.get("holdout_opened") is not False or freeze.get("holdout_opened") is not False:
        raise ValueError("adoption and freeze must both precede sealed holdout opening")
    if adoption.get("official_used_for_selection") is not False:
        raise ValueError("official results must not select the R2 result")
    official = load_object(DATA / "R2_OFFICIAL_REGRESSION.json")
    if official.get("official_used_for_selection") not in (False, None):
        raise ValueError("official regression record claims selection use")
    if (official.get("official") is not True or official.get("R1_compatibility_pass") is not True
            or official.get("regression_errors") != [] or official.get("tuning_after_exposure") is not False):
        raise ValueError("official regression must pass unchanged R1 compatibility without retuning")
    holdout_digest = hashlib.sha256((DATA / "R2_HOLDOUT.json").read_bytes()).hexdigest()
    if official.get("holdout_sha256") != holdout_digest:
        raise ValueError("official regression does not point to the sealed holdout result")
    official_started = load_object(DATA / "R2_OFFICIAL_STARTED.json")
    if (official_started.get("holdout_sha256") != holdout_digest
            or official_started.get("tuning_or_repair_after_official") is not False):
        raise ValueError("official start marker is inconsistent with holdout or no-repair rule")
    return {
        "development_rows": sum(len(rows) for rows in development.values()),
        "d4_rows_fresh": len(d4),
        "d4_rows_reused": len(reused_d4),
        "perturbation_rows": len(perturb_rows),
        "perturbation_parity_rows": len(parity_rows),
        "holdout_rows": len(holdout_rows),
        "adoption": adoption,
    }


def _method_documents() -> list[dict[str, str]]:
    answer = []
    for name in ("docs/R2_METHODS.md", "docs/R2_ROBUST_DERIVATION.md", "docs/LITERATURE.md"):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required public method document is absent: {path}")
        answer.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return answer


def build_manifest(curation_path: Path = DEFAULT_CURATION) -> dict[str, Any]:
    evidence = _validate_actual_data()
    curation = load_object(curation_path)
    if curation.get("schema") != "R2_REPORT_CURATION.v1":
        raise ValueError("curation must use R2_REPORT_CURATION.v1")
    missing = [key for key in CURATED_KEYS if key not in curation]
    if missing:
        raise ValueError(f"curation lacks required keys: {missing}")
    metrics = _metric_rows()
    for row in metrics:
        if row.get("r2") != row.get("r1"):
            raise ValueError(f"{row.get('id', '?')}: frozen R2 must be the exact R1 alias")
    manifest = {
        "schema": "R2_REPORT_MANIFEST.v1",
        "status": "frozen",
        "remote_operations": 0,
        "artifacts": _collect_artifacts(),
        "method_documents": _method_documents(),
        "candidate_decisions": _candidate_rows(evidence["adoption"]),
        "metrics": metrics,
        "evidence_summary": {key: value for key, value in evidence.items() if key != "adoption"},
    }
    for key in CURATED_KEYS:
        manifest[key] = curation[key]
    # Reuse the renderer's stricter end-to-end schema and source-hash checks.
    r2_report.validate_manifest(manifest)
    return manifest


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="validate all evidence and curation without writing the manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_manifest(args.curation)
    if not args.check:
        atomic_json(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
