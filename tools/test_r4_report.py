import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r4_report as R


def review_row(mode, variant, cases=38):
    return {
        "architecture": mode, "variant": variant, "family": "W40",
        "review": {"passed": True, "additional_detected": 2, "lost_R1": 0,
                   "R1_detected": 20, "candidate_detected": 22,
                   "reference_FA_R1": 0, "reference_FA_candidate": 0,
                   "outside_event_FA_R1": 1, "outside_event_FA_candidate": 1,
                   "phases": {"pick": {"cases": 10, "R1_detected": 6,
                                        "candidate_detected": 7}},
                   "failures": [], "every_case_retained": cases},
        "normal_audit_pass": True, "full_loop_parity_pass": True,
        "timing_p99_us": 50.125, "timing_max_us": 900.5,
        "timing_pass": True, "adoption_eligible": True,
        "earlier_disqualification": False, "permanently_disqualified": False,
    }


def stage(stage="validation"):
    cases = {"validation": 38, "development": 38, "d4": 385, "perturb": 34}[stage]
    return {"schema": "R4_STAGE_REVIEW.v1", "stage": stage,
            "diagnostics_cannot_revive": True,
            "results": [review_row(mode, variant, cases)
                        for mode in R.MODES for variant in R.VARIANTS]}


def official():
    records = {}
    for mode in R.MODES:
        records[mode] = {"control_and_collisions": {
            "control": {
                "0.0": ["1.17", "3.91", "0.0", "9.7", "5.1", "0", "0", "0", "0", "정상"],
                "4.0": ["1.15", "3.79", "0.0", "9.7", "14.6", "40", "0", "0", "0", "정상"],
                "8.0": ["3.23", "8.44", "0.0", "9.0", "13.1", "80", "0", "0", "0", "정상"]},
            "collisions": [{"mass": 8.0, "time": 3.0,
                            "cells": ["62", "45", "12", "8", "0회", "10 N"]}]}}
    return {"schema": "R4_OFFICIAL_REGRESSION.v1",
            "architectures": {mode: {"effective_selection": "R1"} for mode in R.MODES},
            "results": records}


def context():
    candidates = []
    perturb_whole = []
    phases = []
    for mode in R.MODES:
        for variant in R.VARIANTS:
            candidates.append({"방식": mode, "후보": variant, "정상 추가 FA": "표본 0, 상승 0",
              "validation": "+2, 손실 0", "development": "+2, 손실 0", "D4": "+6, 손실 0",
              "섭동 무충돌 FA [28조건]": "1→1", "p99/최대 [µs]": "50.125/900.500", "판정": "탈락",
              "실패 관문": "추가 검출 기준 미달"})
            perturb_whole.append({"방식": mode, "후보": variant,
                                  "전체 34조건 충돌 외 FA": "2→3"})
            phases.append({"방식": mode, "후보": variant,
                           "development R1→후보": "pick 6→7/10; transport 5→7/10; place 3→3/10",
                           "D4 R1→후보": "pick 50→51/60; transport 43→49/60; place 33→33/60"})
    metrics = [{"id": "T" + str(index), "name": "지표", "condition": "보존 조건",
                "unit": "단위", "baseline": "1", "v1": "2", "r1": "3", "r4": "3"}
               for index in range(1, 7)]
    curation = {"final": {"servo": "R1", "torque": "R1"}, "conclusion": "R1을 유지한다.",
                "candidates": candidates, "perturb_whole": perturb_whole,
                "phases": phases, "metrics": metrics,
                "holdout_summary": [
                    {"방식": "A 서보", "충돌 검출": "20/30", "정상 4조건 FA": 0,
                     "전체 충돌 외 FA": 0, "용도": "R1 보존 확인; W40 미실행"},
                    {"방식": "B 토크", "충돌 검출": "22/30", "정상 4조건 FA": 0,
                     "전체 충돌 외 FA": 1, "용도": "R1 보존 확인; W40 미실행"}],
                "redesign": [{"family": "W40", "사용": "아니오", "당시 사유": "사유"}],
                "official": R._official_context(official()),
                "normal_audit": {"rows": 128, "extra_samples": 0, "extra_rises": 0},
                "equivalence": {"passed": True, "rows": 16, "fresh_full_loop_simulations": 20,
                                "fresh_w40_rows": 76, "fresh_representative_rows": 4,
                                "reused_rows": 12},
                "reseal": {"invalidated_seed": 371501, "replacement_seed": 371599,
                           "pre_generated_designs": 2, "actual_episodes_before_reseal": 0},
                "literature": {"opened": 10, "full_text": 7,
                               "abstract_bibliography": 2, "publisher_metadata": 1},
                "cause": {"gate_hidden_servo": 43, "gate_hidden_torque": 46,
                          "mean40_any_servo": 17, "mean40_any_torque": 15,
                          "mean40_four_servo": 13, "mean40_four_torque": 14,
                          "all_scores_below_servo": 52, "all_scores_below_torque": 53},
                "perturb_findings": [
                    {"방식": "A 서보", "증가 행": 3, "강성 단독": 2, "강성·마찰 조합": 1,
                     "새 잡음 증가/조건": "0/9"},
                    {"방식": "B 토크", "증가 행": 6, "강성 단독": 4, "강성·마찰 조합": 2,
                     "새 잡음 증가/조건": "0/9"}],
                "timing": [{"방식·소스": label, "최악 행 p99 [µs]": "50.125",
                            "관측 최대 [ms]": "0.900", "1 ms 초과 주기": 0}
                           for label in ("A 서보/R1", "A 서보/W40", "B 토크/R1", "B 토크/W40")],
                "requirements": {"충족": 24, "부분": 4, "판정 불가": 5},
                "r3_r1_timing": {
                    "servo": {"p99_us": "327.670", "max_ms": "11.703", "outlier_count": "8"},
                    "torque": {"p99_us": "350.440", "max_ms": "14.163", "outlier_count": "4"}},
                "limitations": ["유한 검사다."]}
    return {"curation": curation, "manifest": {}, "artifacts": {}, "root": R.ROOT}


class R4ReportTests(unittest.TestCase):
    def test_stage_requires_all_eight_rows(self):
        value = stage()
        self.assertEqual(len(R.stage_map(value, "VALIDATION")), 8)
        value["results"].pop()
        with self.assertRaisesRegex(ValueError, "eight"):
            R.stage_map(value, "VALIDATION")

    def test_stage_rejects_duplicate_and_missing_measurement(self):
        value = stage()
        value["results"][-1] = copy.deepcopy(value["results"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            R.stage_map(value, "VALIDATION")
        value = stage(); del value["results"][0]["review"]["lost_R1"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            R.stage_map(value, "VALIDATION")

    def test_official_evaluator_string_precision_is_literal(self):
        source = official()
        result = R._official_context(source)
        self.assertEqual(result["control"]["servo"]["0.0"][:2], ["1.17", "3.91"])
        self.assertEqual(result["control"]["torque"]["4.0"][4], "14.6")
        rendered = R.render_results(context())
        self.assertIn("1.17 / 3.91", rendered)
        self.assertNotIn("1.170", rendered)
        self.assertIn("14.6", rendered)

    def test_readme_restores_official_rows_and_requirements(self):
        rendered = R.render_readme(context())
        self.assertIn("공식 0/4/8 kg", rendered)
        self.assertIn("공식 8 kg 충돌", rendered)
        self.assertIn("1.17/3.91", rendered)
        self.assertNotIn("1.170", rendered)
        self.assertIn("B 8 kg 3.0 s", rendered)
        self.assertIn("오감지 1회", rendered)
        self.assertIn("docs/REQUIREMENTS_AUDIT.md", rendered)
        for text in ("24충족/4부분/5판정 불가", "G4", "1 ms", "계산 시간"):
            self.assertIn(text, rendered)

    def test_unattempted_methods_are_explicit_in_all_deliverables(self):
        value = context()
        for rendered in (R.render_readme(value), R.render_results(value), R.render_notes(value)):
            self.assertIn("토크 추적 feedforward는 구현하지 않았", rendered)
            self.assertIn("효과는 모른다", rendered)
            self.assertIn("공식 8 kg 미검출 3조건에 W40을 실행하지 않았다", rendered)

    def test_readme_audit_summary_uses_distinct_signal_windows(self):
        rendered = R.render_readme(context())
        summary = rendered.split("## 평가기 점검과 미검출 원인\n", 1)[1].split(
            "## 계산 시간과 미달\n", 1)[0]
        for text in ("### 평가기 점검", "### 공식 8 kg 미검출 원인 측정",
                     "펄스 중 점수", "평가창 점수", "0.80/0.74 (0.80/0.71)",
                     "1.09/1.06", "0.99/0.96 (0.55/0.51)",
                     "1.45/1.55 (1.86/1.70)", "120 ms", "320 ms",
                     "공개본만으로 재현할 수 없다", "독립 검증으로 쓸 수 없다"):
            self.assertIn(text, summary)

    def test_audit_links_are_readme_only_and_preserve_official_precision(self):
        value = context()
        readme = R.render_readme(value)
        self.assertIn("[평가기 점검](#평가기-점검과-미검출-원인)",
                      readme.split("## 용어", 1)[0])
        self.assertIn("[측정 내용](#공식-8-kg-미검출-원인-측정)", readme)
        self.assertIn("미검출([원인 측정](#공식-8-kg-미검출-원인-측정))", readme)
        self.assertIn("1.17/3.91", readme)
        self.assertNotIn("1.170", readme)
        for other in (R.render_results(value), R.render_notes(value)):
            self.assertNotIn("## 평가기 점검과 미검출 원인", other)

    def test_glossary_is_at_the_start_of_readme_and_pdf(self):
        readme = R.render_readme(context())
        first_slide = " ".join(R.slides(context())[0]["bullets"])
        for rendered in (readme.split("## 영상으로 먼저 보기")[0], first_slide):
            for term in ("W40", "CUSUM", "D4", "holdout", "동결"):
                self.assertIn(term, rendered)
            for explanation in ("40 ms", "누적", "385", "선정에 쓰지", "고정"):
                self.assertIn(explanation, rendered)

    def test_failure_column_skips_passing_rows_and_uses_korean_stages(self):
        rows = {name: review_row("servo", "W40_ACTIVE") for name in R.STAGES}
        rows["VALIDATION"]["adoption_eligible"] = False
        rows["VALIDATION"]["earlier_disqualification"] = True
        rows["PERTURB"]["adoption_eligible"] = False
        rows["PERTURB"]["review"]["failures"] = [
            {"reasons": ["extra_outside_event_false_alarm"]}]
        text = R._first_failure(rows)
        self.assertEqual(text, "분석 섭동: 오감지 증가")
        self.assertNotIn("통과", text)

    def test_summary_repair_is_plain_and_seventh_slide_is_short(self):
        value = context()
        value["curation"]["causal_repair"] = {
            "applied": True, "original_w40_gains": {"servo": 4, "torque": 3},
            "revalidation_w40_gains": {"servo": 4, "torque": 4}}
        for rendered in (R.render_readme(value), R.render_results(value)):
            self.assertNotRegex(rendered, r"\b(wrapper|parent|helper|nominal|scratch|certificate|marker|family)\b")
            self.assertNotIn("literal no-store", rendered)
            self.assertNotIn("인과 구현 수리", rendered)
            self.assertNotIn("봉인 기록", rendered)
        slide = R.slides(value)[6]
        self.assertLessEqual(len(slide["bullets"]), 4)
        self.assertLessEqual(sum(len(line) for line in slide["bullets"]), 240)
        self.assertIn("재설계 1회", " ".join(slide["bullets"]))

    def test_supplemental_facts_are_recomputed_from_existing_sources(self):
        loaded = {key: R.load(R.ROOT / path) for key, path in R.ARTIFACTS.items()
                  if key in {"cause_analysis", "perturb_w40_servo", "perturb_w40_torque",
                             "perturb_r1_servo", "perturb_r1_torque"}}
        facts = R._diagnostic_findings(loaded)
        self.assertEqual([r["증가 행"] for r in facts], [3, 6])
        self.assertEqual([r["새 잡음 증가/조건"] for r in facts], ["0/9", "0/9"])
        self.assertEqual([r["강성 단독"] for r in facts], [2, 4])
        self.assertEqual([r["강성·마찰 조합"] for r in facts], [1, 2])

    def test_timing_uses_recorded_rows_and_explicit_decimal_rounding(self):
        loaded = {key: R.load(R.ROOT / path) for key, path in R.ARTIFACTS.items()
                  if key.startswith("timing_")}
        rows = R._timing_summary(loaded)
        self.assertEqual([row["관측 최대 [ms]"] for row in rows],
                         ["0.501", "5.141", "5.257", "0.466"])
        self.assertEqual([row["1 ms 초과 주기"] for row in rows], [0, 1, 3, 0])

    def test_preserved_r3_r1_timing_is_quoted_beside_new_table(self):
        value = context()
        self.assertEqual(R._r3_r1_timing(R.ROOT / "docs/R3_README.md"),
                         value["curation"]["r3_r1_timing"])
        for rendered in (R.render_readme(value), R.render_results(value)):
            timing_section = rendered.split("## 계산 시간과 미달", 1)[1].split("## 요구사항", 1)[0]
            for text in ("327.670 µs", "11.703 ms", "8주기", "350.440 µs", "14.163 ms", "4주기",
                         "R1은 최대 1 ms 목표 미달", "보존된 R3 표"):
                self.assertIn(text, timing_section)

    def test_whole_public_document_file_links_are_checked(self):
        from tools import r4_report_links as L
        self.assertEqual(L.check_public_links(R.ROOT)["missing"], [])

    def test_manifest_curation_tamper_is_rejected(self):
        manifest = {"schema": "R4_REPORT_MANIFEST.v1", "status": "frozen",
                    "artifacts": {}, "curation": {"answer": 2}}
        with patch.object(R, "verify_artifacts", return_value={}), \
             patch.object(R, "curate", return_value={"answer": 1}):
            with self.assertRaisesRegex(ValueError, "curation"):
                R.validate_manifest(manifest)

    def test_artifact_hash_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = {}
            for key, relative in R.ARTIFACTS.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-1.4\n" if path.suffix == ".pdf" else
                                 (b"{}\n" if path.suffix == ".json" else b"text\n"))
                artifacts[key] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "provenance": "fresh"}
            holder = {"artifacts": artifacts}
            R.verify_artifacts(holder, root)
            (root / R.ARTIFACTS["protocol"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                R.verify_artifacts(holder, root)

    def test_rebound_artifact_keeps_current_hash_enforcement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = {}
            for key, relative in R.ARTIFACTS.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-1.4\n" if path.suffix == ".pdf" else
                                 (b"{}\n" if path.suffix == ".json" else b"text\n"))
                artifacts[key] = {
                    "path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "provenance": "fresh",
                }
            rebound = artifacts["equivalence"]
            rebound["prepublication_work_sha256"] = "a" * 64
            R.verify_artifacts({"artifacts": artifacts}, root)
            rebound["sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                R.verify_artifacts({"artifacts": artifacts}, root)
            rebound["sha256"] = hashlib.sha256(
                (root / R.ARTIFACTS["equivalence"]).read_bytes()).hexdigest()
            rebound["prepublication_work_sha256"] = "not-a-hash"
            with self.assertRaisesRegex(ValueError, "prepublication artifact hash"):
                R.verify_artifacts({"artifacts": artifacts}, root)

    def test_public_text_has_no_internal_names_or_status_phrasing(self):
        for rendered in (R.render_readme(context()), R.render_results(context()),
                         R.render_notes(context())):
            R.privacy_check(rendered, "fixture")
            forbidden = ("Cl" + "aude", "Co" + "dex", "G" + "PT", "O" + "MX",
                         "TA" + "SK-" + str(371), chr(47) + "home" + chr(47))
            for word in forbidden:
                self.assertNotIn(word, rendered)

    def test_perturbation_28_and_34_counts_are_not_merged(self):
        value = context()
        value["curation"]["candidates"][4]["섭동 무충돌 FA [28조건]"] = "9→17"
        value["curation"]["perturb_whole"][4]["전체 34조건 충돌 외 FA"] = "10→18"
        rendered = R.render_results(value)
        self.assertIn("섭동 무충돌 FA [28조건]", rendered)
        self.assertIn("9→17", rendered)
        self.assertIn("10→18", rendered)
        self.assertIn("W40 holdout은 실행하지 않았다", rendered)

    def test_pdf_is_eleven_readable_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "presentation.pdf"
            R.make_pdf(path, context())
            self.assertTrue(path.read_bytes().startswith(b"%PDF-"))
            self.assertEqual(R.pdf_pages(path), 11)

    def test_pdf_tables_use_compact_lossless_labels(self):
        pages = R.slides(context())
        candidate = pages[2]["table"][1][0]
        self.assertEqual(candidate["후보"], "W40")
        self.assertEqual(candidate["p99/최대 [µs]"], "50/900")
        self.assertNotIn("validation", candidate)
        phase = pages[5]["table"][1][0]
        self.assertEqual(phase["개발 R1→후보"], "집기 6→7 · 운반 5→7\n놓기 3→3 (각 10)")
        self.assertIn("[N·m]", pages[3]["table"][1][1]["항목"])
        self.assertIn("[mm]", pages[4]["table"][1][1]["항목"])

    def test_revalidation_stage_retains_combined_eight_rows(self):
        value = stage("validation")
        value["stage"] = "revalidation"
        mapped = R.stage_map(value, "REVALIDATION")
        self.assertEqual(len(mapped), 8)
        self.assertEqual(mapped["servo", "W40_ACTIVE"]["review"]["additional_detected"], 2)

    def test_registered_revision_requires_complete_hash_group(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts = {}
            for key, relative in R.ARTIFACTS.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"%PDF-1.4\n" if path.suffix == ".pdf" else
                                 (b"{}\n" if path.suffix == ".json" else b"text\n"))
                artifacts[key] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "provenance": "fresh"}
            addendum = root / R.REVISION_ARTIFACTS[0]
            addendum.parent.mkdir(parents=True, exist_ok=True); addendum.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "inventory"):
                R.verify_artifacts({"artifacts": artifacts}, root)

    def test_metric_pdf_wraps_between_architectures_not_inside_numbers(self):
        self.assertEqual(R._wrap("A 6.144/14.000;", 13), "A 6.144/14.000;")
        self.assertEqual(R._wrap("B 7.042/13.183", 13), "B 7.042/13.183")
        text = R._pdf_metric_value("A 1.174/3.914; B 3.232/8.444")
        self.assertEqual(text.splitlines(), ["A 1.174/3.914;", "B 3.232/8.444"])
        self.assertEqual(R._pdf_metric_value("A 0.2098 / B 11.1507").splitlines(),
                         ["A 0.2098", "B 11.1507"])

    def test_no_holdout_rng_or_builder_is_imported(self):
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertNotIn("default_rng", source)
        self.assertNotIn("_build_holdout_design", source)
        self.assertNotIn("r4_sealed", source)


if __name__ == "__main__":
    unittest.main()
