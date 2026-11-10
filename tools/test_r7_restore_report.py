"""Pure tests for the restored R7 presentation wrapper."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools import r7_report as Base
from tools import r7_restore_report as R
from tools.r7_json import replace_json, write_json


class RestoredR7ReportTests(unittest.TestCase):
    def _record_fixture(self, root: Path):
        def put(relative, value=None):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, {} if value is None else value)
            return R._sha(path)

        for relative in R.SCIENTIFIC_RECORDS:
            put(relative)
        sources = {}
        for key, relative in (("veto_adoption", R.D + "R7_VETO_ADOPTION.json"),
                              ("failure", R.D + "R7_FAILURE.json")):
            sources[key] = {"file": relative, "sha256": put(relative)}
        historical = {}
        for key, relative in (
            ("historical_veto_execution_status", R.D + "R7_EXECUTION_STATUS.json"),
            ("historical_veto_report_curation", R.D + "R7_REPORT_CURATION.json"),
        ):
            historical[key] = {"file": relative, "sha256": put(relative)}
        models = {}
        for index in range(8):
            relative = f"models/model_{index}.json"
            models[relative] = put(relative)
        files = {"my_detector.py": R.SERVO_SHA, "my_servo.py": R.SERVO_CONTROL_SHA,
                 "my_torque.py": R.TORQUE_CONTROL_SHA}
        s4 = {"schema": "R7_ADOPTION.v1",
              "effective_variants": {"servo": "SCALE_STATE", "torque": "R1"},
              "final_assembly": "R7", "final_files": files}
        write_json(root / (R.D + "R7_S4_ADOPTION.json"), s4)
        sources["s4_adoption"] = {"file": R.D + "R7_S4_ADOPTION.json",
                                  "sha256": R._sha(root / (R.D + "R7_S4_ADOPTION.json"))}
        candidate = "experiments/r7_final/A_STATE.py"
        candidate_path = root / candidate
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Base.ROOT / candidate, candidate_path)
        self.assertEqual(R.SERVO_SHA, R._sha(candidate_path))
        for relative in ("my_detector.py", "my_servo.py", "my_torque.py"):
            shutil.copyfile(Base.ROOT / relative, root / relative)
        historical_manifest = R.D + "R7_S4_MEDIA_MANIFEST.json"
        historical_media = {"file": historical_manifest, "sha256": put(historical_manifest)}
        media_sha = {}
        for relative in (
            "artifacts/media/09_r7_servo_positive_comparison.gif",
            "artifacts/media/09_r7_servo_positive_comparison.mp4",
            "artifacts/media/10_r7_servo_negative_comparison.gif",
            "artifacts/media/10_r7_servo_negative_comparison.mp4",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("utf-8"))
            media_sha[relative] = R._sha(path)
        restoration = {
            "schema": "R7_ADOPTION_RESTORATION.v1",
            **historical,
            "user_decision": {"quote": R.USER_QUOTE},
            "final_decision": {"servo": "SCALE_STATE", "torque": "R1",
                               "final_assembly": "R7"},
            "basis": {"protocol_fallback_excludes_document_checks": True,
                      "execution_order_excludes_document_checks": True,
                      "no_candidate_data_gate_change": True,
                      "no_simulation_holdout_official_rerun": True},
            "sources": sources,
            "candidate": {"file": candidate, "sha256": R._sha(candidate_path)},
            "final_files": files,
            "scientific_record_sha256": {name: R._sha(root / name) for name in R.SCIENTIFIC_RECORDS},
            "preserved_models": models,
            "historical_media_manifest": historical_media,
            "restored_media_original_sha256": media_sha,
            "current_validation_record": R.VALIDATION,
            "training_runs": 0,
            "new_physical_simulations": 0,
            "new_holdout_batches": 0,
            "new_official_batches": 0,
            "threshold_updates": 0,
            "remote_operations": 0,
        }
        write_json(root / R.RESTORATION, restoration)
        adoption = {
            **s4,
            "restoration_decision": {"file": R.RESTORATION,
                                     "sha256": R._sha(root / R.RESTORATION)},
            "source_S4_adoption": sources["s4_adoption"],
            "source_veto_adoption": sources["veto_adoption"],
            "source_failure": sources["failure"],
            "historical_publication_veto": True,
            "restored_by_explicit_user_decision": True,
            "user_decision": {"quote": R.USER_QUOTE},
            "no_new_physical_evaluation": True,
            "current_validation_record": R.VALIDATION,
        }
        return restoration, adoption

    def _presentation_context(self, root: Path):
        actual = Base.ROOT
        official = Base.read(actual / "results/v2_protocol/R7_OFFICIAL_REGRESSION.json")
        holdout = Base.read(actual / "results/v2_protocol/R7_HOLDOUT_REVIEW.json")
        historical = Base._historical_context(actual)
        docs = {}
        for mode, variant in (("servo", "R5_A"), ("servo", "R7"),
                              ("torque", "R1"), ("torque", "R7")):
            key = f"R7_BASE_parity_{variant}_{mode}"
            docs[key] = Base.read(actual / f"results/v2_protocol/{key}.json")
        source = {
            "root": root, "adopted": True, "negative": False,
            "effective": {"servo": "SCALE_STATE", "torque": "R1"},
            "official_fresh": True, "official": official, "holdout": holdout,
            "historical": historical,
            "equivalence": "저장 990개와 실제 full-loop 대조 8회가 비트 단위로 같았다.",
            "holdout_rows": Base._holdout_rows(holdout),
            "control_rows": Base._control_rows(official),
            "collision_rows": Base._collision_rows(official),
            "t_rows": Base._t_rows(historical, True),
            "contract": Base.read(actual / "results/v2_protocol/R7_PREFREEZE_CONTRACT.json"),
            "prefinal_contract": Base.read(actual / "results/v2_protocol/R7_PREFREEZE_FINAL_CONTRACT.json"),
            "final_contract": Base.read(actual / "results/v2_protocol/R7_FINAL_CONTRACT_REVIEW.json"),
            "parity_timing_rows": Base._parity_timing_from_base(docs),
        }
        source["inherited_rows"] = Base._inherited_rows(source)
        source["restoration_validation"] = {
            "work": {"full_suite": {"tests_run": 701}},
            "public": {"full_suite": {"tests_run": 581}},
        }
        for name in ("R5_PERTURB_RESULTS.json", "R5_OFFICIAL_REGRESSION.json",
                     "S4_OFFICIAL.json"):
            target = root / "results/v2_protocol" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(Base.ROOT / "results/v2_protocol" / name)
        media = root / "artifacts/media"
        media.mkdir(parents=True)
        cases = []
        for index in (9, 10):
            artifacts = []
            for suffix in ("gif", "mp4"):
                relative = f"artifacts/media/{index:02d}_r7.{suffix}"
                (root / relative).write_bytes(b"x")
                artifacts.append({"file": relative})
            cases.append({"artifacts": artifacts})
        write_json(media / "r7_manifest.json", {"schema": "r7.media-manifest.v1", "cases": cases})
        return source

    def test_cli_bootstraps_from_external_directory(self):
        command = ["env", "-u", "PYTHONPATH", sys.executable, "-I", "-B", "-c",
                   "import runpy,sys; runpy.run_path(sys.argv[1]); print('PASS')", str(Path(R.__file__).resolve())]
        with tempfile.TemporaryDirectory() as name:
            result = subprocess.run(command, cwd=name, text=True, capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_restoration_record_and_canonical_adoption_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            restoration, adoption = self._record_fixture(root)
            R.validate_restoration(root, restoration, adoption)
            adoption["no_new_physical_evaluation"] = False
            with self.assertRaisesRegex(ValueError, "flags"):
                R.validate_restoration(root, restoration, adoption)

    def test_restoration_rejects_changed_scientific_record(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            restoration, adoption = self._record_fixture(root)
            target = root / R.SCIENTIFIC_RECORDS[0]
            target.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scientific record changed"):
                R.validate_restoration(root, restoration, adoption)

    def test_restoration_rejects_nonzero_or_boolean_action_counter(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            restoration, adoption = self._record_fixture(root)
            restoration["training_runs"] = True
            with self.assertRaisesRegex(ValueError, "zero-action"):
                R.validate_restoration(root, restoration, adoption)

    def test_media_binding_rejects_semantic_or_byte_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            restoration, adoption = self._record_fixture(root)
            artifacts = [{"file": relative, "sha256": digest}
                         for relative, digest in restoration["restored_media_original_sha256"].items()]
            binding = {"file": R.D + "R7_ADOPTION.json",
                       "effective_variants": {"servo": "SCALE_STATE", "torque": "R1"}}
            historical = {"schema": "r7.media-manifest.v1",
                          "adoption": {**binding, "sha256": restoration["sources"]["s4_adoption"]["sha256"]},
                          "scope_note": "saved validation", "cases": [{"artifacts": artifacts}]}
            replace_json(root / (R.D + "R7_S4_MEDIA_MANIFEST.json"), historical)
            restoration["historical_media_manifest"]["sha256"] = R._sha(
                root / (R.D + "R7_S4_MEDIA_MANIFEST.json"))
            replace_json(root / R.RESTORATION, restoration)
            adoption["restoration_decision"]["sha256"] = R._sha(root / R.RESTORATION)
            write_json(root / (R.D + "R7_ADOPTION.json"), adoption)
            current = dict(historical)
            current["adoption"] = {**binding, "sha256": R._sha(root / (R.D + "R7_ADOPTION.json"))}
            write_json(root / "artifacts/media/r7_manifest.json", current)
            R.validate_media_restoration(root, restoration, adoption)
            current["scope_note"] = "changed"
            replace_json(root / "artifacts/media/r7_manifest.json", current)
            with self.assertRaisesRegex(ValueError, "beyond adoption"):
                R.validate_media_restoration(root, restoration, adoption)
            current["scope_note"] = "saved validation"
            replace_json(root / "artifacts/media/r7_manifest.json", current)
            media_record = {
                "schema": "R7_RESTORATION_MEDIA.v1", "status": "PASS",
                "restoration_sha256": R._sha(root / R.RESTORATION),
                "adoption_sha256": R._sha(root / (R.D + "R7_ADOPTION.json")),
                "historical_manifest": {"file": R.D + "R7_S4_MEDIA_MANIFEST.json",
                                        "sha256": R._sha(root / (R.D + "R7_S4_MEDIA_MANIFEST.json"))},
                "current_manifest": {"file": "artifacts/media/r7_manifest.json",
                                     "sha256": R._sha(root / "artifacts/media/r7_manifest.json")},
                "original_media_sha256": restoration["restored_media_original_sha256"],
                "only_manifest_adoption_sha_rebound": True,
                "new_rendered_media": 0, "new_physical_simulations": 0,
                "new_holdout_batches": 0, "new_official_batches": 0,
                "remote_operations": 0,
            }
            write_json(root / R.RESTORATION_MEDIA, media_record)
            R.validate_media_restoration(root, restoration, adoption)
            media_record["new_rendered_media"] = True
            replace_json(root / R.RESTORATION_MEDIA, media_record)
            with self.assertRaisesRegex(ValueError, "zero-action"):
                R.validate_media_restoration(root, restoration, adoption)

    def test_fresh_validation_is_hash_bound_and_fail_closed(self):
        root = Base.ROOT
        cases = [{"status": "PASS"} for _ in range(8)]
        side = {"full_suite": {"status": "PASS", "tests_run": 700,
                               "errors": 0, "failures": 0},
                "default_submission_contract": {"status": "PASS", "cases": cases},
                "tested_assembly": {"path": "my_detector.py", "sha256": R.SERVO_SHA}}
        document = {"schema": "R7_RESTORATION_VALIDATION.v1", "status": "PASS",
                    "work": side, "public": {**side},
                    "restoration_sha256": R._sha(root / R.RESTORATION),
                    "adoption_sha256": R._sha(root / (R.D + "R7_ADOPTION.json")),
                    "detector_sha256": R.SERVO_SHA,
                    "new_physical_simulations": 0, "new_holdout_batches": 0,
                    "new_official_batches": 0}
        R.validate_restoration_validation(root, document)
        document["public"]["full_suite"] = {**side["full_suite"], "errors": 1}
        with self.assertRaisesRegex(ValueError, "public"):
            R.validate_restoration_validation(root, document)

    def test_curation_reports_new_records_and_missing_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / R.RESTORATION
            path.parent.mkdir(parents=True)
            write_json(path, {})
            value = R.build_curation(root)
            digest = R._sha(path)
        self.assertEqual(digest, value["sources"][R.RESTORATION]["sha256"])
        self.assertIn(R.D + "R7_VETO_ADOPTION.json", value["pending_required"])
        self.assertNotEqual(Base.CURATION.relative_to(Base.ROOT).as_posix(), R.CURATION)

    def test_rendered_documents_state_restoration_and_keep_precision(self):
        with tempfile.TemporaryDirectory() as name:
            context = self._presentation_context(Path(name))
            texts = R.render_all(context)
        for text in texts:
            self.assertIn(R.USER_QUOTE, text)
            self.assertIn("다시 실행하지 않았다", text)
            self.assertIn("공식 R1과 R7 실측 비교", text)
            self.assertIn("1.17 / 3.91 mm", text)
            self.assertIn("14.6 N·m", text)
            self.assertIn("62 / 45 / 27 / 18", text)
            self.assertIn("49 / 37 / 25 / 18", text)
            self.assertIn("작업 사본 701개", text)
            self.assertIn("기존 677개 기록은 S4 후보 계약", text)
            self.assertNotIn("1.170", text)
            self.assertNotIn("14.600", text)
            Base.safe(text, "restored report")

    def test_readme_retains_required_surfaces_and_media_note(self):
        with tempfile.TemporaryDirectory() as name:
            context = self._presentation_context(Path(name))
            text = R.render_readme(context)
        for token in ("**최종 제출:", "24개 충족", "4개 부분 충족", "5개 판정 불가",
                      "8개 검사는 모두 통과", "총 6건", "0.2098", "11.1507",
                      "11.703 ms", "14.163 ms", "포화는 2/189", "1 ms 초과",
                      "R5 PDF", "R6 PDF", "R3_delay_ms", "R3_false_alarms",
                      "새 holdout 효능 영상이 아니다"):
            self.assertIn(token, text)
        self.assertLess(text.index("**최종 제출:"), text.index("## 판정 복원 경위"))
        self.assertLess(text.index("## 판정 복원 경위"), text.index("| R1 A 서보"))

    def test_readme_problem_summary_has_three_parts_and_eight_priorities(self):
        with tempfile.TemporaryDirectory() as name:
            context = self._presentation_context(Path(name))
            text = R.render_readme(context)
        for token in (
            "## 문제 상황과 해결해야 할 과제", "### 1. 문제 상황",
            "### 2. 해결해야 할 과제", "### 3. 검증 과정에서 생긴 문제와 처리",
            "하중 이양(G4) 미달", "약한 집기·놓기 충돌 놓침", "최대 1 ms 초과",
            "섭동 오감지", "서보 포화", "토크 감지 개선 없음",
            "숨은 조건 일반화 미검증", "계산 비용 증가", "원인(확인/미확정)",
            "1. **하중 이양(G4)**", "2. **약한 집기·놓기 충돌**",
            "3. **최대 1 ms와 계산 비용**", "4. **섭동 오감지**",
            "5. **토크 감지 개선**", "6. **서보 포화 2/189**",
            "7. **숨은 조건 대비**", "8. **선택 항목**",
            "토크 추적 feedforward", "변위·속도", "잡음 시드 1개",
            "분리 가능성은 미확정", "OS 스케줄링 영향으로 보인다",
            "유일한 원인", "시도할 방향", "충돌 뒤 정지·후퇴",
            "D4 추가 검출 +5 (기준 6)", "reference FA 9→10",
            "그립 중 작업대 높이 ±35 mm 밖", "학습한 위상에서만 유효",
            "0 kg에서도 같은 세 조건", "numpy.bool_", "문서 10개",
            "문서 검사는 별도 통과 요건으로 유지한다",
            "docs/R2_README.md#무엇을-시도했고-왜-유지했나",
            "docs/R4_README.md#평가기-점검과-미검출-원인",
            "docs/R4_RESULTS.md#w40-오감지가-늘어난-조건",
            "docs/R5_RESULTS.md", "docs/R6_RESULTS.md", "docs/R7_PUBLICATION_VETO.md",
        ):
            self.assertIn(token, text)
        self.assertNotIn("## 미달과 범위", text)
        self.assertNotIn("운반에서만 켜진다", text)
        self.assertEqual(1, text.count("## 문제 상황과 해결해야 할 과제"))
        shortcut = "[문제 상황과 해결해야 할 과제](#문제-상황과-해결해야-할-과제)"
        self.assertLess(text.index(shortcut), text.index("## 판정 복원 경위"))

    def test_readme_outside_merged_problem_section_is_unchanged(self):
        with tempfile.TemporaryDirectory() as name:
            context = self._presentation_context(Path(name))
            legacy = Base.render_readme(context).replace(
                "R7 기록 수정 재검증", "R7 서보 후보 채택 복원", 1)
            final = f"**최종 제출: {Base._conclusion(context)}**\n\n"
            note = ("## 판정 복원 경위\n\n" + R._history_note() + " "
                    + R._validation_note(context))
            legacy = legacy.replace(final, final + note + "\n\n", 1)
            legacy = R._insert(legacy, "## 요구사항 점검",
                               "## 공식 R1과 R7 실측 비교\n\n" + R._comparison_block(context))
            current = R.render_readme(context)
        shortcut = "[문제 상황과 해결해야 할 과제](#문제-상황과-해결해야-할-과제)\n\n"
        current = current.replace(shortcut, "", 1)
        old_prefix, old_tail = legacy.split("## 미달과 범위\n\n", 1)
        _, old_suffix = old_tail.split("## 이전 라운드와 문서\n", 1)
        new_prefix, new_tail = current.split("## 문제 상황과 해결해야 할 과제\n\n", 1)
        section, new_suffix = new_tail.split("## 이전 라운드와 문서\n", 1)
        self.assertEqual(old_prefix, new_prefix)
        self.assertEqual(old_suffix, new_suffix)
        for token in ("50 %mg", "25 %mg", "+25 %p", "1.1 mg", "0.2098",
                      "0.0000", "11.1507", "0.3802", "증가 0", "1.0045", "1.0140",
                      "2210.500 µs", "11.703 ms", "14.163 ms", "총 6건", "2/189",
                      "A 6→6", "B 9→9", "하드웨어 안전 보장이나 충돌 정지 인증이 아니다"):
            self.assertIn(token, section)

    def test_problem_summary_uses_measured_numbers_not_fixed_display_values(self):
        with tempfile.TemporaryDirectory() as name:
            context = self._presentation_context(Path(name))
            context["historical"]["ref"]["G4_unmet"]["R1"]["servo"]["support_mean_percent"] = 7.654321
            context["holdout"]["entries"][0]["metrics"]["combined_p99_worst_us"] = 123.456
            state = next(row for row in context["historical"]["docs"]["R5_D4_RESULTS"]["entries"]
                         if row["mode"] == "torque" and row["variant"] == "SCALE_STATE")
            state["additional_detected"] = 4
            section = R._problem_section(context)
        self.assertIn("7.6543", section)
        self.assertIn("123.456 µs", section)
        self.assertIn("D4 추가 검출 +4 (기준 6)", section)

    def test_pdf_is_deterministic_and_contains_restoration_page(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            context = self._presentation_context(root)
            first, second = root / "a.pdf", root / "b.pdf"
            R.make_pdf(first, context)
            R.make_pdf(second, context)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            text = Base.pdf_text(first)
        self.assertIn("판정 복원 경위", text)
        self.assertIn("공식 R1과 R7 실측 비교", text)

    def test_module_does_not_use_private_acquisition_modules_or_json_dump(self):
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertNotIn("r7_" + "study", source)
        self.assertNotIn("r7_" + "sealed", source)
        self.assertNotIn("json." + "dump(", source)
        self.assertIn("write_json(path, document)", source)


if __name__ == "__main__":
    unittest.main()
