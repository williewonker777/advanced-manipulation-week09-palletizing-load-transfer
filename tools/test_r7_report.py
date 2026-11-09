"""Pure tests for the R7 public report renderer and strict presentation adapters."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from tools import r7_report as R


class R7ReportTests(unittest.TestCase):
    def _context(self):
        control = []
        tokens = {
            "A 서보": (("0.0", "1.18", "3.94", "5.1"),
                       ("4.0", "1.15", "3.79", "7.6"),
                       ("8.0", "1.17", "3.91", "14.6")),
            "B 토크": (("0.0", "2.09", "4.05", "4.8"),
                       ("4.0", "2.32", "4.42", "6.8"),
                       ("8.0", "3.23", "8.44", "13.1")),
        }
        for mode, rows in tokens.items():
            for mass, rms, maximum, place in rows:
                control.append({"방식": mode, "질량": mass + " kg",
                                "TCP RMS/최대": rms + " / " + maximum + " mm",
                                "포화": "0.0 %", "집기/놓기 계단": "9.7 / " + place + " N·m",
                                "오감지": "0 / 0 / 0", "상태": "정상"})
        collisions = []
        for mode in ("A 서보", "B 토크"):
            for mass in (0, 8):
                for time in (1.42, 3.0, 3.86):
                    collisions.append({"방식": mode, "질량/시각": f"{mass} kg / {time:g} s",
                                       "10 N": "놓침" if time != 3.0 else "62",
                                       "15 N": "놓침" if time == 1.42 else "45",
                                       "40 N": "27", "100 N": "18",
                                       "충돌 세트 FA": "1회" if mode == "B 토크" and mass == 8 and time == 3.0 else "0회",
                                       "최소 검출력": "40 N" if time == 1.42 else "10 N"})
        holdout = [{"방식": mode, "파일": "R1", "충돌 검출": "13→13/30, 추가 +0, 손실 0, 도구 손실 0",
                    "오감지": "전체 0→0; reference 0→0; offset 0→0; 증가 조건 0",
                    "R1 계산 시간": "p99 400.000 / 최대 1200.000 µs / 1 ms 초과 1주기",
                    "시험 계산 시간": "p99 400.000 / 최대 1200.000 µs / 1 ms 초과 1주기",
                    "판정": "탈락" if mode == "A 서보" else "통과"}
                   for mode in ("A 서보", "B 토크")]
        timing = [{"방식": mode, "비교 파일": variant, "에피소드": 2,
                   "감지 p99/최대 [µs]": "350.000 / 900.000",
                   "합산 p99/최대 [µs]": "400.000 / 1200.000",
                   "1 ms 초과": 1, "보존 outlier": 1}
                  for mode, variant in (("A 서보", "R5_A"), ("A 서보", "R7"),
                                        ("B 토크", "R1"), ("B 토크", "R7"))]
        t_rows = [{"항목": f"T{i}", "기준": "기준", "v1": "v1", "R1": "R1", "R7": "R7"}
                  for i in range(1, 7)]
        return {"root": R.ROOT, "adopted": False, "official_fresh": True,
                "negative": False, "effective": {"servo": "R1", "torque": "R1"},
                "equivalence": "저장 990개와 새 full-loop 8회가 비트 단위로 같았다.",
                "holdout": {"entries": [{"metrics": {"combined_max_worst_us": 1200.0}},
                                           {"metrics": {"combined_max_worst_us": 1100.0}}]},
                "holdout_rows": holdout, "control_rows": control,
                "collision_rows": collisions, "parity_timing_rows": timing,
                "t_rows": t_rows,
                "inherited_rows": [{"단계": x, "A 서보": "13→15/30", "B 토크": "11→11/30", "성격": "보존"}
                                   for x in ("validation", "development", "D4", "섭동")],
                "contract": {"full_suite": {"tests_run": 600}},
                "prefinal_contract": {"full_suite": {"tests_run": 620}},
                "final_contract": {"status": "PASS"}, "docs": {}}

    def test_cli_bootstraps_from_external_directory(self):
        command = ["env", "-u", "PYTHONPATH", sys.executable, "-I", "-B", "-c",
                   "import runpy,sys; runpy.run_path(sys.argv[1]); print('PASS')", str(Path(R.__file__).resolve())]
        with tempfile.TemporaryDirectory() as name:
            result = subprocess.run(command, cwd=name, text=True, capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_curation_reports_missing_without_fabricating_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / R.REQUIRED[0]
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
            digest = R.sha(path)
            value = R.build_curation(root)
        self.assertEqual(digest, value["sources"][R.REQUIRED[0]]["sha256"])
        self.assertIn(R.REQUIRED[1], value["pending_required"])

    def test_failure_curation_requires_explicit_failure_and_adoption(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "results/v2_protocol/R7_FAILURE.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
            required = R.required_inputs(root)
        self.assertIn("results/v2_protocol/R7_FAILURE.json", required)
        self.assertIn("results/v2_protocol/R7_ADOPTION.json", required)
        self.assertNotIn("results/v2_protocol/R7_OFFICIAL_REGRESSION.json", required)

    def test_numpy_writer_is_the_only_json_output_sink(self):
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn("write_json(path, build_curation(root))", source)
        self.assertNotIn("json.dump(", source)

    def test_control_preserves_official_string_precision(self):
        control = {mass: ["1.17", "3.91", "0.0", "9.7", "14.6", "79", "0", "0", "0", "정상"]
                   for mass in ("0.0", "4.0", "8.0")}
        collision = [{"mass": mass, "time": time, "cells": ["놓침", "69", "33", "21", "0회", "15 N"]}
                     for mass in (0.0, 8.0) for time in (1.42, 3.0, 3.86)]
        collision[4]["cells"][4] = "1회"
        doc = {"records": {mode: {"control_and_collisions": {"control": control, "collisions": collision}}
                           for mode in R.MODES}}
        rows = R._control_rows(doc)
        self.assertEqual("1.17 / 3.91 mm", rows[0]["TCP RMS/최대"])
        self.assertNotIn("1.170", rows[0]["TCP RMS/최대"])

    def test_collision_table_is_complete_and_keeps_torque_false_alarm(self):
        collisions = [{"mass": mass, "time": time, "cells": ["49", "37", "25", "18", "0회", "10 N"]}
                      for mass in (0.0, 8.0) for time in (1.42, 3.0, 3.86)]
        torque = [{**row, "cells": list(row["cells"])} for row in collisions]
        torque[4]["cells"][4] = "1회"
        doc = {"records": {"servo": {"control_and_collisions": {"collisions": collisions}},
                           "torque": {"control_and_collisions": {"collisions": torque}}}}
        rows = R._collision_rows(doc)
        self.assertEqual(12, len(rows))
        target = next(row for row in rows if row["방식"] == "B 토크" and row["질량/시각"] == "8 kg / 3 s")
        self.assertEqual("1회", target["충돌 세트 FA"])
        self.assertEqual("10 N", target["최소 검출력"])

    def test_holdout_rejects_unmeasured_entries(self):
        with self.assertRaises(ValueError):
            R._holdout_rows({"entries": [{"mode": "servo", "metrics": {}},
                                          {"mode": "torque", "metrics": {}}]})

    def test_optional_prefinal_contract_fixture_requires_full_suite_and_eight_cases(self):
        fixture = {"status": "PASS", "errors": 0, "failures": 0,
                   "full_suite": {"status": "PASS", "tests_run": 620,
                                  "errors": 0, "failures": 0},
                   "default_contract": {"status": "PASS", "default_names_exact": True,
                                        "cases": [{"status": "PASS"} for _ in range(8)]}}
        self.assertIs(fixture, R._validate_prefinal_contract(fixture))
        broken = json.loads(json.dumps(fixture))
        broken["default_contract"]["cases"].pop()
        with self.assertRaisesRegex(ValueError, "pre-freeze contract"):
            R._validate_prefinal_contract(broken)

    def test_parity_timing_uses_all_four_groups_and_all_outliers(self):
        docs = {}
        for mode, variant in (("servo", "R5_A"), ("servo", "R7"),
                              ("torque", "R1"), ("torque", "R7")):
            rows = []
            for index in range(2):
                rows.append({"timing_summary": {
                    "detector": {"wall": {"p99_us": 300 + index, "max_us": 800 + index}},
                    "combined": {"wall": {"p99_us": 400 + index, "max_us": 1200 + index}},
                    "outliers": [{"combined": {"wall_ns": 1_200_000}}]}})
            docs[f"R7_BASE_parity_{variant}_{mode}"] = {"rows": rows}
        result = R._parity_timing_from_base(docs)
        self.assertEqual(4, len(result))
        self.assertTrue(all(row["1 ms 초과"] == 2 for row in result))
        self.assertTrue(all(row["합산 p99/최대 [µs]"] == "401.000 / 1201.000" for row in result))

    def test_readme_contains_all_nine_mandatory_surfaces(self):
        text = R.render_readme(self._context())
        self.assertIn("사전 시험과 별개인 실제 full-loop 대조 8회", text)
        self.assertNotIn("S1 full-loop", text)
        tokens = ("**최종 제출:", "0/4/8 kg", "10/15/40/100 N", "8 kg / 3 s", "1회",
                  "새 holdout 1 batch", "24개 충족", "4개 부분 충족", "5개 판정 불가", "8개 검사는 모두 통과",
                  "평가기 점검과 R1 놓침", "총 6건", "평균 ≥50 %mg", "최소 ≥25 %mg", "+25 %p", "≤1.1 mg",
                  "0.2098", "11.1507", "11.703 ms", "14.163 ms", "포화는 2/189", "p99", "1 ms 초과",
                  "R1](docs/R1_README.md)", "R6](docs/R6_README.md)", "R5 PDF", "R6 PDF",
                  "R3_delay_ms", "R3_false_alarms", "문서 색인")
        for token in tokens:
            self.assertIn(token, text)
        R.safe(text, "README.md")

    def test_failure_branch_renders_explicit_fallback_without_fake_measurement(self):
        context = self._context()
        context.update(negative=True, failure_stage="full-loop 대조", adopted=False,
                       official_fresh=False, official=None, holdout=None)
        text = R.render_readme(context)
        self.assertIn("full-loop 대조 실패", text)
        self.assertIn("R1을 유지", text)
        self.assertNotIn("후보를 채택했고", text)
        self.assertIn("과거 R1 보존값", text)

    def test_rendered_results_distinguish_fresh_and_inherited_evidence(self):
        text = R.render_results(self._context())
        self.assertIn("R7 새 공식 1 batch", text)
        self.assertIn("저장 자료로 승계한 단계", text)
        self.assertIn("새 효능 측정으로 세지", R.render_readme(self._context()))

    def test_index_links_root_and_every_direct_released_document(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in ("README.md", "docs/A.md", "report/B.md", "results/C.pdf"):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
            text = R.render_index(root)
        self.assertIn("[README로 돌아가기](../README.md)", text)
        self.assertIn("[docs/A.md](A.md)", text)
        self.assertIn("[report/B.md](../report/B.md)", text)
        self.assertIn("[results/C.pdf](../results/C.pdf)", text)

    def test_all_three_rendered_documents_resolve_relative_links_and_anchors(self):
        context = self._context()
        rendered = {
            "README.md": R.render_readme(context),
            "docs/R7_RESULTS.md": R.render_results(context),
            "report/PRESENTATION_R7.md": R.render_notes(context),
        }
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            references = [
                *(f"docs/R{index}_README.md" for index in range(1, 7)),
                "docs/V2_RESULTS.md", *(f"docs/R{index}_RESULTS.md" for index in range(2, 7)),
                "docs/R7_METHODS.md", "docs/REQUIREMENTS_AUDIT.md", "docs/R4_README.md",
                "results/presentation_R5.pdf", "results/presentation_R6.pdf",
                "results/presentation.pdf",
            ]
            references += [f"results/v2_protocol/{name}" for name in
                           ("R7_PARITY_RESULT.json", "PROTOCOL_R7.json", "R7_FREEZE.json",
                            "R7_HOLDOUT_REVIEW.json", "R7_OFFICIAL_REGRESSION.json",
                            "R7_ADOPTION.json")]
            references += [f"artifacts/media/{stem}.{suffix}"
                           for _, stem in R.HISTORICAL_MEDIA for suffix in ("gif", "mp4")]
            for relative in references:
                source = R.ROOT / relative
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_file():
                    shutil.copyfile(source, target)
                else:
                    target.write_bytes(b"placeholder")
            for relative, text in rendered.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            (root / "docs/INDEX.md").write_text(R.render_index(root), encoding="utf-8")
            from tools.r4_report_links import check_public_links
            result = check_public_links(root, [*rendered, "docs/INDEX.md"])
            self.assertTrue(result["passed"], result["missing"])
            for relative, text in rendered.items():
                for destination in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                    parsed = urlsplit(destination)
                    if not parsed.fragment or parsed.scheme:
                        continue
                    target = (root / relative).parent / unquote(parsed.path)
                    headings = re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", target.read_text(encoding="utf-8"))
                    slugs = {re.sub(r"[^0-9a-z가-힣 _-]", "", heading.lower()).strip().replace(" ", "-")
                             for heading in headings}
                    self.assertIn(unquote(parsed.fragment), slugs,
                                  f"{relative}: unresolved anchor {destination}")

    def test_adopted_media_requires_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "media manifest"):
                R._media_block(Path(name), True)

    def test_adopted_media_reads_canonical_r7_case_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cases = []
            for index in range(2):
                artifacts = []
                for suffix in (".gif", ".mp4"):
                    relative = f"artifacts/media/{9 + index:02d}_r7{suffix}"
                    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
                    artifacts.append({"file": relative})
                cases.append({"artifacts": artifacts})
            manifest = root / "artifacts/media/r7_manifest.json"
            manifest.write_text(json.dumps({"schema": "r7.media-manifest.v1", "cases": cases}), encoding="utf-8")
            text = R._media_block(root, True)
        self.assertIn("09_r7.gif", text)
        self.assertIn("10_r7.mp4", text)

    def test_privacy_scan_rejects_machine_path(self):
        with self.assertRaises(ValueError):
            R.safe("machine file at /" + "home/name/private", "x")

    def test_table_keeps_rows_and_escapes_pipes(self):
        text = R.md_table(["A"], [{"A": "x|y"}, {"A": "z"}])
        self.assertIn("x\\|y", text)
        self.assertEqual(4, len(text.splitlines()))

    def test_pdf_is_deterministic_and_contains_required_sections(self):
        context = self._context()
        with tempfile.TemporaryDirectory() as name:
            first, second = Path(name) / "a.pdf", Path(name) / "b.pdf"
            R.make_pdf(first, context); R.make_pdf(second, context)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            text = R.pdf_text(first)
        for token in ("R7 최종 판정", "새 holdout", "공식 충돌", "요구사항", "미달"):
            self.assertIn(token, text)

    def test_failure_artifact_links_only_existing_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            target = root / "results/v2_protocol/R7_ADOPTION.json"
            target.parent.mkdir(parents=True); target.write_text("{}", encoding="utf-8")
            text = R.artifact_links({"root": root})
        self.assertIn("R7_ADOPTION.json", text)
        self.assertNotIn("PROTOCOL_R7.json", text)


if __name__ == "__main__":
    unittest.main()
