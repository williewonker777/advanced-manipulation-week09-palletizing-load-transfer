"""Synthetic tests for the R6 public report renderer and strict adapters."""
import tempfile
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path

from tools import r6_report as R


class R6ReportTests(unittest.TestCase):
    def test_cli_module_bootstrap_works_from_an_external_directory(self):
        command = ["env", "-u", "PYTHONPATH", sys.executable, "-I", "-B", "-c",
                   "import runpy,sys; runpy.run_path(sys.argv[1]); "
                   "import tools.r5_report; print('isolated report import PASS')",
                   str(Path(R.__file__).resolve())]
        with tempfile.TemporaryDirectory() as name:
            result = subprocess.run(command, cwd=name, text=True,
                                    capture_output=True, check=False)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("isolated report import PASS", result.stdout)

    def test_negative_artifact_links_do_not_point_to_uncreated_outputs(self):
        text = R.artifact_links({"negative": True})
        self.assertNotIn("](../results/v2_protocol/PROTOCOL_R6.json)", text)
        self.assertNotIn("](../results/v2_protocol/R6_FREEZE.json)", text)
        self.assertIn("R6_S1_FAILURE.json", text)
        self.assertIn("R6_PARITY_RAW_REVIEW.json", text)

    def test_pdf_rows_reserve_space_for_the_longest_wrapped_cell(self):
        cells = [["short", "short"], ["\n".join(["scope"] * 12), "short"]]
        heights = R.table_row_heights(cells, 5)
        self.assertGreater(heights[2], heights[1])
        self.assertLessEqual(sum(heights), .85000001)
        self.assertEqual(3, len(heights))

    def test_curation_marks_missing_required_without_creating_data(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / R.REQUIRED[0]
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
            result = R.build_curation(root)
            self.assertEqual(R.sha(path), result["sources"][R.REQUIRED[0]]["sha256"])
            self.assertIn(R.REQUIRED[1], result["pending_required"])

    def test_negative_curation_requires_failure_evidence_not_sealed_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            failure = root / "results/v2_protocol/R6_S1_FAILURE.json"
            failure.parent.mkdir(parents=True)
            failure.write_text("{}", encoding="utf-8")
            required = R.required_inputs(root)
            self.assertIn("results/v2_protocol/R6_PARITY_RAW_REVIEW.json", required)
            self.assertNotIn("results/v2_protocol/PROTOCOL_R6.json", required)
            self.assertNotIn("results/v2_protocol/R6_PARITY_RESULT.json", required)

    def test_holdout_missing_is_explicitly_unrun(self):
        rows = R._holdout_rows(None)
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["판정"] == "S1 조기 종료" for row in rows))
        self.assertTrue(all(row["충돌 검출"] == "0조건" for row in rows))

    def test_holdout_requires_real_numeric_measurements(self):
        with self.assertRaises(ValueError):
            R._holdout_rows({"entries": [{"mode": "servo", "metrics": {}},
                                          {"mode": "torque", "metrics": {}}]})

    def test_holdout_renders_both_actual_files(self):
        metrics = {"R1_detected_collisions": 23, "tested_detected_collisions": 25,
                   "collision_count": 30, "additional_detected": 2,
                   "lost_R1": 0, "tool_losses": 0, "FA_increased_cases": 0,
                   "reference_FA": 0, "combined_p99_worst_us": 457.878,
                   "combined_max_worst_us": 7198.741, "cycles_over_1ms": 6}
        details = [{"case_id": index, "source_case_id": f"case-{index}",
                    "kind": kind, "R1": {"false_alarms": 0},
                    "tested": {"false_alarms": 0}}
                   for index, kind in enumerate(
                       ["reference"] * 4 + ["offset"] * 4 + ["collision"] * 30)]
        outliers = ([{"case": index, "combined": {"wall_ns": 1_000_001 + index}}
                     for index in range(6)]
                    + [{"case": "boundary", "combined": {"wall_ns": 1_000_000}}])
        combined = {"p99_worst_us": 457.878, "max_worst_us": 7198.741,
                    "samples": 228000}
        branch = {"components": {"combined": combined}, "cycles_over_1ms": 6,
                  "outliers": outliers, "outlier_count": 7,
                  "all_outliers_retained": True}
        timing = {"R1": branch, "tested": branch}
        document = {"entries": [
            {"mode": "servo", "tested_variant": "R6", "status": "PASS",
             "metrics": metrics, "details": details, "timing": timing},
            {"mode": "torque", "tested_variant": "R1", "status": "PASS",
             "metrics": metrics, "details": details, "timing": timing},
        ]}
        rows = R._holdout_rows(document)
        self.assertEqual(["R6", "R6 파일(토크 R1 분기)"], [row["파일"] for row in rows])
        self.assertIn("1 ms 초과 6주기", rows[0]["시험 계산 시간"])
        self.assertIn("전체 0→0", rows[0]["오감지"])

    def test_control_preserves_source_string_precision(self):
        control = {mass: ["1.17", "3.91", "0.0", "9.7", "14.6", "79",
                          "0", "0", "0", "정상"] for mass in ("0.0", "4.0", "8.0")}
        collision = [{"mass": mass, "time": time,
                      "cells": ["놓침", "69", "33", "21", "0회", "15 N"]}
                     for mass in (0.0, 8.0) for time in (1.42, 3.0, 3.86)]
        collision[4]["cells"][4] = "1회"
        document = {"records": {mode: {"control_and_collisions":
                                         {"control": control, "collisions": collision}}
                                for mode in R.MODES}}
        rows = R._control_rows(document)
        self.assertEqual("1.17 / 3.91 mm", rows[0]["TCP RMS/최대"])
        self.assertNotIn("1.170", rows[0]["TCP RMS/최대"])

    def test_collision_table_keeps_all_forces_fa_and_minimum(self):
        collisions = [{"mass": mass, "time": time,
                       "cells": ["49", "37", "25", "18", "0회", "10 N"]}
                      for mass in (0.0, 8.0) for time in (1.42, 3.0, 3.86)]
        torque = [dict(row) for row in collisions]
        torque = [{**row, "cells": list(row["cells"])} for row in torque]
        torque[4]["cells"][4] = "1회"
        document = {"records": {
            "servo": {"control_and_collisions": {"collisions": collisions}},
            "torque": {"control_and_collisions": {"collisions": torque}},
        }}
        rows = R._collision_rows(document)
        self.assertEqual(12, len(rows))
        target = next(row for row in rows
                      if row["방식"] == "B 토크" and row["질량/시각"] == "8 kg / 3 s")
        self.assertEqual("1회", target["충돌 세트 FA"])
        self.assertEqual("10 N", target["최소 검출력"])
        self.assertEqual("18", target["100 N"])

    def test_collision_rejects_missing_explicit_torque_fa(self):
        collisions = [{"mass": mass, "time": time,
                       "cells": ["49", "37", "25", "18", "0회", "10 N"]}
                      for mass in (0.0, 8.0) for time in (1.42, 3.0, 3.86)]
        document = {"records": {mode: {"control_and_collisions": {"collisions": collisions}}
                                for mode in R.MODES}}
        with self.assertRaisesRegex(ValueError, "false alarm"):
            R._collision_rows(document)

    def test_preserved_r1_official_has_full_exact_collision_grid(self):
        document = R._preserved_r1_official(R.ROOT)
        rows = R._collision_rows(document)
        servo = next(row for row in rows
                     if row["방식"] == "A 서보" and row["질량/시각"] == "8 kg / 3 s")
        torque = next(row for row in rows
                      if row["방식"] == "B 토크" and row["질량/시각"] == "8 kg / 3 s")
        self.assertEqual(("62", "45", "27", "18"),
                         tuple(servo[key] for key in ("10 N", "15 N", "40 N", "100 N")))
        self.assertEqual("1회", torque["충돌 세트 FA"])
        self.assertEqual("10 N", torque["최소 검출력"])

    def test_s1_raw_review_timing_keeps_all_four_actual_groups(self):
        timing = [{"mode": mode, "variant": variant, "episodes": 2,
                   "combined_p99_worst_us": 450.0, "combined_max_worst_us": 1200.0,
                   "cycles_over_1ms": 1, "detector_p99_worst_us": 390.0,
                   "detector_max_worst_us": 1100.0, "outliers": [{"case": "x"}],
                   "outlier_count": 1, "all_outliers_retained": True}
                  for mode, variant in (("servo", "R5_A"), ("servo", "R6"),
                                        ("torque", "R1"), ("torque", "R6"))]
        rows = R._parity_timing_rows({"timing": timing})
        self.assertEqual(4, len(rows))
        self.assertEqual(8, sum(row["에피소드"] for row in rows))
        self.assertTrue(all(row["보존 outlier"] == 1 for row in rows))

    def test_equivalence_requires_all_streams_and_eight_episodes(self):
        counts = [124] * 6 + [123] * 2
        aggregate = [{"streams": count, "samples": count * 6000,
                      "passed_streams": count, "all_bit_comparisons_passed": True}
                     for count in counts]
        streams = [{"stream_id": f"validation-servo-{index:03d}", "mode": "servo",
                    "passed": True, "offline_decisions_exact": True,
                    "source_raw_sha256": "a" * 64, "proof_sha256": "b" * 64}
                   for index in range(495)]
        streams += [{"stream_id": f"validation-torque-{index:03d}", "mode": "torque",
                     "passed": True, "offline_decisions_exact": None,
                     "source_raw_sha256": "c" * 64, "proof_sha256": "d" * 64}
                    for index in range(495)]
        saved = {"stream_count": 990, "total_samples": 5940000,
                 "all_990_streams_bit_exact": True, "streams": streams,
                 "aggregate": aggregate}
        parity = {"passed": True, "pair_count": 4, "episodes_run": 8, "rows": [{}] * 4}
        text = R._equivalence_text(saved, parity, {"episodes": 8})
        self.assertIn("990", text)
        with self.assertRaises(ValueError):
            R._equivalence_text({**saved, "all_990_streams_bit_exact": False},
                                parity, {"episodes": 8})

    def test_perturb_fa_keeps_28_reference_and_34_total_scopes_separate(self):
        def row(mode, r1_reference, state_reference, r1_all, state_all):
            details = []
            for index in range(28):
                details.append({"case": {"kind": "reference"},
                                "R1": {"false_alarms": r1_reference if index == 0 else 0},
                                "candidate": {"false_alarms": state_reference if index == 0 else 0}})
            return {"mode": mode, "variant": "SCALE_STATE", "details": details,
                    "R1_FA": r1_all, "candidate_FA": state_all}
        document = {"entries": [row("servo", 6, 6, 6, 6),
                                  row("torque", 9, 10, 10, 11)]}
        adopted = R._perturb_fa_counts(document, True)
        self.assertEqual((6, 6, 6, 6),
                         tuple(adopted["servo"][key] for key in
                               ("r1_reference", "final_reference", "r1_all", "final_all")))
        self.assertEqual((9, 9, 10, 10),
                         tuple(adopted["torque"][key] for key in
                               ("r1_reference", "final_reference", "r1_all", "final_all")))
        fallback = R._perturb_fa_counts(document, False)
        self.assertEqual(9, fallback["torque"]["final_reference"])
        self.assertEqual(10, fallback["torque"]["final_all"])
        self.assertEqual(10, fallback["torque"]["state_reference"])
        self.assertEqual(11, fallback["torque"]["state_all"])

    def test_t6_and_inherited_counts_are_derived_from_historical_results(self):
        historical = R._historical_context(R.ROOT)
        adopted = R._t_rows(historical, True)
        fallback = R._t_rows(historical, False)
        self.assertIn("A 119/61/0", adopted[5]["R6"])
        self.assertEqual(fallback[5]["R1"], fallback[5]["R6"])
        rows = R._inherited_rows({"historical": historical, "adopted": True})
        self.assertEqual("13→15/30", rows[0]["A 서보"])
        self.assertEqual("11→11/30", rows[0]["B 토크"])
        self.assertEqual("13→18/30", rows[1]["A 서보"])
        self.assertEqual("12→12/30", rows[1]["B 토크"])
        self.assertIn("111→119/180", rows[2]["A 서보"])
        self.assertIn("28 reference FA 9→9", rows[3]["B 토크"])

    def test_fallback_keeps_positive_trial_evidence_but_final_t6_is_r1(self):
        historical = R._historical_context(R.ROOT)
        final_rows = R._t_rows(historical, False)
        trial_rows = R._inherited_rows({"historical": historical, "adopted": False})
        self.assertEqual(final_rows[5]["R1"], final_rows[5]["R6"])
        self.assertEqual("13→15/30", trial_rows[0]["A 서보"])
        self.assertEqual("13→18/30", trial_rows[1]["A 서보"])
        self.assertIn("111→119/180", trial_rows[2]["A 서보"])
        self.assertEqual("11→11/30", trial_rows[0]["B 토크"])

    def test_table_keeps_all_rows_and_escapes_pipes(self):
        text = R.md_table(["a", "b"], [{"a": "x|y", "b": 1}, {"a": "z", "b": 2}])
        self.assertEqual(4, len(text.splitlines()))
        self.assertIn("x\\|y", text)

    def test_privacy_scan_rejects_machine_path(self):
        with self.assertRaises(ValueError):
            R.safe("/ho" + "me/example", "fixture")

    def test_historical_media_block_has_four_gif_mp4_pairs(self):
        with tempfile.TemporaryDirectory() as name:
            text = R._media_block(Path(name), False)
        for prefix in ("05", "06", "07", "08"):
            self.assertIn(prefix, text)
        self.assertEqual(4, text.count(".gif"))
        self.assertEqual(4, text.count(".mp4"))

    def test_pdf_has_twenty_readable_pages_and_required_sections(self):
        holdout = [{"방식": mode, "파일": "R6", "충돌 검출": "추가 +2/30",
                    "오감지": "reference 0", "R1 계산 시간": "p99 350 µs",
                    "시험 계산 시간": "p99 450 µs", "판정": "통과"}
                   for mode in ("A 서보", "B 토크")]
        control = [{"방식": mode, "질량": mass, "TCP RMS/최대": "1.17 / 3.91 mm",
                    "포화": "0.0 %", "집기/놓기 계단": "9.7 / 14.6 N·m",
                    "오감지": "0 / 0 / 0", "상태": "정상"}
                   for mode in ("A 서보", "B 토크") for mass in ("0 kg", "4 kg", "8 kg")]
        collision = [{"방식": mode, "질량/시각": f"8 kg / {time} s", "10 N": "49",
                      "15 N": "37", "40 N": "25", "100 N": "18",
                      "충돌 세트 FA": "1회" if mode == "B 토크" and time == 3 else "0회",
                      "최소 검출력": "10 N"}
                     for mode in ("A 서보", "B 토크") for time in (1.42, 3, 3.86)]
        t_rows = [{"항목": f"T{i}", "기준": "1", "v1": "2", "R1": "3", "R6": "3"}
                  for i in range(1, 7)]
        context = {"root": R.ROOT, "adopted": True, "holdout_rows": holdout,
                   "control_rows": control, "collision_rows": collision, "t_rows": t_rows,
                   "inherited_rows": [{"단계": "D4", "A 서보": "111→119/180",
                                       "B 토크": "111→111/180", "성격": "승계"}],
                   "equivalence": "저장 990개와 물리 8회가 비트 단위로 같다."}
        with tempfile.NamedTemporaryFile(suffix=".pdf") as stream:
            with mock.patch.object(R, "_unmet", return_value=["합성 제한"]):
                R.make_pdf(Path(stream.name), context)
                text = R.pdf_text(Path(stream.name))
                pages = subprocess.check_output(["pdfinfo", stream.name], text=True)
        self.assertIn("Pages:           20", pages)
        for token in ("R6 결론", "공식 10/15/40/100 N", "미달", "최종 판정"):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
