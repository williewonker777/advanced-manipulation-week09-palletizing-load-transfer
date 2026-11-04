#!/usr/bin/env python3
"""Synthetic standard-library tests for the fail-closed R3 report generator."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import r3_report as R
from tools import r3_report_manifest as M


class ReportFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "evidence").mkdir()
        (self.root / "results").mkdir()

    def close(self):
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / "evidence" / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def r2_metrics(self):
        return [{"id": f"T{i}", "name": f"metric {i}",
                 "condition": "matched condition", "unit": "unit",
                 "baseline": f"base {i}", "v1": f"v1 {i}",
                 "r1": f"r1 {i}"} for i in range(1, 7)]

    def artifacts(self):
        protocol = {"schema": "R3_PROTOCOL.v1",
                    "validation": [{"mass": mass, "com": [0.0, 0.0, 0.02]}
                                   for mass in (1.27, 3.67, 5.47, 7.17)]}
        documents = {key: {"schema": key + ".fixture"}
                     for key in R.REQUIRED_ARTIFACTS}
        documents["protocol"] = protocol
        documents["r2_report_manifest"] = {"metrics": self.r2_metrics()}
        answer = {key: self.write(key + ".json", value)
                  for key, value in documents.items() if key != "r2_presentation"}
        prefix = self.root / "results/presentation_R2.pdf"
        prefix.write_bytes((R.ROOT / "results/presentation.pdf").read_bytes())
        answer["r2_presentation"] = prefix
        return answer

    def curation(self):
        candidates = []
        for candidate in R.CANDIDATES:
            architecture = "servo" if candidate.startswith("A_") else "torque"
            candidates.append({
                "id": candidate, "method": "registered method",
                "architecture": architecture, "validation_cost": "pass",
                "validation_gain": "diagnostic value", "forced_stability": "fail",
                "actual_stability": "not advanced", "timing": "p99 700 us",
                "decision": "reject", "reason": "local stability gate failed",
            })
        per_mass = []
        for candidate in R.CANDIDATES:
            architecture = "servo" if candidate.startswith("A_") else "torque"
            for mass in (1.27, 3.67, 5.47, 7.17):
                per_mass.append({
                    "candidate": candidate, "architecture": architecture,
                    "mass_kg": mass, "r1_rms_mm": 2.0,
                    "diagnostic_rms_mm": 1.8, "r1_max_mm": 4.0,
                    "diagnostic_max_mm": 3.8, "r1_place_step_nm": 10.0,
                    "diagnostic_place_step_nm": 10.2,
                    "validation_status": "diagnostic only", "final_r3_alias": "R1",
                })
        metrics = []
        for source in self.r2_metrics():
            row = dict(source)
            row.update(r3=source["r1"], status="not improved")
            metrics.append(row)
        return {
            "schema": "R3_REPORT_MANIFEST.v1", "status": "frozen",
            "candidates": candidates, "per_mass_validation": per_mass,
            "metrics": metrics,
            "final_architectures": [
                {"architecture": "servo", "decision": "reject",
                 "selected": "A_handoff_long", "final_alias": "R1",
                 "my_sha256": "same-servo", "r1_sha256": "same-servo",
                 "reason": "registered gate failed"},
                {"architecture": "torque", "decision": "reject",
                 "selected": "B_mass_phase", "final_alias": "R1",
                 "my_sha256": "same-torque", "r1_sha256": "same-torque",
                 "reason": "registered gate failed"},
            ],
            "redesign": [
                {"architecture": mode, "used": 0,
                 "reason": "failure was stability-related, not a justified cost redesign",
                 "recorded_before_development": True,
                 "arbitrary_parameter_search_performed": False}
                for mode in ("servo", "torque")
            ],
            "lie_group_usage": [{
                "method": "pRNE", "section": "7.2", "equation": "Mdot=C+C^T",
                "elastic_1ms_zoh_delay_adaptation":
                    "elastic plant, 1 ms ZOH, one-cycle delay local analysis",
                "comparison": "matched R1 comparison", "decision": "reject",
            }],
            "normal_false_alarms": [
                {"stage": "validation", "architecture": mode, "count": 0,
                 "rise_vs_r1": 0, "status": "pass"}
                for mode in ("servo", "torque")
            ],
            "timing": [
                {"stage": "validation", "architecture": mode, "p99_us": 700.0,
                 "max_us": 1100.0, "outlier_count": 1,
                 "outliers_retained_raw": True, "p99_gate_pass": True,
                 "max_1000_goal_met": False, "status": "p99 pass; max recorded"}
                for mode in ("servo", "torque")
            ],
            "exposure": {
                "holdout": {"opened_after_freeze": True, "batches": 1,
                            "no_tuning_after": True},
                "official": {"executed_last": True, "no_tuning_after": True},
            },
            "stage_outcomes": ["validation completed", "development reviewed",
                               "D4 reviewed", "perturbations reviewed"],
            "document_details": {
                'stability': [dict(candidate=name, forced='2/3', actual='미실행: 비선정', r1='2/3')
                              for name in R.CANDIDATES],
                'costs': [dict(stage='D4', architecture='servo', exceeded=1, total=3,
                               metrics='놓기 계단 1행', masses='4 kg 1행', mass_metrics='4 kg 놓기 계단 1행', worst_ratio=1.056789)],
                'positive_mass_perturbation': [dict(architecture='torque', rows=22, rms_ratio=.653604, max_ratio=.687525)],
                'stability_notes': ['같은 점의 R1도 실패해 이 관문만으로 후보를 가를 수 없다.'],
                'redesign_context': ['결정 기록은 R1 비교보다 먼저 저장됐다. 원인 미식별 사유는 유효하다.'],
                'cost_notes': ['평균은 효능 관문이 아닌 참고값이다.'],
                'causes': ['A는 직접 DOB 보상이 이미 0이었다.', 'B는 P/V 피드백 기여가 하중 보간보다 컸다.']},
            "verification": {"executed": ["local simulation checks"],
                             "not_executed": ["hardware experiment"]},
            "limitations": ["finite local stability analysis",
                            "diagnostic candidates were not adopted"],
            "conclusion": "Both architectures retain the exact R1 alias.",
        }

    def manifest(self):
        artifacts = {key: (path, "reused_exact_source" if key == "r2_presentation"
                                  else "literal_copy" if key == "r2_report_manifest"
                                  else "fresh")
                     for key, path in self.artifacts().items()}
        return R.build_manifest(self.curation(), artifacts, self.root)


class R3ReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ReportFixture()
        self.manifest = self.fixture.manifest()

    def tearDown(self):
        self.fixture.close()

    def test_contract_is_explicit(self):
        contract = R.manifest_contract()
        self.assertEqual(contract["schema"], "R3_REPORT_MANIFEST.v1")
        self.assertEqual(contract["candidate_ids"], list(R.CANDIDATES))
        self.assertEqual(set(contract["artifact_keys"]), R.REQUIRED_ARTIFACTS)

    def test_validate_and_render_have_readable_tables_not_json_cells(self):
        context = R.validate_manifest(self.manifest, self.fixture.root)
        results = R.render_results(context)
        notes = R.render_presentation_notes(context)
        self.assertIn("서보 A 사전 검증", results)
        self.assertIn("토크 B 사전 검증", results)
        self.assertIn("T1–T6", results)
        self.assertNotIn("{'", results)
        self.assertNotIn('| architecture |', results)
        self.assertNotIn('| candidate |', results)
        self.assertIn('0.654', results)
        self.assertNotIn('0.653604', results)
        self.assertIn('같은 강제점 R1', results)
        self.assertIn("## 3. A 후보 판정", notes)
        self.assertIn("## 4. B 후보 판정", notes)
        self.assertGreaterEqual(R.TABLE_FONT_PT, 11.0)
        self.assertGreaterEqual(R.CANDIDATE_TABLE_FONT_PT, 10.5)

    def test_pdf_specs_project_compact_readable_tables(self):
        context = R.validate_manifest(self.manifest, self.fixture.root)
        slides = R._slides(context)
        for index in (2, 3):
            columns, rows = slides[index]["table"]
            self.assertEqual(columns, ["설정", "비용 / 효능", "강제 / 실제 국소",
                                       "계산 p99 / max", "판정"])
            self.assertLessEqual(len(rows), 3)
            self.assertNotIn("validation_cost", columns)
        for index in (4, 5):
            columns, rows = slides[index]["table"]
            self.assertEqual(len(columns), 5)
            self.assertEqual(len(rows), 4)
            self.assertEqual(columns[0], "kg")
            self.assertEqual(columns[-1], "최종 R3")
        for index in (7, 8):
            columns, rows = slides[index]["table"]
            self.assertEqual(columns, ["항목", "조건·단위", "기준", "v1", "R1", "R3"])
            self.assertEqual(len(rows), 3)
        self.assertIn("table", slides[14])
        self.assertNotIn("bullets", slides[14])
        self.assertEqual(slides[9]['table'][0], ['후보', '강제점 통과', '실제 후보점 통과', '같은 강제점 R1'])
        self.assertEqual(len(slides[9]['table'][1]), 5)
        self.assertIn('직접 DOB 보상', ' '.join(slides[1]['bullets']))
        self.assertNotIn('fresh', str(slides[7:9]))
        self.assertIn('R2와 같은 34조건의 보존값', str(slides[7:9]))
        columns, rows = slides[11]['table']
        self.assertEqual(len(columns), 6)
        self.assertEqual(len(rows), len(self.manifest['timing']))
        self.assertIn('p99 / 최대 목표', columns)

    def test_official_cells_are_actual_evaluator_cells_not_alias_numbers(self):
        context = R.validate_manifest(self.manifest, self.fixture.root)
        self.assertEqual(R._official_eight_kg(context), [])
        result = {'results': {}, 'architectures': {}}
        for mode in R.ARCHITECTURES:
            result['results'][mode] = {'control_and_collisions': {
                'control': {mass: ['1.17', '3.91', '0.0', '12', '14.6',
                                   '78', 0, 0, 0, '정상'] for mass in ('0.0', '4.0', '8.0')},
                'collisions': [{'mass': 8., 'time': 3.,
                                'cells': ['62', '45', '20', '10', '0회', '10N']}]}}
            result['architectures'][mode] = {'status': 'PASS'}
        context['artifacts']['official_review'] = result
        rows = R._official_eight_kg(context)
        self.assertEqual(rows[0]['RMS / 최대 mm'], '1.17 / 3.91')
        self.assertEqual(rows[0]['운반 15N / 10N ms'], '45 / 62')
        self.assertEqual(rows[0]['0/4/8kg 무충돌 FA'], '0/0/0')
        self.assertIn('마지막 공식 8 kg 회귀', R.render_results(context))
        self.assertEqual(rows[0]['놓기 step Nm'], '14.6')
        rendered = R.render_results(context)
        self.assertIn('| 1.17 / 3.91 |', rendered)
        self.assertIn('| 14.6 |', rendered)
        self.assertNotIn('| 14.600 |', rendered)

    def test_timing_builder_rejects_incomplete_cycle_total(self):
        with patch.object(M, 'read', return_value={'candidate_grid': [], 'chosen_specs': []}), \
                patch.object(M, 'aggregate_episode_timings', return_value={
                    'results': [{'rows': 1885, 'cycles': 1}]}):
            with self.assertRaisesRegex(ValueError, '11310000'):
                M.build_timing()

    def test_public_fixture_formatting_preserves_ast_and_other_sources(self):
        import ast
        from tools.build_public import public_python_bytes
        source = b'class Fake:\n    def __init__(self):\n        self.last_command = np.zeros(6)\n'
        changed = public_python_bytes('tools/test_r3_stability.py', source)
        self.assertNotEqual(source, changed)
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(changed)))
        self.assertEqual(public_python_bytes('my_servo.py', source), source)

    def test_cjk_pdf_wrapping_preserves_all_characters(self):
        text = '서보 최대 1.006405 / 토크 최대 1.001635; R1 대비 검증'
        wrapped = R._wrap_display(text, 12)
        self.assertEqual(wrapped.replace('\n', ''), text)
        self.assertGreater(wrapped.count('\n'), 1)
        translated = R._prose('forced206+actual112; peak/sat/masked; report17')
        self.assertIn('강제점 검사 206+실제 후보점 검사 112', translated)
        self.assertNotIn('forced', translated)
        self.assertNotIn('actual', translated)
        self.assertIn('최대값/포화/기존 경보·구분 불가', translated)
        self.assertEqual(R._prose('[재현](report/REPRODUCE.md) `servo`'),
                         '[재현](report/REPRODUCE.md) `servo`')
        polished = R._prose('같은 case 추가FA1; score0.951; 40N6방향; reference189조건; pRNE16표본; 1885episode; overlay333; byte 동일')
        for leftover in ('case', 'FA1', 'score0', '40N6', 'reference189', 'pRNE16',
                         '1885episode', 'overlay333', 'byte 동일'):
            self.assertNotIn(leftover, polished)
        self.assertIn('40 N 6방향', polished)
        self.assertEqual(R._prose('one-cycle delay'), '한 주기 지연')

    def test_pdf_wrap_keeps_numeric_tokens_together_when_they_fit(self):
        text = 'abcdefghij 123.456789 abc'
        wrapped = R._wrap_display(text, 9)
        self.assertIn('123.456789', wrapped)
        self.assertEqual(wrapped.replace('\n', ''), text)

    def test_publication_timing_inventory_never_drops_measured_outliers(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            row = dict(architecture='servo', variant='R1', stage='validation', case_id=0,
                timing=dict(combined=dict(wall=dict(samples=6000, p99_us=300., max_us=1200.),
                                         thread_cpu=dict(max_us=1100.)),
                            outliers=[dict(combined=dict(wall_ns=1200000, cpu_ns=1100000))]))
            episode = root / 'episode.json'
            episode.write_text(json.dumps({'results': [row]}))
            # A parity document exists, but is not a measured-episode input.
            (root / 'parity.json').write_text(json.dumps({'results': [{'variant': 'R1'}]}))
            result = M.aggregate_episode_timings([(episode, 1)], root)
            self.assertEqual(result['results'][0]['cycles'], 6000)
            self.assertEqual(result['results'][0]['overrun_cycles'], 1)
            self.assertEqual(result['results'][0]['all_outliers'][0]['combined']['cpu_ns'], 1100000)
            row.pop('timing')
            episode.write_text(json.dumps({'results': [row]}))
            with self.assertRaises(KeyError):
                M.aggregate_episode_timings([(episode, 1)], root)

    def test_hash_mismatch_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["artifacts"]["protocol"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            R.validate_manifest(broken, self.fixture.root)

    def test_failed_normal_false_alarms_are_reportable_not_dropped(self):
        changed = copy.deepcopy(self.manifest)
        changed['normal_false_alarms'][0].update(count=2, rise_vs_r1=1, status='FAIL')
        context = R.validate_manifest(changed, self.fixture.root)
        self.assertIn('실패', R.render_results(context))
        changed['normal_false_alarms'][0]['status'] = 'PASS'
        with self.assertRaisesRegex(ValueError, 'status must match'):
            R.validate_manifest(changed, self.fixture.root)

    def test_r2_literal_copy_and_r1_alias_are_enforced(self):
        broken = copy.deepcopy(self.manifest)
        broken["metrics"][0]["baseline"] = "invented"
        with self.assertRaisesRegex(ValueError, "literal R2 manifest copy"):
            R.validate_manifest(broken, self.fixture.root)
        broken = copy.deepcopy(self.manifest)
        broken["final_architectures"][0]["my_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "exact my/R1 SHA"):
            R.validate_manifest(broken, self.fixture.root)

    def test_missing_candidate_or_mass_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "all five"):
            R.validate_manifest(broken, self.fixture.root)
        broken = copy.deepcopy(self.manifest)
        broken["per_mass_validation"].pop()
        with self.assertRaisesRegex(ValueError, "four validation masses"):
            R.validate_manifest(broken, self.fixture.root)

    def test_redesign_and_outlier_constraints_are_enforced(self):
        broken = copy.deepcopy(self.manifest)
        broken["redesign"][0]["recorded_before_development"] = False
        with self.assertRaisesRegex(ValueError, "prior reason"):
            R.validate_manifest(broken, self.fixture.root)
        broken = copy.deepcopy(self.manifest)
        broken["timing"][0]["outliers_retained_raw"] = False
        with self.assertRaisesRegex(ValueError, "retain all raw outliers"):
            R.validate_manifest(broken, self.fixture.root)

    def test_publication_and_internal_wording_is_rejected(self):
        context = R.validate_manifest(self.manifest, self.fixture.root)
        for text in ("staging state", "push pending", chr(47) + "home/user/evidence",
                     "TASK" + "-369", "Notion notes"):
            with self.assertRaises(ValueError):
                R._privacy_check(R.render_results(context) + text, "fixture")

    def test_generate_then_check_current_and_detect_stale_notes(self):
        manifest_path = self.fixture.root / "R3_REPORT_MANIFEST.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        outputs = (self.fixture.root / "R3_RESULTS.md",
                   self.fixture.root / "PRESENTATION_R3.md",
                   self.fixture.root / "presentation.pdf")
        archive = self.fixture.root / "results/presentation_R2.pdf"
        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        context = R.generate(manifest_path, outputs, self.fixture.root)
        self.assertTrue(R.check_current(context, outputs))
        self.assertTrue(outputs[2].read_bytes().startswith(b"%PDF-"))
        self.assertGreater(outputs[2].stat().st_size, 10_000)
        self.assertEqual(R._pdf_pages(outputs[2]), len(R._slides(context)))
        self.assertEqual(R._pdf_pages(outputs[2]), 21)
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), archive_sha)
        outputs[1].write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            R.check_current(context, outputs)

    def test_no_simulator_or_controller_imports(self):
        text = (Path(__file__).with_name("r3_report.py")).read_text()
        self.assertNotIn("from sim", text)
        self.assertNotIn("import sim", text)
        self.assertNotIn("r3_torque", text)
        self.assertNotIn("r3_servo", text)


if __name__ == "__main__":
    unittest.main()
