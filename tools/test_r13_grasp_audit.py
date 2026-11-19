"""Assembly, isolation and scoring tests for the portable R13 grasp audit."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import r13_grasp_audit as audit


def fake_plan():
    return {
        'schema': 'r13-carried-datum.v1',
        'fit': [{'id': f'f{i}', 'mass': 3. + i, 'com': [.01, -.01, .02],
                 'perturb': {'seed': 100 + i}} for i in range(5)],
        'test': [{'id': name, 'mass': 3.5 + i, 'com': [.01, -.01, .02],
                  'shift': [.006, -.004, .003], 'perturb': {'seed': 200 + i}}
                 for i, name in enumerate('abcd')],
    }


class GeometryAndScoreTest(unittest.TestCase):
    def test_offsets_preserve_lateral_shift_and_reverse_only_height(self):
        case = fake_plan()['test'][0]
        self.assertIsNone(audit.offset(case, 'zero'))
        self.assertEqual(audit.offset(case, 'xy'), [.006, -.004, 0.])
        self.assertEqual(audit.offset(case, 'up'), [.006, -.004, .003])
        self.assertEqual(audit.offset(case, 'down'), [.006, -.004, -.003])

    def test_score_uses_actual_vertical_and_control_errors(self):
        rows = []
        for group in 'abcd':
            rows.extend([
                {'id': group, 'variant': 'zero', 'extra_height_mm': 0.,
                 'true_extra_mm': 0., 'error_mm': 0., 'paired_error_mm': 0.,
                 'accept': False, 'covered': True},
                {'id': group, 'variant': 'xy', 'extra_height_mm': .2,
                 'true_extra_mm': 0., 'error_mm': .2, 'paired_error_mm': .2,
                 'accept': False, 'covered': True},
                {'id': group, 'variant': 'up', 'extra_height_mm': 3.,
                 'true_extra_mm': 3., 'error_mm': 0., 'paired_error_mm': 0.,
                 'accept': True, 'covered': True},
                {'id': group, 'variant': 'down', 'extra_height_mm': -3.,
                 'true_extra_mm': -3., 'error_mm': 0., 'paired_error_mm': 0.,
                 'accept': True, 'covered': True},
            ])
        result = audit.scores(rows)
        self.assertTrue(result['qualified'])
        self.assertEqual(result['height_sign_correct'], 8)
        self.assertEqual(result['height_confident'], 8)
        self.assertAlmostEqual(result['control_max_abs_error_mm'], .2)

    def test_plan_rejects_wrong_counts_duplicate_ids_and_nonfinite_payload(self):
        wrong = fake_plan(); wrong['fit'].pop()
        duplicate = fake_plan(); duplicate['fit'][1]['id'] = duplicate['fit'][0]['id']
        nonfinite = fake_plan(); nonfinite['test'][0]['mass'] = np.nan
        for plan in (wrong, duplicate, nonfinite):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                audit.validate_plan(plan)


class AuditCliContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        self.kit = self.root / 'kit'
        self.output_parent = self.root / 'out'
        for directory in (self.project / 'tools', self.project / 'experiments/r13_carried',
                          self.kit / 'sim', self.kit / 'model', self.output_parent):
            directory.mkdir(parents=True, exist_ok=True)
        for name in ('servo', 'torque', 'detector'):
            (self.project / f'my_{name}.py').write_text(f'# frozen {name}\n')
        (self.project / 'tools/r13_grasp_estimator.py').write_text('# estimator\n')
        (self.project / 'tools/kit_manifest.sha256').write_text('frozen manifest\n')
        (self.project / 'experiments/r13_carried/plan.json').write_text(
            json.dumps(fake_plan()))
        (self.kit / 'evaluate.py').write_text('# original evaluator\n')

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, output):
        argv = ['r13_grasp_audit.py', '--kit', str(self.kit), '--out', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(audit, 'ROOT', self.project), \
                contextlib.redirect_stderr(io.StringIO()):
            audit.main()

    def prepare(self, _kit, overlay, _prefix, mode):
        (overlay / 'tools').mkdir()
        (overlay / 'tools/kit_manifest.sha256').write_bytes(
            (self.project / 'tools/kit_manifest.sha256').read_bytes())
        (overlay / f'{mode}_controller.py').write_text('# transient controller\n')
        (overlay / 'collision_detector.py').write_text('# transient detector\n')

    @staticmethod
    def child(command, **kwargs):
        overlay = Path(kwargs['cwd'])
        mode = 'servo' if (overlay / 'servo_controller.py').exists() else 'torque'
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        assert 'PYTHONPATH' not in kwargs['env']
        assert (overlay / 'grasp_estimator.py').is_file()
        assert (overlay / 'audit.py').is_file()
        assert (overlay / 'plan.json').is_file()
        assert mode in command[-1]
        result = {'calibration': {'mode': mode, 'model': {}},
                  'evaluation': {'mode': mode, 'rows': [], 'metrics': {'qualified': False}}}
        return subprocess.CompletedProcess(command, 0, json.dumps(result), '')

    def test_immutable_overlay_and_separate_archives(self):
        output = self.output_parent / 'portable'
        before = {name: hashlib.sha256((self.project / f'my_{name}.py').read_bytes()).hexdigest()
                  for name in ('servo', 'torque', 'detector')}
        with patch('tools.validate_submission.prepare_overlay', side_effect=self.prepare), \
                patch.object(audit.subprocess, 'run', side_effect=self.child):
            self.invoke(output)
        self.assertEqual(set(path.name for path in output.iterdir()),
                         {'calibration.json', 'evaluation.json', 'summary.json'})
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['scope']['episodes'], 42)
        self.assertFalse(summary['scope']['research_parity'])
        self.assertIn('not applied', summary['scope']['parity_limit'])
        for name, pin in before.items():
            self.assertEqual(summary['pins'][name], pin)

    def test_snapshot_is_bound_to_original_bytes_after_source_mutation(self):
        origin = self.project / 'my_servo.py'
        original = origin.read_bytes()
        blobs, pins = audit.snapshot_files({'servo': origin})
        origin.write_text('# changed after snapshot\n')
        destination = self.output_parent / 'installed.py'
        audit.install_snapshot(blobs['servo'], pins['servo'], destination)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(audit.digest(destination), pins['servo'])
        self.assertNotEqual(audit.digest(origin), pins['servo'])

    def test_missing_bad_kit_existing_inside_and_dangling_outputs_are_rejected(self):
        existing = self.output_parent / 'existing'; existing.mkdir()
        inside = self.kit / 'inside'
        dangling = self.output_parent / 'dangling'; dangling.symlink_to(self.root / 'absent')
        missing_kit = self.root / 'missing-kit'
        for kit, output in ((self.kit, existing), (self.kit, inside),
                            (self.kit, dangling), (missing_kit, self.output_parent / 'new')):
            argv = ['r13_grasp_audit.py', '--kit', str(kit), '--out', str(output)]
            with self.subTest(kit=kit, output=output), patch.object(sys, 'argv', argv), \
                    patch.object(audit, 'ROOT', self.project), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                audit.main()
        self.assertTrue(dangling.is_symlink())
        self.assertFalse(inside.exists())

    def test_symlink_candidate_and_child_failure_leave_output_absent(self):
        detector = self.project / 'my_detector.py'
        detector.unlink(); detector.symlink_to(self.project / 'my_servo.py')
        output = self.output_parent / 'candidate-fail'
        with self.assertRaisesRegex(ValueError, 'regular public candidate'):
            self.invoke(output)
        self.assertFalse(output.exists())

        detector.unlink(); detector.write_text('# detector\n')
        failure = subprocess.CompletedProcess(['python'], 3, '', 'synthetic failure')
        output = self.output_parent / 'child-fail'
        with patch('tools.validate_submission.prepare_overlay', side_effect=self.prepare), \
                patch.object(audit.subprocess, 'run', return_value=failure), \
                self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            self.invoke(output)
        self.assertFalse(output.exists())

    def test_help_works_from_unrelated_working_directory(self):
        script = Path(audit.__file__).resolve()
        with tempfile.TemporaryDirectory() as unrelated:
            result = subprocess.run([sys.executable, '-B', str(script), '--help'], cwd=unrelated,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--kit', result.stdout)
        self.assertIn('--out', result.stdout)


if __name__ == '__main__':
    unittest.main()
