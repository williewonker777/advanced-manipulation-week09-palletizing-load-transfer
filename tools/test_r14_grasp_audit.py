"""Contracts for the portable R14 registered grasp-height audit."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.r14_airborne import registration
from tools import r13_grasp_estimator as old
from tools import r14_grasp_audit as audit
from tools.test_r13_grasp_estimator import manufactured_feature


def fake_plan():
    return {
        'schema': 'r14-airborne.v1',
        'fit': [{'id': f'n{i:02d}', 'mass': 2.8 + .45 * i, 'com': [.01, -.02, .01],
                 'perturb': {'seed': 300 + i}} for i in range(12)],
        'test': [{'id': name, 'mass': 3.4 + i, 'com': [-.01, .02, -.01],
                  'shift': [.006, -.004, .005], 'perturb': {'seed': 400 + i}}
                 for i, name in enumerate('efgh')],
    }


class RegistrationContractTest(unittest.TestCase):
    def test_registered_frame_adds_prior_and_subtracts_normal_expectation(self):
        masses = np.linspace(3., 7., 12)
        fits = [manufactured_feature(mass) for mass in masses]
        raw_model, raw_guard, _ = old.calibration(fits, np.full(6, .05))
        registered_model = registration.calibrate(fits, np.full(6, .05))
        shifted_prior = manufactured_feature(5., 0.)
        shifted_prior['prior_bottom_mm'] = np.array(3.2)
        raw = old.estimate(shifted_prior, raw_model, raw_guard)
        registered = registration.decide(shifted_prior, registered_model)
        self.assertAlmostEqual(raw['extra_height_mm'], 0., places=10)
        self.assertAlmostEqual(registered['expected_normal_bottom_mm'], .2, places=10)
        self.assertAlmostEqual(registered['extra_height_mm'], 3., places=8)
        self.assertAlmostEqual(registered['absolute_bottom_mm'], 3.2, places=8)

    def test_registration_rejects_unknown_calibration_schema(self):
        with self.assertRaisesRegex(ValueError, 'unknown calibration'):
            registration.decide(manufactured_feature(5.), {'schema': 'wrong'})

    def test_policy_metrics_distinguish_raw_failure_from_registered_pass(self):
        rows = []
        for identifier in 'efgh':
            for variant, truth in (('zero', 0.), ('xy', .2), ('up', 3.), ('down', -3.)):
                def result(point, accept):
                    return {'extra_height_mm': point, 'applied_mm': point if accept else 0.,
                            'accept': accept, 'covered': True}
                rows.append({'id': identifier, 'variant': variant, 'truth_mm': truth,
                             'raw': result(0., False),
                             'registered': result(truth, abs(truth) >= 2.5)})
        raw = audit.policy_metrics(rows, 'raw')
        registered = audit.policy_metrics(rows, 'registered')
        self.assertFalse(raw['functional_gate'])
        self.assertTrue(registered['functional_gate'])
        self.assertGreater(raw['point_rmse_mm'], registered['point_rmse_mm'])
        self.assertEqual(registered['large_signed_accurate_actions'], 8)
        self.assertEqual(registered['tiny_nonzero_actions'], 0)

    def test_plan_requires_twelve_normal_and_four_held_groups(self):
        wrong_count = fake_plan(); wrong_count['fit'].pop()
        duplicate = fake_plan(); duplicate['test'][1]['id'] = duplicate['test'][0]['id']
        nonfinite = fake_plan(); nonfinite['fit'][0]['com'][1] = np.inf
        for plan in (wrong_count, duplicate, nonfinite):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                audit.validate_plan(plan)


class AuditCliContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        self.kit = self.root / 'kit'
        self.output_parent = self.root / 'output'
        for directory in (self.project / 'tools', self.project / 'experiments/r14_airborne',
                          self.kit / 'sim', self.kit / 'model', self.output_parent):
            directory.mkdir(parents=True, exist_ok=True)
        for name in ('servo', 'torque', 'detector'):
            (self.project / f'my_{name}.py').write_text(f'# {name}\n')
        (self.project / 'tools/r13_grasp_estimator.py').write_text('# estimator\n')
        (self.project / 'tools/r13_grasp_audit.py').write_text('# base audit\n')
        (self.project / 'tools/kit_manifest.sha256').write_text('manifest\n')
        (self.project / 'experiments/r14_airborne/registration.py').write_text('# registration\n')
        (self.project / 'experiments/r14_airborne/plan.json').write_text(json.dumps(fake_plan()))
        (self.kit / 'evaluate.py').write_text('# evaluator\n')

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, output):
        argv = ['r14_grasp_audit.py', '--kit', str(self.kit), '--out', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(audit, 'ROOT', self.project), \
                contextlib.redirect_stderr(io.StringIO()):
            audit.main()

    def prepare(self, _kit, overlay, _prefix, mode):
        (overlay / 'tools').mkdir()
        (overlay / 'tools/kit_manifest.sha256').write_bytes(
            (self.project / 'tools/kit_manifest.sha256').read_bytes())
        (overlay / f'{mode}_controller.py').write_text('# temporary\n')
        (overlay / 'collision_detector.py').write_text('# temporary\n')

    @staticmethod
    def child(command, **kwargs):
        overlay = Path(kwargs['cwd'])
        mode = 'servo' if (overlay / 'servo_controller.py').exists() else 'torque'
        assert 'PYTHONPATH' not in kwargs['env']
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        assert (overlay / 'tools/r13_grasp_estimator.py').is_file()
        assert (overlay / 'tools/__init__.py').is_file()
        assert (overlay / 'registration.py').is_file()
        assert (overlay / 'grasp_audit_base.py').is_file()
        assert mode in command[-1]
        result = {'calibration': {'mode': mode},
                  'evaluation': {'mode': mode, 'rows': [],
                                 'metrics': {'raw': {}, 'registered': {}}}}
        return subprocess.CompletedProcess(command, 0, json.dumps(result), '')

    def test_immutable_assembly_outputs_separate_archives_and_pins_registration(self):
        output = self.output_parent / 'audit'
        registration_pin = hashlib.sha256(
            (self.project / 'experiments/r14_airborne/registration.py').read_bytes()).hexdigest()
        with patch('tools.validate_submission.prepare_overlay', side_effect=self.prepare), \
                patch.object(audit.subprocess, 'run', side_effect=self.child):
            self.invoke(output)
        self.assertEqual({path.name for path in output.iterdir()},
                         {'calibration.json', 'evaluation.json', 'summary.json'})
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['scope']['episodes'], 56)
        self.assertTrue(summary['scope']['same_signal_comparison'])
        self.assertFalse(summary['scope']['research_parity'])
        self.assertEqual(summary['pins']['registration'], registration_pin)
        self.assertIn('not a measurement', summary['scope']['absolute_height_caveat'])

    def test_preexisting_missing_kit_symlink_candidate_and_child_failure_are_safe(self):
        existing = self.output_parent / 'existing'; existing.mkdir()
        with self.assertRaises(SystemExit):
            self.invoke(existing)
        self.assertEqual(list(existing.iterdir()), [])

        argv = ['r14_grasp_audit.py', '--kit', str(self.root / 'missing-kit'),
                '--out', str(self.output_parent / 'missing-kit-output')]
        with patch.object(sys, 'argv', argv), patch.object(audit, 'ROOT', self.project), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            audit.main()
        self.assertFalse((self.output_parent / 'missing-kit-output').exists())

        dangling = self.output_parent / 'dangling'
        dangling.symlink_to(self.root / 'absent')
        with self.assertRaises(SystemExit):
            self.invoke(dangling)
        self.assertTrue(dangling.is_symlink())

        detector = self.project / 'my_detector.py'
        detector.unlink(); detector.symlink_to(self.project / 'my_servo.py')
        output = self.output_parent / 'symlink'
        with self.assertRaisesRegex(ValueError, 'regular public candidate'):
            self.invoke(output)
        self.assertFalse(output.exists())

        detector.unlink(); detector.write_text('# detector\n')
        failure = subprocess.CompletedProcess(['python'], 2, '', 'synthetic child failure')
        output = self.output_parent / 'failed-child'
        with patch('tools.validate_submission.prepare_overlay', side_effect=self.prepare), \
                patch.object(audit.subprocess, 'run', return_value=failure), \
                self.assertRaisesRegex(RuntimeError, 'synthetic child failure'):
            self.invoke(output)
        self.assertFalse(output.exists())

    def test_help_runs_from_clean_unrelated_cwd(self):
        script = Path(audit.__file__).resolve()
        with tempfile.TemporaryDirectory() as unrelated:
            result = subprocess.run([sys.executable, '-B', str(script), '--help'], cwd=unrelated,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--kit', result.stdout)
        self.assertIn('--out', result.stdout)

    def test_isolated_overlay_import_uses_pinned_base_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            overlay = Path(folder)
            (overlay / 'tools').mkdir()
            (overlay / 'tools/__init__.py').write_text('')
            (overlay / 'audit.py').write_bytes(Path(audit.__file__).read_bytes())
            (overlay / 'grasp_audit_base.py').write_text('MARKER = "pinned"\n')
            environment = {'PATH': os.environ['PATH'], 'PYTHONDONTWRITEBYTECODE': '1'}
            result = subprocess.run(
                [sys.executable, '-B', '-c', 'import audit; print(audit.base.MARKER)'],
                cwd=overlay, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'pinned')


if __name__ == '__main__':
    unittest.main()
