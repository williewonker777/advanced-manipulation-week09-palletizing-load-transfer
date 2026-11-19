"""Isolation, calibration and stage-gate tests for the R14 stock wrapper."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import r14_registered_check as check


def result(passed, failed=None, source_pins=None, calibration_sha=None):
    failed = set(failed or ())
    rows = []
    for identifier in ('sh0', 'sh1', 'sh2', 'sh3', 'sh4', 'sh5'):
        gates = {'status': True, 'G4': identifier not in failed, 'timing': True}
        arm = {'false_alarm_times': [], 'false_alarms': 0, 'saturation_pct': 0.,
               'tool_delay_ms': None}
        rows.append({'id': identifier, 'gates': gates, 'passed': all(gates.values()),
                     'arms': {'r10': dict(arm), 'hybrid': dict(arm), 'registered': dict(arm)},
                     'g4': {}, 'ratios': {}})
    return {'schema': 'r14-stock-handoff-normal.v1', 'stage': 'normal',
            'unloaded': {'passed': True, 'byte_parity': {}},
            'rows': rows, 'passed': passed,
            'pins': {'sources': source_pins or {}, 'calibration': calibration_sha}}


class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        self.kit = self.root / 'kit'
        self.output_parent = self.root / 'output'
        for directory in (self.project, self.kit / 'sim', self.kit / 'model', self.output_parent):
            directory.mkdir(parents=True, exist_ok=True)
        for key, relative in check.SOURCE_PATHS.items():
            path = self.project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'# frozen {key}\n')
        (self.project / 'tools/kit_manifest.sha256').write_text('manifest\n')
        (self.kit / 'evaluate.py').write_text('# evaluator\n')
        self.calibration = self.root / 'calibration.json'
        self._write_calibration_and_plan()

    def tearDown(self):
        self.temp.cleanup()

    def _source_pins(self):
        return {key: check.sha((self.project / relative).read_bytes())
                for key, relative in check.SOURCE_PATHS.items()}

    def _calibration_document(self):
        pins = self._source_pins()
        model = {
            'bias': [[[0.] * 6 for _ in range(227)] for _ in range(2)],
            'center': 5.5, 'range': [2.8, 8.0], 'sigma': [.1] * 6,
            'normal_bottom_intercept_slope_mm': [0., 0.],
            'normal_guard_mm': .2, 'loo_signed_mm': [0.] * 12,
            'schema': 'r14-registration-calibration.v1',
        }
        return {
            'schema': 'r14-grasp-calibration-audit.v1',
            'pins': {
                'torque': pins['r1_torque'], 'detector': pins['my_detector'],
                'estimator': pins['estimator'], 'registration': pins['registration'],
                'plan': pins['airborne_plan'], 'audit': pins['r14_audit'],
                'base_audit': pins['r13_audit'], 'kit_manifest': pins['kit_manifest'],
            },
            'scope': {},
            'modes': {'servo': {}, 'torque': {'mode': 'torque', 'runs': [], 'raw': {},
                                              'registered': model}},
        }

    def _write_calibration_and_plan(self, mutate=None):
        document = self._calibration_document()
        if mutate is not None:
            mutate(document)
        blob = (json.dumps(document, separators=(',', ':')) + '\n').encode()
        self.calibration.write_bytes(blob)
        original = json.loads((check.ROOT / 'experiments/r14_stock_handoff/plan.json').read_text())
        original['calibration']['sha256'] = check.sha(blob)
        (self.project / check.SOURCE_PATHS['plan']).write_text(json.dumps(original))
        return document

    def prepare(self, _kit, overlay, _prefix, mode):
        self.assertEqual(mode, 'torque')
        (overlay / 'tools').mkdir()
        (overlay / 'tools/kit_manifest.sha256').write_bytes(
            (self.project / 'tools/kit_manifest.sha256').read_bytes())
        (overlay / 'torque_controller.py').write_text('# transient\n')
        (overlay / 'collision_detector.py').write_text('# transient\n')

    def child(self, normal_passed):
        calls = []

        def run(command, **kwargs):
            overlay = Path(kwargs['cwd'])
            self.assertNotIn('PYTHONPATH', kwargs['env'])
            self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
            for relative in ('r1_torque.py', 'my_detector.py',
                             'experiments/r14_stock_handoff/controller.py',
                             'experiments/r14_stock_handoff/geometry.py',
                             'experiments/r14_stock_handoff/test_controller.py',
                             'experiments/r14_stock_handoff/qualification.py',
                             'experiments/r14_stock_handoff/run.py',
                             'tools/r13_grasp_estimator.py', 'tools/r14_paired_events.py',
                             'tools/r5_metrics.py', 'calibration.json'):
                self.assertTrue((overlay / relative).is_file(), relative)
            stage = ('impact' if 'tools.r14_paired_events' in command
                     else command[command.index('--stage') + 1])
            output = overlay / command[command.index('--out') + 1]
            if stage == 'normal':
                source_paths = (
                    'r1_torque.py', 'my_detector.py', 'experiments/r12_hybrid/controller.py',
                    'experiments/r14_online/stream.py', 'experiments/r14_airborne/registration.py',
                    'models/r12/counterfactual_coefficients.json',
                    'experiments/r14_stock_handoff/controller.py',
                    'experiments/r14_stock_handoff/detector.py',
                    'experiments/r14_stock_handoff/geometry.py',
                    'experiments/r14_stock_handoff/test_controller.py',
                    'experiments/r14_stock_handoff/plan.json',
                    'experiments/r14_stock_handoff/run.py')
                source_pins = {name: check.sha((overlay / name).read_bytes()) for name in source_paths}
                calibration_sha = check.sha((overlay / 'calibration.json').read_bytes())
                document = result(normal_passed, () if normal_passed else ('sh3', 'sh5'),
                                  source_pins, calibration_sha)
            else:
                normal_path = overlay / command[command.index('--normal-result') + 1]
                public_path = overlay / command[command.index('--public-gates') + 1]
                document = {'schema': 'r14-registered-paired-impact.v1', 'stage': 'impact',
                            'rows': [], 'passed': True,
                            'normal_sha256': check.sha(normal_path.read_bytes()),
                            'public_gates_sha256': check.sha(public_path.read_bytes())}
            output.write_text(json.dumps(document))
            calls.append(stage)
            return subprocess.CompletedProcess(command, 0, '{}\n', '')

        return run, calls

    def invoke(self, output):
        argv = ['r14_registered_check.py', '--kit', str(self.kit),
                '--calibration', str(self.calibration), '--out', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(check, 'ROOT', self.project), \
                contextlib.redirect_stderr(io.StringIO()):
            check.main()

    def test_failed_normal_is_archived_and_impact_never_opens(self):
        output = self.output_parent / 'negative'
        child, calls = self.child(False)
        with patch('tools.validate_submission.prepare_overlay', side_effect=self.prepare), \
                patch.object(check.subprocess, 'run', side_effect=child):
            self.invoke(output)
        self.assertEqual(calls, ['normal'])
        self.assertEqual({path.name for path in output.iterdir()},
                         {'normal.json', 'public_gates.json', 'summary.json'})
        summary = json.loads((output / 'summary.json').read_text())
        self.assertFalse(summary['passed'])
        self.assertFalse(summary['authored_normal_passed'])
        self.assertFalse(summary['public_normal_passed'])
        self.assertEqual(summary['impact_stage'], 'not_opened_normal_failed')
        self.assertTrue(summary['impact_comparator_validated'])
        self.assertEqual(summary['failed_normal_gates'],
                         [{'id': 'sh3', 'gates': ['G4']}, {'id': 'sh5', 'gates': ['G4']}])
        self.assertEqual(summary['failed_public_gates'],
                         [{'id': 'sh3', 'gates': ['authored_case']},
                          {'id': 'sh5', 'gates': ['authored_case']}])
        self.assertEqual(summary['source_mapping']['public'],
                         'experiments/r14_stock_handoff/qualification.py')

    def test_passed_normals_open_strict_paired_comparator_in_same_overlay(self):
        output = self.output_parent / 'positive'
        child, calls = self.child(True)
        with patch('tools.validate_submission.prepare_overlay', side_effect=self.prepare), \
                patch.object(check.subprocess, 'run', side_effect=child):
            self.invoke(output)
        self.assertEqual(calls, ['normal', 'impact'])
        self.assertEqual({path.name for path in output.iterdir()},
                         {'normal.json', 'public_gates.json', 'summary.json', 'impact.json'})
        summary = json.loads((output / 'summary.json').read_text())
        self.assertTrue(summary['authored_normal_passed'])
        self.assertTrue(summary['public_normal_passed'])
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['impact_stage'], 'completed_passed')

    def test_calibration_rejects_schema_mode_pin_nonfinite_and_bounds(self):
        mutations = (
            lambda d: d.update(schema='wrong'),
            lambda d: d['modes']['torque'].update(mode='servo'),
            lambda d: d['pins'].update(registration='0' * 64),
            lambda d: d['modes']['torque']['registered'].update(normal_guard_mm=float('nan')),
            lambda d: d['modes']['torque']['registered'].update(normal_guard_mm=2.5001),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self._write_calibration_and_plan(mutation)
                blobs, pins = self._snapshots()
                with self.assertRaises(ValueError):
                    check.validate_calibration(blobs['calibration'], json.loads(blobs['plan']), pins)

    def _snapshots(self):
        with patch.object(check, 'ROOT', self.project):
            return check.source_inputs(self.calibration)

    def test_calibration_digest_is_bound_by_frozen_plan(self):
        blobs, pins = self._snapshots()
        plan = json.loads(blobs['plan'])
        check.validate_calibration(blobs['calibration'], plan, pins)
        changed = blobs['calibration'] + b' '
        with self.assertRaisesRegex(ValueError, 'does not match'):
            check.validate_calibration(changed, plan, pins)

    def test_timing_is_excluded_but_decisions_and_physics_remain(self):
        value = {'ctrl_p99_us': 700., 'cost': {'tcp_rms': 1.},
                 'registration': {'stream_timing_us': {'p99': 10.},
                                  'accepted': True, 'finish_t': 3.55}}
        stable = check.strip_timing(value)
        self.assertNotIn('ctrl_p99_us', stable)
        self.assertNotIn('stream_timing_us', stable['registration'])
        self.assertEqual(stable['cost'], {'tcp_rms': 1.})
        self.assertEqual(stable['registration']['accepted'], True)
        self.assertEqual(stable['registration']['finish_t'], 3.55)

    def test_public_gate_requires_r10_and_hybrid_event_saturation_and_tool_preservation(self):
        blobs, pins = self._snapshots()
        check.validate_calibration(blobs['calibration'], json.loads(blobs['plan']), pins)
        source_pins = {
            'r1_torque.py': pins['r1_torque'], 'my_detector.py': pins['my_detector'],
            'experiments/r12_hybrid/controller.py': pins['hybrid'],
            'experiments/r14_online/stream.py': pins['stream'],
            'experiments/r14_airborne/registration.py': pins['registration'],
            'models/r12/counterfactual_coefficients.json': pins['coefficients'],
            'experiments/r14_stock_handoff/controller.py': pins['controller'],
            'experiments/r14_stock_handoff/detector.py': pins['detector'],
            'experiments/r14_stock_handoff/geometry.py': pins['geometry'],
            'experiments/r14_stock_handoff/test_controller.py': pins['controller_test'],
            'experiments/r14_stock_handoff/plan.json': pins['plan'],
            'experiments/r14_stock_handoff/run.py': pins['qualification'],
        }
        normal = result(True, (), source_pins, pins['calibration'])
        sh5 = normal['rows'][-1]['arms']
        sh5['r10'].update(false_alarm_times=[], saturation_pct=0., tool_delay_ms=11.)
        sh5['hybrid'].update(false_alarm_times=[3.879, 3.906], saturation_pct=.1,
                             tool_delay_ms=11.)
        sh5['registered'].update(false_alarm_times=[3.879, 3.906], saturation_pct=.1,
                                 tool_delay_ms=11.)
        public = check.public_normal_gates(normal, pins)
        gates = public['rows'][-1]['gates']
        self.assertFalse(gates['FA_r10_20ms'])
        self.assertTrue(gates['FA_hybrid_20ms'])
        self.assertFalse(gates['saturation_r10'])
        self.assertTrue(gates['saturation_hybrid'])
        self.assertTrue(gates['tool_r10'])
        self.assertTrue(gates['tool_hybrid'])
        self.assertFalse(public['passed'])

    def test_false_event_matching_is_one_to_one_with_twenty_ms_limit(self):
        self.assertTrue(check.match_false_events([1.01, 2.0], [1.0, 2.019])['passed'])
        self.assertFalse(check.match_false_events([1.0, 1.005], [1.002])['passed'])
        self.assertFalse(check.match_false_events([1.021], [1.0])['passed'])

    def test_bad_paths_symlink_calibration_and_child_failure_leave_output_absent(self):
        existing = self.output_parent / 'existing'; existing.mkdir()
        with self.assertRaises(SystemExit):
            self.invoke(existing)
        original_kit = self.kit
        self.kit = self.root / 'missing-kit'
        with self.assertRaises(SystemExit):
            self.invoke(self.output_parent / 'missing-kit-output')
        self.assertFalse((self.output_parent / 'missing-kit-output').exists())
        self.kit = original_kit
        dangling = self.output_parent / 'dangling'
        dangling.symlink_to(self.root / 'absent')
        with self.assertRaises(SystemExit):
            self.invoke(dangling)
        self.assertTrue(dangling.is_symlink())
        link = self.root / 'calibration-link.json'; link.symlink_to(self.calibration)
        original = self.calibration; self.calibration = link
        with self.assertRaises(SystemExit):
            self.invoke(self.output_parent / 'linked')
        self.calibration = original
        failure = subprocess.CompletedProcess(['python'], 2, '', 'synthetic runner failure')
        output = self.output_parent / 'failed'
        with patch('tools.validate_submission.prepare_overlay', side_effect=self.prepare), \
                patch.object(check.subprocess, 'run', return_value=failure), \
                self.assertRaisesRegex(RuntimeError, 'synthetic runner failure'):
            self.invoke(output)
        self.assertFalse(output.exists())

    def test_help_runs_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as unrelated:
            result_ = subprocess.run([sys.executable, '-B', str(Path(check.__file__).resolve()), '--help'],
                                     cwd=unrelated, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result_.returncode, 0, result_.stderr)
        self.assertIn('--calibration', result_.stdout)


if __name__ == '__main__':
    unittest.main()
