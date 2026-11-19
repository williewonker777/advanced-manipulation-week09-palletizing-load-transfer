"""Counterexamples for the strict R14 matched impact comparator."""
import tempfile
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import numpy as np

from tools import r14_paired_events as paired


def event(*, outcome='detected', delay=30., false=(), pre=False,
          tool_present=False, tool_detected=False, tool_prealarm=False, tool_delay=None):
    return {'outcome': outcome, 'delay_ms': delay, 'prealarm': pre,
            'false_alarms': len(false), 'false_alarm_times': list(false),
            'tool_present': tool_present, 'tool_detected': tool_detected,
            'tool_prealarm': tool_prealarm, 'tool_delay_ms': tool_delay}


def passing_inputs():
    events = {name: event() for name in paired.ARMS}
    prefixes = {name: {'passed': True} for name in paired.ARMS}
    forces = {name: {'passed': True} for name in paired.ARMS}
    statuses = {name: 'ok' for name in paired.ARMS}
    return events, prefixes, forces, statuses


class PairedComparatorTest(unittest.TestCase):
    def test_public_import_uses_neutral_qualification_without_run_module(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for package in ('tools', 'sim', 'experiments',
                            'experiments/r14_stock_handoff'):
                path = root/package
                path.mkdir(exist_ok=True)
                (path/'__init__.py').write_text('')
            shutil.copy2(Path(paired.__file__), root/'tools/r14_paired_events.py')
            shutil.copy2(Path(paired.__file__).with_name('r5_metrics.py'),
                         root/'tools/r5_metrics.py')
            (root/'sim/config.py').write_text('DT=.001\n')
            (root/'sim/runner.py').write_text('COLLISION_HOLD=.08\nCOLLISION_RAMP=.02\n')
            (root/'experiments/r14_stock_handoff/qualification.py').write_text(
                "KEYS=('q','q_d','tau','e_mm','f_table','f_tool','qpos','r','detect','f_ext')\n")
            result = subprocess.run(
                [sys.executable, '-B', '-c',
                 'import tools.r14_paired_events as p;print(p.qualification.__name__)'],
                cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={'PYTHONDONTWRITEBYTECODE': '1', 'PATH': '/usr/bin:/bin'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(),
                             'experiments.r14_stock_handoff.qualification')
            self.assertFalse((root/'experiments/r14_stock_handoff/run.py').exists())

    def test_exact_public_force_window_and_prefix_through_onset(self):
        t = np.arange(6000)*.001
        collision = [3.905, [8., -5., -15.]]
        force = paired.expected_force(t, collision)
        realized = paired.force_realization(t, force, collision)
        self.assertTrue(realized['passed'])
        # The decimal onset is not exactly representable; the public runner's
        # first nominal onset sample can therefore carry a subnormal ramp.
        self.assertAlmostEqual(realized['active_first_s'], 3.905, places=12)
        self.assertAlmostEqual(realized['active_last_s'], 4.024, places=12)
        bad = force.copy(); bad[3906] += 1e-12
        self.assertFalse(paired.force_realization(t, bad, collision)['passed'])

        normal = {'t': t}; forced = {'t': t.copy()}
        for key in paired.PREFIX_KEYS:
            if key == 't':
                continue
            normal[key] = np.zeros((6000, 6)) if key in {'q','q_d','tau','r'} else np.zeros(6000)
            forced[key] = normal[key].copy()
        forced['q'][3904, 0] = 1e-15
        prefix = paired.equal_prefix(normal, forced, collision[0])
        self.assertFalse(prefix['passed'])
        self.assertFalse(prefix['fields']['q'])

    def test_pass_requires_real_baseline_and_candidate_detections(self):
        inputs = passing_inputs()
        self.assertTrue(paired.assess(*inputs)['passed'])
        for arm, outcome in (('r10', 'missed'), ('registered', 'indistinguishable')):
            with self.subTest(arm=arm):
                inputs = passing_inputs()
                inputs[0][arm] = event(outcome=outcome, delay=None)
                result = paired.assess(*inputs)
                self.assertFalse(result['passed'])

    def test_delay_prealarm_force_prefix_and_status_fail_closed(self):
        for mutate, gate in (
                (lambda x: x[0].__setitem__('registered', event(delay=35.001)), 'delay_r10_5ms'),
                (lambda x: x[0].__setitem__('registered', event(pre=True)), 'no_active_prealarm'),
                (lambda x: x[1].__setitem__('hybrid', {'passed': False}), 'matched_prefix'),
                (lambda x: x[2].__setitem__('r10', {'passed': False}), 'force_realized'),
                (lambda x: x[3].__setitem__('registered', 'diverged'), 'status')):
            with self.subTest(gate=gate):
                inputs = passing_inputs(); mutate(inputs)
                result = paired.assess(*inputs)
                self.assertFalse(result['gates'][gate])
                self.assertFalse(result['passed'])

    def test_false_events_must_match_r10_and_hybrid_one_to_one(self):
        inputs = passing_inputs()
        inputs[0]['registered'] = event(false=(1.0, 1.005))
        inputs[0]['r10'] = event(false=(1.002,))
        inputs[0]['hybrid'] = event(false=(1.0, 1.006))
        result = paired.assess(*inputs)
        self.assertFalse(result['gates']['FA_r10_20ms'])
        self.assertTrue(result['gates']['FA_hybrid_20ms'])

    def test_tool_loss_delay_and_added_prealarm_are_rejected(self):
        mutations = (
            event(),
            event(tool_present=True, tool_detected=True, tool_delay=16.),
            event(tool_present=True, tool_detected=True, tool_prealarm=True, tool_delay=10.),
        )
        for candidate in mutations:
            with self.subTest(candidate=candidate):
                inputs = passing_inputs()
                reference = event(tool_present=True, tool_detected=True, tool_delay=10.)
                inputs[0]['r10'] = reference; inputs[0]['hybrid'] = reference
                inputs[0]['registered'] = candidate
                self.assertFalse(paired.assess(*inputs)['passed'])

    def test_official_event_result_rejects_normal_coincident_rise(self):
        t = np.arange(6000)*.001
        normal = np.zeros(6000, bool); normal[3920:3930] = True
        forced = normal.copy()
        collision = [3.905, [8., -5., -15.]]
        scored = paired.event_result(t, forced, {'collision': collision}, np.zeros(6000), normal)
        self.assertEqual(scored['outcome'], 'indistinguishable')


if __name__ == '__main__':
    unittest.main()
