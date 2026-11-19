"""Historical R9 records bind frozen R1 controllers and the FAST2 archive."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.validate_submission import check_release, prepare_overlay

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/r9'
FAST2 = 'experiments/r9_final/FAST2.py'
FAST2_SHA256 = '2db6351f98a139137f20d835f940adb4e823aef273cc84117d37ee982611062d'
R1_CONTROLLERS = {'servo': 'r1_servo.py', 'torque': 'r1_torque.py'}


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.official = json.loads((ROOT / 'results/r9/official_regression.json').read_text())

    def test_binds_frozen_r9_submission_and_requires_all_cells(self):
        self.assertEqual(sha(FAST2), FAST2_SHA256)
        historical = json.loads((ROOT / 'results/v2_protocol/R8_OFFICIAL_REGRESSION.json').read_text())['records']
        for mode in ('servo', 'torque'):
            row = self.official[mode]
            self.assertEqual(row['source_sha256'], sha(R1_CONTROLLERS[mode]))
            self.assertEqual(row['detector_sha256'], sha(FAST2))
            self.assertEqual(len(row['control_and_collisions']['collisions']), 6)
            self.assertEqual(row['episodes'], 29)
            self.assertEqual(row['control_and_collisions'], historical[mode]['control_and_collisions'])
        modified = copy.deepcopy(self.official)
        modified['torque']['control_and_collisions']['collisions'][0]['cells'][0] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'torque.control_and_collisions'):
            check_release(modified, self.official)

    def test_release_rejects_different_kit_or_submission_names(self):
        check_release(self.official, self.official)
        for field, value in [('kit_integrity', {}), ('self_contained_default_names', False)]:
            modified = copy.deepcopy(self.official)
            modified['servo'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                check_release(modified, self.official)

    def test_cli_protects_kit_and_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            kit, keep = Path(d) / 'licensed', Path(d) / 'previous'
            kit.mkdir()
            keep.mkdir()
            sentinel = keep / 'measured.txt'
            sentinel.write_text('retain this result')
            for out in (keep, kit / 'new-output'):
                r = subprocess.run([sys.executable, str(ROOT / 'tools/validate_submission.py'),
                                    '--kit', str(kit), '--out', str(out)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 2)
            self.assertEqual(sentinel.read_text(), 'retain this result')
            self.assertFalse(any(kit.iterdir()))

    def test_overlay_refuses_link_in_protected_tree_before_running_kit(self):
        with tempfile.TemporaryDirectory() as d:
            kit, overlay = Path(d) / 'kit', Path(d) / 'execution'
            (kit / 'sim').mkdir(parents=True)
            (kit / 'model').mkdir()
            (kit / 'sim/bad').symlink_to(kit / 'missing')
            overlay.mkdir()
            with self.assertRaisesRegex(ValueError, 'symlink'):
                prepare_overlay(kit, overlay, 'my', 'servo')
            self.assertFalse((overlay / 'evaluate.py').exists())

    def test_evidence_is_for_complete_streams_and_all_outliers(self):
        design = json.loads((ROOT / 'results/v2_protocol/R8_HOLDOUT_REVIEW.json').read_text())
        for mode in ('servo', 'torque'):
            replay = json.loads((RESULTS / f'{mode}_r8_replay.json').read_text())
            self.assertEqual(replay['mode'], mode)
            self.assertEqual(replay['cycles'], 114 * 6000)
            self.assertEqual(replay['streams'], len({r['case'] for r in replay['rows']}))
            self.assertEqual(replay['streams'], 114)
            self.assertTrue(replay['identical'] and all(r['identical'] for r in replay['rows']))
            self.assertEqual(replay['design_sha256'], design['design_sha256'])
            self.assertEqual(replay['candidate_sha256'], self.official[mode]['detector_sha256'])
            self.assertEqual(sum(r['extra_cycles'] for r in replay['rows']) > 0, mode == 'servo')
        timing = json.loads((RESULTS / 'servo_timing.json').read_text())
        self.assertEqual(timing['source_sha256'], {
            'C1': sha('experiments/r8_final/C1.py'), 'R9': sha(FAST2)})
        self.assertTrue(timing['identical'] and timing['timing_gate'])
        for r in timing['rows']:
            for v in ('C1', 'R9'):
                self.assertEqual(r[v]['cycles'], 6000)
                self.assertEqual(r[v]['wall_ge_1ms'], sum(w >= 1000. for _, w, _ in r[v]['outliers']))

    def test_deadline_probe_binds_sources_environment_and_every_overrun(self):
        record = json.loads((RESULTS / 'servo_deadline_probe.json').read_text())
        self.assertEqual(record['sources'], {
            'detector': sha(FAST2), 'controller': sha('r1_servo.py'),
            'kit_manifest': sha('tools/kit_manifest.sha256'), 'probe': sha('tools/r9_deadline_probe.py')})
        self.assertTrue(record['gc_policy_unchanged'])
        self.assertFalse(record['growing_recorder_control'])
        self.assertGreaterEqual(record['environment']['affinity_cpu_count'], 1)
        self.assertIn('scheduler', record['environment'])
        self.assertEqual(record['cycles'], 6000 * len(record['episodes']))
        for i, episode in enumerate(record['episodes']):
            outliers = [r for r in record['outliers'] if r['episode'] == i]
            for axis in ('wall', 'cpu'):
                self.assertEqual(episode[f'{axis}_ge_1ms'], sum(r[f'{axis}_us'] >= 1000 for r in outliers))
            for r in outliers:
                self.assertTrue(0 <= r['cycle'] < 6000)
                self.assertEqual(len(r['gc_starts']), 2)

    def test_saturation_development_is_not_a_release_decision(self):
        for tag in ('d4', 'perturbed'):
            result = json.loads((RESULTS / f'servo_headroom_{tag}.json').read_text())
            for name, expected in result['sources'].items():
                self.assertEqual(sha(name), expected)
            self.assertEqual(result['release_detector_sha256'], sha(FAST2))
        release = json.loads((RESULTS / 'servo_headroom_release.json').read_text())
        self.assertFalse(release['adopt_release_pair'])
        self.assertEqual(release['pins']['design'], sha('results/r9/servo_headroom_release_design.json'))
        self.assertEqual(release['pins']['FAST2.py'], sha(FAST2))
        cross = json.loads((RESULTS / 'servo_headroom_attribution.json').read_text())['rows']
        for name in ('R1', 'HEADROOM1'):
            for event in ('reference_r5', 'collision_r5'):
                self.assertEqual(cross[name + '_C1'][event], cross[name + '_FAST2'][event])
        self.assertEqual(cross['R1_FAST2']['collision_r5']['outcome'], 'detected')
        self.assertEqual(cross['HEADROOM1_FAST2']['collision_r5']['outcome'], 'indistinguishable')

    def test_feedback_governor_stopped_on_cost_and_recovery_before_holdout(self):
        result = json.loads((RESULTS / 'servo_feedback.json').read_text())
        self.assertFalse(result['adopt_feedback_governor'] or result['scope']['holdout_opened'])
        self.assertEqual(set(result['failed_gates']), {'normal_cost_ratios', 'catchup'})
        for key, name in [('plan', 'results/r9/servo_feedback_design.json'),
                          ('governor.py', 'experiments/r9_saturation/feedback/governor.py'),
                          ('FAST2.py', FAST2), ('r1_servo.py', 'r1_servo.py')]:
            self.assertEqual(result['pins'][key], sha(name))
