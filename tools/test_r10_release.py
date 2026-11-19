"""The live three-file release is tied to real official, replay, and timing evidence."""
import hashlib
import json
from pathlib import Path
import unittest

from tools.validate_submission import CURRENT_RELEASE, check_release
from tools.r10_startup import conservative_summary, measurement_order

ROOT = Path(__file__).resolve().parents[1]
R9 = ROOT / 'results/r9'
R10 = ROOT / 'results/r10'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CurrentReleaseTests(unittest.TestCase):
    def test_live_source_and_all_official_cells(self):
        self.assertEqual(CURRENT_RELEASE, 'r10')
        official = read(R10 / 'official_regression.json')
        prior = read(R9 / 'official_regression.json')
        for mode in ('servo', 'torque'):
            live = official[mode]
            self.assertEqual(live['source_sha256'], sha(ROOT / f'my_{mode}.py'))
            self.assertEqual(live['source_sha256'], sha(ROOT / f'r1_{mode}.py'))
            self.assertEqual(live['detector_sha256'], sha(ROOT / 'my_detector.py'))
            self.assertNotEqual(live['detector_sha256'], prior[mode]['detector_sha256'])
            self.assertTrue(live['self_contained_default_names'])
            self.assertEqual(live['episodes'], 29)
            self.assertEqual(live['kit_integrity']['files'], 20)
            cells = live['control_and_collisions']
            self.assertEqual([len(c) for c in cells['control'].values()], [10] * 3)
            self.assertEqual([len(r['cells']) for r in cells['collisions']], [6] * 6)
            prior[mode]['detector_sha256'] = live['detector_sha256']
        check_release(official, prior)

    def test_full_replays_preserve_every_branch_row(self):
        for mode in ('servo', 'torque'):
            record = read(R10 / f'{mode}_r8_replay.json')
            prior = read(R9 / f'{mode}_r8_replay.json')
            self.assertEqual(record['candidate_sha256'], sha(ROOT / 'my_detector.py'))
            for field in ('mode', 'design_sha256', 'streams', 'cycles', 'rows'):
                self.assertEqual(record[field], prior[field])
            self.assertEqual(record['streams'], 114)
            self.assertEqual(record['cycles'], 684000)
            self.assertTrue(record['identical'] and all(r['identical'] for r in record['rows']))

    def test_startup_gain_recomputed_and_all_outliers_preserved(self):
        plan = ROOT / 'experiments/r10_scratch/PREREGISTERED_PLAN.json'
        bounds = read(plan)['timing']
        for sample in ('initial', 'repeat'):
            record = read(R10 / f'startup_{sample}.json')
            self.assertEqual(record['sources'], {'plan': sha(plan), 'candidate': sha(ROOT/'my_detector.py'),
                                                'baseline': sha(ROOT/'experiments/r9_final/FAST2.py')})
            self.assertTrue(record['eligible'] and all(record['gates'].values()))
            cold = record['cold']
            self.assertEqual(len(cold), bounds['cold_subprocess_pairs'])
            gain = min(r['baseline']['combined_ms'] for r in cold) / max(r['candidate']['combined_ms'] for r in cold)
            self.assertEqual(gain, record['cold_speedup_conservative'])
            self.assertGreaterEqual(gain, bounds['cold_speedup_min'])
            self.assertEqual(record['scratch_fallback']['active_constraint_count'], 32)
            self.assertTrue(record['scratch_fallback']['passed'])
            self.assertEqual(len(record['loops']), 6)
            for row in record['loops']:
                self.assertTrue(row['identical'])
                self.assertEqual(row['baseline']['digests'], row['candidate']['digests'])
                for variant in ('baseline', 'candidate'):
                    s = row[variant]
                    self.assertEqual(bool(s['outliers']), max(s['max_wall_us'], s['max_cpu_us']) >= 1000)
            self.assertEqual(record['candidate_worst_p99_us'], max(r['candidate']['p99_wall_us'] for r in record['loops']))
            self.assertLessEqual(record['candidate_worst_p99_us'], bounds['finite_worst_p99_us_max'])

    def test_new_detector_does_not_change_ringdown_physics(self):
        current, baseline = read(R10 / 'release_ringdown.json'), read(R10 / 'r1_ringdown.json')
        for mode in ('servo', 'torque'):
            self.assertEqual(current['modes'][mode]['detector_sha256'], sha(ROOT/'my_detector.py'))
            self.assertEqual(baseline['modes'][mode]['detector_sha256'], sha(ROOT/'experiments/r9_final/FAST2.py'))
            baseline['modes'][mode]['detector_sha256'] = current['modes'][mode]['detector_sha256']
        self.assertEqual(current, baseline)

    def test_distributed_startup_tool_and_deadline_observations(self):
        timed = read(R10 / 'public_startup.json')
        self.assertEqual(timed['sources']['tool'], sha(ROOT/'tools/r10_startup.py'))
        self.assertEqual(timed['sources']['current'], sha(ROOT/'my_detector.py'))
        self.assertEqual(timed['sources']['FAST2'], sha(ROOT/'experiments/r9_final/FAST2.py'))
        self.assertEqual([r['variant'] for r in timed['rows']], measurement_order(timed['pairs']))
        self.assertEqual(timed['conservative'], conservative_summary(timed['rows']))
        for row in timed['rows']:
            self.assertEqual(row['preimport_numeric_modules'], [])
            self.assertAlmostEqual(row['headline_ms'], row['detector_import_ms'] + row['first_reset_ms'])
        for mode in ('servo', 'torque'):
            probe = read(R10 / f'{mode}_deadline.json')
            self.assertEqual(probe['sources']['detector'], sha(ROOT/'my_detector.py'))
            self.assertEqual(probe['sources']['controller'], sha(ROOT/f'my_{mode}.py'))
            self.assertEqual(probe['sources']['probe'], sha(ROOT/'tools/r9_deadline_probe.py'))
            self.assertEqual(probe['cycles'], 6000 * len(probe['episodes']))
            self.assertTrue(probe['gc_policy_unchanged'])
            self.assertFalse(probe['growing_recorder_control'])
            for i, episode in enumerate(probe['episodes']):
                outliers = [r for r in probe['outliers'] if r['episode'] == i]
                for clock in ('wall', 'cpu'):
                    self.assertEqual(episode[f'{clock}_ge_1ms'], sum(r[f'{clock}_us'] >= 1000 for r in outliers))
