"""R10 tests must keep historical R1 evidence distinct from later controllers."""
import json
from pathlib import Path
import unittest

from tools.r10_ringdown import digest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results/r10'


class EvidenceTests(unittest.TestCase):
    def test_diagnostics_bind_original_sources_and_failures(self):
        record = json.loads((RESULTS/'r1_ringdown.json').read_text())
        self.assertEqual(record['plan_sha256'], digest(RESULTS/'ringdown_plan.json'))
        self.assertEqual(record['analysis_sha256'], digest(ROOT/'tools/r10_ringdown.py'))
        self.assertEqual(set(record['modes']), {'servo','torque'})
        quiet, pulses, failures, tail_fails = 0, 0, [], []
        for mode, data in record['modes'].items():
            self.assertEqual(data['controller_sha256'], digest(ROOT/f'r1_{mode}.py'))
            self.assertEqual(data['detector_sha256'], '2db6351f98a139137f20d835f940adb4e823aef273cc84117d37ee982611062d')
            self.assertEqual(data['kit_manifest_sha256'], digest(ROOT/'tools/kit_manifest.sha256'))
            self.assertEqual(len(data['rows']), 3)
            for load in data['rows']:
                quiet += all(w['quiet_both_windows'] for w in load['normal'].values())
                for pulse in load['pulses']:
                    self.assertTrue(pulse['excited'])
                    self.assertEqual(len(pulse['tail_abs_deg']), 6)
                    pulses += 1
                    if pulse['torque_pct_rated']['failed_joints']:
                        failures.append((mode, load['payload'], pulse['name'],
                                         pulse['torque_pct_rated']['failed_joints']))
                    if not pulse['returned']:
                        tail_fails.append((mode, pulse['name']))
        self.assertEqual((quiet, pulses), (6, 12))
        self.assertEqual(failures, [('servo', 'empty', 'transport', [4])])
        self.assertEqual(tail_fails.count(('torque','stationary')), 3)
        self.assertEqual(tail_fails.count(('servo','stationary')), 0)
