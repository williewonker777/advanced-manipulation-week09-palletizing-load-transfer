"""A smooth trend must not be mistaken for oscillation, or missing force for decay."""
from copy import deepcopy
import json
import unittest

import numpy as np

from tools.r10_ringdown import DESIGN, band_metrics, compare, normal


class VibrationTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(DESIGN.read_text())
        self.t = np.arange(6000) * self.spec['dt']
        self.limits = np.full(6, 100.)
        z = np.zeros((6000, 6))
        self.base = dict(t=self.t, q=z.copy(), q_d=z.copy(), tau=z.copy(), f_ext=np.zeros(6000))
        self.pulse = self.spec['pulses'][1]

    def trace(self, decay):
        forced = deepcopy(self.base)
        start = round(self.pulse['t'] * 1000) + 1
        end = start + 118
        pos = self.t[end:] - self.t[end]
        forced['f_ext'][start:end] = np.linalg.norm(self.pulse['force'])
        forced['q'][end:, 3] = np.deg2rad(.17 * np.exp(-decay*pos)*np.sin(2*np.pi*40*pos))
        return forced

    def test_affine_drift_removed_and_known_band_rms_recovered(self):
        t = self.t[:200]
        a = np.column_stack([2.3+t, 5+t+0.4*np.sin(2*np.pi*40*t), *[0*t for _ in range(4)]])
        r = band_metrics(a, .001, [15,150])
        self.assertLess(r['rms'][0], 1e-12)
        self.assertIsNone(r['peak_hz'][0])
        self.assertAlmostEqual(r['rms'][1], .4/np.sqrt(2), places=3)
        self.assertEqual(r['peak_hz'][1], 40.)
        with self.assertRaises(ValueError):
            band_metrics(np.ones((200, 1)), .001, [15, 150])

    def test_sustained_and_decaying_oscillation_are_distinguished(self):
        for decay, wanted in [(0, 'flagged'), (30., 'within_screen')]:
            row = compare(self.base, self.trace(decay), self.pulse, self.limits, self.spec)
            self.assertEqual(row['screen'], wanted)
            self.assertEqual(row['angle_deg']['failed_joints'], [4] if decay==0 else [])

    def test_unexcited_and_invalid_pairs_do_not_pass(self):
        forced = self.trace(30.)
        forced['q'][:] = 0.
        row = compare(self.base, forced, self.pulse, self.limits, self.spec)
        self.assertEqual(row['screen'], 'inconclusive')
        forced['tau'][0,1] = 1e-15
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            compare(self.base, forced, self.pulse, self.limits, self.spec)
        forced['tau'][0,1] = -0.0
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            compare(self.base, forced, self.pulse, self.limits, self.spec)
        with self.assertRaisesRegex(ValueError, 'pulse'):
            compare(self.base, self.base, self.pulse, self.limits, self.spec)

    def test_common_oscillation_flagged_by_unforced_control(self):
        self.base['tau'][:, 4] = 2*np.sin(2*np.pi*45*self.t)
        self.assertFalse(normal(self.base, self.limits, self.spec)['torque_pct_rated']['quiet_both_windows'])
        self.base['q_d'][-11,4] = .001
        with self.assertRaisesRegex(ValueError, 'stationary'):
            normal(self.base, self.limits, self.spec)
