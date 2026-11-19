"""Frozen R9 FAST2 regressions, separated from the live scratch detector."""
import hashlib
from pathlib import Path
import unittest

import numpy as np

import my_detector as live
from experiments.r8_final import C1
from experiments.r9_final import FAST2
from sim import trajectory as T

ROOT = Path(__file__).resolve().parents[1]
FAST2_SHA256 = '2db6351f98a139137f20d835f940adb4e823aef273cc84117d37ee982611062d'


class FastServoTests(unittest.TestCase):
    def test_archive_is_exact_promoted_fast2(self):
        path = ROOT / 'experiments/r9_final/FAST2.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), FAST2_SHA256)

    def test_live_com_derivative_matches_independent_frozen_oracle(self):
        rng = np.random.default_rng(74039)
        for mass, com in ((1.37, (.01, -.02, -.015)), (7.61, (-.03, .012, .017))):
            bank = live.Membership(mass, com, live._r8_model(False))
            oracle = FAST2.Membership(mass, com, FAST2._r8_model(False))
            offset_models = []
            for axis in oracle.axes_local.T:
                pair = []
                for sign in (-1., 1.):
                    model = FAST2.Nominal()
                    model.set_payload(mass, np.asarray(com) + sign*oracle.epsilon*axis)
                    pair.append(model)
                offset_models.append(pair)
            for t in rng.uniform(.01, 5.99, 100):
                args = [v + rng.normal(0., scale, 6)
                        for v, scale in zip(T.reference(t), (.015, .5, 8.))]
                bank.full.inverse_dynamics(*args)
                expected = [-(plus.inverse_dynamics(*args)[0] - minus.inverse_dynamics(*args)[0]) /
                            (2 * oracle.epsilon) for minus, plus in offset_models]
                np.testing.assert_allclose(bank._com_sensitivity(args), expected, rtol=0, atol=1e-7)

    def test_stream_order_checked_even_after_absorbing_alarm(self):
        bank = FAST2.Membership(0., (0., 0., 0.), FAST2._r8_model(True))
        zero = np.zeros(6)
        self.assertFalse(bank.update(0., zero, zero, False, 0., True))
        self.assertFalse(bank.update(.001, zero, zero, False, 0., False))
        for t in (.001, .0021, .003):
            with self.subTest(t=t), self.assertRaises(ValueError):
                bank.update(t, zero, zero, False, 0., False)

    def test_reference_predictions_are_frozen_for_both_grip_values(self):
        mass, com = 5.789, (-.023, .012, -.008)
        new = FAST2.FixedMonitor(FAST2.R5_PAYLOADS['servo'])
        old = C1.FixedMonitor(C1.R5_PAYLOADS['servo'])
        normal = FAST2.NormalResidualModel(**FAST2.MODEL_PARAMETERS['servo'])
        new._reference_grid = FAST2._ReferenceGrid(mass, com, normal, new)
        wrench = np.array([8., -11., 4., 0., 0., 1.])
        # In/out of fitted phases, both arbitrary grip values; off-grid or
        # changed known payload must take the original scalar calculation.
        for t, m in ((0., mass), (1.4, mass), (2.513, mass), (3.92300000001, mass),
                     (3.72, 1.234), (5.7, mass)):
            for grip in (0, 1):
                args = t, m, com, np.zeros(6), np.zeros(6), wrench, True, grip
                self.assertEqual(new.update(*args), old.update(*args))
                np.testing.assert_array_equal(new.last_prediction, old.last_prediction)
                self.assertEqual(new.last_scale, old.last_scale)
                self.assertEqual(new.last_ema, old.last_ema)


if __name__ == '__main__':
    unittest.main()
