import unittest

import numpy as np
from sim import config as C

from experiments.r22_release.state import ReleaseBalance


class ReleaseBalanceTests(unittest.TestCase):
    def samples(self, mass, before, after, initially_gripped):
        state = ReleaseBalance(mass)
        zero = np.zeros(len(C.K))
        axis = zero.copy()
        axis[0] = 1.0
        values = []
        for sample in range(160):
            switched = sample >= 80
            row = (after if switched else before) * axis
            values.append(state.feed(axis * sample * C.DT * 2.25, zero, row,
                                     (not initially_gripped) if switched else initially_gripped))
        return values[80:]

    def test_empty_command_switch_does_not_clear_history(self):
        self.assertTrue(np.allclose(self.samples(0, 2, 2, True), 0, atol=1e-9))

    def test_release_removes_only_outgoing_modeled_momentum(self):
        self.assertTrue(np.allclose(self.samples(7, 3, 1, True), 0, atol=1e-9))

    def test_loaded_pickup_keeps_original_reset(self):
        values = self.samples(7, 1, 3, False)
        self.assertIsNone(values[0])
        self.assertAlmostEqual(values[-1], 0, places=9)
