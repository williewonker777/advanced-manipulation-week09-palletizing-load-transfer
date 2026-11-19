import unittest

import numpy as np
from sim import config as C
from experiments.r21_yaw_continuity.state import ReleaseBalance


class ReleaseMomentumTests(unittest.TestCase):
    def run_switch(self, mass, before, after, initial):
        obs = ReleaseBalance(mass)
        zero = np.zeros(len(C.K))
        axis = np.zeros_like(zero)
        axis[0] = 1
        value = []
        for i in range(160):
            change = i >= 80
            row = (after if change else before) * axis
            value.append(obs.feed(axis * (i * C.DT * 2.25), zero, row,
                                  not initial if change else initial))
        return value[80:]

    def test_empty_gripper_command_change_preserves_window(self):
        self.assertTrue(np.allclose(self.run_switch(0, 2, 2, True), 0, atol=1e-9))

    def test_release_subtracts_outgoing_momentum(self):
        self.assertTrue(np.allclose(self.run_switch(7, 3, 1, True), 0, atol=1e-9))

    def test_loaded_pickup_still_resets(self):
        result = self.run_switch(7, 1, 3, False)
        self.assertIsNone(result[0])
        self.assertAlmostEqual(result[-1], 0, places=9)
