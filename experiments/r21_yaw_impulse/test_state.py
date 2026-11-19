import unittest

import numpy as np
from sim import config as C
from experiments.r21_yaw_impulse.state import AVERAGE, SEPARATION, YawBalance


class YawBalanceTests(unittest.TestCase):
    def make(self):
        self.zero = np.zeros(len(C.K))
        self.row = self.zero.copy()
        self.row[0] = 1.
        return YawBalance()

    def test_constant_velocity_in_any_direction_is_not_a_collision(self):
        obs = self.make()
        for k in range(100):
            value = obs.feed(self.row * (k * C.DT * -2), self.zero, self.row, False)
        self.assertAlmostEqual(value, 0., places=9)

    def test_finite_impulse_tracks_signed_constant_external_yaw(self):
        for torque in (-5., 3.):
            obs = self.make()
            for k in range(100):
                q = self.row * (.5 * torque * (k * C.DT) ** 2)
                value = obs.feed(q, self.zero, self.row, False)
                if k < AVERAGE + SEPARATION:
                    self.assertIsNone(value)
            self.assertAlmostEqual(value, torque, places=9)

    def test_grip_change_restarts_full_window(self):
        obs = self.make()
        for k in range(80):
            obs.feed(self.zero, self.zero, self.row, False)
        for k in range(AVERAGE + SEPARATION):
            self.assertIsNone(obs.feed(self.zero, self.zero, self.row, True))
        self.assertEqual(obs.feed(self.zero, self.zero, self.row, True), 0.)
