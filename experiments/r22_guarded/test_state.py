import unittest

import numpy as np

from experiments.r21_yaw_impulse.state import AVERAGE
from experiments.r22_guarded.candidate import GuardedReleaseBalance


class GuardedReleaseBalanceTest(unittest.TestCase):
    def sample(self, balance, grip, alarm=False, mass_scale=1.0):
        balance.pre_release_alarm = alarm
        th = np.linspace(0.0, 0.05, 6) * self.tick
        tau = np.linspace(0.1, 0.6, 6)
        row = np.linspace(1.0, 2.0, 6) * mass_scale
        self.tick += 1
        return balance.feed(th, tau, row, grip)

    def fill(self, balance, grip=True):
        for _ in range(AVERAGE + 3):
            self.sample(balance, grip)

    def setUp(self):
        self.tick = 0

    def test_quiet_positive_mass_release_resets(self):
        balance = GuardedReleaseBalance(8.0)
        self.fill(balance)
        self.sample(balance, False, alarm=False, mass_scale=0.5)
        self.assertFalse(balance.release_continuity_selected)
        self.assertTrue(balance.use_reset_bank)
        self.assertEqual(balance.integral, 0.0)
        self.assertEqual(len(balance.history), 0)

    def test_active_positive_mass_release_preserves_outgoing_state(self):
        balance = GuardedReleaseBalance(4.0)
        self.fill(balance)
        old_integral = balance.integral
        self.sample(balance, False, alarm=True, mass_scale=0.5)
        self.assertTrue(balance.release_continuity_selected)
        self.assertFalse(balance.use_reset_bank)
        self.assertNotEqual(balance.integral, 0.0)
        self.assertNotEqual(balance.integral, old_integral)

    def test_empty_mass_switch_remains_continuous(self):
        balance = GuardedReleaseBalance(0.0)
        self.fill(balance)
        old_integral = balance.integral
        self.sample(balance, False, alarm=False)
        self.assertIsNone(balance.release_continuity_selected)
        self.assertFalse(balance.use_reset_bank)
        self.assertNotEqual(balance.integral, 0.0)
        self.assertNotEqual(balance.integral, old_integral)

    def test_pickup_resets_bank_and_second_cycle_can_choose_again(self):
        balance = GuardedReleaseBalance(8.0)
        self.fill(balance)
        self.sample(balance, False, alarm=False, mass_scale=0.5)
        self.assertTrue(balance.use_reset_bank)
        self.sample(balance, True, alarm=False)
        self.assertFalse(balance.use_reset_bank)
        self.assertIsNone(balance.release_continuity_selected)
        self.fill(balance)
        self.sample(balance, False, alarm=True, mass_scale=0.5)
        self.assertTrue(balance.release_continuity_selected)


if __name__ == "__main__":
    unittest.main()
