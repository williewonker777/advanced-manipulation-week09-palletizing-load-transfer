import unittest

from experiments.r20_slow_handoff.state import SlowHandoff


class SlowStateTest(unittest.TestCase):
    def test_cannot_arm_on_fast_only_repetitions_or_late_slow(self):
        s = SlowHandoff()
        for aux, slow in [(1, 1), (1, 0), (0, 0), (0, 1), (0, 1)]:
            self.assertFalse(s.update(aux, slow, False, 2, True))

    def test_slow_at_fall_hands_off_until_incumbent_takes_over(self):
        s = SlowHandoff()
        trace = [(1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 1, 0)]
        self.assertEqual([s.update(*row, 2, True) for row in trace], [False, True, True, True, False, False])

    def test_existing_slow_clear_counter_and_disarm_do_not_delay_next_onset(self):
        s = SlowHandoff()
        for _ in range(2):
            s.update(True, False, False, 2, True)
            self.assertTrue(s.update(False, True, False, 2, True))
            self.assertTrue(s.update(False, False, False, 2, True))
            self.assertFalse(s.update(False, False, False, 2, True))
        s.update(True, False, False, 2, True)
        self.assertFalse(s.update(False, True, False, 2, False))
        self.assertFalse(s.update(False, True, False, 2, True))


if __name__ == '__main__':
    unittest.main()
