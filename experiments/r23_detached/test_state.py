import unittest

from experiments.r23_detached.state import OnsetGate


class OnsetTests(unittest.TestCase):
    def test_first_high_cannot_be_attributed(self):
        g = OnsetGate()
        self.assertFalse(any(g.feed((8., 2.), True, False, 5.) for _ in range(20)))
        for _ in range(3):
            self.assertFalse(g.feed((0., 0.), False, False, 5.))
        self.assertTrue(g.feed((8., 2.), True, False, 5.))
        self.assertTrue(g.feed((8., 2.), True, True, 5.))
        self.assertFalse(g.feed((1., 1.), False, False, 5.))

    def test_incumbent_or_missing_history_invalidate_arming(self):
        for interrupted in (((0., 0.), False, True, 5.), (None, False, False, 5.)):
            g = OnsetGate()
            for _ in range(3):
                g.feed((2., 0.), False, False, 5.)
            self.assertFalse(g.feed(*interrupted))
            self.assertFalse(g.feed((8., 2.), True, False, 5.))
