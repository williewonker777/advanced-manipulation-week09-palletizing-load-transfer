import unittest
from tools.r3_resources import is_potential_simulator


class TestR3Resources(unittest.TestCase):
    def test_light_orchestrators_are_counted_before_large_allocations(self):
        self.assertTrue(is_potential_simulator('python tools/r3_extended.py --stage d4', 1000))
        self.assertTrue(is_potential_simulator('python tools/r3_replay_timing.py', 1000))
        self.assertTrue(is_potential_simulator('python unknown.py', 128 * 1024))
        self.assertFalse(is_potential_simulator('python ordinary_read.py', 1024))
