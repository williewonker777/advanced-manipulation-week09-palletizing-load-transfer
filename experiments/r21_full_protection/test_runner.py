import unittest

import numpy as np

from experiments.r21_full_protection import runner


class TestHelpers(unittest.TestCase):
    def test_intervals_and_edge_hits(self):
        bits = np.zeros(50, dtype=bool)
        bits[10:13] = True
        bits[30:31] = True
        items = runner.intervals(bits)
        self.assertEqual(items, [[10, 13], [30, 31]])
        self.assertTrue(runner.interval_hits(items, 13, 31))
        self.assertFalse(runner.interval_hits(items, 13, 30))

    def test_force_off_by_one(self):
        t = np.arange(6000) / 1000.0
        signed = runner.expected_signed(t, (1.38, (5.0, 0.0, 0.0)))
        nonzero = np.flatnonzero(np.linalg.norm(signed, axis=1) > 0)
        self.assertEqual(int(nonzero[0]), 1381)
        self.assertEqual(int(nonzero[-1]), 1499)
        true = (t >= 1.38) & (t < 1.70)
        self.assertEqual(int(np.flatnonzero(true)[0]), 1380)
        self.assertEqual(int(np.flatnonzero(true)[-1]), 1699)

    def test_twin_window_is_inclusive_twenty_samples(self):
        reference = {"detect_intervals": [[120, 121]]}
        self.assertTrue(runner.has_twin(reference, 100))
        self.assertTrue(runner.has_twin(reference, 140))
        self.assertFalse(runner.has_twin(reference, 99))
        self.assertFalse(runner.has_twin(reference, 141))

    def test_matrix_is_fixed_and_unique(self):
        cases = runner.build_cases("my_torque")
        self.assertEqual(len(cases), 385)
        self.assertEqual(len({case["case_id"] for case in cases}), 385)
        self.assertEqual(sum(case["kind"] == "reference" for case in cases), 189)
        self.assertEqual(sum(case["kind"] == "collision" for case in cases), 180)
        self.assertEqual(sum(case["kind"] == "offset" for case in cases), 16)


if __name__ == "__main__":
    unittest.main()
