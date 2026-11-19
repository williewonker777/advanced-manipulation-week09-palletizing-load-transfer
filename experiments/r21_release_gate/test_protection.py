import unittest

from experiments.r21_release_gate.protection import selected_cases


class ProtectionSelectionTest(unittest.TestCase):
    def test_reused_D4_screen_is_fixed_and_stratified(self):
        references, impacts, offsets = selected_cases()
        self.assertEqual([len(references), len(impacts), len(offsets)], [7, 12, 16])
        self.assertEqual({row["force_n"] for row in impacts}, {10., 15., 40.})
        self.assertEqual({row["direction"] for row in impacts}, {"+x", "-x", "+y", "-y"})
        self.assertEqual({row["mass"] for row in references}, {0., 1.3, 2., 3.7, 4., 6.25, 8.})
        self.assertEqual(len({row["case_id"] for row in (*references, *impacts, *offsets)}), 35)


if __name__ == "__main__":
    unittest.main()
