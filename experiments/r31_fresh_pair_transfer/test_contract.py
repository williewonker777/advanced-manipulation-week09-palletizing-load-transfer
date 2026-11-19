import unittest

from experiments.r31_fresh_pair_transfer import generate


class FrozenGenerationTest(unittest.TestCase):
    def test_generation_is_deterministic_and_disjoint(self):
        first = generate.build_cases()
        self.assertEqual(first, generate.build_cases())
        self.assertEqual(len(first), 5)
        self.assertEqual(len({row["id"] for row in first}), 5)
        for row in first:
            a, b = row["impacts"]
            self.assertLess(a[1], round((b[0] - a[0]) * 1000))
            self.assertLessEqual(row["long"][1], 230)


if __name__ == "__main__":
    unittest.main()
