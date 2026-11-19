import unittest

from experiments.r33_carry_vertical.witness import VerticalCarryWitness


class WitnessTest(unittest.TestCase):
    def feed(self, witness, values, incumbent=False):
        return [witness.extra(a, b, True, incumbent) for a, b in values]

    def test_requires_signed_agreement_and_persistence(self):
        witness = VerticalCarryWitness(1.0, window_samples=2, persistence_samples=2)
        self.assertEqual(self.feed(witness, [(0, 0), (0, 0), (2, -2), (2, -2)]),
                         [False, False, False, False])
        witness.reset()
        output = self.feed(witness, [(0, 0), (0, 0), (2, 2), (4, 4), (4, 4)])
        self.assertEqual(output, [False, False, False, False, True])

    def test_incumbent_owns_same_excursion(self):
        witness = VerticalCarryWitness(1.0, window_samples=2, persistence_samples=1)
        output = self.feed(witness, [(0, 0), (0, 0), (2, 2), (2, 2)], incumbent=True)
        self.assertFalse(any(output))

    def test_trailing_reversal_does_not_rearm(self):
        witness = VerticalCarryWitness(1.0, window_samples=2, persistence_samples=1)
        values = [(0, 0)] * 4 + [(3, 3)] * 5 + [(0, 0)] * 5
        output = self.feed(witness, values)
        rises = [i for i, value in enumerate(output) if value and (i == 0 or not output[i - 1])]
        self.assertEqual(len(rises), 1)

    def test_unknown_fails_closed(self):
        witness = VerticalCarryWitness(1.0, window_samples=2, persistence_samples=1)
        self.assertFalse(witness.extra(None, 1.0, True, False))
        self.assertFalse(witness.extra(1.0, 1.0, False, False))


if __name__ == "__main__":
    unittest.main()
