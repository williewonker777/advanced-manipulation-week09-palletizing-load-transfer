import unittest

import numpy as np

from experiments.r25_weak.world_horizontal import (WorldHorizontalWitness, confidence_score,
                                                     unexplained_horizontal)
from experiments.r15_delayed_wrench.observer import LAG, Sample


class Observer:
    def __init__(self, vectors):
        self.vectors = iter(vectors)

    def update(self, t, encoder, jts, grip):
        value = next(self.vectors)
        return None if value is None else Sample(t, t - .019, np.r_[value, 0.0])


class Packet:
    last_table_allowed = False
    last_table_wrench = np.zeros(6)


def witness(vectors, **changes):
    value = object.__new__(WorldHorizontalWitness)
    value.observer = Observer(vectors)
    value.threshold = changes.get("threshold", 3.0)
    value.support_uncertainty_per_n = changes.get("support_uncertainty_per_n", .2)
    value.persistence = changes.get("persistence", 2)
    value.hold_samples = changes.get("hold", 3)
    value.rearm_samples = changes.get("rearm", 4)
    from collections import deque
    value.support = deque([np.zeros(4)] * (LAG + 1), maxlen=LAG + 1)
    value.previous = None
    value.above = value.hold = value.quiet = 0
    value.armed = True
    value.last_vector = value.last_raw = np.zeros(2)
    value.last_score = 0.0
    value.runtime_ns = []
    return value


class TestWorldHorizontalMechanics(unittest.TestCase):
    def test_support_is_time_aligned_and_subtracted(self):
        raw = np.array([8.0, -2.0])
        support = np.array([5.5, -1.0, 12.0])
        result = unexplained_horizontal(raw, support, True, .6)
        residual = raw - support[:2]
        expected_norm = max(np.linalg.norm(residual) - (.6 * 12.0 - np.linalg.norm(support[:2])), 0)
        self.assertAlmostEqual(np.linalg.norm(result), expected_norm)
        self.assertGreater(float(np.dot(result, residual)), 0.0)

    def test_no_admitted_support_cannot_remove_force(self):
        raw = np.array([-4.0, 3.0])
        np.testing.assert_array_equal(unexplained_horizontal(raw, [9, 9, 99], False, .6), raw)

    def test_confidence_penalty_requires_observed_support_normal(self):
        vector = np.array([6.0, 8.0])
        self.assertEqual(confidence_score(vector, [0, 0, 12], False, .2), 10.0)
        self.assertEqual(confidence_score(vector, [0, 0, 12], True, .2), 7.6)

    def test_opposite_sign_vectors_remain_distinct(self):
        positive = np.array([4.0, -3.0])
        negative = -positive
        self.assertLess(float(np.dot(positive, negative)), 0.0)
        self.assertEqual(np.linalg.norm(positive), np.linalg.norm(negative))

    def test_alarm_is_bounded_until_a_real_quiet_rearm(self):
        vectors = [[4.0, 0.0]] * 10 + [[0.0, 0.0]] * 4 + [[-4.0, 0.0]] * 3
        probe = witness(vectors)
        outputs = [probe(i * .001, np.zeros(6), np.zeros(6), True, Packet())
                   for i in range(len(vectors))]
        self.assertEqual(sum(outputs[:10]), 3)
        self.assertEqual(sum(outputs[10:14]), 0)
        self.assertEqual(sum(outputs[14:]), 2)


if __name__ == "__main__":
    unittest.main()
