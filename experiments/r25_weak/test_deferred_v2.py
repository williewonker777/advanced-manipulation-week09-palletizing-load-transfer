import unittest

from experiments.r25_weak.world_horizontal_v2 import DeferredAdmission


class DeferredAdmissionTest(unittest.TestCase):
    def test_quiet_incumbent_reports_candidate_immediately(self):
        gate = DeferredAdmission()
        self.assertTrue(gate.feed(True, False, False))

    def test_incumbent_consumes_pending_without_suppression(self):
        gate = DeferredAdmission(ceiling=5, hold=3)
        self.assertFalse(gate.feed(True, False, True))
        self.assertFalse(gate.feed(False, False, True))
        self.assertFalse(gate.feed(False, True, True))
        self.assertFalse(gate.pending)
        self.assertEqual(gate.hold, 0)

    def test_unresolved_pending_has_finite_release(self):
        gate = DeferredAdmission(ceiling=3, hold=2)
        outputs = [gate.feed(i == 0, False, True) for i in range(5)]
        self.assertEqual(outputs, [False, False, True, True, False])

    def test_falling_incumbent_evidence_releases_without_dead_time(self):
        gate = DeferredAdmission(ceiling=100, hold=2)
        self.assertFalse(gate.feed(True, False, True))
        self.assertTrue(gate.feed(False, False, False))


if __name__ == "__main__":
    unittest.main()
