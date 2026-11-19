import unittest
from tools.r25_event_ledger import failure_reason


class EventCauseTest(unittest.TestCase):
    def test_no_attributable_edge_has_priority_over_quiet(self):
        self.assertEqual(failure_reason(dict(strict=False, attributable_rise_indices=[],
                                            duplicate_rises=0, quiet_before=False)),
                         "no_attributable_rise")

    def test_duplicate_and_quiet_are_not_the_same_failure(self):
        one = dict(strict=False, attributable_rise_indices=[10],
                   duplicate_rises=0, quiet_before=False)
        self.assertEqual(failure_reason(one), "insufficient_preceding_quiet")
        self.assertEqual(failure_reason(dict(one, attributable_rise_indices=[10, 40],
                                             duplicate_rises=1)), "multiple_rises")
        self.assertIsNone(failure_reason(dict(one, strict=True, quiet_before=True)))
