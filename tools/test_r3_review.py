import unittest
from tools.r3_review import summarize


class TestR3Review(unittest.TestCase):
    def test_outcomes_and_offsets_not_confused(self):
        row = dict(false_alarms=0, saturation_pct=0, diverged=False,
                   tool_contact_peak_n=0, tool_contact_delay_ms=None)
        result = summarize([dict(row, kind='reference'),
                            dict(row, kind='collision', collision_outcome='missed'),
                            dict(row, kind='offset', false_alarms=1,
                                 tool_contact_peak_n=2)])
        self.assertEqual(result['counts']['missed'], 1)
        self.assertEqual(result['reference_false_alarms'], 0)
        self.assertEqual(result['offset_false_alarms'], 1)
        self.assertEqual(result['tool_contact_misses'], 1)
