import unittest
from experiments.r32_conservative_release.diagnose import causal_feasibility
class FeasibilityTest(unittest.TestCase):
    def test_r31_stock_first_rise_is_irreversibly_inside_quiet(self):
        result=causal_feasibility(3481,3500,22)
        self.assertEqual(result['quiet_interval'],[3478,3500])
        self.assertEqual(result['minimum_advance_ms'],4)
        self.assertFalse(result['end_or_rearm_after_rise_can_fix'])
    def test_ending_can_only_help_when_rise_precedes_quiet(self):
        self.assertTrue(causal_feasibility(3477,3500,22)['end_or_rearm_after_rise_can_fix'])
if __name__=='__main__':unittest.main()
