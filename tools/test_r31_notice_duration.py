import unittest
from tools.r31_notice_duration import audit, primary


class PrimaryWidth(unittest.TestCase):
    def test_only_first_attributable_alarm_not_an_unrelated_late_rise(self):
        arm={'score':{'strict':True,'valid':{'prefix':True},'delay_ms':7},
             'alarm_intervals':[[500,526],[1007,1011],[1020,1180]]}
        self.assertEqual(primary(arm,1.),dict(start_tick=1007,stop_exclusive=1011,width_ms=4,delay_ms=7))
        arm['score']['delay_ms']=8
        with self.assertRaises(ValueError): primary(arm,1.)
    def test_no_duration_credit_for_missed_or_bad_twin(self):
        self.assertIsNone(primary({'score':{'strict':False,'valid':{'prefix':True}}},1.))
        with self.assertRaises(ValueError): primary({'score':{'strict':False,'valid':{'prefix':False}}},1.)
    def test_no_subset_as_broad_evidence(self):
        with self.assertRaises(ValueError):audit({'schema':'other'})
