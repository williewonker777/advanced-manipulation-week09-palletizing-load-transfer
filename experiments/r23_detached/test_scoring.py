import unittest

from experiments.r23_detached.exposed import compare


def record(rises,quiet=True):
    return {'event':{'quiet_before':quiet,'high_before':not quiet,'exactly_one':len(rises)==1,
                     'delays_ms':list(rises),'raw_rise_indices':[3800+i for i in rises]},
            'none_false_s':[],'force_false_s':[],'stuck_tail':{'none':False,'force':False}}


class PreservationTests(unittest.TestCase):
    def test_added_duplicate_is_not_a_success(self):
        checks=compare(record([20,30]),record([25]),'single')
        self.assertFalse(checks['no_extra_duplicates'])
        self.assertFalse(checks['no_strict_loss_or_delay_over_5ms'])

    def test_real_earlier_detection_not_a_regression(self):
        self.assertTrue(all(compare(record([29]),record([]),'single').values()))

    def test_complete_negative_or_stuck_is_not_free_credit(self):
        ref=record([]);new=record([])
        new['none_false_s']=[4.1];new['stuck_tail']['none']=True
        gates=compare(new,ref,'single')
        self.assertFalse(gates['no_added_false'])
        self.assertFalse(gates['no_added_tail'])
