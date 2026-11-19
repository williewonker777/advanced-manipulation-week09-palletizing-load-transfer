import unittest
import numpy as np

from .event import ReleasedEvent
from .study import gate_impact


class ReleasedEventTests(unittest.TestCase):
    def setUp(self):
        self.e=ReleasedEvent({'horizontal':4.,'positive_vertical':4.,'negative_vertical':3.},4,8)

    def feed(self,f,slow=(0,0,0),fast=(0,0,0),cf=False,slow_count=0,fast_count=0,learned=False):
        return self.e.feed(None if f is None else np.asarray(f,float),np.asarray(slow,float),
                           np.asarray(fast,float),cf,slow_count,fast_count,learned)

    def test_unavailable_sensor_cannot_mute_real_alarms(self):
        for cf,learned in ((True,False),(False,True),(False,False)):
            self.assertEqual(self.feed(None,cf=cf,learned=learned),cf or learned)

    def test_signed_release_mark_is_not_an_ongoing_force_flag(self):
        for sign in (1,-1):
            self.e.reset();self.assertFalse(self.feed((0,0,0)))
            self.assertFalse(any(self.feed((sign*9,0,0)) for _ in range(30)))
            hits=[self.feed((sign*3,0,0))]+[self.feed((0,0,0)) for _ in range(16)]
            self.assertEqual(hits,[True]*8+[False]*9)

    def test_incumbent_that_owns_the_force_is_not_reported_twice(self):
        self.feed((0,0,0))
        self.assertFalse(self.feed((0,0,9)))
        self.assertTrue(self.feed((0,0,10),cf=True,fast_count=8))
        for _ in range(10):self.feed((0,0,10))
        self.assertFalse(any(self.feed((0,0,1))for _ in range(10)))

    def test_recoil_does_not_veto_learned_or_independent_slow_source(self):
        self.feed((0,0,0))
        for _ in range(10):self.feed((0,0,-9))
        self.feed((0,0,-4))
        for _ in range(9):self.feed((0,0,0),slow=(0,0,-5),fast=(0,0,5))
        args={'slow':(0,0,-5),'fast':(0,0,5),'cf':True,'fast_count':8}
        self.assertFalse(self.feed((0,0,0),**args))
        self.assertTrue(self.feed((0,0,0),learned=True,**args))
        self.assertTrue(self.feed((0,0,0),slow_count=4,**args))

    def test_occurrence_is_not_claimed_if_late_or_long_vs_fast_controller(self):
        ref={'first_genuine_delay_ms':38.,'active_samples':68,'duplicates':0,'official_false_s':[],
             'prealarm':False,'tail':False,'tool_delay_ms':None,'signed_force_exact':True,
             'force_free_prefix_exact':True,'detect_prefix_exact':True}
        self.assertTrue(all(gate_impact(ref,dict(ref,active_samples=73,first_genuine_delay_ms=43.)).values()))
        self.assertFalse(gate_impact(ref,dict(ref,active_samples=75))['active_within_5_samples'])
        self.assertFalse(gate_impact(ref,dict(ref,first_genuine_delay_ms=44.))['delay_within_5ms'])


if __name__=='__main__':
    unittest.main()
