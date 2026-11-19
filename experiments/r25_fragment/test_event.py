import unittest
import numpy as np

from experiments.r25_fragment.event import SignedEvent
from experiments.r25_fragment.study import contained


class EventTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([8.,0.]);self.f=np.array([7.,0.,0.]);self.event=SignedEvent(predict=True)

    def sample(self,up=False,x=None,f=None,grip=True):
        return self.event.feed(up,self.x if x is None else x,self.f if f is None else f,grip,8,18,8.)

    def prime(self):
        for _ in range(24):self.assertFalse(self.sample(True))

    def test_steady_untriggered_input_cannot_initiate_alarm(self):
        self.assertFalse(any(self.sample() for _ in range(50)))

    def test_signed_support_continues_only_a_confirmed_event(self):
        self.prime()
        self.assertTrue(self.sample())
        self.assertTrue(all(self.sample() for _ in range(22)))
        for i in range(1,9):
            self.sample(x=self.x*(1-i/8),f=self.f*(1-i/8))
        self.assertFalse(self.sample())
        self.assertFalse(self.sample())

    def test_signed_reversal_exits_and_does_not_restart(self):
        self.prime();self.assertTrue(self.sample())
        for _ in range(8):value=self.sample(x=-self.x,f=-self.f)
        self.assertFalse(value)
        self.assertFalse(self.sample())

    def test_two_independent_estimates_must_concur(self):
        self.prime()
        for _ in range(8):self.sample(x=self.x,f=-self.f)
        self.assertFalse(self.sample(x=self.x,f=-self.f))

    def test_grip_switch_cannot_carry_status_across_models(self):
        self.prime();self.assertFalse(self.sample(grip=False))

    def test_missing_measurements_fail_closed(self):
        self.prime()
        self.assertFalse(self.event.feed(False,self.x,None,True,8,18,8.))
        self.assertFalse(self.sample())

    def test_matching_false_times_cannot_be_reused(self):
        self.assertTrue(contained([.1,.2],[.1005]))
        self.assertFalse(contained([.1],[.1,.1001]))
        self.assertFalse(contained([.1],[.11]))
