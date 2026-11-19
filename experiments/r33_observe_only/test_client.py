from unittest import TestCase
from unittest.mock import patch
from .client import ObserveOnly, Research, Observation

class AdvisoryTest(TestCase):
    def test_unqualified_advice_never_becomes_alarm(self):
        p=object.__new__(ObserveOnly);p._advisory_live=False
        p.parent_event=False;p.pick_event=True;p.carry_event=False
        with patch.object(Research,'update',return_value=('raw',True)) as fn:
            self.assertEqual(p.update(*(0,)*6),('raw',False))
            self.assertEqual(p.observation,Observation(False,True,False,True))
            self.assertEqual(p.update(*(0,)*6),('raw',False))
            self.assertFalse(p.observation.advisory_started)
            self.assertEqual(fn.call_count,2)
    def test_parent_returns_even_without_advisory(self):
        p=object.__new__(ObserveOnly);p._advisory_live=False
        p.parent_event=True;p.pick_event=p.carry_event=False
        with patch.object(Research,'update',return_value=('raw',True)):
            self.assertEqual(p.update(*(0,)*6),('raw',True))
            self.assertEqual(p.observation,Observation(True,False,False,False))
