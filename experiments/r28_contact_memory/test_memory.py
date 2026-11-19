from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from experiments.r28_contact_memory.candidate import AirContact, Detector


class MemoryTests(unittest.TestCase):
    def make(self):
        d=object.__new__(Detector)
        d.grip_episode=NS(bound=15.0,release=11.25,persistence=2,report_samples=8)
        with patch.object(AirContact,'reset',lambda self:None):
            d.reset()
        ep=d.grip_episode
        self.assertEqual((ep.bound,ep.release,ep.persistence,ep.report_samples),(15.,11.25,2,8))
        return ep

    def test_temporarily_ineligible_does_not_erase_low_observation(self):
        ep=self.make()
        self.assertFalse(ep.feed([10,0,0],True,False))
        self.assertFalse(ep.feed([12,0,0],False,False))
        self.assertFalse(ep.feed([16,0,0],True,False))
        self.assertTrue(ep.feed([16,0,0],True,False))

    def test_ineligible_does_break_onset_and_cancel_delivery(self):
        ep=self.make()
        ep.feed([6,0,0],True,False)
        self.assertFalse(ep.feed([16,0,0],True,False))
        self.assertFalse(ep.feed([16,0,0],False,False))
        self.assertFalse(ep.feed([16,0,0],True,False))
        self.assertTrue(ep.feed([16,0,0],True,False))
        self.assertFalse(ep.feed([16,0,0],False,False))
        self.assertFalse(ep.feed([16,0,0],True,False))
