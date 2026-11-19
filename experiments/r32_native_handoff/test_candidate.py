import unittest
import numpy as np
from .candidate import Handoff


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.base = np.array([1., 3.])
        self.on = np.array([[5., 7.], [8., -21.]])
        self.off = np.array([[1., 7.], [3., -21.]])
        self.h = Handoff()
        self.h.start(0, self.base, np.array([3., 5.]))

    def feed(self, fast, slow, associated=False):
        return self.h.feed(self.base, True, True, fast, slow, associated)

    def test_loaded_oblique_and_native_ring_are_distinct(self):
        self.assertEqual(self.feed(self.on,self.on,True),(True,True))
        self.assertEqual(self.feed(self.off,self.on,True),(False,True))
        self.assertEqual(self.feed(self.on,self.on,False),(False,False))
        self.assertEqual(self.feed(self.on,self.on,True),(False,False))

    def test_no_transfer_on_discharge_or_gap(self):
        self.assertEqual(self.feed(self.on,self.on),(True,False))
        self.assertEqual(self.feed(self.off,self.on,True),(False,False))
        self.assertEqual(self.feed(self.on,self.on,True),(False,False))

    def test_different_generation_cannot_take_over(self):
        self.assertEqual(self.feed(self.on,self.on,True),(True,True))
        self.assertEqual(self.h.feed(self.base.copy(),True,True,self.on,self.on,True),(False,False))

    def test_incomplete_or_missing_clears_both(self):
        self.assertEqual(self.feed(self.on,self.on,True),(True,True))
        self.assertEqual(self.h.feed(self.base,True,False,self.on,self.on,True),(False,False))
        self.assertEqual(self.feed(self.on,self.on,True),(False,False))
        self.h.start(0,self.base,np.array([3.,5.]))
        self.assertEqual(self.feed(None,None,True),(False,False))

    def test_no_notice_means_no_extra_signal(self):
        self.h.clear()
        self.assertEqual(self.feed(self.on,self.on,True),(False,False))
