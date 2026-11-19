import unittest
from unittest.mock import patch
import numpy as np
from experiments.r18_single_pass.recovery import Detector as Recovery
from experiments.r19_impulse_end.detector import Detector
from experiments.r19_impulse_end.state import ImpulseEndState


class ImpulseEndTest(unittest.TestCase):
 def test_minimum_history_and_valid_end(self):
  s=ImpulseEndState();self.assertTrue(s.update(True,np.full(6,.5)))
  for _ in range(6):self.assertTrue(s.update(False,np.full(6,.5)))
  self.assertFalse(s.end_confirmed);self.assertFalse(s.update(False,np.full(6,.5)));self.assertTrue(s.end_confirmed)
 def test_raw_r18_high_blocks_end(self):
  s=ImpulseEndState()
  for _ in range(8):self.assertTrue(s.update(True,np.zeros(6)))
  self.assertFalse(s.end_confirmed)
 def test_strong_and_alternating_unsigned_do_not_end(self):
  for rows in ([np.full(6,1.1)]*8,[np.array([1.2,0,0,0,0,0])]*8):
   s=ImpulseEndState()
   for i,row in enumerate(rows):self.assertTrue(s.update(i==0,row))
   self.assertTrue(s.active);self.assertGreater(s.axis_sums.max(),8)
 def test_reset_and_validation(self):
  s=ImpulseEndState();s.update(True,np.ones(6));self.assertFalse(s.update(False,np.ones(6),armed=False));self.assertEqual(s.samples,0)
  with self.assertRaises(ValueError):s.update(True,np.array([1,2,3]))
 def test_nonloaded_fallback_is_not_wrapped(self):
  detector=Detector.__new__(Detector);impl=object();sentinel=lambda *a: a
  detector._impl=impl;detector.update=sentinel
  with patch.object(Recovery,'reset',return_value=None):detector.reset()
  self.assertIs(detector._impl,impl);self.assertIs(detector.update,sentinel)


if __name__=='__main__':unittest.main()
