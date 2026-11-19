import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.unloaded_aware import Detector

class TestOwnership(unittest.TestCase):
 def item(self,mass,normal=0.0,support=0.0,valid=False):
  x=object.__new__(Detector);x.box_mass=mass;x.normal_resolution=.5
  x._shared=SimpleNamespace(last_table_allowed=True,last_table_wrench=np.array([0.,0.,normal]))
  x.ctrl=SimpleNamespace(shared_estimator=SimpleNamespace(lambda_support=support,lambda_support_valid=valid))
  x.grip_force=SimpleNamespace(update=lambda *a:SimpleNamespace(force_n=np.array([0.,20.,0.])))
  x.grip_episode=mock.Mock();x.grip_episode.feed.return_value=False
  x.grip_force_value=None;x.grip_force_extra=False;x.load_ownership=False;x.load_branch='unknown';return x
 def update(self,x):
  with mock.patch.object(C1,'update',return_value=(np.zeros(6),False)):
   Detector.update(x,1.4,None,None,None,None,True)
  return x.load_ownership,x.load_branch
 def test_unresolved_requires_normal_below_resolution(self):
  self.assertEqual(self.update(self.item(0.0,normal=.4)),(True,'unresolved'))
  self.assertEqual(self.update(self.item(0.0,normal=.6)),(False,'unresolved'))
 def test_resolved_requires_valid_half_support(self):
  self.assertEqual(self.update(self.item(.1,support=.5,valid=True)),(True,'resolved'))
  self.assertEqual(self.update(self.item(.1,support=.49,valid=True)),(False,'unknown'))

if __name__=='__main__':unittest.main()
