"""Document the frozen f764 numerical-contract discrepancy without changing it."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r28_landing_contact.candidate import Detector


class FrozenFiniteContractTest(unittest.TestCase):
    def item(self, allowed):
        item=object.__new__(Detector)
        item.box_mass=0.0;item.normal_resolution=0.5
        item._shared=SimpleNamespace(last_table_allowed=allowed,
            last_table_wrench=np.array([0.0,0.0,float('nan')]))
        item.ctrl=SimpleNamespace(shared_estimator=SimpleNamespace(
            lambda_support=float('nan'),lambda_support_valid=False))
        item.grip_force=SimpleNamespace(update=lambda *args:SimpleNamespace(
            force_n=np.array([0.0,20.0,0.0])))
        item.grip_episode=mock.Mock();item.grip_episode.feed.return_value=False
        item.grip_force_value=None;item.grip_force_extra=False
        item.load_ownership=False;item.load_branch='unknown';item.table_normal_n=None
        return item

    def update(self,item):
        with mock.patch.object(C1,'update',return_value=(np.zeros(6),False)):
            Detector.update(item,1.4,None,None,None,None,True)
        return item

    def test_false_geometry_is_admissible_air_not_a_missing_sample(self):
        item=self.update(self.item(False))
        self.assertEqual(item.table_normal_n,0.0)
        self.assertTrue(item.load_ownership)

    def test_frozen_true_geometry_nan_is_incorrectly_converted_to_zero(self):
        item=self.update(self.item(True))
        # This assertion documents the frozen defect. The preregistered contract
        # instead requires this missing/nonfinite measurement to fail closed.
        self.assertEqual(item.table_normal_n,0.0)
        self.assertTrue(item.load_ownership)


if __name__=='__main__':unittest.main()
