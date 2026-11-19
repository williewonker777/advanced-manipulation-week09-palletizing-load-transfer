import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.support_aware import Detector


class SupportAwareTest(unittest.TestCase):
    def detector(self, support, valid=True):
        item = object.__new__(Detector)
        item.ctrl = SimpleNamespace(shared_estimator=SimpleNamespace(
            lambda_support=support, lambda_support_valid=valid))
        item.grip_force = SimpleNamespace(update=lambda *args: SimpleNamespace(
            force_n=np.array([0.0, 20.0, 0.0])))
        item.grip_episode = mock.Mock()
        item.grip_episode.feed.return_value = True
        item.grip_force_value = None
        item.grip_force_extra = False
        item.arm_support_owned = False
        return item

    def update(self, item):
        with mock.patch.object(C1, "update", return_value=(np.zeros(6), False)):
            Detector.update(item, 1.4, None, None, None, None, True)
        return item.grip_episode.feed.call_args.args[1]

    def test_half_or_more_valid_support_owns_source(self):
        self.assertTrue(self.update(self.detector(0.5)))
        self.assertTrue(self.update(self.detector(1.0)))

    def test_table_owned_or_uncertain_support_passes_c1_only(self):
        self.assertFalse(self.update(self.detector(0.49)))
        self.assertFalse(self.update(self.detector(1.0, valid=False)))


if __name__ == "__main__":
    unittest.main()
