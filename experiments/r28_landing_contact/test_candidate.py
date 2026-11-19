import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r28_landing_contact.candidate import Detector, OwnedHorizontalEpisode


class OwnedHorizontalEpisodeTest(unittest.TestCase):
    def test_onset_must_be_contiguous_while_owned(self):
        event = OwnedHorizontalEpisode(10.0, 0.8, 2, 3)
        self.assertFalse(event.feed(np.zeros(3), True, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), False, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertFalse(event.feed(np.zeros(3), True, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertTrue(event.feed(np.array([0.0, 11.0, 0.0]), True, False))

    def test_ownership_loss_terminates_pending_pulse_and_requires_rearm(self):
        event = OwnedHorizontalEpisode(10.0, 0.8, 2, 3)
        event.feed(np.zeros(3), True, False)
        event.feed(np.array([0.0, 11.0, 0.0]), True, False)
        self.assertTrue(event.feed(np.array([0.0, 11.0, 0.0]), True, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), False, False))
        self.assertFalse(event.feed(np.array([0.0, 11.0, 0.0]), True, False))


class DetectorOwnershipTest(unittest.TestCase):
    def item(self, mass, table_normal, support=1.0, valid=True):
        item = object.__new__(Detector)
        item.box_mass = mass
        item.normal_resolution = 0.5
        item._shared = SimpleNamespace(
            last_table_allowed=True,
            last_table_wrench=np.array([0.0, 0.0, table_normal]))
        item.ctrl = SimpleNamespace(shared_estimator=SimpleNamespace(
            lambda_support=support, lambda_support_valid=valid))
        item.grip_force = SimpleNamespace(update=lambda *args: SimpleNamespace(
            force_n=np.array([0.0, 20.0, 0.0])))
        item.grip_episode = mock.Mock()
        item.grip_episode.feed.return_value = False
        item.grip_force_value = None
        item.grip_force_extra = False
        item.load_ownership = False
        item.load_branch = "unknown"
        item.table_normal_n = None
        return item

    def update(self, item):
        with mock.patch.object(C1, "update", return_value=(np.zeros(6), False)):
            Detector.update(item, 1.4, None, None, None, None, True)
        return item.load_ownership

    def test_table_contact_veto_applies_to_resolved_and_unresolved_load(self):
        self.assertTrue(self.update(self.item(0.0, 0.5)))
        self.assertFalse(self.update(self.item(0.0, 0.5001)))
        self.assertTrue(self.update(self.item(5.0, 0.5)))
        self.assertFalse(self.update(self.item(5.0, 0.5001)))

    def test_unknown_resolved_support_fails_closed(self):
        self.assertFalse(self.update(self.item(5.0, 0.0, valid=False)))


if __name__ == "__main__":
    unittest.main()
