import unittest

import numpy as np

from experiments.r17_realtime import contract
from experiments.r17_realtime.controller import Controller, ScheduledReferenceMixin
from sim import config as C


class FakeBase:
    registered_height_m = .001

    def reset(self):
        self.calls = []

    def _reference_at(self, t):
        self.calls.append(float(t)); value = np.array([float(t)])
        return value, value+1., value+2.

    def _full_pose(self, q, dz):
        return q+dz


class Fake(ScheduledReferenceMixin, FakeBase):
    def reset(self):
        FakeBase.reset(self)
        self._scheduled_reference = __import__('collections').OrderedDict()
        self.scheduled_hits = self.scheduled_misses = 0
        self.scheduled_first_active_s = None


class ControllerTest(unittest.TestCase):
    def test_contract(self):
        self.assertIs(contract.loaded_class, Controller)

    def test_semantic_neighbor_reuses_next_runner_tick(self):
        obj = Fake(); obj.reset()
        t = 3.8
        obj._reference_at(t+C.DT)
        obj._reference_at(round((t+C.DT)/C.DT)*C.DT)
        self.assertEqual(obj.scheduled_misses, 1)
        self.assertEqual(obj.scheduled_hits, 1)

    def test_off_grid_and_zero_height_bypass(self):
        obj = Fake(); obj.reset()
        obj._reference_at(3.8002)
        self.assertEqual(obj.scheduled_misses, 0)
        obj.registered_height_m = 0.
        obj._reference_at(3.8)
        self.assertEqual(obj.scheduled_misses, 0)

    def test_noop_pose_threshold(self):
        obj = Fake(); obj.reset(); q = np.array([1.])
        self.assertIs(obj._full_pose(q, 1e-12), q)


if __name__ == "__main__":
    unittest.main()
