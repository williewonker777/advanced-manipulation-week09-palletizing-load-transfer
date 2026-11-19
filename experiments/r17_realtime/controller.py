"""Causal scheduled-node reuse below the frozen balanced C reference stack."""
from collections import OrderedDict

from experiments.r14_stock_handoff.controller import height_weight
from experiments.r16_balanced.controller import Controller as Balanced
from sim import config as C


class ScheduledReferenceMixin:
    CACHE_LIMIT = 6
    GRID_TOLERANCE_S = 1e-12
    IK_NOOP_POSITION_M = 2e-11

    def reset(self):
        self._scheduled_reference = OrderedDict()
        self.scheduled_hits = self.scheduled_misses = 0
        self.scheduled_first_active_s = None
        super().reset()
        # Phase-table construction is reset-time nominal work, not online state.
        self._scheduled_reference.clear()
        self.scheduled_hits = self.scheduled_misses = 0
        self.scheduled_first_active_s = None

    @staticmethod
    def _scheduled_tick(t):
        tick = int(round(float(t)/float(C.DT)))
        scheduled = tick*float(C.DT)
        return tick, scheduled

    def _full_pose(self, base_q, dz):
        # The frozen solver starts from base_q and exits unchanged when its
        # initial position error is below 2e-11 m. Avoid only that proven no-op.
        if abs(float(dz)) < self.IK_NOOP_POSITION_M:
            return base_q
        return super()._full_pose(base_q, dz)

    def _reference_at(self, t):
        t = float(t)
        if getattr(self, "registered_height_m", 0.0) == 0.0 or height_weight(t) == 0.0:
            return super()._reference_at(t)
        tick, scheduled = self._scheduled_tick(t)
        if abs(t-scheduled) > self.GRID_TOLERANCE_S:
            return super()._reference_at(t)
        if self.scheduled_first_active_s is None:
            self.scheduled_first_active_s = t
        if tick in self._scheduled_reference:
            self.scheduled_hits += 1
            value = self._scheduled_reference.pop(tick)
            self._scheduled_reference[tick] = value
            return value
        self.scheduled_misses += 1
        value = super()._reference_at(scheduled)
        self._scheduled_reference[tick] = value
        while len(self._scheduled_reference) > self.CACHE_LIMIT:
            self._scheduled_reference.popitem(last=False)
        return value


class Controller(ScheduledReferenceMixin, Balanced):
    """Frozen balanced C with bounded semantic-grid `_reference_at` reuse."""
