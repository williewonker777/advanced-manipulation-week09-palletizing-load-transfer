"""Causal signed vertical-innovation ownership, independent of task timing."""
from collections import deque
import math

import numpy as np


class VerticalCarryWitness:
    """Offer one optional notice for a dual-estimator physical excursion."""

    def __init__(self, threshold_N, window_samples=40, persistence_samples=2):
        if not math.isfinite(threshold_N) or threshold_N <= 0:
            raise ValueError("a finite positive normal-only threshold is required")
        self.threshold = float(threshold_N)
        self.window = int(window_samples)
        self.persistence = int(persistence_samples)
        if self.window < 1 or self.persistence < 1:
            raise ValueError("positive window and persistence are required")
        self.values = deque(maxlen=2 * self.window)
        self.reset_source()

    def reset_source(self):
        self.above = 0
        self.owned = False
        self.settling = False
        self.direction = 0.0
        self.peak = 0.0
        self.baseline = None
        self.native_seen = False

    def reset(self):
        self.values.clear()
        self.reset_source()

    def _innovation(self):
        if len(self.values) != self.values.maxlen:
            return None
        values = np.asarray(self.values)
        before = np.mean(values[:self.window], axis=0)
        now = np.mean(values[self.window:], axis=0)
        delta = now - before
        if not np.isfinite(delta).all() or delta[0] * delta[1] <= 0:
            return 0.0, 0.0, before, now
        return float(np.sign(delta[0])), float(np.min(np.abs(delta))), before, now

    def extra(self, projection_z, shared_z, eligible, incumbent):
        """Return an additive Boolean; unknown/ineligible observations fail closed."""
        if (not eligible or projection_z is None or shared_z is None
                or not math.isfinite(projection_z) or not math.isfinite(shared_z)):
            self.reset()
            return False
        self.values.append((float(projection_z), float(shared_z)))
        innovation = self._innovation()
        if innovation is None:
            return False
        direction, score, before, now = innovation
        if self.settling:
            if score < self.threshold / 2:
                self.settling = False
                self.above = 0
            return False
        if not self.owned:
            self.above = self.above + 1 if score > self.threshold else 0
            if self.above < self.persistence:
                return False
            self.owned = True
            self.direction = direction
            self.peak = score
            self.baseline = before.copy()
            self.native_seen = bool(incumbent)
        else:
            if incumbent:
                self.native_seen = True
            displacement = self.direction * (now - self.baseline)
            along = float(np.min(displacement))
            self.peak = max(self.peak, along)
            if 2 * along <= self.peak:
                self.owned = False
                self.settling = True
                return False
        return bool(self.owned and not self.native_seen)
