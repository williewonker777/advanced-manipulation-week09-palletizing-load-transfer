"""Signed detached-force acquisition and directional source attribution."""
from collections import deque

import numpy as np


class OnsetEvent:
    """Report an acquisition transient while exposing the sustained level separately."""
    def __init__(self, level_bounds, change_bounds, lag, pulse_samples):
        self.level_bounds = level_bounds
        self.change_bounds = change_bounds
        self.lag = int(lag)
        self.pulse_samples = int(pulse_samples)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.lag + 1)
        self.owner = None
        self.owner_bound = 0.0
        self.pulse = 0
        self.was_cf = False
        self.attributed_cf = False
        self.sustained = False
        self.onset = False
        self.signed_level_n = 0.0
        self.signed_change_n = 0.0

    def classify(self, force):
        xy = float(np.hypot(*force[:2]))
        z = float(force[2])
        vertical = "positive_vertical" if z >= 0.0 else "negative_vertical"
        if abs(z) / self.level_bounds[vertical] >= xy / self.level_bounds["horizontal"]:
            direction = np.array([0.0, 0.0, 1.0 if z >= 0.0 else -1.0])
            return vertical, direction, self.level_bounds[vertical]
        direction = np.r_[force[:2] / xy, 0.0]
        return "horizontal", direction, self.level_bounds["horizontal"]

    def feed(self, force, slow, fast, cf, slow_count, fast_count, learned):
        force = np.asarray(force, float)
        self.history.append(force.copy())
        kind, direction, level_bound = self.classify(force)
        level = float(force @ direction)
        ratio = level / level_bound
        self.sustained = ratio > 1.0
        self.signed_level_n = level
        self.onset = False

        if self.owner is not None:
            projection = float(force @ self.owner)
            if projection < 0.8 * self.owner_bound:
                self.owner = None
                self.attributed_cf = False
            elif ratio > 1.0 and float(direction @ self.owner) <= 0.0:
                self.owner = None
                self.attributed_cf = False

        if len(self.history) == self.history.maxlen:
            change = force - self.history[0]
            signed_change = float(change @ direction)
            self.signed_change_n = signed_change
            if (self.sustained and signed_change > self.change_bounds[kind]
                    and (self.owner is None or float(direction @ self.owner) <= 0.0)):
                self.owner = direction.copy()
                self.owner_bound = level_bound
                self.onset = True
                if not (cf or learned):
                    self.pulse = self.pulse_samples

        # Attribute only a rising counterfactual edge with the same measured
        # direction. Learned sources are always independent and never masked.
        if self.owner is not None and cf and not self.was_cf and not learned:
            s = float(np.asarray(slow) @ self.owner)
            v = float(np.asarray(fast) @ self.owner)
            direct = (self.sustained and
                      ((slow_count and s > self.owner_bound)
                       or (fast_count and v > 0.0 and self.signed_level_n > self.owner_bound)))
            recoil = (not self.sustained and fast_count and not slow_count
                      and s > self.owner_bound and v < 0.0)
            self.attributed_cf = bool(direct or recoil)
        if not cf:
            self.attributed_cf = False
        self.was_cf = bool(cf)

        extra = self.pulse > 0
        self.pulse = max(0, self.pulse - 1)
        return extra, bool(self.attributed_cf)
