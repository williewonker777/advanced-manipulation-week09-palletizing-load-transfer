"""Finite impulse statistic for world-planar force, with explicit availability and hysteresis."""
from collections import deque
import math


def load_class(mass, grip):
    if not math.isfinite(mass) or not 0 <= mass <= 8:
        return None
    if not grip or mass == 0:
        return 0
    return 1 if mass <= 1 else 2 if mass <= 4 else 3


class Evidence:
    def __init__(self):
        self.history = deque(maxlen=12)
        self.sx = self.sy = 0.
        self.above = self.quiet = 0
        self.active = False
        self.score = None
        self.group = None

    def feed(self, vector, group, bound):
        if vector is None or group is None or group != self.group:
            self.history.clear()
            self.sx = self.sy = 0.
            self.above = self.quiet = 0
            self.active = False
            self.score = None
        self.group = group
        if vector is None or group is None:
            return False
        x, y = float(vector[0]), float(vector[1])
        if not math.isfinite(x) or not math.isfinite(y) or math.isnan(bound) or bound <= 0:
            raise ValueError('expected finite planar signal and positive bound')
        if len(self.history) == self.history.maxlen:
            a, b = self.history[0]
            self.sx -= a
            self.sy -= b
        self.history.append((x, y))
        self.sx += x
        self.sy += y
        if len(self.history) == self.history.maxlen:
            self.score = math.hypot(self.sx, self.sy) / len(self.history)
            self.above = self.above + 1 if self.score > bound else 0
            self.quiet = self.quiet + 1 if self.score < .8 * bound else 0
            if self.above >= 3:
                self.active = True
            elif self.quiet >= 8:
                self.active = False
        return self.active
