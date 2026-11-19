"""Finite-history predictors have no knowledge of trial times, force schedules or phase labels."""
from collections import deque
import math


class SupportHistory:
    width, taps = 8, 5

    def __init__(self):
        self.pressure = deque([0.]*(self.width*self.taps), maxlen=self.width*self.taps)
        self.sums = [0.]*self.taps

    def push(self, normal):
        # Accumulate exact non-overlapping, sliding causal bands; oldest crosses band boundaries.
        x = float(normal)
        for i in range(self.taps):
            previous = self.pressure[(i+1)*self.width-1]
            self.sums[i] += x - previous
            x = previous
        self.pressure.appendleft(float(normal))
        return [s/self.width for s in self.sums]


class ConeHistory:
    def __init__(self, width):
        self.window = deque(maxlen=width)
        self.width = width

    def signed_credit(self, projected, measured, cone):
        self.window.append((projected, measured, cone))
        signed, observed, capacity = (sum(row[k] for row in self.window)/self.width for k in range(3))
        if signed*observed <= 0 or not math.isfinite(capacity):
            return 0.
        return math.copysign(min(abs(observed),max(0.,capacity)), signed)
