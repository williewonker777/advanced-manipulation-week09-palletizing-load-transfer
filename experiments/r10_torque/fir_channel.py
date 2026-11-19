"""Causal two-window wrench innovation calibrated only on normal episodes."""
from collections import deque
from pathlib import Path
import time

import numpy as np


class FIRFeature:
    FAST = 8
    GAP = 4
    SLOW = 8
    LENGTH = FAST + GAP + SLOW

    def __init__(self):
        self.values = deque(maxlen=self.LENGTH)
        self.last = np.zeros(6)

    def update(self, wrench):
        self.values.append(np.asarray(wrench, float).copy())
        if len(self.values) < self.LENGTH:
            self.last = np.zeros(6)
        else:
            values = np.asarray(self.values)
            self.last = values[-self.FAST:].mean(axis=0) - values[:self.SLOW].mean(axis=0)
        return self.last.copy()


class FIRChannel:
    """Frozen normal-scaled detector; no clock phase or matched trace input."""
    PERSISTENCE = 2

    def __init__(self, model):
        with np.load(Path(model), allow_pickle=False) as source:
            self.scale = source["scale"]
            self.threshold = float(source["threshold"])
        if self.scale.shape != (6,) or np.any(~np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError("six finite positive normal scales required")
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("finite positive threshold required")
        self.feature = FIRFeature()
        self.statistic = 0.0
        self.count = 0
        self.runtime_ns = []

    def update(self, wrench, base_alarm=False):
        started = time.perf_counter_ns()
        value = self.feature.update(wrench)
        self.statistic = float(np.linalg.norm(value / self.scale))
        over = self.statistic > self.threshold and not bool(base_alarm)
        self.count = self.count + 1 if over else 0
        self.runtime_ns.append(time.perf_counter_ns() - started)
        return self.count >= self.PERSISTENCE
