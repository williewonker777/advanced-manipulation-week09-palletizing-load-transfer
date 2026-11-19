"""Normal-trained 6-sample empty-shadow alignment, with unchanged event gate."""
from __future__ import annotations

from collections import deque

import numpy as np

from experiments.r34_released_source.core import (BIN_COUNT, CLEAR, DT, Gate,
                                                    HORIZON, RISE, bin_index)


SHADOW_DELAY = 6


class Stream:
    def __init__(self, model):
        self.model = model
        self.gate = Gate()
        self.current_bin = None
        self.shadow_delay = deque(maxlen=SHADOW_DELAY + 1)
        self.force = deque(maxlen=HORIZON)
        self.shadow = deque(maxlen=HORIZON)
        self.force_impulse = self.shadow_impulse = 0.0
        self.aligned_shadow = None

    def feed(self, valid, displacement, force_z, shadow_z):
        item = bin_index(displacement, self.model.EDGES) if valid else None
        finite = bool(item is not None and self.model.SUPPORTED[item]
                      and np.isfinite(force_z) and np.isfinite(shadow_z))
        if not finite or item != self.current_bin:
            self.current_bin = item if finite else None
            self.shadow_delay.clear()
            self.force.clear()
            self.shadow.clear()
        if finite:
            self.shadow_delay.append(float(shadow_z))
        aligned = bool(finite and len(self.shadow_delay) == SHADOW_DELAY + 1)
        self.aligned_shadow = self.shadow_delay[0] if aligned else None
        if aligned:
            self.force.append(float(force_z - self.model.FORCE_MEDIAN[item]))
            self.shadow.append(float(self.aligned_shadow - self.model.SHADOW_MEDIAN[item]))
        complete = bool(aligned and len(self.force) == HORIZON)
        self.force_impulse = DT * sum(self.force) if complete else 0.0
        self.shadow_impulse = DT * sum(self.shadow) if complete else 0.0
        return self.gate.feed(
            complete, self.force_impulse, self.shadow_impulse,
            self.model.FORCE_THRESHOLD[item] if complete else 1.0,
            self.model.SHADOW_THRESHOLD[item] if complete else 1.0)


def batch(trace, model):
    stream = Stream(model)
    output, force, shadow, bins, aligned = [], [], [], [], []
    for eligible, displacement, force_z, shadow_z in zip(
            trace["eligible"], trace["displacement"], trace["force_z"], trace["shadow_z"]):
        output.append(stream.feed(bool(eligible), displacement, force_z, shadow_z))
        force.append(stream.force_impulse)
        shadow.append(stream.shadow_impulse)
        bins.append(-1 if stream.current_bin is None else stream.current_bin)
        aligned.append(np.nan if stream.aligned_shadow is None else stream.aligned_shadow)
    return (np.asarray(output, bool), np.asarray(force), np.asarray(shadow),
            np.asarray(bins, int), np.asarray(aligned))
