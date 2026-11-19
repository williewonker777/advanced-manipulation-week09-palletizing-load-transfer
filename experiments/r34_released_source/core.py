"""Batch and streaming logic for the released empty-shadow observer."""
from __future__ import annotations

from collections import deque

import numpy as np


BIN_COUNT = 8
HORIZON = 25
RISE = 3
CLEAR = 8
DT = 0.001


def bin_index(displacement, edges):
    if not np.isfinite(displacement) or displacement < edges[0] or displacement > edges[-1]:
        return None
    return min(max(int(np.searchsorted(edges, displacement, side="right") - 1), 0),
               len(edges) - 2)


class Gate:
    def __init__(self):
        self.over = 0
        self.low = 0
        self.active = False

    def feed(self, valid, force, shadow, force_threshold, shadow_threshold):
        high = bool(valid and force > force_threshold and shadow > shadow_threshold)
        low = bool(not valid or (force < 0.6 * force_threshold
                                 and shadow < 0.6 * shadow_threshold))
        self.over = self.over + 1 if high else 0
        self.low = self.low + 1 if low else 0
        if self.over >= RISE:
            self.active = True
            self.low = 0
        elif self.low >= CLEAR:
            self.active = False
        return self.active


class Stream:
    def __init__(self, model):
        self.model = model
        self.gate = Gate()
        self.current_bin = None
        self.force = deque(maxlen=HORIZON)
        self.shadow = deque(maxlen=HORIZON)
        self.force_impulse = self.shadow_impulse = 0.0

    def feed(self, valid, displacement, force_z, shadow_z):
        item = bin_index(displacement, self.model.EDGES) if valid else None
        valid = bool(item is not None and self.model.SUPPORTED[item]
                     and np.isfinite(force_z) and np.isfinite(shadow_z))
        if not valid or item != self.current_bin:
            self.current_bin = item if valid else None
            self.force.clear()
            self.shadow.clear()
        if valid:
            self.force.append(float(force_z - self.model.FORCE_MEDIAN[item]))
            self.shadow.append(float(shadow_z - self.model.SHADOW_MEDIAN[item]))
        complete = bool(valid and len(self.force) == HORIZON)
        self.force_impulse = DT * sum(self.force) if complete else 0.0
        self.shadow_impulse = DT * sum(self.shadow) if complete else 0.0
        return self.gate.feed(
            complete, self.force_impulse, self.shadow_impulse,
            self.model.FORCE_THRESHOLD[item] if complete else 1.0,
            self.model.SHADOW_THRESHOLD[item] if complete else 1.0)


def batch(trace, model):
    stream = Stream(model)
    output = []
    force = []
    shadow = []
    bins = []
    for eligible, displacement, force_z, shadow_z in zip(
            trace["eligible"], trace["displacement"], trace["force_z"], trace["shadow_z"]):
        output.append(stream.feed(bool(eligible), displacement, force_z, shadow_z))
        force.append(stream.force_impulse)
        shadow.append(stream.shadow_impulse)
        bins.append(-1 if stream.current_bin is None else stream.current_bin)
    return (np.asarray(output, bool), np.asarray(force), np.asarray(shadow),
            np.asarray(bins, int))
