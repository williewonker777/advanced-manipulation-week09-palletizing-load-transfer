"""Wait for a corroborated new force plateau; a native event retains full authority."""
from collections import deque

import numpy as np

from experiments.r27_load_change.event import Changes, DIRECTIONS, WIDTH, sensor


def channels(detector, grip):
    native_error = sensor(detector, grip)
    if native_error is None:
        return None
    projected=detector.r27_projection.from_integrated(detector)
    return None if projected is None else np.array([native_error,projected],dtype=float)


class PairedWindows:
    def __init__(self):
        self.windows = [Changes(), Changes()]

    def feed(self, values):
        means = [w.feed(None if values is None else values[i]) for i,w in enumerate(self.windows)]
        if means[0] is None or means[1] is None:
            return None
        before, now = np.stack(means, axis=1)
        changes = (now-before) @ DIRECTIONS.T
        return before, now, changes


class Settled:
    def __init__(self, model):
        self.bounds = np.asarray(model['joint_bounds_N'],float)
        self.each = np.asarray(model['separate_bounds_N'],float)
        if (self.bounds.shape!=(8,) or self.each.shape!=(2,8) or min(self.bounds.min(),self.each.min())<=0
                or not np.isfinite(self.bounds).all() or not np.isfinite(self.each).all()):
            raise ValueError('positive signed normal-only envelopes for two independent estimates required')
        self.windows = PairedWindows()
        self.native_window = deque(maxlen=WIDTH)
        self.clear()

    def clear(self):
        self.state = 'quiet'
        self.axis = None
        self.reference = self.peak = self.change_peak = None
        self.native_seen = False
        self.signal_left = 0
        self.notified = False

    def feed(self, values, native):
        self.notified = False
        if values is None:
            self.native_window.clear()
        self.native_window.append(bool(native))
        obj = self.windows.feed(values)
        if obj is None:
            self.clear()
            return bool(native)
        before, now, change = obj
        if self.state == 'quiet':
            ratios = np.min(change,axis=0) / self.bounds
            axis = int(np.argmax(ratios))
            if ratios[axis]>1. and np.all(now @ DIRECTIONS[axis] > 0):
                self.axis = axis
                self.state='rising'
                self.reference=before @ DIRECTIONS[axis]
                self.peak=now @ DIRECTIONS[axis]-self.reference
                self.change_peak=change[:,axis].copy()
                self.native_seen=any(self.native_window)
        else:
            self.native_seen |= bool(native)
            value = now @ DIRECTIONS[self.axis] - self.reference
            self.peak = np.maximum(self.peak,value)
            self.change_peak=np.maximum(self.change_peak,change[:,self.axis])
            if np.any(2*value <= self.peak):
                self.clear()
            elif (self.state=='rising' and np.all(change[:,self.axis] <= self.each[:,self.axis])
                  and np.all(2*change[:,self.axis] <= self.change_peak)):
                self.state='settled'
                if not self.native_seen:
                    self.signal_left=4  # the incumbent's native four-sample decision width, not a duration estimate
                    self.notified=True
        if native:
            self.signal_left=0
        active = self.signal_left>0
        if active:self.signal_left-=1
        return bool(native or active)
