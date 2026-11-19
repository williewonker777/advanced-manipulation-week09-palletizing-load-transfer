"""Adjacent-window vector innovation, separate source ownership and actual Boolean output."""
from collections import deque
import math

import numpy as np


WIDTH = 40  # Four native persistence intervals of one normal-model 10-tick EMA time scale.
DIRECTIONS = np.stack((np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)), axis=1)


class Changes:
    def __init__(self):
        self.clear()

    def clear(self):
        self.past = deque()
        self.present = deque()
        self.a = np.zeros(2)
        self.b = np.zeros(2)

    def feed(self, value):
        if value is None:
            if self.present:
                self.clear()
            return None
        v = np.asarray(value, float)
        if v.shape != (2,) or not np.isfinite(v).all():
            self.clear()
            return None
        if len(self.present) == WIDTH:
            old = self.present.popleft()
            self.b -= old
            if len(self.past) == WIDTH:
                self.a -= self.past.popleft()
            self.past.append(old)
            self.a += old
        self.present.append(v.copy())
        self.b += v
        if len(self.past) != WIDTH:
            return None
        return self.a / WIDTH, self.b / WIDTH


def sensor(detector, grip):
    n = detector.normal_model
    packet = detector._shared
    if (not grip or n is None or not n.last_active or packet.last_table_allowed
            or not math.isfinite(packet.load_fraction) or packet.load_fraction < 0.8):
        return None
    return n.last_error


class Event:
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, float)
        if self.bounds.shape != (8,) or not np.isfinite(self.bounds).all() or np.min(self.bounds) <= 0:
            raise ValueError('eight finite positive fitted normal-only bounds needed')
        self.windows = Changes()
        self.clear()

    def clear(self):
        self.owned = False
        self.baseline = self.direction = None
        self.peak = 0.
        self.reported_native = False
        self.native_was_high = False

    def feed(self, value, native):
        means = self.windows.feed(value)
        if means is None:
            self.clear()
            self.native_was_high = bool(native)
            return bool(native)
        before, now = means
        change = now - before
        if not self.owned:
            scores = (DIRECTIONS @ change) / self.bounds
            axis = int(np.argmax(scores))
            # Moving back toward the ordinary model prediction is not proof of a new impact.
            if scores[axis] > 1 and float(np.dot(now, change)) > 0 and np.dot(now, now) > np.dot(before, before):
                self.owned = True
                self.baseline = before.copy()
                self.direction = change / np.linalg.norm(change)
                self.peak = float(np.dot(now - self.baseline, self.direction))
                self.reported_native = bool(native or self.native_was_high)
        else:
            level = float(np.dot(now - self.baseline, self.direction))
            self.peak = max(self.peak, level)
            # Nearer the preceding level than the excursion peak; no fixed alarm or inhibition time.
            if 2 * level <= self.peak:
                self.clear()
            elif native:
                self.reported_native = True
        self.native_was_high = bool(native)
        return bool(native or (self.owned and not self.reported_native))
