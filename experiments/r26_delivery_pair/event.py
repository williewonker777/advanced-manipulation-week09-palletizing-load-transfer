"""Signed detached-force events and attribution of an old fast-channel recoil."""
import numpy as np


class ReleasedEvent:
    """A causal event ends when its signed force is closer to zero than to its peak.

    Only an unreported event adds an edge. A fast counterfactual edge after
    unloading can instead be identified as recoil; slow/learned sources and a
    newly measured force remain independent. No absolute-time or label input.
    """
    def __init__(self, bounds, persistence, release):
        self.b = bounds
        self.persist, self.release = persistence, release
        self.reset()

    def reset(self):
        self.direction = self.recoil = None
        self.bound = self.peak = 0.
        self.samples = self.cool = self.pulse = 0
        self.seen = self.falling = self.suppress_fast = self.was_cf = False
        self.armed = False

    def classify(self, f):
        xy = float(np.hypot(*f[:2]))
        z = float(f[2])
        b = self.b['positive_vertical' if z >= 0. else 'negative_vertical']
        if abs(z)/b >= xy/self.b['horizontal']:
            return abs(z) / b, np.array([0., 0., 1. if z >= 0. else -1.]), b
        return xy/self.b['horizontal'], np.r_[f[:2] / xy, 0.], self.b['horizontal']

    def start(self, u, b, f, incumbent):
        self.direction, self.bound = u, b
        self.peak, self.samples = float(f @ u), 1
        self.seen, self.falling = bool(incumbent), False
        self.recoil, self.suppress_fast = None, False

    def feed(self, f, slow, fast, cf, slow_count, fast_count, learned):
        incumbent = bool(cf or learned)
        if f is None:
            self.reset()
            return incumbent
        ratio,u,b = self.classify(f)
        if self.direction is None:
            if ratio < .8:
                self.armed = True
            if ratio > 1. and self.armed:
                self.start(u,b,f,incumbent)
                self.armed = False
        else:
            projection = float(f @ self.direction)
            if not self.falling:
                self.samples += int(projection > self.bound)
                self.seen |= incumbent
                self.peak = max(self.peak, projection)
                if 2.*projection < self.peak or ratio < .8:
                    self.falling = True
                    if self.samples >= self.persist and not self.seen:
                        self.pulse = self.release
                        self.recoil = self.direction.copy()
                        self.cool = 0
            # Break on a genuine return-to-normal or new signed direction,
            # not on elapsed time since the last report.
            if ratio < .8:
                self.direction = None
                self.armed = True
            elif ratio > 1. and float(u @ self.direction) <= 0.:
                self.start(u,b,f,incumbent)
        # Positive recoil should not be attributed twice to a negative impulse
        # (and conversely). Require evidence from ALL three public signals at
        # the actual onset of the old fast route. Do not blanket-veto the base.
        if self.recoil is not None:
            s, v = float(slow @ self.recoil), float(fast @ self.recoil)
            if s <= self.bound and v >= 0.:
                self.cool += 1
            else:
                self.cool = 0
            if self.cool >= self.release:
                self.recoil = None
            elif (cf and not self.was_cf and not learned and ratio <= 1.
                    and fast_count and not slow_count and s > self.bound and v < 0.):
                self.suppress_fast = True
        if not cf or slow_count >= self.persist:
            self.suppress_fast = False
        self.was_cf = bool(cf)
        extra = self.pulse > 0
        self.pulse = max(0,self.pulse-1)
        return bool(learned or (cf and not self.suppress_fast) or extra)
