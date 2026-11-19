"""Causal identity for one signed yaw event across estimator route dropouts."""
from __future__ import annotations

import math

import numpy as np

from experiments.r25_composed.detector import Detector as R25


class SignedYawOwnership:
    """Classify an inactive sample against the owned event and zero hypotheses.

    Ownership can only start from the frozen detector's yaw alarm.  Its signed
    yaw mean and the independent learned horizontal-residual mean are learned
    while that alarm is present.  A route handoff is admitted only when both
    current measurements are closer to their event means than to zero.  Once
    admitted, signed yaw owns the event until zero becomes nearer.  A sign
    reversal, grip/model switch, non-finite sample, or zero-closer sample
    releases ownership immediately.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.bridge = False
        self.mean = 0.0
        self.support_mean = None
        self.n = 0
        self.grip = None

    def _begin(self, value, support, grip):
        self.active = True
        self.bridge = False
        self.mean = value
        self.support_mean = None if support is None else support.copy()
        self.n = 1
        self.grip = grip

    @staticmethod
    def _support(value):
        if value is None:
            return None
        value = np.asarray(value, dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all():
            return None
        return value

    @staticmethod
    def _nearer_event(value, mean):
        return bool(2.0 * float(np.dot(value, mean)) > float(np.dot(mean, mean)))

    def feed(self, observed, value, support, grip, blocked=False):
        grip = bool(grip)
        support = self._support(support)
        if blocked or value is None or not math.isfinite(float(value)):
            self.reset()
            self.grip = grip
            return False
        value = float(value)
        if self.grip is not None and self.grip != grip:
            self.reset()
        self.grip = grip
        if observed:
            if not self.active or self.mean * value <= 0.0:
                self._begin(value, support, grip)
            else:
                self.n += 1
                self.mean += (value - self.mean) / self.n
                if support is not None:
                    if self.support_mean is None:
                        self.support_mean = support.copy()
                    else:
                        self.support_mean += (support - self.support_mean) / self.n
                self.bridge = False
            return False
        if not self.active:
            return False
        # Equal covariance and prior odds: event is nearer than zero iff
        # (value-mean)^2 < value^2, equivalently 2*value*mean > mean^2.
        yaw_agrees = bool(2.0 * value * self.mean > self.mean * self.mean)
        # A route handoff becomes owned only when the independent learned
        # horizontal residual agrees with the same event-vs-zero hypothesis.
        # Once established, ownership follows signed yaw until zero is nearer;
        # instantaneous support noise cannot create extra output edges.
        support_agrees = bool(
            self.bridge or
            (support is not None and self.support_mean is not None and
             self._nearer_event(support, self.support_mean)))
        self.bridge = bool(yaw_agrees and support_agrees)
        if not self.bridge:
            self.reset()
            self.grip = grip
        return self.bridge


class YawEventOwnershipMixin:
    """Keep a redundantly witnessed signed yaw event continuous; never initiate one."""

    def reset(self):
        super().reset()
        self.yaw_identity = SignedYawOwnership()
        self.yaw_identity_bridge = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        # R25 pickup completion intentionally consumes its yaw route at a real
        # payload-model switch.  Event identity must never undo that correction.
        blocked = bool(getattr(self, "pick_completion_active", False))
        normal = getattr(self, "normal_model", None)
        support = (None if normal is None or not normal.last_active
                   else normal.last_error)
        self.yaw_identity_bridge = self.yaw_identity.feed(
            bool(self.yaw_active), self.yaw_value, support, grip, blocked=blocked)
        return residual, bool(report or self.yaw_identity_bridge)


class Detector(YawEventOwnershipMixin, R25):
    """Frozen R25 plus causal signed-yaw event ownership."""
