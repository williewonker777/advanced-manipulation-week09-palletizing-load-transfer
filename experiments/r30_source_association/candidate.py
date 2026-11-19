"""Assign the first optional notice by measured yaw observability; never mask a new yaw rise."""
from __future__ import annotations

import math
import numpy as np
from experiments.r27_load_change.manifold_event import Coordinates
from experiments.r29_disjoint_events.candidate import Detector as Composed
from experiments.r29_loaded_rising_certificate.candidate import (
    NATIVE_NOTICE_SAMPLES, rising_certificate, rising_means,
)


class MeasuredCoordinates(Coordinates):
    """Expose the exact coordinates already evaluated, without doing another inverse solve."""
    def __init__(self):
        super().__init__()
        self.last = None

    def feed(self, encoder, eligible):
        self.last = super().feed(encoder, eligible)
        return self.last


def prefer_planar_notice(changes, yaw_column):
    """For both planar innovations require more energy orthogonal to yaw than along it.

    J_xy[:,0] is the translational Jacobian column at the sensor witness pose.
    Its dot product with force is the torque about joint 1. The scale of this
    column cancels in the comparison. Ties, opposing normals and unknowns use
    the existing detector/plateau, not an early notice.
    """
    if changes is None or yaw_column is None:return False
    delta = np.asarray(changes, dtype=float)
    col = np.asarray(yaw_column, dtype=float)
    if delta.shape != (2,2) or col.shape != (2,) or not np.isfinite(delta).all() or not np.isfinite(col).all():
        return False
    cx, cy = map(float, col)
    if not math.hypot(cx, cy) > 0:return False
    along = delta[:,0]*cx + delta[:,1]*cy
    across = -delta[:,0]*cy + delta[:,1]*cx
    return bool(np.isfinite(along).all() and np.isfinite(across).all()
                and np.all(np.abs(across)>np.abs(along)) and across[0]*across[1]>0)


class Detector(Composed):
    def reset(self):
        super().reset()
        self.r27_coords = MeasuredCoordinates()
        self.r30_owner = None
        self.r30_notice_left = 0
        self.r30_optional_starts = 0
        self.r30_deferred = 0

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, reported = super().update(t, th, thd, tau_m, tau_s, grip)
        gate = self.r27_gate
        if gate.state == 'quiet':
            self.r30_owner = None
            self.r30_notice_left = 0
        native = bool(self.r29_active_routes or self.r29_native_consumed)
        if (self.r30_owner == 'planar' and gate.state == 'settled'
                and self.r27_certificate and reported and not native):
            # Only our redundant optional plateau is consumed. Native yaw is never suppressed here.
            reported = False

        if self.r30_owner is None and gate.state == 'rising' and not native:
            levels = self._levels(gate)
            if rising_certificate(gate, self.r27_normals, self.r27_coords.last, levels, grip,
                                  native, self.r29_routes_complete):
                before, now = rising_means(gate)
                if (prefer_planar_notice(now-before, self.r27_projection.jp[:2,0])
                        and self.r29_link.start(gate.axis, gate.reference, gate.peak, grip)):
                    self.r30_owner = 'planar'
                    self.r30_notice_left = NATIVE_NOTICE_SAMPLES
                    self.r30_optional_starts += 1
                else:
                    self.r30_owner = 'incumbent'
                    self.r30_deferred += 1
        optional = self.r30_notice_left > 0
        if optional:self.r30_notice_left -= 1
        return residual, bool(reported or optional)
