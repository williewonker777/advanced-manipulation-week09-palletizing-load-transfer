"""Publish one pose-certified notice while the existing two-force excursion rises."""
from __future__ import annotations

import numpy as np

from experiments.r27_load_change.event import DIRECTIONS, WIDTH
from experiments.r27_load_change.manifold_event import Coordinates
from experiments.r27_load_change.settled_event import channels
from experiments.r29_disjoint_events.candidate import Detector as Composed


NATIVE_NOTICE_SAMPLES = 4  # The inherited native decision width, not a new duration threshold.


def rising_means(gate):
    """Return the exact adjacent 40-sample means already owned by ``Settled``."""
    pairs = []
    for window in gate.windows.windows:
        if len(window.past) != WIDTH or len(window.present) != WIDTH:
            return None
        before = window.a / WIDTH
        now = window.b / WIDTH
        if (np.asarray(before).shape != (2,) or np.asarray(now).shape != (2,)
                or not np.isfinite(before).all() or not np.isfinite(now).all()):
            return None
        pairs.append((before, now))
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def rising_certificate(gate, normals, coords, levels, grip, native_active, routes_complete):
    """No threshold: both measured levels must exceed the frozen local normal set
    while both existing signed 40-sample changes are still strictly increasing.
    """
    if (not grip or native_active or not routes_complete or gate.state != "rising"
            or gate.axis is None or gate.native_seen):
        return False
    means = rising_means(gate)
    if means is None or levels is None:
        return False
    before, now = means
    axis = int(gate.axis)
    if axis not in range(len(DIRECTIONS)):
        return False
    signed_change = (now - before) @ DIRECTIONS[axis]
    return bool(np.all(signed_change > 0.0)
                and normals.unexpected(coords, levels, axis))


class Detector(Composed):
    """Keep the linked and detached mechanisms; move only an optional notice.

    ``super().update`` is called exactly once.  The added observer reads the two
    existing means after that call.  It never changes the underlying C3 update.
    """

    def reset(self):
        super().reset()
        self.r29_rising_coords = Coordinates()
        self.r29_rising_announced = False
        self.r29_rising_signal_left = 0
        self.r29_rising_starts = 0
        self.r29_rising_start_times = []
        self.r29_rising_axes = []
        self.r29_plateau_samples_suppressed = 0

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)

        pair = channels(self, grip)
        coords = self.r29_rising_coords.feed(th, pair is not None)
        levels = self._levels(self.r27_gate)
        native_active = bool(self.r29_active_routes or self.r29_native_consumed)

        if self.r27_gate.state == "quiet":
            self.r29_rising_announced = False
            self.r29_rising_signal_left = 0

        if (self.r29_rising_announced and self.r27_gate.state == "settled"
                and self.r27_certificate and report and not native_active):
            # The same excursion was already announced while rising.  Suppress
            # only its later optional plateau pulse; a native route is invariant.
            report = False
            self.r29_plateau_samples_suppressed += 1

        if (not self.r29_rising_announced
                and rising_certificate(self.r27_gate, self.r27_normals, coords,
                                       levels, grip, native_active,
                                       self.r29_routes_complete)
                and self.r29_link.start(self.r27_gate.axis,
                                        self.r27_gate.reference,
                                        self.r27_gate.peak, grip)):
            self.r29_rising_announced = True
            self.r29_rising_signal_left = NATIVE_NOTICE_SAMPLES
            self.r29_rising_starts += 1
            self.r29_rising_start_times.append(float(t))
            self.r29_rising_axes.append(int(self.r27_gate.axis))

        active = self.r29_rising_signal_left > 0
        if active:
            self.r29_rising_signal_left -= 1
        return residual, bool(report or active)
