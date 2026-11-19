"""Link a pose-certified optional notice to the exact same continuous C3 route."""
from __future__ import annotations

import numpy as np

from experiments.r27_load_change import manifold
from experiments.r27_load_change.manifold_event import DIRECTIONS
from experiments.r27_load_change.settled_event import channels


LINKED_NATIVE_ROUTES = frozenset(("r8_base", "normal"))


class ContinuousForceLink:
    """Measured force identity with fail-open native-route provenance."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.active = False
        self.native_linked = False
        self.grip = False
        self.axis = None
        self.reference = None
        self.peak = None

    def start(self, axis, reference, peak, grip):
        reference = np.asarray(reference, float)
        peak = np.asarray(peak, float)
        if (axis not in range(len(DIRECTIONS)) or reference.shape != (2,) or peak.shape != (2,)
                or not np.isfinite(reference).all() or not np.isfinite(peak).all()
                or np.any(peak <= 0.0) or not grip):
            self.clear()
            return False
        self.active = True
        self.native_linked = False
        self.grip = bool(grip)
        self.axis = int(axis)
        self.reference = reference.copy()
        self.peak = peak.copy()
        return True

    def observe(self, levels, grip):
        if not self.active or bool(grip) != self.grip:
            self.clear()
            return False
        levels = np.asarray(levels, float)
        if levels.shape != (2, 2) or not np.isfinite(levels).all():
            self.clear()
            return False
        value = levels @ DIRECTIONS[self.axis] - self.reference
        if np.any(2.0 * value <= self.peak):
            self.clear()
            return False
        return True

    def consume(self, native, routes, complete):
        if not native:
            return False
        active_routes = frozenset(name for name, value in routes.items() if value)
        if not self.active or not complete:
            self.clear()
            return False
        if not self.native_linked:
            if active_routes != LINKED_NATIVE_ROUTES:
                self.clear()
                return False
            self.native_linked = True
        elif not active_routes or not active_routes <= LINKED_NATIVE_ROUTES:
            self.clear()
            return False
        return True


class Detector(manifold.Detector):
    """R27 pose certificate plus conservative continuous-force/native identity."""

    def reset(self):
        super().reset()
        self.r29_link = ContinuousForceLink()
        self.r29_optional_starts = 0
        self.r29_native_links = 0
        self.r29_consumed_samples = 0
        self.r29_native_consumed = False
        self.r29_routes_complete = True
        self.r29_active_routes = ()

    @staticmethod
    def _levels(gate):
        values = []
        for window in gate.windows.windows:
            if not window.present:
                return None
            values.append(window.b / len(window.present))
        return np.asarray(values, float)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, native = manifold.Base.update(self, t, th, thd, tau_m, tau_s, grip)
        pair = channels(self, grip)
        coords = self.r27_coords.feed(th, pair is not None)
        extra_or_native = self.r27_gate.feed(pair, native)
        levels = self._levels(self.r27_gate)
        if self.r27_gate.notified:
            self.r27_certificate = self.r27_normals.unexpected(
                coords, levels, self.r27_gate.axis)
            if self.r27_certificate and self.r29_link.start(
                    self.r27_gate.axis, self.r27_gate.reference,
                    self.r27_gate.peak, grip):
                self.r29_optional_starts += 1
        elif not extra_or_native or native:
            self.r27_certificate = False

        continuous = self.r29_link.observe(levels, grip)
        routes, complete = self._r25_route_provenance(native)
        self.r29_routes_complete = complete
        self.r29_active_routes = tuple(sorted(name for name, value in routes.items() if value))
        was_linked = self.r29_link.native_linked
        consume = bool(continuous and self.r29_link.consume(native, routes, complete))
        if consume and not was_linked:
            self.r29_native_links += 1
        self.r29_native_consumed = consume
        self.r29_consumed_samples += int(consume)
        optional = bool(self.r27_certificate and extra_or_native)
        return residual, bool(optional or (native and not consume))
