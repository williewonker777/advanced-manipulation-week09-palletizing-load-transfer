"""A contact notice is an edge. Give the course Boolean a causal, measured level as well."""
from dataclasses import dataclass
import numpy as np

from experiments.r27_load_change.event import WIDTH
from experiments.r31_held_identity.candidate import Detector as EventDetector


@dataclass(frozen=True)
class Decision:
    event_started: bool
    event_output: bool
    measured_continuation: bool
    alarm_level: bool


class Excursion:
    """Once retired, one announcement can never re-open its continuation.

    Compare signed loading in the two force estimates with their own measured
    peak.  The midpoint is the equidistant boundary between the unloaded and
    observed loaded state (the same construction as the inherited load gate).
    Full spatial compatibility, including transverse height, is required for
    continuation. There is no task clock, period of silence or calibrated hold.
    """
    def __init__(self):
        self.token = None
        self.peak = None

    def begin(self, token):
        self.token = token
        self.peak = token.size.copy() if token is not None else None

    def end(self):
        self.begin(None)
        return False

    def feed(self, grip, routes_complete, generation, instant, mean, associated):
        token = self.token
        if token is None:
            return False
        if (not grip or not routes_complete or (generation is not token.generation and not associated)
                or instant is None or mean is None):
            return self.end()
        fast = token.coordinates(instant)
        smooth = token.coordinates(mean)
        if fast is None or smooth is None:
            return self.end()
        _, now, fast_cross, _ = fast
        _, level, cross, _ = smooth
        self.peak = np.maximum(self.peak, level)
        # Either the slow evidence has stopped explaining the original spatial
        # ray, or BOTH current estimates corroborate discharge below mid-load.
        if (not np.all((level > cross) & (level > 0))
                or np.all(2*now <= self.peak)
                or np.all((fast_cross > np.abs(now)) & (fast_cross > self.peak))):
            return self.end()
        return True


def observations(detector):
    windows = detector.r27_gate.windows.windows
    z = detector.z_window
    if (detector.r27_projection.last_force is None or len(z.present) != WIDTH
            or any(len(w.present) != WIDTH for w in windows)):
        return None, None
    instant = np.column_stack(([w.present[-1] for w in windows], z.present[-1]))
    level = np.column_stack(([w.b / WIDTH for w in windows], z.b / WIDTH))
    return instant, level


class Detector(EventDetector):
    """Same sensors/controller; actual returned Boolean is the sustained alarm level.

    `decision.event_started` is separately available even when a distinct
    attributed event arrives before the older alarm level becomes low.
    It is not an extra Boolean rise in the course evaluator.
    """
    def reset(self):
        super().reset()
        self.excursion = Excursion()
        self.previous_event = False
        self.decision = Decision(False, False, False, False)

    def update(self, t, th, thd, tm, ts, grip):
        before = self.planar_notices
        linked = self.associated_native_samples
        residual, event = super().update(t, th, thd, tm, ts, grip)
        if self.planar_notices > before:
            self.excursion.begin(self.source)
        continuing = False
        if self.excursion.token is not None:
            instant, level = observations(self)
            continuing = self.excursion.feed(grip, self.r29_routes_complete, self.r27_gate.reference,
                                             instant, level, self.associated_native_samples > linked)
        out = bool(event or continuing)
        self.decision = Decision(bool(event and not self.previous_event), bool(event), continuing, out)
        self.previous_event = bool(event)
        return residual, out
