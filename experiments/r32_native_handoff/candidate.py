"""Track an announced signed load and an uninterrupted handoff to a native detector."""
from dataclasses import dataclass

import numpy as np

from experiments.r27_load_change.event import DIRECTIONS, WIDTH
from experiments.r31_held_identity.candidate import Detector as EventDetector


@dataclass(frozen=True)
class Decision:
    event_started: bool
    event_output: bool
    measured_load: bool
    linked_native: bool
    alarm_level: bool


class Handoff:
    """Start only on a true, forwarded notice. No clock, late restarts or added edges."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.generation = None
        self.axis = None
        self.peak = None
        self.loading = self.linked = False
        return False, False

    def start(self, axis, reference, peak):
        self.clear()
        if (axis is None or axis not in range(len(DIRECTIONS)) or reference is None
                or peak is None or np.shape(reference) != (2,) or np.shape(peak) != (2,)
                or not np.isfinite(reference).all() or not np.isfinite(peak).all()
                or not np.all(np.asarray(peak) > 0)):
            return
        self.generation = reference          # gate assigns a fresh, immutable baseline for each excursion
        self.axis = DIRECTIONS[axis]
        self.peak = np.array(peak, copy=True)
        self.loading = True

    def feed(self, generation, grip, complete, fast, slow, associated):
        if (self.generation is None or not grip or not complete
                or (generation is not None and generation is not self.generation)
                or fast is None or slow is None):
            return self.clear()
        if self.loading:
            if generation is not self.generation:
                self.loading = False
            else:
                # The two independent force estimates project onto the originally
                # certified signed axis. A vertical component of an oblique force
                # is not, by itself, evidence that the horizontal load disappeared.
                current = np.asarray(fast) @ self.axis - self.generation
                mean = np.asarray(slow) @ self.axis - self.generation
                if not (np.isfinite(current).all() and np.isfinite(mean).all()):
                    return self.clear()
                self.peak = np.maximum(self.peak, mean)
                self.loading = bool(np.all(2 * mean > self.peak) and np.any(2 * current > self.peak))
        # Native response may ring after the force is gone. Mark it separately;
        # do not call it measured pressure. Never bridge across a sampled gap.
        self.linked = bool(associated and (self.linked or self.loading))
        if not (self.loading or self.linked):
            return self.clear()
        return self.loading, self.linked


def observations(gate):
    windows = gate.windows.windows
    if len(windows) != 2 or any(len(w.present) != WIDTH for w in windows):
        return None, None
    return (np.asarray([w.present[-1] for w in windows]),
            np.asarray([w.b / WIDTH for w in windows]))


class Detector(EventDetector):
    """Level-returning research view. Every parent notice remains available separately."""
    def reset(self):
        super().reset()
        self.handoff = Handoff()
        self.previous_event = False
        self.decision = Decision(False, False, False, False, False)

    def update(self, t, th, thd, tm, ts, grip):
        lead, plateau = self.planar_notices, self.r29_optional_starts
        linked = self.associated_native_samples
        residual, event = super().update(t, th, thd, tm, ts, grip)
        start = bool(event and not self.previous_event)
        gate = self.r27_gate
        if start and (self.planar_notices > lead or self.r29_optional_starts > plateau):
            self.handoff.start(gate.axis, gate.reference, gate.peak)
        loading = native = False
        if self.handoff.generation is not None:
            fast, slow = observations(gate)
            loading, native = self.handoff.feed(
                gate.reference, grip, self.r29_routes_complete, fast, slow,
                self.associated_native_samples > linked or self.r29_native_consumed)
        level = bool(event or loading or native)
        self.decision = Decision(start, bool(event), loading, native, level)
        self.previous_event = bool(event)
        return residual, level


class NoticeCompatible(Detector):
    """Drop-in R31 *notice* with read-only diagnostic levels. Not a duration-scoring trick."""
    def update(self, *args):
        residual, _ = super().update(*args)
        return residual, self.decision.event_output
