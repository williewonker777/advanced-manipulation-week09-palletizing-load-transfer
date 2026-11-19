"""Causal onset/output/ownership state for the R27 loaded-force observer."""
from __future__ import annotations

from collections import deque

import numpy as np


class LoadedForceOnset:
    """Emit one bounded onset pulse; use continuous evidence only for ownership."""

    def __init__(self, model):
        if model.get("schema") != "r27-loaded-observer-model.v1":
            raise ValueError("unknown loaded-force observer model")
        self.onset_lag = int(model["onset_lag_samples"])
        self.sustain_lag = int(model["sustain_lag_samples"])
        self.grip_stability = int(model["grip_stability_samples"])
        self.persistence = int(model["onset_persistence_samples"])
        self.pending_samples = int(model["pending_samples"])
        self.pulse_samples = int(model["pulse_samples"])
        self.rearm_samples = int(model["rearm_quiet_samples"])
        self.onset_threshold = float(model["onset_threshold_N"])
        self.sustain_threshold = float(model["sustain_threshold_N"])
        self.release_threshold = float(model["ownership_release_N"])
        values = (self.onset_lag, self.sustain_lag, self.grip_stability,
                  self.persistence, self.pending_samples, self.pulse_samples,
                  self.rearm_samples)
        if any(value <= 0 for value in values):
            raise ValueError("positive integer state parameters required")
        if not (0 < self.onset_lag < self.sustain_lag):
            raise ValueError("ordered onset and sustain lags required")
        if not (np.isfinite([self.onset_threshold, self.sustain_threshold,
                             self.release_threshold]).all()
                and 0 < self.release_threshold < self.sustain_threshold
                and self.onset_threshold > 0):
            raise ValueError("finite ordered thresholds required")
        self.history = deque(maxlen=self.sustain_lag + 1)
        self.reset()

    def reset(self):
        self.history.clear()
        self.previous_tick = None
        self.previous_delta = None
        self.above = 0
        self.pending = 0
        self.pulse = 0
        self.owner = False
        self.baseline = None
        self.direction = None
        self.owner_quiet = 0
        self.last_onset_score = 0.0
        self.last_sustain_score = 0.0
        self.last_projection = 0.0
        self.onset_count = 0

    def _cancel_pending(self):
        self.pending = 0
        self.above = 0
        self.baseline = None
        self.direction = None

    def feed(self, tick, force, grip_run, incumbent):
        value = np.asarray(force, float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("finite three-axis public force required")
        tick = int(tick)
        if (int(grip_run) < self.grip_stability
                or (self.previous_tick is not None and tick != self.previous_tick + 1)):
            self.reset()
        self.previous_tick = tick
        if int(grip_run) < self.grip_stability:
            return False
        self.history.append(value.copy())
        if len(self.history) <= self.sustain_lag:
            return False

        delta = value - self.history[-1 - self.onset_lag]
        self.last_onset_score = float(np.linalg.norm(delta))
        coherent = bool(
            self.last_onset_score > self.onset_threshold
            and (self.previous_delta is None
                 or float(delta @ self.previous_delta) > 0.0))
        self.previous_delta = delta.copy()

        if self.owner:
            self.last_projection = float((value - self.baseline) @ self.direction)
            self.owner_quiet = (self.owner_quiet + 1
                                if self.last_projection < self.release_threshold else 0)
            if self.owner_quiet >= self.rearm_samples:
                self.owner = False
                self.owner_quiet = 0
                self.baseline = None
                self.direction = None
        elif self.pending:
            if incumbent:
                self._cancel_pending()
            else:
                self.pending -= 1
                if self.pending == 0:
                    sustain = value - self.baseline
                    self.last_sustain_score = float(np.linalg.norm(sustain))
                    aligned = float(sustain @ self.direction)
                    if (self.last_sustain_score > self.sustain_threshold
                            and aligned > 0.0):
                        self.direction = sustain / self.last_sustain_score
                        self.owner = True
                        self.pulse = self.pulse_samples
                        self.onset_count += 1
                    else:
                        self._cancel_pending()
        else:
            self.above = self.above + 1 if coherent else 0
            if self.above >= self.persistence and not incumbent:
                self.baseline = self.history[0].copy()
                self.direction = delta / self.last_onset_score
                self.pending = self.pending_samples
                self.above = 0

        # The extra never extends a simultaneously live incumbent report.
        active = bool(self.pulse > 0 and not incumbent)
        if self.pulse:
            self.pulse -= 1
        return active
