"""Signed lateral residual onset after normal vertical-coupling subtraction."""
from __future__ import annotations

from collections import deque

import numpy as np


def lateral_residual(current, previous, coefficients):
    delta = np.asarray(current, float) - np.asarray(previous, float)
    table = np.asarray(coefficients, float)
    if delta.shape != (3,) or table.shape != (2, 2) or not (
            np.isfinite(delta).all() and np.isfinite(table).all()):
        raise ValueError("finite force vectors and 2x2 coupling coefficients required")
    predicted = np.array([1.0, delta[2]]) @ table
    return delta[:2] - predicted


class LoadedLateralOnset:
    """One bounded onset pulse plus non-output event ownership telemetry."""

    def __init__(self, model):
        if model.get("schema") != "r27-loaded-lateral-model.v1":
            raise ValueError("unknown loaded lateral model")
        self.onset_coefficients = np.asarray(model["onset_xy_from_z"], float)
        self.sustain_coefficients = np.asarray(model["sustain_xy_from_z"], float)
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
        if (self.onset_coefficients.shape != (2, 2)
                or self.sustain_coefficients.shape != (2, 2)):
            raise ValueError("two finite 2x2 coupling models required")
        if not (0 < self.onset_lag < self.sustain_lag):
            raise ValueError("ordered positive lags required")
        if not (0 < self.release_threshold < self.sustain_threshold
                and self.onset_threshold > 0):
            raise ValueError("ordered positive thresholds required")
        self.history = deque(maxlen=self.sustain_lag + 1)
        self.reset()

    def reset(self):
        self.history.clear()
        self.previous_tick = None
        self.previous_residual = None
        self.above = 0
        self.pending = 0
        self.pulse = 0
        self.owner = False
        self.baseline = None
        self.direction = None
        self.owner_quiet = 0
        self.last_onset_residual = np.zeros(2)
        self.last_sustain_residual = np.zeros(2)
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
            raise ValueError("finite public three-axis force required")
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

        onset = lateral_residual(
            value, self.history[-1 - self.onset_lag], self.onset_coefficients)
        self.last_onset_residual = onset.copy()
        self.last_onset_score = float(np.linalg.norm(onset))
        coherent = bool(
            self.last_onset_score > self.onset_threshold
            and (self.previous_residual is None
                 or float(onset @ self.previous_residual) > 0.0))
        self.previous_residual = onset.copy()

        if self.owner:
            sustain = lateral_residual(value, self.baseline, self.sustain_coefficients)
            self.last_projection = float(sustain @ self.direction)
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
                    sustain = lateral_residual(
                        value, self.baseline, self.sustain_coefficients)
                    self.last_sustain_residual = sustain.copy()
                    self.last_sustain_score = float(np.linalg.norm(sustain))
                    if (self.last_sustain_score > self.sustain_threshold
                            and float(sustain @ self.direction) > 0.0):
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
                self.direction = onset / self.last_onset_score
                self.pending = self.pending_samples
                self.above = 0

        active = bool(self.pulse > 0 and not incumbent)
        if self.pulse:
            self.pulse -= 1
        return active
