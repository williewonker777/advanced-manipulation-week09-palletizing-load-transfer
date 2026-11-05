"""Fixed sequential statistics used by the preregistered R4 detectors.

The implementation is deliberately smaller than the cited general methods:
parameters are fixed offline, the statistic is one-sided, and no online
adaptation or probabilistic coverage claim is made.
"""

import numpy as np


R4_PHASE_BOUNDARIES = (0.0, 1.4, 1.7, 3.6, 3.9, 6.0)


class SequentialMonitor:
    """Causal fixed-parameter monitor with four-sample onset hysteresis."""

    def __init__(self, spec, calibration):
        self.spec = dict(spec)
        family = self.spec.get("family")
        if family not in {"W40", "CUSUM_ACTIVE", "CUSUM_PHASE"}:
            raise ValueError("unknown R4 sequential-monitor family")
        expected = 5 if family == "CUSUM_PHASE" else 1
        self.location = np.asarray(calibration.get("location"), dtype=float)
        self.scale = np.asarray(calibration.get("scale"), dtype=float)
        self.threshold = np.asarray(calibration.get("threshold"), dtype=float)
        if any(value.shape != (expected,) for value in
               (self.location, self.scale, self.threshold)):
            raise ValueError("R4 calibration array length differs from its family")
        if (not np.all(np.isfinite(self.location))
                or not np.all(np.isfinite(self.scale))
                or np.any(self.scale <= 0.0)
                or np.any(np.isnan(self.threshold))
                or np.any(self.threshold < 0.0)):
            raise ValueError("invalid R4 calibration value")
        if family == "W40" and int(self.spec.get("window_samples", 0)) != 40:
            raise ValueError("W40 requires the preregistered 40-sample window")
        self._buffer = np.zeros(40, dtype=float)
        self.reset()

    def reset(self):
        self.statistic = 0.0
        self.phase = -1
        self.detected = False
        self._over_count = 0
        self._clear_count = 0
        self._buffer_count = 0
        self._buffer_cursor = 0
        self._buffer_sum = 0.0

    @staticmethod
    def phase_at(t):
        phase = int(np.searchsorted(R4_PHASE_BOUNDARIES, float(t), side="right") - 1)
        return min(max(phase, 0), 4)

    def _hard_reset(self, phase):
        self.statistic = 0.0
        self.phase = int(phase)
        self.detected = False
        self._over_count = 0
        self._clear_count = 0
        self._buffer_count = 0
        self._buffer_cursor = 0
        self._buffer_sum = 0.0

    def update(self, t, score, active):
        family = self.spec["family"]
        phase = self.phase_at(t) if family == "CUSUM_PHASE" else 0
        eligible = True if family == "CUSUM_PHASE" else bool(active)
        value = float(score)
        if not eligible or not np.isfinite(value):
            self._hard_reset(phase)
            return False
        if family == "CUSUM_PHASE" and phase != self.phase:
            self._hard_reset(phase)
        else:
            self.phase = phase

        index = phase if family == "CUSUM_PHASE" else 0
        if family == "W40":
            if self._buffer_count < 40:
                self._buffer[self._buffer_count] = value
                self._buffer_count += 1
                self._buffer_sum += value
            else:
                self._buffer_sum += value - self._buffer[self._buffer_cursor]
                self._buffer[self._buffer_cursor] = value
                self._buffer_cursor = (self._buffer_cursor + 1) % 40
            self.statistic = self._buffer_sum / self._buffer_count
        else:
            increment = ((value - self.location[index]) / self.scale[index]
                         - float(self.spec["drift"]))
            self.statistic = max(0.0, self.statistic + increment)

        over = self.statistic > self.threshold[index]
        self._over_count = self._over_count + 1 if over else 0
        self._clear_count = 0 if over else self._clear_count + 1
        if self._over_count >= 4:
            self.detected = True
        elif self._clear_count >= 8:
            self.detected = False
        return bool(self.detected)
