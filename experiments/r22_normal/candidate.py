"""R21 with one-sample early-wrench spikes rejected before thresholding.

The stock course collision is a 20 ms ramp, so a horizontal collision signal is
temporally coherent.  The inherited R20 branch instead thresholded a single
unfiltered sample.  This branch keeps its frozen threshold and state machine,
but applies a causal three-sample component-wise median before the absolute
horizontal maximum.  R10 and the R21 yaw-balance branch are unchanged.
"""
from collections import deque

import numpy as np

import my_detector
from experiments.r21_yaw_impulse.coarse import Detector as R21
from experiments.r21_yaw_impulse.frozen import Detector as Global


class Detector(R21):
    """Composable replacement for the R21 detector; only R20 horizontal input changes."""

    def reset(self):
        super().reset()
        self.horizontal_samples = deque(maxlen=3)
        self.horizontal_raw = np.zeros(2)
        self.horizontal_filtered = np.zeros(2)

    @staticmethod
    def robust_horizontal(samples):
        values = np.asarray(samples, dtype=float)
        if values.shape != (3, 2) or not np.all(np.isfinite(values)):
            raise ValueError("three finite horizontal wrench samples required")
        return np.median(values, axis=0)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        # Bypass only the inherited R20/R21 composition; the unchanged R10
        # detector still owns the shared estimator and its baseline result.
        residual, baseline = my_detector.Detector.update(
            self, t, th, thd, tau_m, tau_s, grip)

        self.horizontal_raw = np.asarray(self._shared.last_residual_wrench[:2], float)
        self.horizontal_samples.append(self.horizontal_raw.copy())
        threshold = self.threshold(float(t))
        if threshold is None or len(self.horizontal_samples) < 3:
            over = False
            self.horizontal_score = 0.0
        else:
            self.horizontal_filtered = self.robust_horizontal(self.horizontal_samples)
            self.horizontal_score = float(np.max(np.abs(self.horizontal_filtered)) / threshold)
            over = bool(self.horizontal_score > 1.0)
        self.horizontal_count = self.horizontal_count + 1 if over else 0
        self.horizontal_clear = 0 if over else self.horizontal_clear + 1
        if self.horizontal_count >= 1:
            self.horizontal_detected = True
        elif self.horizontal_clear >= 8:
            self.horizontal_detected = False

        idx = int(round(1000 * float(t))) // 100
        self.bounds = self._by_tick[idx] if 0 <= idx < 60 else Global.bounds
        packet = self._shared
        self.yaw_value = self.yaw.feed(th, tau_s - packet.bias, packet.mass_prev[0], grip)
        if self.yaw_value is None or t < .08:
            self.yaw_count = self.yaw_quiet = 0
            self.yaw_active = False
        else:
            above = abs(self.yaw_value) > self.bounds[int(bool(grip))]
            self.yaw_count = self.yaw_count + 1 if above else 0
            self.yaw_quiet = 0 if above else self.yaw_quiet + 1
            if self.yaw_count >= 3:
                self.yaw_active = True
            elif self.yaw_quiet >= 8:
                self.yaw_active = False
        return residual, bool(baseline or self.horizontal_detected or self.yaw_active)
