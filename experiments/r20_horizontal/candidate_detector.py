"""Normal-bounded horizontal residual detector, OR'ed with unchanged R10.

The auxiliary channel uses the same world-frame horizontal residual already
computed by R10.  Its threshold is 1.02 times the largest absolute x/y
residual in the early-transfer portion of a normal-only 4-mass x 5-COM
calibration grid.  The phase window is a public trajectory transfer phase, not
a collision time.  Both horizontal directions share the threshold.  A release
branch was rejected because a prior collision produced a larger later normal
transfer residual than the official 10 N cases.
"""
import numpy as np

import my_detector


# The early-transfer normal maximum was 18.0150567392 N.
PHASES = (
    (1.35, 1.75, 18.3753578739),
)
# The threshold is a phase-wide maximum bound, so no additional consecutive
# exceedance assumption is needed to preserve the calibration negatives.
PERSISTENCE = 1
CLEAR_SAMPLES = 8


class Detector(my_detector.Detector):
    """Preserve every baseline alarm and add a causal horizontal branch."""

    def reset(self):
        super().reset()
        self.horizontal_count = 0
        self.horizontal_clear = 0
        self.horizontal_detected = False
        self.horizontal_score = 0.0

    @staticmethod
    def threshold(t):
        for start, stop, value in PHASES:
            if start <= t < stop:
                return value
        return None

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
        threshold = self.threshold(float(t))
        if threshold is None:
            over = False
            self.horizontal_score = 0.0
        else:
            horizontal = np.asarray(self._shared.last_residual_wrench[:2], float)
            self.horizontal_score = float(np.max(np.abs(horizontal)) / threshold)
            over = bool(self.horizontal_score > 1.0)
        self.horizontal_count = self.horizontal_count + 1 if over else 0
        self.horizontal_clear = 0 if over else self.horizontal_clear + 1
        if self.horizontal_count >= PERSISTENCE:
            self.horizontal_detected = True
        elif self.horizontal_clear >= CLEAR_SAMPLES:
            self.horizontal_detected = False
        return residual, bool(baseline or self.horizontal_detected)
