"""Shape a yaw alarm from its measured mechanical falling edge.

The R22 continuity witness still owns re-arming.  This layer only ends the
visible yaw event after its absolute disturbance falls ten percent from the
event peak for three samples.  It cannot start another event until the
underlying normal-bounded witness first goes quiet and crosses again.
"""
from experiments.r22_release.candidate import Detector as Continuous

FALL_RATIO = 0.90
FALL_SAMPLES = 3


class Detector(Continuous):
    def reset(self):
        super().reset()
        self.release_event_active = False
        self.release_event_peak = 0.0
        self.release_fall_count = 0
        self._release_yaw_was_active = False

    def update(self, t, *args):
        residual, _ = super().update(t, *args)
        yaw = bool(self.yaw_active)
        magnitude = 0.0 if self.yaw_value is None else abs(float(self.yaw_value))
        if yaw and not self._release_yaw_was_active:
            self.release_event_active = True
            self.release_event_peak = magnitude
            self.release_fall_count = 0
        if yaw and self.release_event_active:
            self.release_event_peak = max(self.release_event_peak, magnitude)
            below = magnitude <= FALL_RATIO * self.release_event_peak
            self.release_fall_count = self.release_fall_count + 1 if below else 0
            if self.release_fall_count >= FALL_SAMPLES:
                self.release_event_active = False
        elif not yaw:
            self.release_event_active = False
            self.release_event_peak = 0.0
            self.release_fall_count = 0
        self._release_yaw_was_active = yaw
        baseline = bool(self._shared.detected or self.horizontal_detected)
        return residual, bool(baseline or self.release_event_active)
