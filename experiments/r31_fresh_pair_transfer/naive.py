"""Diagnostic-only one-event-per-grip ablation."""
from experiments.r30_yaw_headroom.candidate import Detector as Headroom


class Detector(Headroom):
    def reset(self):
        super().reset()
        self.r31f_state = "idle"

    def update(self, *args):
        residual, report = super().update(*args)
        if not bool(args[-1]):
            self.r31f_state = "idle"
            return residual, report
        if self.r31f_state == "idle":
            if report:
                self.r31f_state = "active"
            return residual, report
        if self.r31f_state == "active":
            if not report:
                self.r31f_state = "closed"
            return residual, report
        return residual, False
