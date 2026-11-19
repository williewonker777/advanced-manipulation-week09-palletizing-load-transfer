"""A base-axis torque witness, not a general six-axis safety detector."""
import math

from experiments.r20_horizontal.candidate_detector import Detector as Horizontal
from experiments.r21_yaw_impulse.state import YawBalance


class Detector(Horizontal):
    bounds = (math.inf, math.inf)  # diagnostic default until calibrated
    model_scope = 'R10 torque controller only'

    def reset(self):
        super().reset()
        self.yaw = YawBalance()
        self.yaw_value = None
        self.yaw_count = self.yaw_quiet = 0
        self.yaw_active = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
        p = self._shared
        self.yaw_value = self.yaw.feed(th, tau_s - p.bias, p.mass_prev[0], grip)
        # Initial zeroing is already performed by the unmodified R10 observer.
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
        return residual, bool(detected or self.yaw_active)
