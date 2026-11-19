"""Plain R10 detector plus a support-aware cyclic base-yaw witness."""
import json
from pathlib import Path

import numpy as np

import my_detector
from experiments.r21_support_yaw.support import admissible_yaw, unexplained_yaw
from experiments.r21_yaw_impulse.coarse_calibration import BINS, bin_index
from experiments.r21_yaw_impulse.state import YawBalance


class Detector(my_detector.Detector):
    def reset(self):
        super().reset()
        doc = json.loads(Path(__file__).with_name("normal_bins.json").read_bytes())
        table = np.asarray(doc["threshold_Nm"], dtype=float)
        if doc.get("schema") != "r21-servo-yaw-normal-bins.v1" or table.shape != (2, BINS):
            raise ValueError("frozen servo normal bounds required")
        if not np.all(np.isfinite(table)) or np.any(table <= 0.0):
            raise ValueError("invalid servo normal bounds")
        self._bins = table
        self.yaw = YawBalance()
        self.yaw_value = None
        self.yaw_ambiguity = 0.0
        self.yaw_excess = None
        self.yaw_count = self.yaw_quiet = 0
        self.yaw_active = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
        packet = self._shared
        self.yaw_value = self.yaw.feed(th, tau_s - packet.bias, packet.mass_prev[0], grip)
        if self.yaw_value is None or t < 0.08:
            self.yaw_ambiguity = 0.0
            self.yaw_excess = None
            self.yaw_count = self.yaw_quiet = 0
            self.yaw_active = False
        else:
            self.yaw_ambiguity = admissible_yaw(packet, th, tau_s, grip)
            self.yaw_excess = unexplained_yaw(self.yaw_value, self.yaw_ambiguity)
            bound = self._bins[int(bool(grip)), bin_index(t)]
            above = abs(self.yaw_excess) > bound
            self.yaw_count = self.yaw_count + 1 if above else 0
            self.yaw_quiet = 0 if above else self.yaw_quiet + 1
            if self.yaw_count >= 3:
                self.yaw_active = True
            elif self.yaw_quiet >= 8:
                self.yaw_active = False
        return residual, bool(detected or self.yaw_active)
