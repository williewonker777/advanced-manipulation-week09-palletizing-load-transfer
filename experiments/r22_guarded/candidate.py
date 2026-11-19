"""Sensor-guarded release continuity composed onto the frozen R22 core."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r21_yaw_impulse.state import YawBalance
from experiments.r22_combined.core import Detector as Core
from experiments.r22_release.state import ReleaseBalance


NORMAL_BOUNDS = Path(__file__).resolve().parents[1] / "r21_yaw_impulse/normal_bins.json"


class GuardedReleaseBalance(ReleaseBalance):
    """Continue a loaded release only when the preceding sensor report was active."""

    def __init__(self, mass):
        super().__init__(mass)
        self.pre_release_alarm = False
        self.release_continuity_selected = None
        self.use_reset_bank = False

    def feed(self, th, tau, mass_row, grip):
        switching = self.prev is not None and self.grip != bool(grip)
        releasing_payload = switching and self.loaded and not bool(grip)
        picking_payload = switching and self.loaded and bool(grip)
        if releasing_payload and not self.pre_release_alarm:
            self.release_continuity_selected = False
            self.use_reset_bank = True
            value = YawBalance.feed(self, th, tau, mass_row, grip)
            self._positions.append(self.prev)
            return value
        if releasing_payload:
            self.release_continuity_selected = True
            self.use_reset_bank = False
        elif picking_payload:
            self.release_continuity_selected = None
            self.use_reset_bank = False
        return super().feed(th, tau, mass_row, grip)


class Detector(Core):
    """R22 core with sensor-evidenced continuity at positive-mass release."""

    def reset(self):
        super().reset()
        model = json.loads(NORMAL_BOUNDS.read_bytes())
        table = np.asarray(model["threshold_Nm"], dtype=float)
        if (model["schema"] != "r21-yaw-normal-bins.v1" or table.shape != (2, 60)
                or not np.isfinite(table).all() or not (table > 0).all()):
            raise ValueError("invalid frozen R21 normal-only yaw bounds")
        self._normal_by_tick = table.T
        self._continuity_by_tick = self._by_tick.copy()
        self.yaw = GuardedReleaseBalance(self.box_mass)
        self._previous_combined_report = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        self._by_tick = (self._normal_by_tick if self.yaw.use_reset_bank
                         else self._continuity_by_tick)
        self.yaw.pre_release_alarm = bool(self._previous_combined_report)
        residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
        self._previous_combined_report = bool(detected)
        return residual, detected
