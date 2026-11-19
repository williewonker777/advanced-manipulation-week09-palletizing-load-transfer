"""Causal pickup model-switch continuity trial on the selected R23 detector."""
from __future__ import annotations

import numpy as np

from experiments.r21_yaw_impulse.state import AVERAGE
from experiments.r22_guarded.candidate import GuardedReleaseBalance
from experiments.r23_detached.free import Detector as Free


class PickupContinuousBalance(GuardedReleaseBalance):
    """Keep one active yaw event continuous as the known payload is attached.

    The model's generalized momentum changes algebraically when the grip state
    selects the loaded inertia.  If yaw evidence was already active on the
    preceding sensor report, compensate that modeled momentum step instead of
    erasing the measured event history.  Quiet pickups keep the inherited reset.
    """

    def __init__(self, mass):
        super().__init__(mass)
        self.pre_pick_yaw_active = False
        self.pick_continuity_selected = None

    def feed(self, th, tau, mass_row, grip):
        switching = self.prev is not None and self.grip != bool(grip)
        picking_payload = switching and self.loaded and bool(grip)
        can_continue = picking_payload and self.pre_pick_yaw_active and (
            len(self._positions) == AVERAGE + 1)
        if can_continue:
            dq = (self._positions[-1] - self._positions[0]) / (AVERAGE * self.dt)
            self.integral += float(
                (np.asarray(mass_row, dtype=float) - self.mass_row) @ dq)
            self.mass_row = np.asarray(mass_row, dtype=float).copy()
            self.grip = True
            self.pick_continuity_selected = True
        elif picking_payload:
            self.pick_continuity_selected = False
        return super().feed(th, tau, mass_row, grip)


class Detector(Free):
    """Selected R23 free detector plus sensor-evidenced pickup continuity."""

    def reset(self):
        super().reset()
        self.yaw = PickupContinuousBalance(self.box_mass)
        self.pick_event_active = False
        self.pick_event_quiet = 0
        self.pick_continuity_consumed = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        self.yaw.pre_pick_yaw_active = bool(self.yaw_active)
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        if (self.yaw.pick_continuity_selected and
                not self.pick_continuity_consumed):
            self.pick_event_active = True
            self.pick_continuity_consumed = True
        packet = self._shared
        continuation = max(float(packet.last_score),
                           float(packet.last_fast_score)) > 0.55
        if self.pick_event_active and (report or continuation):
            self.pick_event_quiet = 0
        elif self.pick_event_active:
            self.pick_event_quiet += 1
            if self.pick_event_quiet >= packet.release_persistence:
                self.pick_event_active = False
        return residual, bool(report or self.pick_event_active)
