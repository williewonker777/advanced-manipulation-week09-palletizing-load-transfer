"""Complete a reported pre-pick yaw event at a physically consistent model switch."""
from __future__ import annotations

import math

from experiments.r24_composed.detector import Detector as R24Joint


def signed_completion(before, after):
    """Return the consumed sign when compensation preserves sign and reduces magnitude."""
    if not (math.isfinite(before) and math.isfinite(after)):
        return 0
    if before == 0.0 or before * after <= 0.0 or abs(after) > abs(before):
        return 0
    return 1 if before > 0.0 else -1


class PickupCompletionMixin:
    """Turn a compensated, already-reported pickup impulse into a completed event.

    The inherited detector is updated exactly once.  Its yaw observer and model
    compensation remain untouched.  When the actual grip/model transition shows
    that the signed yaw estimate is already progressing toward zero, only that
    reported yaw route is consumed; all other detector routes remain live.  The
    route is immediately released for an above-bound opposite-signed yaw event.
    """

    def reset(self):
        super().reset()
        self.pick_completion_active = False
        self.pick_completion_decided = False
        self.pick_completion_sign = 0
        self.pick_completion_selected = False
        self.pick_completion_reason = None

    def _non_yaw_report(self):
        packet = self._shared
        normal = getattr(self, "normal_model", None)
        normal_active = bool(normal is not None and normal.detected)
        light = getattr(getattr(self, "light_witness", None), "state", None)
        detached = getattr(getattr(self, "detached", None), "evidence", None)
        return bool(
            packet.detected
            or normal_active
            or self.horizontal_detected
            or (light is not None and light.active)
            or (detached is not None and detached.active)
        )

    def update(self, t, th, thd, tau_m, tau_s, grip):
        before = self.yaw_value
        pre_active = bool(self.yaw_active)
        selection_before = self.yaw.pick_continuity_selected
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)

        selected_now = (selection_before is not True and
                        self.yaw.pick_continuity_selected is True)
        if selected_now and not self.pick_completion_decided:
            self.pick_completion_decided = True
            sign = signed_completion(float(before), float(self.yaw_value))
            if pre_active and sign:
                self.pick_completion_active = True
                self.pick_completion_sign = sign
                self.pick_completion_selected = True
                self.pick_completion_reason = "same-sign compensated momentum receded"
                # R24's generic latch is replaced only after causal completion.
                self.pick_event_active = False
            else:
                self.pick_completion_reason = "signed completion confidence absent"

        if not self.pick_completion_active:
            return residual, report

        value = self.yaw_value
        bound = float(self.bounds[int(bool(grip))])
        reversal = bool(
            value is not None
            and math.isfinite(float(value))
            and self.pick_completion_sign * float(value) < -bound
        )
        if reversal:
            self.pick_completion_active = False
            self.pick_completion_reason = "above-bound opposite-signed yaw released route"
            return residual, report

        if not self.yaw_active:
            self.pick_completion_active = False
            self.pick_completion_reason = "consumed yaw route physically cleared"
        return residual, self._non_yaw_report()


class Detector(PickupCompletionMixin, R24Joint):
    """R24-compatible detector with one causal pickup-completion update."""

