"""Consume only a confidently same-source yaw reacquisition, never its gap."""
from __future__ import annotations

import math

from experiments.r25_composed.detector import Detector as R25
from experiments.r26_long.candidate import SignedYawOwnership


class YawReacquisitionMixin:
    """Separate continuous event identity from the detector's Boolean alarm.

    R25 is updated exactly once.  A native yaw fall is always exposed.  If the
    frozen signed identity remains certified during that gap, a later yaw-only
    reacquisition of the same source is consumed.  Any independently live R25
    route is preserved.  Unknown route provenance fails open.
    """

    def reset(self):
        super().reset()
        self.yaw_acquisition_identity = SignedYawOwnership()
        self.yaw_acquisition_episode_active = False
        self.yaw_reacquisition_consumed = False
        self.yaw_reacquisition_count = 0
        self.yaw_reacquisition_provenance_complete = True
        self._yaw_acquisition_previous_active = False
        self._yaw_acquisition_gap_certified = False

    @staticmethod
    def _same_signed_value(identity, value, grip):
        """Use the frozen identity's signed event-vs-zero decision."""
        return bool(
            identity.active
            and identity.grip == bool(grip)
            and value is not None
            and math.isfinite(float(value))
            and identity.mean * float(value) > 0.0
            and 2.0 * float(value) * identity.mean > identity.mean * identity.mean
        )

    @classmethod
    def _same_signed_source(cls, identity, value, grip):
        return bool(identity.bridge and cls._same_signed_value(identity, value, grip))

    def _r25_route_provenance(self, report):
        """Return effective R25 route states and whether their OR is complete.

        The reconstruction includes the outer continuation route and the
        pickup-completion ownership rule, not merely PickupCompletionMixin's
        `_non_yaw_report`.  A mismatch with R25's actual output is uncertainty,
        so suppression is forbidden for that sample.
        """
        try:
            packet = self._shared
            normal = self.normal_model
            light = self.light_witness.state
            detached = self.detached.evidence
            continuation = self.continuation
            pickup_completion = bool(self.pick_completion_active)
            routes = {
                "r8_base": bool(self.r8_base_detect),
                "r8_extra": bool(self.r8_extra),
                "directional": bool(packet.detected),
                "normal": bool(normal is not None and normal.detected),
                "horizontal": bool(self.horizontal_detected),
                "yaw": bool(self.yaw_active and not pickup_completion),
                "light": bool(light.active),
                "detached": bool(detached.active),
                "pickup_hold": bool(self.pick_event_active and not pickup_completion),
                "continuation": bool(continuation.bridge),
            }
        except (AttributeError, TypeError, ValueError):
            return {}, False
        return routes, bool(any(routes.values()) == bool(report))

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, report = super().update(t, th, thd, tau_m, tau_s, grip)
        yaw_now = bool(self.yaw_active)
        blocked = bool(getattr(self, "pick_completion_active", False))
        normal = getattr(self, "normal_model", None)
        support = (None if normal is None or not normal.last_active
                   else normal.last_error)
        identity = self.yaw_acquisition_identity

        reacquired = bool(
            yaw_now
            and not self._yaw_acquisition_previous_active
            and self._yaw_acquisition_gap_certified
            and not blocked
            and self._same_signed_source(identity, self.yaw_value, grip)
        )
        if self.yaw_reacquisition_consumed and yaw_now:
            if blocked or not self._same_signed_value(identity, self.yaw_value, grip):
                self.yaw_reacquisition_consumed = False
        elif not yaw_now:
            self.yaw_reacquisition_consumed = False
        if reacquired:
            self.yaw_reacquisition_consumed = True
            self.yaw_reacquisition_count += 1

        bridge = identity.feed(yaw_now, self.yaw_value, support, grip, blocked=blocked)
        self._yaw_acquisition_gap_certified = bool(not yaw_now and bridge)
        self.yaw_acquisition_episode_active = bool(identity.active)
        self._yaw_acquisition_previous_active = yaw_now

        routes, complete = self._r25_route_provenance(report)
        self.yaw_reacquisition_provenance_complete = complete
        independent = bool(any(value for name, value in routes.items() if name != "yaw"))
        consume = bool(
            self.yaw_reacquisition_consumed
            and complete
            and routes.get("yaw", False)
            and not independent
        )
        return residual, bool(report and not consume)


class Detector(YawReacquisitionMixin, R25):
    """Frozen R25 with signed same-event yaw reacquisition ownership."""

