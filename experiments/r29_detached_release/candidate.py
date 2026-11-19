"""Expose return to normal only for an already-delivered, detached-only force event."""
import math
from experiments.r22_tangent.frozen import Detector as Tangent
from experiments.r26_yaw_acquisition.candidate import Detector as C3


class Endpoint:
    """The acquired long-window route and its short input must both certify low force."""

    def __init__(self):
        self.low = 0
        self.published = False

    def update(self, report, detached, vector, bound, mean_quiet, sole):
        if not detached:
            self.__init__()
            return bool(report)
        try:
            x, y = vector
            finite = all(math.isfinite(v) for v in (x, y, bound)) and bound > 0
            inside = finite and math.hypot(x, y) < .8 * bound
        except (TypeError, ValueError):
            inside = False
        self.low = self.low + 1 if inside else 0
        retire = self.published and sole and mean_quiet and self.low >= 3
        answer = bool(report and not retire)
        self.published |= answer
        return answer


class Detector(C3):
    def reset(self):
        super().reset()
        self.detached_endpoint = Endpoint()

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, native = super().update(t, th, thd, tau_m, tau_s, grip)
        routes, complete = self._r25_route_provenance(native)
        witness = self.detached
        detached = bool(not grip and witness.evidence.active and
                        (not witness.require_onset or witness.gate.accepted))
        sole = complete and routes.get('detached', False) and not any(
            live for route, live in routes.items() if route != 'detached')
        report = self.detached_endpoint.update(native, detached, witness.vector,
                    Tangent.bounds_tangent[0], witness.evidence.quiet > 0, sole)
        return residual, report
