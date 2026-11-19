"""A completed detached notice cannot reopen in the same uninterrupted native acquisition."""
import math
from experiments.r29_detached_release.candidate import Detector as Released, Endpoint


class AcquisitionEndpoint(Endpoint):
    def __init__(self):
        super().__init__()
        self.retired = False

    def update(self, report, detached, vector, bound, mean_quiet, sole):
        answer = super().update(report, detached, vector, bound, mean_quiet, sole)
        # The parent resets through self.__init__ when the witness drops. Its next
        # acquisition is a different event even one tick later. No dead-time is imposed.
        if not detached or not sole or not report:
            return answer
        try:
            x, y = vector
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(bound) and bound > 0):
                return answer
        except (TypeError,ValueError):
            return answer
        if not answer and self.published:
            self.retired = True
        return bool(answer and not self.retired)


class Detector(Released):
    def reset(self):
        super().reset()
        self.detached_endpoint = AcquisitionEndpoint()
