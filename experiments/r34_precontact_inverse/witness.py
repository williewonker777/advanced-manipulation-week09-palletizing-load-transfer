"""Composable precontact inverse-load source owner."""
import math

from experiments.r34_precontact_inverse.geometry import Approach


class PrecontactInverseWitness:
    def __init__(self, threshold_N, persistence=3):
        self.threshold = float(threshold_N)
        self.persistence = int(persistence)
        if not math.isfinite(self.threshold) or self.threshold <= 0 or self.persistence < 1:
            raise ValueError("finite positive model required")
        self.approach = Approach()
        self.reset_source()

    def reset_source(self):
        self.above = 0
        self.owned = False
        self.native_seen = False
        self.last_score = 0.0

    def extra(self, sample, grip, incumbent):
        eligible = self.approach.feed(None if sample is None else sample.bottom_z, grip)
        if sample is None or not eligible or not math.isfinite(float(sample.wrench[2])):
            self.reset_source()
            return False
        score = max(-float(sample.wrench[2]), 0.0)
        self.last_score = score
        if self.owned:
            if incumbent:
                self.native_seen = True
            if score < self.threshold / 2:
                self.reset_source()
                return False
            return not self.native_seen
        self.above = self.above + 1 if score > self.threshold else 0
        if self.above < self.persistence:
            return False
        self.owned = True
        self.native_seen = bool(incumbent)
        return not self.native_seen
