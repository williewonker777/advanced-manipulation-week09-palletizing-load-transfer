"""Disclosed R27 pulse-restart intervention, not an enrolled or default detector."""
import json
from pathlib import Path

import numpy as np

from experiments.r27_landing_integration.unloaded_aware import Detector as FrozenV3


class ContiguousOwnedEpisode:
    """Exact diagnostic semantics frozen in plan.json; no parameter changes."""
    def __init__(self, bound, release_fraction, persistence, report_samples):
        self.bound = float(bound)
        self.release = float(release_fraction) * self.bound
        self.persistence = int(persistence)
        self.report_samples = int(report_samples)
        self.reset()

    def reset(self):
        self.armed = False; self.active = False; self.over = 0
        self.pulse = 0; self.seen = False

    def feed(self, vector, owned, incumbent):
        if vector is None:
            self.over = 0
            self.pulse = 0
            return False
        horizontal = float(np.hypot(vector[0], vector[1]))
        if horizontal < self.release:
            self.armed = True; self.active = False; self.over = 0; self.seen = False
        if not owned:
            if not self.active:
                self.over = 0
            self.pulse = 0
            return False
        if self.active:
            self.seen |= bool(incumbent)
        elif self.armed:
            self.over = self.over + 1 if horizontal > self.bound else 0
            if self.over >= self.persistence:
                self.active = True; self.armed = False; self.seen = bool(incumbent)
                if not self.seen:
                    self.pulse = self.report_samples
        extra = self.pulse > 0
        self.pulse = max(0, self.pulse - 1)
        return extra


class Detector(FrozenV3):
    def reset(self):
        super().reset()
        path = Path(__file__).resolve().parents[1] / 'r27_landing_integration/model.json'
        model = json.loads(path.read_bytes())
        self.grip_episode = ContiguousOwnedEpisode(
            model['horizontal_bound_N'], model['release_fraction'],
            model['onset_consecutive_samples'], self._shared.release_persistence)
