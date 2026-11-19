"""Course-simulation candidate: R14 coupled reports plus detached 3-D event ownership."""
import json
from pathlib import Path

import numpy as np

from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r26_delivery_pair.event import ReleasedEvent
from experiments.r26_delivery_pair.observer import ReleasedVector


class Detector(Coupled):
    def reset(self):
        super().reset()
        self.released_vector = ReleasedVector(self.nom, self.nom_box)
        bounds = json.loads(Path(__file__).with_name('bounds.json').read_bytes())
        self.detached_event = ReleasedEvent(bounds, self._shared.persistence,
                                           self._shared.release_persistence)

    def update(self, t, th, thd, motor, spring, closed):
        residual, incumbent = super().update(t, th, thd, motor, spring, closed)
        f = self.released_vector.feed(t, th, spring, closed)
        if f is None:
            self.detached_event.reset()
            return residual, incumbent
        k = min(round(float(t)*1000),len(self.wrench_coef)-1)
        b = np.array([1.,float(self.box_mass)])
        slow = self._shared.last_residual_wrench[:3] - self.wrench_coef[k,:3] @ b
        fast = self._shared.last_fast_innovation[:3] - self.fast_coef[k,:3] @ b
        learned = bool(self.normal_model.detected or self.r5_extra or self.r8_extra)
        return residual, self.detached_event.feed(f,slow,fast,self.cf_detected,
                                                  self.cf_alarm,self.cf_fast,learned)
