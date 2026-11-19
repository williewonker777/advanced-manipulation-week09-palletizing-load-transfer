"""C2: detached signed-force acquisition layered over the unchanged R14 sources."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r26_delivery_pair.observer import ReleasedVector
from experiments.r26_impulse_onset.event import OnsetEvent


class Detector(Coupled):
    def reset(self):
        super().reset()
        here = Path(__file__).resolve().parent
        level = json.loads((here.parent / "r26_delivery_pair" / "bounds.json").read_bytes())
        model = json.loads((here / "model.json").read_bytes())
        if (model.get("schema") != "r26-detached-onset-model.v1"
                or model.get("fit_episode_count") != 7
                or model.get("lag_samples") != self._shared.release_persistence):
            raise ValueError("frozen detached-onset model contract differs")
        self.released_vector = ReleasedVector(self.nom, self.nom_box)
        self.detached_onset = OnsetEvent(level, model["change_floor_n"],
                                        model["lag_samples"],
                                        self._shared.release_persistence)
        self.detached_trace = []

    def update(self, t, th, thd, motor, spring, closed):
        residual, _ = super().update(t, th, thd, motor, spring, closed)
        force = self.released_vector.feed(t, th, spring, closed)
        learned = bool(self.normal_model.detected or self.r5_extra or self.r8_extra)
        if force is None:
            self.detached_onset.reset()
            self.detached_trace.append((False, False, 0.0, 0.0))
            return residual, bool(self.cf_detected or learned)
        k = min(round(float(t) * 1000), len(self.wrench_coef) - 1)
        basis = np.array([1.0, float(self.box_mass)])
        slow = self._shared.last_residual_wrench[:3] - self.wrench_coef[k, :3] @ basis
        fast = self._shared.last_fast_innovation[:3] - self.fast_coef[k, :3] @ basis
        extra, attributed = self.detached_onset.feed(
            force, slow, fast, self.cf_detected, self.cf_alarm, self.cf_fast, learned)
        self.detached_trace.append((self.detached_onset.sustained,
                                    self.detached_onset.onset,
                                    self.detached_onset.signed_level_n,
                                    self.detached_onset.signed_change_n))
        return residual, bool(learned or (self.cf_detected and not attributed) or extra)
