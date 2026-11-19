"""Observation memory with source-owned empty-arm unloading/rebound attribution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r28_contact_memory.candidate import Detector as Memory
from sim import config as C


class Detector(Memory):
    """Suppress only CF episodes certified as the same measured unloading source."""

    def reset(self):
        super().reset()
        self.release_bounds = json.loads(
            (Path(__file__).resolve().parents[1] / "r26_delivery_pair" / "bounds.json").read_bytes())
        self.empty_inverse = NormalForce(
            self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
        self.empty_inverse_value = None
        self.was_grip = False
        self.was_cf_release = False
        self.release_open = False
        self.unloading_owned = False
        self.rebound_pending = False
        self.suppress_cf_episode = False
        self.settle_samples = 0
        self.r29_suppressed = False

    def _ratios(self, vector):
        if vector is None:
            return None
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            return None
        horizontal = float(np.hypot(vector[0], vector[1])) / self.release_bounds["horizontal"]
        key = "positive_vertical" if vector[2] >= 0.0 else "negative_vertical"
        vertical = abs(float(vector[2])) / self.release_bounds[key]
        return horizontal, vertical, key

    def _other_source(self, incumbent):
        normal = getattr(self, "normal_model", None)
        learned = bool(getattr(normal, "detected", False)
                       or getattr(self, "r5_extra", False) or getattr(self, "r8_extra", False))
        event = getattr(self, "detached_event", None)
        detached = bool(getattr(event, "pulse", 0))
        memory = bool(getattr(self, "grip_force_extra", False))
        return bool(learned or detached or memory or (incumbent and not self.cf_detected))

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.empty_inverse.update(t, th, tau_s, False)
        self.empty_inverse_value = None if sample is None else sample.force_n.copy()
        opened = self.was_grip and not bool(grip)
        self.was_grip = bool(grip)
        if opened:
            self.release_open = True
            self.unloading_owned = False
            self.rebound_pending = False
            self.suppress_cf_episode = False
            self.settle_samples = 0
        if grip:
            self.release_open = False
            self.unloading_owned = False
            self.rebound_pending = False
            self.suppress_cf_episode = False
            self.settle_samples = 0

        cf = bool(self.cf_detected)
        cf_rise = cf and not self.was_cf_release
        other = self._other_source(incumbent)
        ratios = self._ratios(self.empty_inverse_value)
        normal_vector = ratios is not None and max(ratios[:2]) < 1.0
        negative_unload = (ratios is not None and ratios[2] == "negative_vertical"
                           and ratios[1] > 1.0 and ratios[1] >= ratios[0])

        if self.release_open and not cf and self.cf_alarm == 0 and self.cf_fast == 0 and normal_vector:
            self.settle_samples += 1
            if self.settle_samples >= self._shared.release_persistence:
                self.unloading_owned = False
                self.rebound_pending = False
        else:
            self.settle_samples = 0

        if self.release_open and self.rebound_pending and ratios is not None and not normal_vector:
            if not negative_unload:
                self.unloading_owned = False
                self.rebound_pending = False

        if cf_rise and self.release_open and not other:
            if not self.unloading_owned and negative_unload:
                self.unloading_owned = True
                self.rebound_pending = True
                self.suppress_cf_episode = True
            elif self.rebound_pending and normal_vector:
                self.suppress_cf_episode = True
                self.rebound_pending = False
                self.unloading_owned = False
            else:
                self.suppress_cf_episode = False
        if not cf:
            self.suppress_cf_episode = False
        self.was_cf_release = cf
        self.r29_suppressed = bool(cf and self.suppress_cf_episode and not other)
        return residual, bool(other or (cf and not self.r29_suppressed))
