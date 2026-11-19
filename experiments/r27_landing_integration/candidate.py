"""C1 plus one normal-calibrated, gripped world-force acquisition source."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r26_delivery_pair.candidate import Detector as C1
from sim import config as C


MODEL = Path(__file__).with_name("model.json")


class GripForceEpisode:
    """Own one measured horizontal-force episode without muting the incumbent."""

    def __init__(self, bound, release_fraction, persistence, report_samples):
        self.bound = float(bound)
        self.release = float(release_fraction) * self.bound
        self.persistence = int(persistence)
        self.report_samples = int(report_samples)
        self.reset()

    def reset(self):
        self.armed = False
        self.active = False
        self.over = 0
        self.pulse = 0
        self.seen = False

    def feed(self, vector, grip, incumbent):
        if vector is None:
            return False
        horizontal = float(np.hypot(vector[0], vector[1]))
        if horizontal < self.release:
            self.armed = True
            self.active = False
            self.over = 0
            self.seen = False
        if not grip:
            return False
        if self.active:
            self.seen |= bool(incumbent)
        elif self.armed:
            self.over = self.over + 1 if horizontal > self.bound else 0
            if self.over >= self.persistence:
                self.active = True
                self.armed = False
                self.seen = bool(incumbent)
                if not self.seen:
                    self.pulse = self.report_samples
        extra = self.pulse > 0
        self.pulse = max(0, self.pulse - 1)
        return extra


class Detector(C1):
    """Preserve C1 and add a physically gripped inverse-force witness."""

    def reset(self):
        super().reset()
        model = json.loads(MODEL.read_bytes())
        if (model.get("schema") != "r27-gripped-world-force-bound.v1"
                or model.get("onset_consecutive_samples") != 2
                or not np.isfinite(model.get("horizontal_bound_N", np.nan))
                or model["horizontal_bound_N"] <= 0
                or not 0 < model.get("release_fraction", 0) < 1):
            raise ValueError("invalid frozen R27 landing model")
        self.grip_force = NormalForce(
            self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D,
            jts_hz=C.JTS_CUTOFF)
        self.grip_episode = GripForceEpisode(
            model["horizontal_bound_N"], model["release_fraction"],
            model["onset_consecutive_samples"],
            self._shared.release_persistence)
        self.grip_force_value = None
        self.grip_force_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.grip_force.update(t, th, tau_s, bool(grip))
        self.grip_force_value = None if sample is None else sample.force_n.copy()
        self.grip_force_extra = self.grip_episode.feed(
            self.grip_force_value, bool(grip), incumbent)
        return residual, bool(incumbent or self.grip_force_extra)
