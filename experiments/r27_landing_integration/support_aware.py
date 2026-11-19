"""Separate C1 integration: admit v1 force acquisition only while arm carries load."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r27_landing_integration.candidate import GripForceEpisode
from sim import config as C


MODEL = Path(__file__).with_name("model.json")
ARM_SUPPORT_OWNERSHIP = 0.5


class Detector(C1):
    """Never alter C1; add acquisition only under valid carried-load ownership."""

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
        self.arm_support_owned = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.grip_force.update(t, th, tau_s, bool(grip))
        self.grip_force_value = None if sample is None else sample.force_n.copy()
        support = getattr(self.ctrl, "shared_estimator", None)
        self.arm_support_owned = bool(
            support is not None and support.lambda_support_valid
            and np.isfinite(support.lambda_support)
            and support.lambda_support >= ARM_SUPPORT_OWNERSHIP)
        qualified_grip = bool(grip) and self.arm_support_owned
        self.grip_force_extra = self.grip_episode.feed(
            self.grip_force_value, qualified_grip, incumbent)
        return residual, bool(incumbent or self.grip_force_extra)
