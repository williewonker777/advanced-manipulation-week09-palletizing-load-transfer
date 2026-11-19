"""C1 plus contiguous air-owned horizontal acquisition; C1 always passes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r26_delivery_pair.candidate import Detector as C1
from sim import config as C


R27 = Path(__file__).resolve().parents[1] / "r27_landing_integration"
GRAVITY_M_S2 = 9.81


class OwnedHorizontalEpisode:
    """One optional event whose onset and pending report require live ownership."""

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

    def feed(self, vector, owned, incumbent):
        if not owned:
            self.over = 0
            self.pulse = 0
            self.active = False
            self.armed = False
            return False
        if vector is None:
            self.over = 0
            return False
        horizontal = float(np.hypot(vector[0], vector[1]))
        if horizontal < self.release:
            self.armed = True
            self.active = False
            self.over = 0
            self.seen = False
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
    """Preserve C1 and add only contiguous, air-owned horizontal acquisition."""

    def reset(self):
        super().reset()
        force = json.loads((R27 / "model.json").read_bytes())
        resolution = json.loads((R27 / "unloaded_resolution.json").read_bytes())
        if (force.get("schema") != "r27-gripped-world-force-bound.v1"
                or resolution.get("schema") != "r27-unresolved-payload-resolution.v1"):
            raise ValueError("invalid frozen R27 source models")
        self.normal_resolution = float(resolution["table_normal_resolution_N"])
        if not np.isfinite(self.normal_resolution) or self.normal_resolution <= 0:
            raise ValueError("positive finite table-normal resolution required")
        self.grip_force = NormalForce(
            self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D,
            jts_hz=C.JTS_CUTOFF)
        self.grip_episode = OwnedHorizontalEpisode(
            force["horizontal_bound_N"], force["release_fraction"],
            force["onset_consecutive_samples"], self._shared.release_persistence)
        self.grip_force_value = None
        self.grip_force_extra = False
        self.load_ownership = False
        self.load_branch = "unknown"
        self.table_normal_n = None

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.grip_force.update(t, th, tau_s, bool(grip))
        self.grip_force_value = None if sample is None else sample.force_n.copy()
        payload_weight = float(self.box_mass) * GRAVITY_M_S2
        support = getattr(self.ctrl, "shared_estimator", None)
        table_normal = (float(max(0.0, self._shared.last_table_wrench[2]))
                        if self._shared.last_table_allowed else 0.0)
        self.table_normal_n = table_normal if np.isfinite(table_normal) else None
        air = self.table_normal_n is not None and self.table_normal_n <= self.normal_resolution
        if not np.isfinite(payload_weight) or payload_weight < 0:
            load_owned = False
            self.load_branch = "unknown"
        elif payload_weight <= self.normal_resolution:
            load_owned = True
            self.load_branch = "unresolved"
        else:
            load_owned = bool(
                support is not None and support.lambda_support_valid
                and np.isfinite(support.lambda_support)
                and support.lambda_support >= 0.5)
            self.load_branch = "resolved" if load_owned else "unknown"
        self.load_ownership = bool(grip) and load_owned and air
        self.grip_force_extra = self.grip_episode.feed(
            self.grip_force_value, self.load_ownership, incumbent)
        return residual, bool(incumbent or self.grip_force_extra)
