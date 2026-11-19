"""Opt-in post-release +Z detector with a freshly initialized empty-model source."""
from __future__ import annotations

import numpy as np

import my_torque
from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r34_released_source import core_b, model_b
from sim import config as C


class ReleasedSourceHook:
    """Composable sensor-only hook; the owning detector calls its parent separately."""

    def __init__(self, owner):
        self.normal_force = NormalForce(
            owner.nom, owner.nom_box, dt=C.DT,
            viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
        self.shadow = my_torque.SharedEstimator(
            owner.nom, owner.nom_box, owner.box_mass, owner.box_com)
        self.stream = core_b.Stream(model_b)
        self.was_gripped = False
        self.released = False
        self.release_separation = None
        self.extra = False

    def update(self, owner, t, th, thd, tau_s, grip):
        packet = owner._shared
        separation = float(packet._point_jacobian(np.asarray(th, float))[1][2]
                           - packet.table_heights[1])
        if grip:
            self.was_gripped = True
            self.released = False
            self.release_separation = None
            self.stream = core_b.Stream(model_b)
        elif self.was_gripped and not self.released:
            self.released = True
            self.release_separation = separation
            self.shadow.reset()
            self.shadow.bias = np.asarray(packet.bias, float).copy()
            self.shadow.bias_weight = float(packet.bias_weight)
        if self.released:
            self.shadow.update(t, th, thd, tau_s, False)
        sample = self.normal_force.update(t, th, tau_s, grip, project=True)
        valid = bool(self.released and sample is not None
                     and hasattr(self.shadow, "last_residual_wrench"))
        displacement = separation - self.release_separation if valid else 0.0
        force_z = float(sample.force_n[2]) if valid else 0.0
        shadow_z = float(self.shadow.last_residual_wrench[2]) if valid else 0.0
        self.extra = bool(self.stream.feed(valid, displacement, force_z, shadow_z))
        return self.extra


class Detector(Parent):
    """One unchanged R31 update OR the optional released-source hook."""

    def reset(self):
        super().reset()
        self.released_source = ReleasedSourceHook(self)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, parent = super().update(t, th, thd, tau_m, tau_s, grip)
        extra = self.released_source.update(self, t, th, thd, tau_s, grip)
        return residual, bool(parent or extra)
