"""Deferred dual-observer release notice composed with unchanged R10 servo detection."""
from __future__ import annotations

import numpy as np

import my_detector
from experiments.r15_delayed_wrench.observer import NormalForce
from sim import config as C


HORIZONTAL_N = 5.0
DIRECTION_COSINE = 0.8
PERSISTENCE = 3
REPORT_SAMPLES = 8


def coherent_horizontal(force, wrench, table_allowed):
    if table_allowed or force is None:
        return False, None
    force = np.asarray(force, float); wrench = np.asarray(wrench, float)
    if (force.shape != (3,) or wrench.shape[0] < 2
            or not np.isfinite(force).all() or not np.isfinite(wrench[:2]).all()):
        return False, None
    a, b = force[:2], wrench[:2]
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = float(a @ b / (an * bn)) if an > 0 and bn > 0 else -1.0
    return bool(an > HORIZONTAL_N and bn > HORIZONTAL_N and cosine >= DIRECTION_COSINE), {
        "delayed_horizontal_N": an, "shared_horizontal_N": bn, "cosine": cosine}


class DeferredEpisode:
    def __init__(self): self.reset()

    def reset(self):
        self.onset = 0; self.pending = False; self.owned = False
        self.release = 0; self.pulse = 0

    def feed(self, evidence, valid, incumbent):
        if self.pulse:
            self.pulse -= 1
            return True
        if self.pending:
            self.owned |= bool(incumbent)
            if not valid:
                return False
            self.release = 0 if evidence else self.release + 1
            if self.release < PERSISTENCE:
                return False
            report = not self.owned
            self.onset = 0; self.pending = False; self.owned = False; self.release = 0
            if report:
                self.pulse = REPORT_SAMPLES - 1
                return True
            return False
        if not valid:
            self.onset = 0
            return False
        self.onset = self.onset + 1 if evidence else 0
        if self.onset >= PERSISTENCE:
            self.pending = True
            self.owned = bool(incumbent)
            self.onset = 0
        return False


class Detector(my_detector.Detector):
    def reset(self):
        super().reset()
        self.release_force = NormalForce(self.nom, self.nom_box, dt=C.DT,
                                         viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
        self.release_episode = DeferredEpisode()
        self.release_evidence = False; self.release_features = None; self.release_extra = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.release_force.update(t, th, tau_s, bool(grip))
        force = None if sample is None else sample.force_n
        self.release_evidence, self.release_features = coherent_horizontal(
            force, self._shared.last_wrench, self._shared.last_table_allowed)
        self.release_extra = self.release_episode.feed(
            self.release_evidence, sample is not None, bool(incumbent))
        return residual, bool(incumbent or self.release_extra)
