"""Causal signed world-horizontal force with measured support compensation."""
from collections import deque
from time import perf_counter_ns

import numpy as np

from sim import config as C
from experiments.r15_delayed_wrench.observer import LAG, NormalForce


def unexplained_horizontal(raw, table, allowed, friction_mu):
    """Remove only force inside the measured unilateral support friction set."""
    raw = np.asarray(raw, float)
    table = np.asarray(table, float)
    if not allowed:
        return raw.copy()
    residual = raw - table[:2]
    used = float(np.linalg.norm(table[:2]))
    remaining = max(float(friction_mu) * max(float(table[2]), 0.0) - used, 0.0)
    magnitude = float(np.linalg.norm(residual))
    return residual * (max(magnitude - remaining, 0.0) / magnitude) if magnitude else residual


def confidence_score(vector, table, allowed, support_uncertainty_per_n):
    """Normal-only fitted uncertainty while support is genuinely observed."""
    magnitude = float(np.linalg.norm(vector))
    penalty = (float(support_uncertainty_per_n) * max(float(table[2]), 0.0)
               if allowed else 0.0)
    return max(magnitude - penalty, 0.0)


class WorldHorizontalWitness:
    """Small callable force witness; no schedule or payload identity is consumed."""

    def __init__(self, empty, loaded, *, threshold, support_uncertainty_per_n=.2,
                 persistence=2, hold=8, rearm=25):
        self.observer = NormalForce(empty, loaded, dt=C.DT, viscous_spring=C.D,
                                    jts_hz=C.JTS_CUTOFF)
        self.threshold = float(threshold)
        self.support_uncertainty_per_n = float(support_uncertainty_per_n)
        self.persistence = int(persistence)
        self.hold_samples = int(hold)
        self.rearm_samples = int(rearm)
        self.support = deque(maxlen=LAG + 1)
        self.previous = None
        self.above = self.hold = self.quiet = 0
        self.armed = True
        self.last_vector = np.zeros(2)
        self.last_raw = np.zeros(2)
        self.last_score = 0.0
        self.runtime_ns = []

    def __call__(self, t, encoder, jts, grip, packet):
        started = perf_counter_ns()
        allowed = bool(getattr(packet, "last_table_allowed", False))
        table = np.asarray(getattr(packet, "last_table_wrench", np.zeros(6)), float)
        support = (np.r_[table[:3], 1.0] if allowed and np.isfinite(table[:3]).all()
                   else np.zeros(4))
        self.support.append(support)
        sample = self.observer.update(t, encoder, jts, grip)
        corrected = None
        if sample is not None and len(self.support) == self.support.maxlen:
            self.last_raw = np.asarray(sample.force_n[:2], float).copy()
            delayed = self.support[0]
            corrected = unexplained_horizontal(self.last_raw, delayed[:3], bool(delayed[3]),
                                                getattr(packet, "friction_mu", 0.0))
            if np.isfinite(corrected).all():
                self.last_vector = corrected.copy()
            else:
                corrected = None

        eligible = bool(grip) and corrected is not None
        self.last_score = (confidence_score(corrected, table[:3], allowed,
                                            self.support_uncertainty_per_n)
                           if eligible else 0.0)
        coherent = (eligible and self.last_score > self.threshold and
                    (self.previous is None or float(np.dot(corrected, self.previous)) > 0.0))
        self.above = self.above + 1 if coherent else 0
        self.quiet = 0 if coherent else self.quiet + 1
        if self.above >= self.persistence and self.hold == 0 and self.armed:
            self.hold = self.hold_samples
            self.armed = False
        active = self.hold > 0
        if self.hold:
            self.hold -= 1
        if self.quiet >= self.rearm_samples:
            self.armed = True
        elif not coherent and not active:
            self.above = 0
        self.previous = (corrected.copy() if eligible and
                         np.linalg.norm(corrected) > self.threshold else None)
        self.runtime_ns.append(perf_counter_ns() - started)
        return active


class WorldHorizontalMixin:
    """Cooperative OR-only adapter preserving every incumbent report."""

    world_horizontal_threshold = float("inf")
    world_horizontal_support_uncertainty = .2

    def reset(self):
        super().reset()
        self.world_horizontal = WorldHorizontalWitness(
            self.nom, self.nom_box, threshold=self.world_horizontal_threshold,
            support_uncertainty_per_n=self.world_horizontal_support_uncertainty)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = super().update(t, th, thd, tau_m, tau_s, grip)
        extra = self.world_horizontal(t, th, tau_s, grip, self._shared)
        return residual, bool(incumbent or extra)
