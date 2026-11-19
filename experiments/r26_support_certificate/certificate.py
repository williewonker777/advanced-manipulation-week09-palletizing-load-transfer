"""Worst-case signed attribution outside a unilateral Coulomb support disk."""
from collections import deque

import numpy as np

from experiments.r15_delayed_wrench.observer import LAG


ONSET_N = 8.394415779390961


def signed_cone_lower_bound(raw_xy, friction_mu, maximum_normal_n):
    """Lower bound along the observed signed direction for any support vector."""
    raw = np.asarray(raw_xy, float)
    if raw.shape != (2,) or not np.isfinite(raw).all():
        return 0.0
    magnitude = float(np.linalg.norm(raw))
    radius = max(float(friction_mu), 0.0) * max(float(maximum_normal_n), 0.0)
    return max(magnitude - radius, 0.0)


class SupportConeCertificate:
    """Certificate using the worst observed admissible normal over observer lag."""

    def __init__(self, threshold=ONSET_N, lag=LAG):
        self.threshold = float(threshold)
        self.normals = deque(maxlen=int(lag) + 1)
        self.previous = None
        self.last_lower_bound = 0.0
        self.last_maximum_normal = 0.0

    def feed(self, raw_xy, existing_score, packet):
        table = np.asarray(getattr(packet, "last_table_wrench", np.zeros(6)), float)
        allowed = bool(getattr(packet, "last_table_allowed", False))
        normal = max(float(table[2]), 0.0) if allowed and np.isfinite(table[2]) else 0.0
        self.normals.append(normal)
        self.last_maximum_normal = max(self.normals, default=0.0)
        raw = np.asarray(raw_xy, float)
        self.last_lower_bound = signed_cone_lower_bound(
            raw, getattr(packet, "friction_mu", 0.0), self.last_maximum_normal)
        condition = (raw.shape == (2,) and np.isfinite(raw).all() and
                     np.isfinite(existing_score) and float(existing_score) > self.threshold and
                     self.last_lower_bound > self.threshold)
        coherent = condition and (self.previous is None or
                                  float(np.dot(raw, self.previous)) > 0.0)
        self.previous = raw.copy() if condition else None
        return bool(coherent)


def lifecycle_eligible(phase, physical_eligible, certified):
    """Certification gates onset only; its loss cannot create a trailing edge."""
    return bool(physical_eligible and (certified if phase == "quiet" else True))
