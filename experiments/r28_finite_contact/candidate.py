"""R28 opt-in model: verify observations before unilateral force projection."""
from __future__ import annotations

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector as C1
from experiments.r28_landing_contact.candidate import Detector as Frozen, GRAVITY_M_S2


def finite_scalar(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def admissible_normal(packet):
    """A definite 'out of table geometry' is different from a missing measurement."""
    allowed = getattr(packet, 'last_table_allowed', None)
    if not isinstance(allowed, (bool, np.bool_)):
        return None
    if not allowed:
        return 0.0
    wrench = getattr(packet, 'last_table_wrench', None)
    if wrench is None:
        return None
    try:
        raw = finite_scalar(wrench[2])
    except (IndexError, KeyError, TypeError):
        return None
    return None if raw is None else max(0.0, raw)


def support_owner(mass, resolution, support):
    mass = finite_scalar(mass)
    if mass is None or mass < 0:
        return False, 'unknown'
    if mass * GRAVITY_M_S2 <= resolution:
        return True, 'unresolved'
    valid = getattr(support, 'lambda_support_valid', None)
    fraction = finite_scalar(getattr(support, 'lambda_support', None))
    if not isinstance(valid, (bool, np.bool_)) or not valid or fraction is None or fraction < 0.5:
        return False, 'unknown'
    return True, 'resolved'


class Detector(Frozen):
    """Identical observer and C1 pass-through on normal finite sensor inputs."""

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, incumbent = C1.update(self, t, th, thd, tau_m, tau_s, grip)
        sample = self.grip_force.update(t, th, tau_s, bool(grip))
        value = getattr(sample, 'force_n', None)
        vector = np.asarray(value) if value is not None else None
        observed = (vector is not None and vector.shape == (3,) and vector.dtype.kind in 'fiu'
                    and np.isfinite(vector).all())
        self.grip_force_value = vector.copy() if observed else None
        self.table_normal_n = admissible_normal(self._shared)
        load_owned, self.load_branch = support_owner(
            self.box_mass, self.normal_resolution, getattr(self.ctrl, 'shared_estimator', None))
        self.load_ownership = bool(grip) and load_owned and bool(observed) and (
            self.table_normal_n is not None and self.table_normal_n <= self.normal_resolution)
        self.grip_force_extra = self.grip_episode.feed(self.grip_force_value, self.load_ownership, incumbent)
        return residual, bool(incumbent or self.grip_force_extra)
