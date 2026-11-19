"""Preserve the yaw impulse history across a modeled payload release."""
from collections import deque

import numpy as np

from experiments.r21_yaw_impulse.state import AVERAGE, YawBalance


class ReleaseBalance(YawBalance):
    """Remove outgoing modeled momentum instead of clearing sensor history."""

    def __init__(self, mass):
        super().__init__()
        self.loaded = float(mass) > 0
        self._positions = deque(maxlen=AVERAGE + 1)

    def feed(self, th, tau, mass_row, grip):
        switching = self.prev is not None and self.grip != bool(grip)
        if switching and not self.loaded:
            # An empty gripper command changes no modeled payload.
            self.grip = bool(grip)
        elif switching and not grip and len(self._positions) == self._positions.maxlen:
            # Use only positions preceding the release.  The inertia-row change
            # is already part of the public R10 model packet.
            dq = (self._positions[-1] - self._positions[0]) / (AVERAGE * self.dt)
            self.integral += float((np.asarray(mass_row) - self.mass_row) @ dq)
            self.mass_row = np.asarray(mass_row).copy()
            self.grip = False
        value = super().feed(th, tau, mass_row, grip)
        self._positions.append(self.prev)
        return value
