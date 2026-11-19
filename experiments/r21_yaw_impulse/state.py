"""Estimate external base-axis impulse without differentiating joint velocity."""
from collections import deque

import numpy as np
from sim import config as C

AVERAGE = 12
SEPARATION = 32


class YawBalance:
    def __init__(self):
        self.dt = float(C.DT)
        self.elastic = np.asarray(C.ELASTIC, bool)
        k, d = C.K[self.elastic], C.D[self.elastic]
        self.decay = d / (d + k * self.dt)
        self.in_gain = self.dt / (d + k * self.dt)
        self.phi = None
        self.prev = None
        self.mass_row = None
        self.grip = None
        self.integral = 0.
        self.terms = deque(maxlen=AVERAGE)
        self.history = deque(maxlen=SEPARATION + 1)

    def feed(self, th, tau, mass_row, grip):
        # D*phi' + K*phi = -tau_link. Only public encoder & calibrated JTS.
        measured = np.asarray(tau)[self.elastic]
        q = np.asarray(th, dtype=float).copy()
        if self.phi is None:
            self.phi = -measured / C.K[self.elastic]
        else:
            self.phi = self.decay * self.phi - self.in_gain * measured
        q[self.elastic] += self.phi
        row = np.asarray(mass_row, dtype=float)
        if self.prev is None or self.grip != bool(grip):
            self.terms.clear()
            self.history.clear()
            self.integral = 0.
            value = None
        else:
            # M[0] dq/dt is vertical angular momentum of links plus known carried
            # payload. For the cyclic base yaw, gravity has no generalized torque.
            p = float((self.mass_row + row) @ (q - self.prev)) / (2 * self.dt)
            self.integral += float(tau[0]) * self.dt
            self.terms.append(p - self.integral)
            value = None
            if len(self.terms) == AVERAGE:
                self.history.append(sum(self.terms) / AVERAGE)
                if len(self.history) > SEPARATION:
                    value = (self.history[-1] - self.history[0]) / (SEPARATION * self.dt)
        self.mass_row = row.copy()
        self.grip = bool(grip)
        self.prev = q
        return value
