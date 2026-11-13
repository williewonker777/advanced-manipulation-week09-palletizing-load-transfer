"""Normal-only experimental parallel J1 channel, never a controller input.

Three samples of motor tracking error yield a slope delayed by one millisecond.
Adding the current analytic reference speed does not remove that error lag.
No contact state, true link state, sensor nuisance parameters or labels are read.
"""
from collections import deque

import numpy as np

import r1_detector
from sim import config as C, trajectory as T


class NoiseEstimator(r1_detector.SharedEstimator):
    def reset(self):
        super().reset()
        self.error_history = deque(maxlen=3)

    def _estimate_state(self, t, th, thd, tau):
        q_ref, qd_ref, _ = T.reference(t)
        error = np.asarray(th) - q_ref
        self.error_history.append(error.copy())
        if len(self.error_history) == 3:
            thd = qd_ref + (self.error_history[-1] - self.error_history[0]) / (2*C.DT)
        return super()._estimate_state(t, th, thd, tau)


def grid_index(t):
    index = int(round(float(t) / C.DT))
    if 0 <= index < 6000 and abs(index*C.DT-float(t)) <= 1e-12:
        return index
    raise ValueError("normal-only mean model requires the registered 1 ms grid")


class NormalChannel:
    """One independent state; fit/calibration arrays are supplied after sealing."""
    def __init__(self, masses, means, thresholds):
        self.masses = np.asarray(masses, float).copy()
        self.means = np.asarray(means, float).copy()
        self.thresholds = np.asarray(thresholds, float).copy()
        if (self.masses.shape != (2,) or self.means.shape != (2, 6000)
                or self.thresholds.shape != (240,)
                or not np.all(np.isfinite(self.means))
                or not np.all(np.isfinite(self.thresholds))
                or self.masses[1] <= self.masses[0]
                or np.any(self.thresholds < 0.5)):
            raise ValueError("invalid normal calibration")
        for array in (self.masses, self.means, self.thresholds):
            array.setflags(write=False)
        self.reset()

    def reset(self):
        self.count = self.clear = 0
        self.detected = False

    def update(self, t, mass, residual):
        index = grid_index(t)
        if not 3.6 <= t < 4.0:
            self.reset()
            return False
        weight = (float(mass)-self.masses[0])/(self.masses[1]-self.masses[0])
        expected = self.means[0, index] + weight*(self.means[1, index]-self.means[0, index])
        over = bool(abs(float(residual)-expected) > self.thresholds[index//25])
        self.count = self.count + 1 if over else 0
        self.clear = 0 if over else self.clear + 1
        if self.count >= 4:
            self.detected = True
        elif self.clear >= 8:
            self.detected = False
        return bool(self.detected)
