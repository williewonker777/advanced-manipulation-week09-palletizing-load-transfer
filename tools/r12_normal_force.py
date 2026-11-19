"""Causal, delayed virtual vertical force for the course's nominal 6-axis arm.

Uses only motor encoder, joint-torque sensor, grip and the two public nominal
models. It is an estimate of *all* vertical external wrench, not a contact
classifier. The delay is part of the output contract. No force clipping.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


N = 31
# Encoder is a nodal state. Rigid-axis JTS at that state was formed from
# the previous solved interval. f_table[j] is logged after solving step j.
STATE_LAG = 2 + N // 2
LAG = STATE_LAG + 1
X = np.linspace(-1., 1., N)
KERNEL = (1. - X * X) ** 2
KERNEL /= KERNEL.sum()


@dataclass(frozen=True)
class Sample:
    time: float
    effective_time: float
    vertical_n: float


def vertical_displacement(model, q):
    """Virtual displacement that admits vertical force and rejects other wrenches."""
    import mujoco
    model.d.qpos[:] = q
    mujoco.mj_kinematics(model.m, model.d)
    mujoco.mj_comPos(model.m, model.d)
    jp, jw = np.zeros((3, 6)), np.zeros((3, 6))
    mujoco.mj_jacSite(model.m, model.d, jp, jw, model.m.site('tcp').id)
    jac = np.vstack((jp, jw))
    if np.linalg.cond(jac) > 1e4:
        return None
    return np.linalg.solve(jac, [0., 0., 1., 0., 0., 0.])


class NormalForce:
    """A fresh instance per episode, owning its nominal workspaces."""
    def __init__(self, empty, loaded, *, dt, viscous_spring, jts_hz):
        if not np.isclose(dt, .001, rtol=0, atol=1e-12):
            raise ValueError('this design requires 1 ms samples')
        self.empty, self.loaded, self.dt = empty, loaded, dt
        self.elastic = np.asarray(empty.elastic, bool)
        self.k = np.asarray(empty.K)[self.elastic]
        d = np.asarray(viscous_spring)[self.elastic]
        if np.any(self.k <= 0) or np.any(d <= 0) or jts_hz <= 0:
            raise ValueError('strictly positive elastic and sensor parameters required')
        self.spring_memory = np.exp(-dt * self.k / d)
        self.unfilter = 1. / np.expm1(2. * np.pi * jts_hz * dt)
        self.prev_th = self.prev_sensor = self.deflection = None
        self.clock = self.prev_grip = None
        self.pos = deque(maxlen=3)
        self.residual = deque(maxlen=N)
        self.midpos = deque(maxlen=N)
        self.offsets = []
        self.bias = None

    def update(self, t, encoder, jts, grip, *, project=True):
        """Return Sample or None (no full window/calibration, singular, or no load).

        Call on every regular sample. project=False skips only the output
        projection; it neither skips samples nor changes the integration.
        """
        th, raw = np.asarray(encoder, float), np.asarray(jts, float)
        if (not np.isfinite(t) or th.shape != (6,) or raw.shape != (6,)
                or not np.isfinite(th).all() or not np.isfinite(raw).all()):
            raise ValueError('finite time and two finite six-axis signals required')
        if self.clock is not None and abs(t - self.clock - self.dt) > 1e-8:
            raise ValueError('a dropped or reordered sample requires a new observer')
        self.clock = t
        # A change of payload invalidates every differentiation/averaging stencil
        # that straddles it. Bias belongs to the unloaded arm, not the payload.
        if self.prev_grip != bool(grip):
            self.pos.clear()
            self.residual.clear()
            self.midpos.clear()
            self.prev_th = self.prev_sensor = self.deflection = None
        self.prev_grip = bool(grip)
        if self.prev_th is None:
            self.prev_th, self.prev_sensor = th.copy(), raw.copy()
            return None
        aligned = raw + self.unfilter * (raw - self.prev_sensor)
        previous = self.prev_th.copy()
        self.prev_th, self.prev_sensor = th.copy(), raw.copy()
        equilibrium = aligned[self.elastic] / self.k
        self.deflection = (equilibrium if self.deflection is None else
                           equilibrium + self.spring_memory * (self.deflection - equilibrium))
        previous[self.elastic] -= self.deflection
        self.pos.append((previous, aligned))
        if len(self.pos) < 3:
            return None
        (qa, _), (q, tau), (qb, _) = self.pos
        v = (qb - qa) / (2. * self.dt)
        acc = ((qb - q) - (q - qa)) / self.dt**2
        model = self.loaded if grip else self.empty
        self.residual.append(model.inverse_dynamics(q, v, acc) - tau)
        self.midpos.append(q)
        if len(self.residual) != N:
            return None
        state_time = t - STATE_LAG * self.dt
        # MuJoCo's nominal inverse dynamics is affine in acceleration. Integrate
        # the residual before projection; endpoints and slope of the kernel
        # vanish, so isolated quantization differences are not used as force.
        averaged = KERNEL @ np.asarray(self.residual)
        if self.bias is None:
            if not grip and .075 <= state_time < .425:
                self.offsets.append(averaged)
            elif state_time >= .425 and len(self.offsets) >= 300:
                self.bias = np.median(self.offsets, axis=0)
                self.offsets.clear()
        if not (project and grip) or self.bias is None:
            return None
        s = vertical_displacement(model, self.midpos[N // 2])
        if s is None:
            return None
        return Sample(float(t), float(t-LAG*self.dt), float(s @ (averaged - self.bias)))
