"""R1 plus two smooth Cartesian-normal feedforward coordinates."""
from __future__ import annotations

import numpy as np

import r1_servo
import r1_torque
from experiments.r11_datum.controller import frame
from sim import trajectory as T


def smooth(t, start, stop):
    """C2 transition and its first two time derivatives."""
    if not stop > start:
        raise ValueError("positive transition duration required")
    s = np.clip((np.asarray(t)-start)/(stop-start), 0.0, 1.0)
    return np.stack((s**3*(10.0-15.0*s+6.0*s*s),
                     30.0*s*s*(1.0-s)**2/(stop-start),
                     60.0*s*(1.0-s)*(1.0-2.0*s)/(stop-start)**2), axis=-1)


def profiles(t):
    """Transient velocity bump and independent equilibrium plateau, mm units."""
    t = np.asarray(t)
    center, width = T.T_RELEASE-0.080, 0.120
    x = (t-center)/width
    keep = np.abs(x) < 1.0
    clipped = np.clip(x, -1.0, 1.0)
    h = 1.0-clipped*clipped
    transient = np.stack((h**3,
                          -6.0*clipped*h*h/width,
                          (-6.0+36.0*clipped**2-30.0*clipped**4)/width**2),
                         axis=-1)*keep[..., None]

    start, full = T.T_RELEASE-0.220, T.T_RELEASE-0.080
    off_start, off_stop = T.T_RELEASE+0.020, T.T_RELEASE+0.120
    rising = smooth(t, start, full)
    falling = np.array([1.0, 0.0, 0.0])-smooth(t, off_start, off_stop)
    plateau = np.where((t < full)[..., None], rising,
                       np.where((t <= off_start)[..., None],
                                np.array([1.0, 0.0, 0.0]), falling))
    return np.stack((transient, plateau), axis=-1)


class _NormalLoad:
    amplitudes_mm = (0.0, 0.0)

    def reset(self):
        super().reset()
        amplitudes = np.asarray(self.amplitudes_mm, dtype=float)
        if amplitudes.shape != (2,) or not np.isfinite(amplitudes).all():
            raise ValueError("two finite normal coefficients required")
        self._amplitudes_m = 1e-3*amplitudes*np.clip(self.box_mass/2.0, 0.0, 1.0)
        _, _, jacobian = frame(self.nom, T.reference(T.T_RELEASE)[0], True)
        if np.linalg.cond(jacobian) > 1e5:
            raise ValueError("placement Jacobian is ill-conditioned")
        self._vertical = np.linalg.solve(jacobian, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def _reference(self, t, *args):
        reference = super()._reference(t, *args)
        correction = profiles(t) @ self._amplitudes_m
        return tuple(value + gain*self._vertical
                     for value, gain in zip(reference, correction))


class Servo(_NormalLoad, r1_servo.Controller):
    pass


class Torque(_NormalLoad, r1_torque.Controller):
    pass
