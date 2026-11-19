"""Small pitch excitation about the *nominal TCP*. Private simulated experiment, not a submission."""
from functools import lru_cache
import numpy as np
from sim import config, trajectory
from sim.nominal import Nominal
import r1_servo
import r1_torque
from experiments.r11_datum.controller import frame

START, STOP, AMPLITUDE = 1.98, 3.40, np.deg2rad(1.5)
DT = .001


def angle(t):
    if not START < t < STOP:
        return 0.
    u = (t - START) / (STOP - START)
    return AMPLITUDE * np.sin(np.pi * u)**4 * np.sin(6 * np.pi * u)


def solve(nom, q0, theta):
    if theta == 0.:
        return np.zeros(6)
    p, r = frame(nom, q0)
    c, s = np.cos(theta), np.sin(theta)
    goal = r @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    q = q0.copy()
    for _ in range(6):
        x, rot, jac = frame(nom, q, True)
        # sin(angle)*axis is a stable local orientation error at <= 1.5 deg.
        error = np.r_[p - x, .5 * np.cross(rot.T, goal.T).sum(axis=0)]
        if np.linalg.norm(error) < 2e-13:
            return q - q0
        q += np.linalg.solve(jac, error)
    raise ValueError('small TCP-preserving tilt did not converge')


@lru_cache(maxsize=1)
def table():
    if abs(config.DT - DT) > 1e-12:
        raise ValueError('probe requires the native 1ms clock')
    lo, hi = round(START/DT), round(STOP/DT)
    nom = Nominal()
    d = np.array([solve(nom, trajectory.reference(k*DT)[0], angle(k*DT)) for k in range(lo, hi+1)])
    p = np.pad(d, ((2, 2), (0, 0)))
    v = (p[:-4] - 8*p[1:-3] + 8*p[3:-1] - p[4:])/(12*DT)
    a = (-p[:-4] + 16*p[1:-3] - 30*p[2:-2] + 16*p[3:-1] - p[4:])/(12*DT**2)
    v[[0,-1]] = 0.
    a[[0,-1]] = 0.
    data = np.stack((d, v, a))
    data.flags.writeable = False
    return data


def correction(t):
    if not START < t < STOP:
        return np.zeros((3,6))
    data = table()
    k = (t-START)/DT
    nearest = round(k)
    if abs(k-nearest) < 2e-11:
        return data[:,nearest]
    i = int(np.floor(k))
    s = k-i
    q, v, a = data[:,i]
    q1, v1, a1 = data[:,i+1]
    c1 = DT*v
    c2 = .5*DT**2*a
    e, f, g = q1-(q+c1+c2), DT*v1-(c1+2*c2), DT**2*a1-2*c2
    c3, c4, c5 = 10*e-4*f+.5*g, -15*e+7*f-g, 6*e-3*f+.5*g
    return np.array([q+s*(c1+s*(c2+s*(c3+s*(c4+s*c5)))),
                     (c1+s*(2*c2+s*(3*c3+s*(4*c4+s*5*c5))))/DT,
                     (2*c2+s*(6*c3+s*(12*c4+s*20*c5)))/DT**2])


class _Probe:
    def reset(self):
        super().reset()
        if self.box_mass > 0:
            table()  # build only from nominal taught poses; outside control loop

    def _reference(self, t, *args):
        original = super()._reference(t, *args)
        ta = t + self.dt
        if self.box_mass <= 0 or not START < ta < STOP:
            return original
        delta = correction(ta)
        return tuple(q + d for q, d in zip(original, delta))


class Servo(_Probe, r1_servo.Controller):
    pass


class Torque(_Probe, r1_torque.Controller):
    pass
