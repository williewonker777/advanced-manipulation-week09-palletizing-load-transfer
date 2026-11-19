"""R1 plus frozen geometric coordinates and independent normal torque feedforward."""
from __future__ import annotations

import numpy as np

import r1_servo
import r1_torque
from experiments.r11_datum.controller import frame
from experiments.r11_load_ilc.controller import profiles as geometric_profiles
from sim import trajectory as T


def smooth(t, start, stop):
    if not stop > start:
        raise ValueError('positive transition duration required')
    s = np.clip((np.asarray(t)-start)/(stop-start), 0.0, 1.0)
    return np.stack((s**3*(10.0-15.0*s+6.0*s*s),
                     30.0*s*s*(1.0-s)**2/(stop-start),
                     60.0*s*(1.0-s)*(1.0-2.0*s)/(stop-start)**2), axis=-1)


def pressure_profile(t):
    """Unit C2 pressure plateau; value/velocity/acceleration columns."""
    t = np.asarray(t)
    start, full = T.T_RELEASE-0.220, T.T_RELEASE-0.080
    off_start, off_stop = T.T_RELEASE+0.020, T.T_RELEASE+0.120
    rising = smooth(t, start, full)
    falling = np.array([1.0, 0.0, 0.0])-smooth(t, off_start, off_stop)
    return np.where((t < full)[..., None], rising,
                    np.where((t <= off_start)[..., None],
                             np.array([1.0, 0.0, 0.0]), falling))


class _NormalWrench:
    geometric_amplitudes_mm = (0.0, 0.0)
    pressure_fraction = 0.0

    def reset(self):
        super().reset()
        geometry = np.asarray(self.geometric_amplitudes_mm, dtype=float)
        pressure = float(self.pressure_fraction)
        if (geometry.shape != (2,) or not np.isfinite(geometry).all()
                or not np.isfinite(pressure)):
            raise ValueError('two geometric coefficients and one pressure required')
        mass_scale = np.clip(self.box_mass/2.0, 0.0, 1.0)
        self._geometry_m = 1e-3*geometry*mass_scale
        self._pressure = pressure*mass_scale
        _, _, jacobian = frame(self.nom, T.reference(T.T_RELEASE)[0], True)
        if np.linalg.cond(jacobian) > 1e5:
            raise ValueError('placement Jacobian is ill-conditioned')
        self._vertical = np.linalg.solve(jacobian, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.max_abs_pressure_torque_nm = np.zeros(6)

    def _reference(self, t, *args):
        reference = super()._reference(t, *args)
        correction = geometric_profiles(t) @ self._geometry_m
        return tuple(value + gain*self._vertical
                     for value, gain in zip(reference, correction))

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        command = super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
        q = np.asarray(th)-self.inv_k*np.asarray(tau_s)
        _, _, jacobian = frame(self.nom, q, True)
        fraction = self._pressure*pressure_profile(t)[0]
        torque = jacobian[2]*fraction*self.box_mass*9.81
        self.max_abs_pressure_torque_nm = np.maximum(
            self.max_abs_pressure_torque_nm, np.abs(torque))
        if self.MODE == 'servo':
            position, velocity, feedforward_velocity, feedforward_torque = command
            return (position, velocity, feedforward_velocity,
                    feedforward_torque+torque)
        return command+torque


class Servo(_NormalWrench, r1_servo.Controller):
    pass


class Torque(_NormalWrench, r1_torque.Controller):
    pass
