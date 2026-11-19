"""Research-only grasp-datum registration. No plant, force labels, or alarms."""
import mujoco
import numpy as np

import r1_servo
import r1_torque
from sim import trajectory as T


def smooth(t, start, stop):
    """C2 unit transition, with derivatives in inverse seconds."""
    if not stop > start:
        raise ValueError('positive transition duration required')
    s = float(np.clip((t - start) / (stop - start), 0., 1.))
    return np.array([s**3*(10.-15.*s+6.*s*s),
                     30.*s*s*(1.-s)**2 / (stop-start),
                     60.*s*(1.-s)*(1.-2.*s) / (stop-start)**2])


def frame(nom, q, jacobian=False):
    """Public nominal TCP frame; link angles supplied by the caller."""
    nom.d.qpos[:] = q
    mujoco.mj_kinematics(nom.m, nom.d)
    site = nom.m.site('tcp').id
    pos = nom.d.site_xpos[site].copy()
    rot = nom.d.site_xmat[site].reshape(3, 3).copy()
    if not jacobian:
        return pos, rot
    mujoco.mj_comPos(nom.m, nom.d)
    jv = np.zeros((3, 6))
    jw = np.zeros_like(jv)
    mujoco.mj_jacSite(nom.m, nom.d, jv, jw, site)
    return pos, rot, np.vstack((jv, jw))


def normal_datum(pick, observed, place_rot):
    p0, r0 = pick
    pm, rm = observed
    rotated = place_rot @ r0.T
    dz = (rotated @ (pm - p0))[2]
    # sin(angle)*axis: first order for a measured, tightly bounded correction.
    drift = .5 * sum((np.cross(r0[:, i], rm[:, i]) for i in range(3)), np.zeros(3))
    tilt = rotated @ drift
    tilt[2] = 0.
    linear = np.array([0., 0., dz])
    scale = min(1., .0015 / max(abs(dz), 1e-15),
                np.deg2rad(.4) / max(np.linalg.norm(tilt), 1e-15))
    return scale * np.r_[linear, tilt], float(scale)


class _Datum:
    def reset(self):
        super().reset()
        self._was_gripped = False
        self._dq = None
        self.datum = None
        self._pick = frame(self.nom, T.reference(T.T_GRASP)[0])
        _, self._place_rot, self._place_jac = frame(self.nom, T.reference(T.T_RELEASE)[0], True)
        if np.linalg.cond(self._place_jac) > 1.e5:
            raise ValueError('placement Jacobian too ill-conditioned for registration')
        self._start = max(p[0] for p in T.WAYPOINTS if p[0] < T.T_RELEASE)

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        if self.box_mass > 0 and grip and not self._was_gripped:
            measured = frame(self.nom, np.asarray(th) - self.inv_k * tau_s)
            twist, scale = normal_datum(self._pick, measured, self._place_rot)
            self._dq = np.linalg.solve(self._place_jac, twist)
            self.datum = (twist.copy(), scale)
        self._was_gripped = bool(grip)
        return super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)

    def _reference(self, t, *args):
        reference = super()._reference(t, *args)
        if self._dq is None or t < self._start or t >= T.T_RELEASE + .100:
            return reference
        weights = (smooth(t, self._start, T.T_RELEASE) if t <= T.T_RELEASE else
                   np.array([1., 0., 0.]) - smooth(t, T.T_RELEASE, T.T_RELEASE + .100))
        return tuple(v + w * self._dq for v, w in zip(reference, weights))


class Servo(_Datum, r1_servo.Controller):
    pass


class Torque(_Datum, r1_torque.Controller):
    pass
