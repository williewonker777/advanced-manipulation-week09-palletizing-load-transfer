"""Prepare the stored normal contact energy *before* scheduled detachment.

The last free-box state cannot be corrected after a weld opens. Change only a
submillimetre normal reference on a positively confirmed support handoff.
"""
import numpy as np

from experiments.r15_early_com.controller import Controller as Early
from sim import trajectory as T


GRAVITY = 9.81
NORMAL_ACCELERATION_BUDGET = .05 * GRAVITY


def _smooth(s):
    return s**3 * (10. + s * (-15. + 6.*s))


def _smooth_d(s):
    return 30.*s**2*(1.-s)**2


def release_motion(t, start, release, settle):
    """Return displacement, speed, acceleration in SI, all C2 at joins."""
    t = float(t)
    horizon = release - start
    if not 0. < horizon or not 0. < settle:
        raise ValueError('positive approach and return horizons required')
    if t <= start or t >= release + settle:
        return 0., 0., 0.
    # Peak approach acceleration is (1.875/2) * the acceleration budget.
    speed = .5 * NORMAL_ACCELERATION_BUDGET * horizon
    if t < release:
        s = (t-start)/horizon
        primitive = s**4*(2.5+s*(-3.+s))
        return speed*horizon*primitive, speed*_smooth(s), speed*_smooth_d(s)/horizon
    u = (t-release)/settle
    height = .5*speed*horizon
    p = 1.-_smooth(u)
    dp = -_smooth_d(u)
    ddp = -60.*u*(1.-u)*(1.-2.*u)
    v = u*(1.+u**2*(-6.+u*(8.-3.*u)))
    dv = 1.+u**2*(-18.+u*(32.-15.*u))
    ddv = u*(-36.+u*(96.-60.*u))
    return (height*p + speed*settle*v,
            height*dp/settle + speed*dv,
            height*ddp/settle**2 + speed*ddv/settle)


class Controller(Early):
    def reset(self):
        super().reset()
        self.relief_start = None
        self.relief_axis = None
        self.relief_position_m = 0.

    def _reference(self, t):
        q, qd, qdd = super()._reference(t)
        blend = self.parameters.blend_duration
        if (self.relief_start is None and self.box_mass > 0. and self.hybrid_active
                and T.T_RELEASE - 2.*blend <= t <= T.T_RELEASE - blend):
            # _reference is called before this tick's support update by the
            # parent. Thus only previously confirmed, not future contact is used.
            anchor = self._reference_at(T.T_RELEASE)[0]
            _, _, jac = self._flange_pose_jacobian(anchor)
            axis = np.linalg.lstsq(jac, np.array([0., 0., 1., 0., 0., 0.]), rcond=1e-8)[0]
            if (np.isfinite(axis).all() and np.linalg.norm(axis) <= 25.
                    and np.linalg.norm(jac@axis-[0.,0.,1.,0.,0.,0.]) < 1e-6):
                self.relief_start, self.relief_axis = float(t), axis
        if self.relief_start is None:
            return q, qd, qdd
        z, v, a = release_motion(t, self.relief_start, T.T_RELEASE,
                                 2.*self.parameters.release_return)
        self.relief_position_m = z
        if z == v == a == 0.:
            return q, qd, qdd
        return q+self.relief_axis*z, qd+self.relief_axis*v, qdd+self.relief_axis*a
