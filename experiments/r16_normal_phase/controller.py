"""Normal-only, endpoint-preserving phase lead on a registered handoff."""
import numpy as np

from experiments.r14_stock_handoff.controller import Controller as Pose
from experiments.r14_stock_handoff.controller import MAX_HEIGHT_MM
from experiments.r15_early_com.controller import Controller as Early
from sim import config as C, trajectory as T
from sim.nominal import Nominal


def ease(x):
    s = max(0., min(float(x), 1.))
    return s**3*(10.+s*(-15.+6.*s))


def phase_lead(t, start, release, blend):
    if t <= start or t >= release:
        return 0.
    midpoint = .5*(release+start)
    taper = release-2.5*blend
    return .5*blend*ease((t-start)/(midpoint-start))*ease((release-t)/(release-taper))


class Controller(Early):
    def reset(self):
        super().reset()
        self._correction = None
        if self.box_mass <= 0.:
            return
        self._first = round(3.6/C.DT)
        self._last = round(T.T_RELEASE/C.DT)
        dt = float(C.DT)
        # Work on dedicated scratch state. Do not advance or mutate the shared
        # online estimator, and do not solve nonlinear IK in the 1kHz loop.
        kin = Pose.__new__(Pose)
        kin.nom = Nominal()
        kin.ik_max_position_error_m = kin.ik_max_orientation_error_rad = 0.
        n = self._last-self._first+1
        delta = np.zeros((n,6)); commanded = np.zeros(n)
        cap = MAX_HEIGHT_MM*.001/4.
        for i in range(1,n-1):
            t = (self._first+i)*dt
            advance = phase_lead(t,3.6,T.T_RELEASE,self.parameters.blend_duration)
            if not advance:
                continue
            base = self._base_reference(t)[0]
            ahead = self._base_reference(t+advance)[0]
            z = kin.nom.tcp_position(ahead)[2] - kin.nom.tcp_position(base)[2]
            dz = cap*float(np.tanh(z/cap))
            if abs(dz) < 1.e-10:
                continue
            delta[i] = kin._full_pose(base,dz)-base
            commanded[i] = dz
        velocity = np.zeros_like(delta); acceleration = np.zeros_like(delta)
        velocity[1:-1] = (delta[2:]-delta[:-2])/(2.*dt)
        acceleration[1:-1] = (delta[2:]-2.*delta[1:-1]+delta[:-2])/(dt*dt)
        self._correction = (delta,velocity,acceleration)
        self.phase_max_shift_m = float(np.max(abs(commanded)))
        self.phase_ik_error_m = float(kin.ik_max_position_error_m)

    def _reference(self,t):
        q,qd,qdd = super()._reference(t)
        if self._correction is None:
            return q,qd,qdd
        k = round(float(t)/C.DT)
        if not self._first < k < self._last:
            return q,qd,qdd
        dq,dv,da = self._correction
        i = k-self._first
        return q+dq[i],qd+dv[i],qdd+da[i]
