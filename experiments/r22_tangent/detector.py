"""A separate research witness, not a default controller or an R21 change."""
import math

from sim import config as C
from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r21_yaw_impulse.coarse import Detector as Yaw
from experiments.r22_tangent.state import Evidence, load_class


class Detector(Yaw):
    bounds_tangent = (math.inf,) * 4

    def reset(self):
        super().reset()
        self.tangent_force = NormalForce(self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)
        self.tangent = Evidence()

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, baseline = super().update(t, th, thd, tau_m, tau_s, grip)
        s = self.tangent_force.update(t, th, tau_s, grip)
        g = load_class(self.box_mass, grip)
        flag = self.tangent.feed(None if s is None else s.force_n, g, math.inf if g is None else self.bounds_tangent[g])
        return residual, bool(baseline or flag)
