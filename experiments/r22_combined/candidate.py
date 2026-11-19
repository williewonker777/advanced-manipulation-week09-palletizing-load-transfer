"""Core plus the pre-fit world-tangential force channel. No plant data or force schedule."""
import math

from sim import config as C
from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r22_combined.core import Detector as Core
from experiments.r22_tangent.frozen import Detector as TangentCalibration
from experiments.r22_tangent.state import Evidence, load_class


class Detector(Core):
    def reset(self):
        super().reset()
        self.tangent = Evidence()
        self.tangent_force = NormalForce(self.nom, self.nom_box, dt=C.DT, viscous_spring=C.D, jts_hz=C.JTS_CUTOFF)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        r, core = super().update(t, th, thd, tau_m, tau_s, grip)
        sample = self.tangent_force.update(t, th, tau_s, grip)
        group = load_class(self.box_mass, grip)
        bound = math.inf if group is None else TangentCalibration.bounds_tangent[group]
        flag = self.tangent.feed(None if sample is None else sample.force_n, group, bound)
        return r, bool(core or flag)
