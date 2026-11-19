"""Direct exposed comparator: unchanged R23 detached XY measurement on R14."""
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r23_detached.candidate import ReleasedForce


class Detector(Coupled):
    def reset(self):
        super().reset()
        self.xy = ReleasedForce(self.box_mass,self.nom,self.nom_box,True)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual, incumbent = super().update(t,th,thd,tau_m,tau_s,grip)
        add = self.xy.feed(t,th,tau_s,grip,incumbent)
        return residual, bool(incumbent or add)
