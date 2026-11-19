"""Two causal force estimates on the first observable lift away from the grasp height."""
from dataclasses import dataclass
import numpy as np
from experiments.r15_delayed_wrench.observer import N, NormalForce
from sim import config as C

@dataclass(frozen=True)
class Sample:
    height: float
    inverse: float
    momentum: float

class LiftSignal:
    """No intended trajectory or simulator contact state. All distances are in metres."""
    def __init__(self):
        self.previous=False;self.origin=None;self.done=False;self.observer=None

    def feed(self, det,t,th,tau_s,grip):
        grip=bool(grip)
        if not grip:
            self.previous=False;self.done=False;self.origin=None;self.observer=None
            return None
        model=det.nom
        if not self.previous:
            q=det._shared.q_prev
            self.origin=None if q is None else float(model.tcp_position(q)[2])
            bias=getattr(det,'static_bias',None)
            if self.origin is not None and bias is not None:
                self.observer=NormalForce(model,det.nom_box,dt=C.DT,viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)
                self.observer.bias=np.asarray(bias).copy()
            self.previous=True
        if self.done or self.observer is None:return None
        sample=self.observer.update(t,th,tau_s,True)
        if sample is None:return None
        # The inverse solver has already aligned the nominal loaded workspace
        # with the middle of its causal encoder stencil; no second Jacobian.
        loaded=self.observer.loaded
        h=float(loaded.d.site_xpos[loaded.m.site('tcp').id,2])-self.origin
        if h>.100:
            self.done=True;self.observer=None;return None
        if h<.003:return None
        return Sample(h,float(sample.force_n[2]),float(det._shared.last_wrench[2]))
