"""Hand off strong or still-rising pickup loads; report only otherwise-unclaimed small settled loads."""
from collections import deque
import numpy as np

from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r33_pick_vertical.candidate import parameters
from experiments.r33_pick_vertical.signal import LiftSignal
from experiments.r33_pick_vertical.state import expected

class Route:
    def __init__(self):
        p=parameters()
        self.signal=LiftSignal();self.medians=np.asarray(p['medians']);self.floor=p['margin_N']
        self.limit={-1:p['negative_limit_N'],1:p['positive_limit_N']}
        self.blocks=deque(maxlen=16);self.finished=False
        self.reset()
    def reset(self):
        self.announced=self.competing=False
        self.sign=self.quiet=0
    def feed(self,det,t,th,thd,tau_m,tau_s,grip,parent):
        s=self.signal.feed(det,t,th,tau_s,grip)
        if not grip:
            self.blocks.clear();self.finished=False;self.reset();return False
        if parent:self.competing=True
        if s is not None and s.height>.035:self.finished=True
        if s is None or self.finished:
            self.blocks.clear();return False
        d=np.array((s.inverse,s.momentum))-expected(s.height,self.medians)
        if not np.isfinite(d).all():
            self.blocks.clear();return False
        self.blocks.append(d)
        if len(self.blocks)<16:return False
        x=np.asarray(self.blocks);a=x[:8].mean(0);b=x[8:].mean(0)
        sign=1 if b[0]>=0 else -1;lo=self.limit[sign]
        if np.min(sign*b)<lo/2:
            self.quiet+=1
            if self.quiet>=8:
                self.reset();self.competing=bool(parent)
            return False
        self.quiet=0
        if self.sign and sign!=self.sign:self.reset();self.competing=bool(parent)
        self.sign=sign
        cap=det._shared.force_threshold[2]
        if sign*b[0]>=cap or sign*a[0]>=cap:
            self.competing=True
        if self.competing or self.announced or min(np.min(sign*a),np.min(sign*b))<=lo:return False
        # A causal plateau certificate rejects the early low-amplitude part of
        # larger forces, without a fixed refractory interval or force schedule.
        if abs(b[0]-a[0])>max(self.floor,abs(b[0])/8):return False
        self.announced=True;return True

class Detector(Parent):
    def reset(self):
        super().reset();self.pick_route=Route() if getattr(self.ctrl,'MODE',None)=='torque' else None
    def update(self,t,th,thd,tau_m,tau_s,grip):
        r,native=super().update(t,th,thd,tau_m,tau_s,grip)
        extra=self.pick_route is not None and self.pick_route.feed(self,t,th,thd,tau_m,tau_s,grip,native)
        return r,bool(native or extra)
