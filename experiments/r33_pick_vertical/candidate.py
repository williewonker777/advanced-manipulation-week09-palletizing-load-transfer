"""Optional R31 plus an event notification from a measured, support-cleared vertical pickup."""
import hashlib
import json
from pathlib import Path

from experiments.r31_held_identity.candidate import Detector as Parent
from .signal import LiftSignal
from .state import Evidence

CALIBRATION='fc2e4054e58e442d1cc2c3c872bcd8d05a590bdf55e3377c2c120683687b088f'

def parameters():
    path=Path(__file__).with_name('calibration.json')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CALIBRATION:raise ValueError('pickup calibration changed')
    return json.loads(path.read_bytes())

class Route:
    def __init__(self):
        p=parameters()
        self.signal=LiftSignal();self.evidence=Evidence(p['medians'])
        self.limits={1:p['positive_limit_N'],-1:p['negative_limit_N']}
        self.reset_event();self.ended=False
    def reset_event(self):
        self.sign=0;self.pending=0;self.owned=False;self.quiet=0;self.native=False
    def feed(self,det,t,th,thd,tau_m,tau_s,grip,parent):
        sample=self.signal.feed(det,t,th,tau_s,grip)
        if not grip:
            self.ended=False;self.reset_event();self.evidence.buffer.clear();return False
        if sample is not None and sample.height>.035:self.ended=True
        if self.ended:return False
        if parent:self.native=True
        value=self.evidence.feed(sample)
        if value is None:return False
        s=1 if value>=0 else -1
        high=abs(value)>self.limits[s]
        if high:
            if self.sign and self.sign!=s:
                self.owned=False;self.native=bool(parent);self.pending=0
            self.sign=s;self.pending+=1;self.quiet=0
            if self.pending>=3 and not self.owned:
                self.owned=True
                return not self.native
        else:
            self.pending=0
            self.quiet=self.quiet+1 if abs(value)<self.limits[s]*.5 else 0
            if self.quiet>=8:self.reset_event();self.native=bool(parent)
        return False

class Detector(Parent):
    def reset(self):
        super().reset();self.pick_route=Route() if getattr(self.ctrl,'MODE',None)=='torque' else None
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,native=super().update(t,th,thd,tau_m,tau_s,grip)
        alarm=(self.pick_route is not None and self.pick_route.feed(self,t,th,thd,tau_m,tau_s,grip,native))
        return residual,bool(native or alarm)
