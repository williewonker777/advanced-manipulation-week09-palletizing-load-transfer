"""Research-only localized weak shear supplement to the R31 alarm."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.r31_held_identity.candidate import Detector as Parent
from .field import Bank,RAYS,Windows,coordinates,row_vector,scores
from .signal import Signal

CALIBRATION='9000b3f34c32d4cc4437aa19ba7b44d121c9d855dc056cb57b3618f8d0f7143f'
def parameters():
    f=Path(__file__).with_name('calibration.json')
    b=f.read_bytes()
    if hashlib.sha256(b).hexdigest()!=CALIBRATION:raise ValueError('changed normal force field')
    return json.loads(b)

class Event:
    def __init__(self,p):
        self.bank=Bank(p['bank']);self.limits=np.asarray(p['thresholds_N'])
        self.win=Windows();self.region=0;self.reset_source()
    def reset_source(self):
        self.native=self.reported=False;self.quiet=self.remaining=0
    def feed(self,s,native,native_limit,persistence):
        if s is None:
            self.win.clear();self.region=0;self.reset_source();return False
        if s.mode!=self.region:
            self.win.clear();self.reset_source();self.region=s.mode
        if native:self.native=True
        center=self.bank.center(s.mode,coordinates(s.mode,s.progress_m,s.z_m))
        if center is None or (s.mode==1 and s.progress_m<.003):
            self.win.clear();self.remaining=0;return False
        blocks=self.win.feed(s.mode,row_vector(s)-center)
        if blocks is None:return False
        a,b=blocks;sc,past,present=scores(a,b);limits=self.limits[s.mode-1]
        ratio=sc/limits;idx=int(np.argmax(ratio))
        if np.max(ratio)<.45:
            self.quiet+=1
            if self.quiet>=8:self.reset_source();self.native=bool(native)
            return False
        self.quiet=0
        # The original route gets a veto for strong forces or an excursion
        # it has already noticed. Do not manufacture a second alarm on its tail.
        if max(np.linalg.norm(a[:3]),np.linalg.norm(b[:3]))>=native_limit/2:
            self.native=True
        if self.native or ratio[idx]<=1:return False
        if np.linalg.norm(b[:2])<=abs(b[2]):return False
        if not self.reported:
            if abs(present[0,idx]-past[0,idx])>max(.75,present[0,idx]/8):return False
            self.reported=True;self.remaining=persistence
        if self.remaining>0:self.remaining-=1;return True
        return False

class Route:
    def __init__(self):
        self.sensor=Signal();self.event=Event(parameters());self.latest=None
    def feed(self,d,t,th,thd,tm,ts,grip,parent):
        s=self.sensor.feed(d,t,th,ts,grip)
        if s is not None and s.mode==1 and self.sensor.stage.max_z-self.sensor.stage.start[2]>.060:
            s=None
        self.latest=s
        return self.event.feed(s,bool(parent),float(d._shared.force_threshold[0]),d._shared.persistence)

class Detector(Parent):
    def reset(self):
        super().reset();self.shear=Route() if getattr(self.ctrl,'MODE',None)=='torque' else None
        self.parent_event=self.shear_event=False
    def update(self,t,th,thd,tm,ts,grip):
        r,base=super().update(t,th,thd,tm,ts,grip)
        self.parent_event=bool(base);self.shear_event=self.shear is not None and self.shear.feed(self,t,th,thd,tm,ts,grip,base)
        return r,bool(base or self.shear_event)
