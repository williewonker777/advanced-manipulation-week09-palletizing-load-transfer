"""R31 plus two non-overlapping weak vertical-source notifications on measured first lift and transit."""
from collections import deque
import numpy as np

from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r33_pick_clearance.candidate import Route as OriginalLift
from experiments.r33_carry_vertical.candidate import MODEL
from experiments.r33_carry_vertical.witness import VerticalCarryWitness


class FirstLift(OriginalLift):
    """An impact-induced return through the pickup band is not another first departure."""
    def reset_lift(self):
        super().reset_lift();self.departed=False

    def classify(self,h,*args):
        if self.departed and h<=.035:
            self.over=True;self.blocks.clear();return False
        self.departed |= h>.035
        return super().classify(h,*args)


class Travel:
    """Choose travel by observed lift deceleration and actual departure around the base axis."""
    def __init__(self):
        self.station=None;self.azimuth=None;self.heights=deque(maxlen=9)
        self.peak=0.;self.ready=False

    def feed(self,det,grip,native):
        if not grip:
            if self.station is not None:self.__init__()
            return False
        if self.ready:return True
        q=det._shared.q_prev
        if q is None:return False
        xyz=det.nom.tcp_position(q)
        if self.station is None:
            self.station=xyz.copy();self.azimuth=float(q[0])
        height=float(xyz[2]-self.station[2]);self.heights.append(height)
        rise=height-self.heights[0];self.peak=max(self.peak,rise)
        arc=abs(float(q[0])-self.azimuth)*np.linalg.norm(xyz[:2])
        if (height>=det._shared.height_band and self.peak>0 and rise<=.25*self.peak
                and arc>=.003 and not native):
            self.ready=True
        return self.ready


class AxialCarry:
    """Only a settled source predominantly axial in both sensors may supplement the parent."""
    def __init__(self):
        self.witness=VerticalCarryWitness(MODEL['threshold_N'],MODEL['window_samples'],MODEL['persistence_samples'])
        self.ring=deque(maxlen=2*MODEL['window_samples']);self.base=None
        self.claimed=False;self.remaining=0

    def feed(self,det,eligible,parent):
        projection=det.r27_projection.last_force;packet=det._shared
        valid=(eligible and not packet.last_table_allowed and projection is not None
               and all(w.present for w in det.r27_gate.windows.windows))
        v=np.stack((projection,packet.last_residual_wrench[:3])) if valid else None
        w=self.witness
        if v is None or not np.isfinite(v).all():
            w.reset();self.ring.clear();self.base=None;self.remaining=0;return False
        self.ring.append(v.copy())
        was=w.owned;extra=w.extra(v[0,2],v[1,2],True,parent)
        if not w.owned:
            self.base=None;self.remaining=0;return False
        arr=np.asarray(self.ring)
        if not was:
            self.base=arr[:w.window].mean(0);self.claimed=False;self.remaining=0
        if not extra or self.base is None:return False
        a=arr[-16:-8].mean(0)-self.base
        recent=arr[-8:].mean(0);b=recent-self.base
        sign=w.direction;along=sign*b[:,2]
        cap=packet.out_of_height_fz_threshold if sign>0 else packet.force_threshold[2]
        if max(sign*a[0,2],along[0])>=cap:
            self.claimed=True;self.remaining=0;return False
        if (not np.all(along>np.linalg.norm(b[:,:2],axis=1))
                or not np.all(sign*a[:,2]>w.threshold) or not np.all(along>w.threshold)
                or not np.all(sign*recent[:,2]>0)):
            return False
        if not self.claimed:
            if abs(b[0,2]-a[0,2])>max(w.threshold/4,abs(b[0,2])/8):return False
            self.claimed=True;self.remaining=packet.persistence
        if self.remaining>0:
            self.remaining-=1;return True
        return False


class Detector(Parent):
    def reset(self):
        super().reset()
        self.lift=FirstLift() if getattr(self.ctrl,'MODE',None)=='torque' else None
        self.travel=Travel();self.carry=AxialCarry()
        self.parent_event=self.pick_event=self.carry_event=False

    def update(self,t,th,thd,tau_m,tau_s,grip):
        r,parent=super().update(t,th,thd,tau_m,tau_s,grip)
        self.parent_event=bool(parent);self.pick_event=self.carry_event=False
        if self.lift is not None:
            travel=self.travel.feed(self,grip,parent)
            self.pick_event=bool(self.lift.feed(self,t,th,thd,tau_m,tau_s,grip,parent)) and not travel
            self.carry_event=self.carry.feed(self,travel and bool(grip),bool(parent))
        return r,bool(parent or self.pick_event or self.carry_event)
