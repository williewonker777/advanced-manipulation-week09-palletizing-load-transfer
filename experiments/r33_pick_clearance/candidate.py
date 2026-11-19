"""Causal pickup handoff, using pre-clearance sensor history but no pre-clearance alarms."""
from collections import deque
import numpy as np

from sim import config as C
from experiments.r15_delayed_wrench.observer import NormalForce
from experiments.r31_held_identity.candidate import Detector as Parent
from experiments.r33_pick_vertical.candidate import parameters
from experiments.r33_pick_vertical.state import expected


class Route:
    def __init__(self):
        p=parameters()
        self.medians=np.asarray(p['medians']);self.floor=p['margin_N']
        self.limit={-1:p['negative_limit_N'],1:p['positive_limit_N']}
        self.blocks=deque(maxlen=16);self.reset_lift()

    def reset_source(self):
        self.native=self.reported=False
        self.owner=self.quiet=self.remaining=0

    def reset_lift(self):
        self.origin=None;self.previous=False;self.over=False;self.observer=None
        self.blocks.clear();self.reset_source()

    def classify(self,h,force,momentum,parent,cap,persistence):
        if parent:self.native=True
        if h<0 or h>.1:
            self.blocks.clear();self.remaining=0;return False
        d=np.array((force[2],momentum))-expected(h,self.medians)
        if not np.isfinite(d).all() or not np.isfinite(force).all():
            self.blocks.clear();self.remaining=0;return False
        self.blocks.append((d[0],d[1],force[0],force[1]))
        if len(self.blocks)!=16:return False
        x=np.asarray(self.blocks);a=x[:8].mean(0);b=x[8:].mean(0)
        sign=1 if b[0]>=0 else -1;lo=self.limit[sign]
        if np.min(sign*b[:2])<lo/2:
            self.remaining=0;self.quiet+=1
            if self.quiet>=8:self.reset_source();self.native=bool(parent)
            return False
        self.quiet=0
        if self.owner and sign!=self.owner:self.reset_source();self.native=bool(parent)
        if h<=.035:self.owner=sign
        if max(sign*a[0],sign*b[0])>=cap:self.native=True
        if h<.003 or self.owner!=sign or self.native:return False
        if min(np.min(sign*a[:2]),np.min(sign*b[:2]))<=lo:return False
        if abs(b[0])<=np.linalg.norm(b[2:]):return False
        if not self.reported:
            if abs(b[0]-a[0])>max(self.floor,abs(b[0])/8):return False
            self.reported=True;self.remaining=persistence
        if self.remaining>0:
            self.remaining-=1;return True
        return False

    def feed(self,det,t,th,thd,tau_m,tau_s,grip,parent):
        if not grip:
            if self.previous:self.reset_lift()
            return False
        packet=det._shared
        if not self.previous:
            self.previous=True
            q=packet.q_prev;bias=getattr(det,'static_bias',None)
            if q is not None and bias is not None:
                self.origin=float(det.nom.tcp_position(q)[2])
                self.observer=NormalForce(det.nom,det.nom_box,dt=C.DT,viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)
                self.observer.bias=np.array(bias,copy=True)
        if parent:self.native=True
        if self.over or self.observer is None:return False
        s=self.observer.update(t,th,tau_s,True)
        if s is None:return False
        nom=self.observer.loaded
        h=float(nom.d.site_xpos[nom.m.site('tcp').id,2])-self.origin
        if h>.1:
            self.over=True;self.observer=None;return False
        return self.classify(h,s.force_n,packet.last_wrench[2],parent,packet.force_threshold[2],packet.persistence)

class Detector(Parent):
    def reset(self):
        super().reset();self.pick_route=Route() if getattr(self.ctrl,'MODE',None)=='torque' else None
    def update(self,t,th,thd,tau_m,tau_s,grip):
        r,parent=super().update(t,th,thd,tau_m,tau_s,grip)
        extra=self.pick_route is not None and self.pick_route.feed(self,t,th,thd,tau_m,tau_s,grip,parent)
        return r,bool(parent or extra)
