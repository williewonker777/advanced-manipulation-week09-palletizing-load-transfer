"""Identify motor transmission from past encoder/command/JTS data, before any grasp."""
import math

import numpy as np

from sim import config as C


class DriveEstimate:
    width, windows = 48, 20

    def __init__(self):
        self.last_velocity = self.last_jts = None
        self.begin_velocity = self.begin_jts = None
        self.command = np.zeros(3);self.resistance=np.zeros(3)
        self.x=[];self.y=[];self.i=self.blocks=0
        self.closed=False;self.gain=None;self.excitation=None

    def feed(self, velocity, torque, jts, grip):
        if self.closed:return
        if grip:
            self.closed=True
            return
        v=np.asarray(velocity)[:3];j=np.asarray(jts)[:3]
        if self.last_velocity is None:
            self.begin_velocity=v.copy();self.begin_jts=j.copy()
        self.last_velocity=v;self.last_jts=j
        self.command+=np.asarray(torque)[:3]
        self.resistance+=j+C.VISCOUS[:3]*v+C.COULOMB[:3]*np.tanh(v/C.FRICTION_EPS)
        self.i+=1
        if self.i < self.width:return
        duration=self.width*float(C.DT);lag=float(C.DT)+1./(2.*math.pi*float(C.JTS_CUTOFF))
        y=self.resistance/self.width+(C.B[:3]*(v-self.begin_velocity)+lag*(j-self.begin_jts))/duration
        if self.blocks:
            self.x.append(self.command/self.width);self.y.append(y)
        self.blocks+=1;self.begin_velocity=v.copy();self.begin_jts=j.copy()
        self.command.fill(0.);self.resistance.fill(0.);self.i=0
        if len(self.x)==self.windows:
            self.closed=True
            x=np.asarray(self.x);y=np.asarray(self.y);x=x-x.mean(0);y=y-y.mean(0)
            xx=np.sum(x*x,axis=0);self.excitation=np.sqrt(xx/self.windows)
            if np.all(self.excitation>1.) and np.isfinite(xx).all():
                gains=np.sum(x*y,axis=0)/xx
                if np.isfinite(gains).all():self.gain=gains


def departure(gain, low, high):
    if gain is None:return False
    values=np.asarray(gain,dtype=float);lo=np.asarray(low);hi=np.asarray(high)
    return bool(values.shape==(3,) and np.isfinite(values).all() and np.count_nonzero((values<lo)|(values>hi))>=2)
