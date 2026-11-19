"""Public horizontal wrench at the moving flange, with a freely rotating base frame."""
from dataclasses import dataclass

import mujoco
import numpy as np
from sim import config as C
from experiments.r15_delayed_wrench.observer import KERNEL, N, NormalForce


@dataclass(frozen=True)
class Sample:
    mode: int
    progress_m: float
    z_m: float
    inverse: np.ndarray
    point: np.ndarray
    momentum: np.ndarray
    free_moment: np.ndarray
    support_N: float

class Surface:
    """Stage is selected from observed TCP, measured excursion and grip edges, not time."""
    def __init__(self):
        self.start=self.end=None
        self.max_z=None
        self.moved=self.returned=self.detached=False
    def feed(self,xyz,grip):
        if grip:
            if self.start is None or self.detached:
                self.start=xyz.copy();self.end=None;self.max_z=xyz[2]
                self.moved=self.returned=self.detached=False
            self.max_z=max(self.max_z,xyz[2])
            height=xyz[2]-self.start[2]
            if np.linalg.norm(xyz[:2]-self.start[:2])>=.16:self.moved=True
            if not self.moved:
                return (1,float(height)) if -.002<=height<=.060 else (0,0.)
            depth=self.max_z-xyz[2]
            if depth>=.018:self.returned=True
            return (2,float(depth)) if self.returned else (0,0.)
        if self.start is None:return 0,0.
        if not self.detached:
            self.detached=True;self.end=xyz.copy()
        gap=float(np.linalg.norm(xyz-self.end))
        return (3,gap) if gap<=.075 else (0,0.)

class Signal:
    def __init__(self):
        self.stage=Surface();self.sensor=None
        self.jp=np.zeros((3,6));self.jw=self.jp.copy()
    def feed(self,d,t,th,tau_s,grip):
        q=d._shared.q_prev
        if q is None:return None
        nom=d.nom;xyz=nom.tcp_position(q)
        mode,progress=self.stage.feed(xyz,grip)
        if self.sensor is None:
            bias=getattr(d,'static_bias',None)
            if self.stage.start is None or bias is None:return None
            self.sensor=NormalForce(nom,d.nom_box,dt=C.DT,viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)
            self.sensor.bias=np.array(bias,copy=True);self.body=nom.m.body('link_6').id
        self.sensor.update(t,th,tau_s,bool(grip),project=False)
        if not mode or len(self.sensor.residual)!=N:return None
        qmid=self.sensor.midpos[N//2]
        nom.d.qpos[:]=qmid
        mujoco.mj_kinematics(nom.m,nom.d);mujoco.mj_comPos(nom.m,nom.d)
        # The public empty-arm inertial point stays on the flange in BOTH
        # modes. The full loaded model enters the dynamics, not the assumed
        # location of an unknown contact.
        mujoco.mj_jacBodyCom(nom.m,nom.d,self.jp,self.jw,self.body)
        j=np.vstack((self.jp,self.jw))
        if np.linalg.cond(j)>1e4:return None
        tau=KERNEL@np.asarray(self.sensor.residual)-self.sensor.bias
        full=np.linalg.solve(j.T,tau)
        weighted=self.jp.T/np.asarray(C.JTS_NOISE)[:,None]
        if np.linalg.cond(weighted)>1e3:return None
        point=np.linalg.lstsq(weighted,tau/np.asarray(C.JTS_NOISE),rcond=None)[0]
        angle=qmid[0];co,si=np.cos(angle),np.sin(angle)
        rot=np.array([[co,si,0.],[-si,co,0.],[0.,0.,1.]])
        return Sample(mode,progress,float(xyz[2]),rot@full[:3],rot@point,rot@d._shared.last_wrench[:3],
                      rot@full[3:],float(d._shared.last_table_wrench[2]))
