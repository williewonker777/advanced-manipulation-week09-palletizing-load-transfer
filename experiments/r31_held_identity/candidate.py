"""Assign individual native acquisitions to a previously announced measured force, not to a clock."""
import math

import mujoco
import numpy as np

from experiments.r15_delayed_wrench.observer import KERNEL, N
from experiments.r27_load_change.event import Changes
from experiments.r29_loaded_rising_certificate.candidate import rising_means
from experiments.r30_detached_lifetime.candidate import AcquisitionEndpoint
from experiments.r30_yaw_headroom.candidate import Detector as Headroom


class SpatialProjection:
    """Return the identical xy solve used by the parent while retaining its already-computed z.

    The nominal model and scratch arrays are shared with the original projection.
    One Jacobian/SVD/solve is evaluated, never a second plant integration.
    """
    def __init__(self, previous):
        self.nom,self.jp,self.jw,self.site=previous.nom,previous.jp,previous.jw,previous.site
        self.last_force=None

    def from_integrated(self, detector):
        self.last_force=None
        witness=detector.witness
        if (witness is None or witness.bias is None or len(witness.residual)!=N
                or len(witness.midpos)!=N or not witness.prev_grip):
            return None
        self.nom.d.qpos[:]=witness.midpos[N//2]
        mujoco.mj_kinematics(self.nom.m,self.nom.d);mujoco.mj_comPos(self.nom.m,self.nom.d)
        mujoco.mj_jacSite(self.nom.m,self.nom.d,self.jp,self.jw,self.site)
        jac=np.vstack((self.jp,self.jw))
        if np.linalg.cond(jac)>1e4:return None
        self.last_force=np.linalg.solve(jac.T,KERNEL@np.asarray(witness.residual)-witness.bias)[:3]
        return self.last_force[:2]


class Source:
    """A horizontal-force ray measured independently in two channels in a single gate generation."""
    def __init__(self, generation, past, current, yaw):
        self.generation=generation; self.before=np.array(past,float)
        delta=np.asarray(current)-self.before
        self.size=np.linalg.norm(delta[:,:2],axis=1)
        self.rays=delta[:,:2]/self.size[:,None]
        self.yaw=yaw

    @classmethod
    def start(cls,generation,past,now,yaw):
        if generation is None or yaw is None or not math.isfinite(yaw) or past is None or now is None:
            return None
        a,b=np.asarray(past),np.asarray(now)
        if (a.shape!=(2,3) or b.shape!=(2,3) or not np.isfinite(a).all() or not np.isfinite(b).all()
                or not np.all(np.linalg.norm(b[:,:2]-a[:,:2],axis=1)>0)):
            return None
        return cls(generation,a,b,yaw)

    def coordinates(self,levels):
        if levels is None:return None
        levels=np.asarray(levels)
        if levels.shape!=(2,3) or not np.isfinite(levels).all():return None
        delta=levels-self.before
        along=np.sum(self.rays*delta[:,:2],axis=1)
        side=self.rays[:,0]*delta[:,1]-self.rays[:,1]*delta[:,0]
        perpendicular=np.hypot(side,delta[:,2])
        return delta,along,perpendicular,side

    def agrees(self,generation,levels,routes,yaw,column):
        if (self.generation is not generation or not routes or not set(routes) <= {'yaw','normal','directional','r8_base'}
                or not ({'yaw','normal'}&set(routes))):
            return False
        data=self.coordinates(levels)
        if data is None:return False
        delta,along,off,_=data
        # Both sensors must remain closer to the originally measured force than to zero,
        # and the common ray must explain more than all transverse force together.
        if not np.all((2*along>self.size)&(along>off)):
            return False
        if 'yaw' in routes:
            col=np.asarray(column)
            if col.shape!=(2,) or not np.isfinite(col).all() or yaw is None or not math.isfinite(yaw):
                return False
            # Generalized joint-1 momentum must move with both measured planar projections.
            if not np.all((float(yaw)-self.yaw)*(delta[:,:2]@col)>0):
                return False
        return True

    def distinct_load(self,levels):
        data=self.coordinates(levels)
        if data is None:return True   # missing compatibility information fails open
        delta,along,off,side=data
        # Merely returning to zero along the original ray is not a new positive
        # excitation. Opposing or perpendicular displacement has to exceed the
        # originally certified onset and agree between the two instruments.
        return bool(np.all(-along>self.size) or (np.all(off>self.size) and np.all(off>np.abs(along))
                    and float(side[0]*side[1]+delta[0,2]*delta[1,2])>0))


class NativeAcquisitions:
    """Never consume an already exposed route or an unrelated newly active route."""
    def __init__(self):
        self.previous=set()
        self.accounted=set()

    def feed(self,routes,complete,grip,generation,levels,token,yaw,column):
        live=set(routes); new=live-self.previous
        self.previous=live
        self.accounted.intersection_update(live)
        if not complete or not grip or (token is not None and levels is None):
            self.accounted.clear()
            return False
        if token is None:
            return False
        if self.accounted and (generation is not None and generation is not token.generation
                               or token.distinct_load(levels)):
            self.accounted.clear()
        if new and token.agrees(generation,levels,live,yaw,column):
            self.accounted.update(new)
        return bool(live and live<=self.accounted)


class Detector(Headroom):
    def reset(self):
        super().reset()
        self.detached_endpoint=AcquisitionEndpoint()
        self.r27_projection=SpatialProjection(self.r27_projection)
        self.z_window=Changes()
        self.source=None
        self.native_acquisitions=NativeAcquisitions()
        self.associated_native_samples=0

    def update(self,t,th,thd,tm,ts,grip):
        count=self.planar_notices;width=self.pending_width
        residual,reported=super().update(t,th,thd,tm,ts,grip)
        optional=bool(width or self.planar_notices>count)
        gate=self.r27_gate;projection=self.r27_projection
        usable=bool(projection.last_force is not None and all(w.present for w in gate.windows.windows))
        z=self.z_window.feed([self._shared.last_residual_wrench[2],projection.last_force[2]] if usable else None)
        means=rising_means(gate) if z is not None and (self.source is not None or count<self.planar_notices) else None
        past=levels=None
        if means is not None:
            past=np.column_stack((means[0],z[0]));levels=np.column_stack((means[1],z[1]))
        if self.planar_notices>count:
            self.source=Source.start(gate.reference,past,levels,self.yaw_value)
        associated=self.native_acquisitions.feed(self.r29_active_routes,self.r29_routes_complete,grip,gate.reference,
                                                 levels,self.source,self.yaw_value,projection.jp[:2,0])
        if associated:
            self.associated_native_samples+=1
            reported=optional
        elif self.source is not None and self.r29_active_routes and self.r29_routes_complete and grip:
            # The older broad link may have elided this raw native rise. With an
            # optional ownership claim we must expose uncertain/new raw sources.
            reported=True
        if (not grip or (self.source is not None and gate.reference is not self.source.generation
                         and not self.native_acquisitions.accounted)):
            self.source=None
        return residual,bool(reported)
