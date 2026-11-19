"""Explain table yaw only in a transmission regime supported by both DC impulse and AC-slope evidence."""
import json
from pathlib import Path

import numpy as np

from sim import config as C
from experiments.r22_guarded.candidate import GuardedReleaseBalance
from experiments.r23_detached.free import Detector as Free
from experiments.r24_loaded.drive import DriveEstimate
from experiments.r24_loaded.memory import MemoryYaw, MODEL as SUPPORT

MODEL=Path(__file__).with_name('dc_drive.json')


def measurements(est):
    if est.gain is None:return None,None
    mean_x=np.mean(est.x,axis=0);mean_y=np.mean(est.y,axis=0)
    valid=np.isfinite(mean_x)&np.isfinite(mean_y)&(np.abs(mean_x)>C.COULOMB[:3])
    dc=np.divide(mean_y, mean_x, out=np.zeros_like(mean_y), where=valid)
    return np.stack((est.gain,dc)),valid


def changed(value,valid,lo,hi):
    if value is None:return False
    same_side=np.all(value<lo,axis=0)|np.all(value>hi,axis=0)
    return bool(np.any(valid & np.all(np.isfinite(value),axis=0) & same_side))


class SelectiveYaw(MemoryYaw):
    def __init__(self,mass,packet,weights):
        super().__init__(mass,packet,weights);self.admitted=False

    def feed(self,th,tau,mass_row,grip):
        if self.admitted:
            return super().feed(th,tau,mass_row,grip)
        # Use the same balance and past history when inactive, not a second
        # observer and not a filtered/suppressed R23 output.
        value=GuardedReleaseBalance.feed(self,th,tau,mass_row,grip)
        self.raw_value=value;self.reaction=0.
        return value


class Detector(Free):
    def reset(self):
        super().reset()
        self.drive=None;self.changed_drive=False
        if getattr(self.ctrl,'MODE',None)!='torque' or not self.box_mass>0:return
        m=json.loads(MODEL.read_bytes());s=json.loads(SUPPORT.read_bytes());w=s['coefficients_Nm_per_N']
        self.lo=np.asarray(m['low']);self.hi=np.asarray(m['high']);self.good=np.asarray(m['eligible'],dtype=bool)
        if (m['schema']!='week09-r24-dc-slope-calibration.v1' or s['schema']!='week09-r24-load-response-model.v1'
                or self.lo.shape!=(2,3) or self.hi.shape!=(2,3) or self.good.shape!=(3,) or not (self.hi>self.lo).all()
                or not np.isfinite([self.lo,self.hi]).all() or len(w)!=5 or not np.isfinite(w).all()):
            raise ValueError('invalid frozen torque-port models')
        self.drive=DriveEstimate()
        self.yaw=SelectiveYaw(self.box_mass,self._shared,w)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        if self.drive is not None and not self.drive.closed:
            self.drive.feed(thd,tau_m,tau_s,grip)
            if self.drive.closed:
                data,valid=measurements(self.drive)
                self.changed_drive=changed(data, None if valid is None else valid & self.good,self.lo,self.hi)
                self.yaw.admitted=self.changed_drive
        return super().update(t,th,thd,tau_m,tau_s,grip)
