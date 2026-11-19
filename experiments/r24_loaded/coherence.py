"""Confirm drive departure from disjoint identification windows on the same motor."""
import json
from pathlib import Path

import numpy as np

from experiments.r24_loaded.drive import departure
from experiments.r24_loaded.gated import Detector as Multi

MODEL=Path(__file__).with_name('split_drive.json')


def estimates(estimator):
    if estimator.gain is None or len(estimator.x)!=20:return None
    x=np.asarray(estimator.x);y=np.asarray(estimator.y);g=[estimator.gain]
    for k in (0,1):
        a=x[k::2]-x[k::2].mean(0);b=y[k::2]-y[k::2].mean(0)
        power=np.sum(a*a,axis=0)
        if not (np.sqrt(power/len(a))>1.).all():return None
        g.append(np.sum(a*b,axis=0)/power)
    result=np.asarray(g)
    return result if np.isfinite(result).all() else None


def repeated(gain,low,high):
    if gain is None:return False
    return bool(np.any(np.all(gain<low,axis=0)|np.all(gain>high,axis=0)))


class Identification:
    def __init__(self,base,low,high):
        self.base=base;self.low=low;self.high=high;self.gain=None;self.confirmed=False

    @property
    def closed(self):
        return self.base.closed

    def feed(self,*args):
        self.base.feed(*args)
        if self.base.closed:
            self.gain=self.base.gain
            self.confirmed=repeated(estimates(self.base),self.low,self.high)


class Detector(Multi):
    def reset(self):
        super().reset()
        if self.drive is None:return
        m=json.loads(MODEL.read_bytes());lo=np.asarray(m['low']);hi=np.asarray(m['high'])
        if (m['schema']!='week09-r24-split-drive-model.v1' or lo.shape!=(3,3) or hi.shape!=(3,3)
                or not np.isfinite([lo,hi]).all() or not (lo<hi).all()):raise ValueError('invalid split drive model')
        self.drive=Identification(self.drive,lo,hi)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        # The drive packet must be consumed exactly once. From here dispatch
        # directly to the common R23 response estimator rather than Multi.update.
        from experiments.r24_loaded.memory import Response
        if self.drive is not None and not self.drive.closed:
            self.drive.feed(thd,tau_m,tau_s,grip)
            if self.drive.closed and (self.drive.confirmed or departure(self.drive.gain,self._low,self._high)):
                self.changed_drive=True
                self._memory_yaw.__dict__.update(self.yaw.__dict__)
                self.yaw=self._memory_yaw
        return Response.update(self,t,th,thd,tau_m,tau_s,grip)
