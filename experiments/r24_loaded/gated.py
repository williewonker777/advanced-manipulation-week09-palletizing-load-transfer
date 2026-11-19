"""Use the support-response correction only when an independently commissioned drive regime has departed."""
import json
from pathlib import Path

import numpy as np

from experiments.r24_loaded.drive import DriveEstimate, departure
from experiments.r24_loaded.memory import Response

DRIVE_MODEL=Path(__file__).with_name('nominal_drive.json')


class Detector(Response):
    def reset(self):
        super().reset()
        self.drive=DriveEstimate() if self.box_mass>0 and getattr(self.ctrl,'MODE',None)=='torque' else None
        self.changed_drive=False
        if self.drive is None:return
        m=json.loads(DRIVE_MODEL.read_bytes());lo=np.asarray(m['low']);hi=np.asarray(m['high'])
        if (m['schema']!='week09-r24-motor-port-calibration.v1' or lo.shape!=(3,) or hi.shape!=(3,)
                or not np.isfinite([lo,hi]).all() or not (lo<hi).all()):raise ValueError('invalid nominal drive calibration')
        self._low,self._high=lo,hi
        # Reuse the original R23 yaw state throughout commissioning, keeping its
        # history intact. Changing class below only supplies the post-lift logic.
        from experiments.r22_guarded.candidate import GuardedReleaseBalance
        self._memory_yaw=self.yaw;self.yaw=GuardedReleaseBalance(self.box_mass)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        if self.drive is not None and not self.drive.closed:
            self.drive.feed(thd,tau_m,tau_s,grip)
            if self.drive.closed and departure(self.drive.gain,self._low,self._high):
                self.changed_drive=True
                # Share the correctly warmed unmodified balance history; keep the
                # feature-history fields and packet specific to the support model.
                self._memory_yaw.__dict__.update(self.yaw.__dict__)
                self.yaw=self._memory_yaw
        return super().update(t,th,thd,tau_m,tau_s,grip)
