"""R20 plus a normal-only, time-indexed horizontal load-transfer observer."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from experiments.r20_horizontal.candidate_detector import Detector as R20

MODEL=Path(__file__).with_name('model.json')


def features(mass,com):
 m=float(mass);c=np.asarray(com,float)
 if c.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(m):raise ValueError('known finite payload required')
 return np.r_[1.,m,m*m,c,m*c]


class Detector(R20):
    """Experimental opt-in; baseline and R20 alarms remain an exact OR."""
    def reset(self):
        super().reset();doc=json.loads(MODEL.read_bytes())
        if doc.get('schema')!='r21-pick-normal-model.v1' or not doc.get('normal_only'):
            raise ValueError('frozen normal-only model required')
        self.pick_start,self.pick_stop=map(float,doc['window_s']);self.pick_dt=float(doc['dt_s'])
        self.pick_scale=np.asarray(doc['scale'],float);self.pick_coef=np.asarray(doc['coefficients'],float)
        self.pick_threshold=float(doc['threshold_N']);self.pick_persistence=int(doc['persistence_samples'])
        self.pick_domain=(np.asarray(doc['masses'],float),np.asarray(doc['coms'],float))
        self.pick_feature=features(self.box_mass,self.box_com)
        masses,coms=self.pick_domain;c=np.asarray(self.box_com,float)
        self.pick_supported=bool(masses.min()<=self.box_mass<=masses.max() and
                                 np.all(c>=coms.min(0)) and np.all(c<=coms.max(0)))
        self.pick_history=[];self.pick_count=self.pick_clear=0;self.pick_detected=False
        self.pick_error=np.zeros(2);self.pick_smoothed=np.zeros(2)

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,baseline=super().update(t,th,thd,tau_m,tau_s,grip)
        active=self.pick_supported and self.pick_start<=float(t)<self.pick_stop
        if active:
            k=int(round((float(t)-self.pick_start)/self.pick_dt));k=max(0,min(k,len(self.pick_coef)-1))
            predicted=np.einsum('f,fa->a',self.pick_feature/self.pick_scale,self.pick_coef[k])
            self.pick_error=np.asarray(self._shared.last_residual_wrench[:2],float)-predicted
            self.pick_history.append(self.pick_error.copy());self.pick_history=self.pick_history[-4:]
            if len(self.pick_history)==4:
                self.pick_smoothed=np.mean(self.pick_history,axis=0)
                over=bool(np.max(np.abs(self.pick_smoothed))>self.pick_threshold)
            else:over=False
        else:
            self.pick_history=[];self.pick_error.fill(0.);self.pick_smoothed.fill(0.);over=False
        self.pick_count=self.pick_count+1 if over else 0;self.pick_clear=0 if over else self.pick_clear+1
        if self.pick_count>=self.pick_persistence:self.pick_detected=True
        elif self.pick_clear>=8:self.pick_detected=False
        return residual,bool(baseline or self.pick_detected)
