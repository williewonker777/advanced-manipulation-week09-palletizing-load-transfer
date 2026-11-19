"""Defer weak servo notice and require two settled observer windows before rearm."""
import numpy as np
import my_detector
from experiments.r15_delayed_wrench.observer import N,NormalForce
from experiments.r31_servo_release.candidate import coherent_horizontal,HORIZONTAL_N,PERSISTENCE,REPORT_SAMPLES
from sim import config as C

SETTLE_SAMPLES=2*N

class OwnedSource:
    def __init__(self):self.reset()
    def reset(self):
        self.onset=0;self.pending=False;self.owned=False;self.release=0;self.pulse=0;self.awaiting_settle=False;self.settle=0
    def feed(self,evidence,valid,settled,incumbent):
        if self.awaiting_settle:
            self.settle=self.settle+1 if valid and settled else 0
            if self.settle>=SETTLE_SAMPLES:
                self.awaiting_settle=False;self.settle=0
        if self.pulse:
            self.pulse-=1;return True
        if self.awaiting_settle:return False
        if self.pending:
            self.owned|=bool(incumbent)
            if not valid:return False
            self.release=0 if evidence else self.release+1
            if self.release<PERSISTENCE:return False
            report=not self.owned;self.onset=0;self.pending=False;self.owned=False;self.release=0
            self.awaiting_settle=True;self.settle=1 if settled else 0
            if report:self.pulse=REPORT_SAMPLES-1;return True
            return False
        if not valid:self.onset=0;return False
        self.onset=self.onset+1 if evidence else 0
        if self.onset>=PERSISTENCE:self.pending=True;self.owned=bool(incumbent);self.onset=0
        return False

class Detector(my_detector.Detector):
    def reset(self):
        super().reset();self.release_force=NormalForce(self.nom,self.nom_box,dt=C.DT,viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)
        self.source_owner=OwnedSource();self.release_extra=False;self.release_evidence=False;self.release_settled=False
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,incumbent=super().update(t,th,thd,tau_m,tau_s,grip)
        sample=self.release_force.update(t,th,tau_s,bool(grip));force=None if sample is None else sample.force_n
        self.release_evidence,features=coherent_horizontal(force,self._shared.last_wrench,self._shared.last_table_allowed)
        valid=sample is not None and force is not None and np.isfinite(force).all() and np.isfinite(self._shared.last_wrench[:2]).all()
        self.release_settled=bool(valid and np.linalg.norm(force[:2])<=HORIZONTAL_N and np.linalg.norm(self._shared.last_wrench[:2])<=HORIZONTAL_N)
        self.release_extra=self.source_owner.feed(self.release_evidence,valid,self.release_settled,bool(incumbent))
        return residual,bool(incumbent or self.release_extra)
