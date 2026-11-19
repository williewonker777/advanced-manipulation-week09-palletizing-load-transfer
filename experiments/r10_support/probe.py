"""Analysis probe comparing public spring/gravity evidence with frozen C1.

Only the runner joins this public-signal trace to simulator table force.  The
probe itself has no plant/table/case access and is suitable for evaluating the
causal information boundary.
"""
from __future__ import annotations
import numpy as np
from experiments.r8_final.C1 import Detector as C1
from sim import config as C, trajectory as T


class Detector(C1):
    def reset(self):
        super().reset()
        self.inv_k=np.zeros(6)
        elastic=np.asarray(self.nom.elastic,bool)
        self.inv_k[elastic]=1.0/np.asarray(self.nom.K)[elastic]
        self.bias=None
        self.dynamic_bias=None
        self.q_previous=None
        self.velocity=np.zeros(6)
        self.trace=[]

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,detected=super().update(t,th,thd,tau_m,tau_s,grip)
        q=np.asarray(th,float)-self.inv_k*np.asarray(tau_s,float)
        empty=self.nom.gravity(q)
        full=self.nom_box.gravity(q)
        sensed=np.asarray(tau_s,float)-empty
        if self.bias is None:
            self.bias=sensed.copy()
        elif t<T.T_GRASP-.08:
            self.bias += .002*(sensed-self.bias)
        centered=sensed-self.bias
        payload=full-empty
        denom=float(payload@payload)+1e-9
        raw=float(centered@payload/denom)
        orth=centered-raw*payload
        orth_ratio=float(np.linalg.norm(orth)/(np.linalg.norm(payload)+1e-9))
        qd=np.zeros(6) if self.q_previous is None else (q-self.q_previous)/C.DT
        self.q_previous=q.copy()
        self.velocity += .08*(qd-self.velocity)
        qdd_ref=T.reference(t+C.DT)[2]
        dynamic_empty=self.nom.inverse_dynamics(q,self.velocity,qdd_ref)
        dynamic_full=self.nom_box.inverse_dynamics(q,self.velocity,qdd_ref)
        dynamic_sensed=np.asarray(tau_s,float)-dynamic_empty
        if self.dynamic_bias is None:
            self.dynamic_bias=dynamic_sensed.copy()
        elif t<T.T_GRASP-.08:
            self.dynamic_bias += .002*(dynamic_sensed-self.dynamic_bias)
        dynamic_payload=dynamic_full-dynamic_empty
        dynamic_raw=float((dynamic_sensed-self.dynamic_bias)@dynamic_payload/
                          (float(dynamic_payload@dynamic_payload)+1e-9))
        shared=self._shared
        if t>=3.65:
            self.trace.append(dict(t=float(t), gravity_load=raw,
                gravity_support=float(1.0-raw), orth_ratio=orth_ratio,
                dynamic_load=dynamic_raw, dynamic_support=float(1.0-dynamic_raw),
                c1_detect=bool(detected), observer_load=float(shared.lambda_direct),
                observer_support=float(1.0-shared.lambda_support),
                observer_valid=bool(shared.lambda_support_valid),
                wrench_fz=float(shared.last_wrench[2])))
        return residual,detected
