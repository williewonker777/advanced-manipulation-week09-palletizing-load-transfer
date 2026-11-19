"""Fit one predeclared, zero-intercept causal contact response to 24 no-external-force simulations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r23_detached.free import Detector as Free
from experiments.r23_holdout import evaluator as lab
from experiments.r24_loaded.response_state import SupportHistory

HERE=Path(__file__).resolve().parent
PLAN=HERE/'response_plan.json'


class Recorder(Free):
    def reset(self):
        super().reset()
        self.lag=SupportHistory(); self.was_away=False; self.observations=[]

    def update(self,t,th,thd,tau_m,tau_s,grip):
        out=super().update(t,th,thd,tau_m,tau_s,grip);s=self._shared
        if not grip:
            if self.was_away:self.lag=SupportHistory()
            self.was_away=False
        elif not s.last_table_allowed:
            self.was_away=True
        if grip and self.was_away:
            u=max(0., float(s.last_table_wrench[2])) if s.last_table_allowed else 0.
            features=self.lag.push(u)
            if s.last_table_allowed and self.yaw_value is not None and max(features)>1e-6:
                self.observations.append(features+[float(self.yaw_value)])
        return out


def one(job):
    mass,com,gain=job
    d=Recorder();_,log=lab.run_episode({'mass':mass,'com':com}, {'kind':'actuator_skew','joint_gain':gain},lambda:d,(None,None))
    if not lab.certified.exact_force(log,(None,None))['passed']:raise ValueError('not force free')
    rows=np.asarray(d.observations,float).reshape((-1,6));x,y=rows[:,:5],rows[:,5]
    return {'mass_kg':mass,'com_m':com,'joint_gain':gain,'samples':len(rows),'physics_sha256':lab.certified.physical_hash(log)},x.T@x,x.T@y,float(y@y)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    plan=json.loads(PLAN.read_bytes());jobs=[(m,c,g) for m,c in zip(plan['mass_kg'],plan['com_m'],strict=True) for g in plan['normal_gains']]
    with ProcessPoolExecutor(a.workers) as pool:stats=list(pool.map(one,jobs))
    xx=sum(v[1] for v in stats);xy=sum(v[2] for v in stats);yy=sum(v[3] for v in stats)
    ridge=.02*np.trace(xx)/5.;w=np.linalg.solve(xx+ridge*np.eye(5),xy)
    doc={'schema':'week09-r24-load-response-model.v1','plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'training':'normal only',
         'tap_width_samples':8,'coefficients_Nm_per_N':w.tolist(),'ridge':float(ridge),'records':[v[0] for v in stats],
         'sum_squared_fit_before':yy,'sum_squared_fit_after':float(yy-2*w@xy+w@xx@w)}
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print('calibration',len(stats),sum(r['samples'] for r in doc['records']),w,doc['sum_squared_fit_before'],doc['sum_squared_fit_after'])


if __name__=='__main__':main()
