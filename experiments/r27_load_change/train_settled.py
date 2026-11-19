"""Fit joint and marginal directional support on precisely the prior normal-only split."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as Base
from experiments.r27_load_change.projection import Projection
from experiments.r27_load_change.settled_event import PairedWindows, channels

HERE=Path(__file__).resolve().parent


class Scan(Base):
    def reset(self):
        super().reset()
        self.r27_projection=Projection(self.box_mass,self.box_com)
        self.windows=PairedWindows()
        self.joint=np.zeros(8)
        self.each=np.zeros((2,8))
        self.valid=0

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,native=super().update(t,th,thd,tau_m,tau_s,grip)
        obj=self.windows.feed(channels(self,grip))
        if obj is not None:
            self.valid+=1
            change=obj[2]
            self.joint=np.maximum(self.joint,change.min(axis=0))
            self.each=np.maximum(self.each,change)
        return residual,native


def episode(spec):
    case,mech=spec;det=Scan()
    _,log=lab.run_episode(case,mech,lambda:det,(None,None))
    if not lab.exact_force(log,(None,None))['passed'] or not det.valid:
        raise ValueError('valid unforced trajectory required')
    return dict(id=case['id'],mechanics=mech['kind'],valid=det.valid,
                joint=det.joint.tolist(),each=det.each.tolist(),physical=lab.certified.trace_signature({'none':log}))


def envelope(rows,mechanics,key):
    per=np.array([np.max([r[key] for r in rows if r['mechanics']==m],axis=0) for m in mechanics])
    return (np.max(per,axis=0)+np.ptp(per,axis=0)).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args();reg=json.loads((HERE/'registration.json').read_bytes());plan=json.loads(lab.PLAN.read_bytes())
    tasks=[(c,plan['variants'][m])for c in reg['fit_normals'] for m in reg['mechanics']]
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(episode,tasks))
    old=json.loads((HERE/'model.json').read_bytes())
    if {(x['id'],x['variant']):x['physics'] for x in old['normals']}!={(x['id'],x['mechanics']):x['physical']for x in rows}:
        raise ValueError('did not use the identical original normal-only split and physics')
    out=dict(schema='r27-two-stream-settled-normal.v1',registration_sha256=hashlib.sha256((HERE/'settled_registration.json').read_bytes()).hexdigest(),
             joint_bounds_N=envelope(rows,reg['mechanics'],'joint'),separate_bounds_N=envelope(rows,reg['mechanics'],'each'),episodes=len(rows),rows=rows)
    with a.out.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(len(rows),out['joint_bounds_N'])


if __name__=='__main__':
    main()
