"""Normal-only calibration and declared D4 development captures, never simulator truth at runtime."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r24_loaded.d4 import load_cases
from experiments.r31_held_identity.candidate import Detector as R31
from .signal import LiftSignal
from .probe import TARGET

class Probe(R31):
    def reset(self):
        super().reset();self.lift=LiftSignal();self.rows=[];Probe.instance=self
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,alarm=super().update(t,th,thd,tau_m,tau_s,grip)
        s=self.lift.feed(self,t,th,tau_s,grip)
        if s is not None:self.rows.append((round(t*1000),s.height,s.inverse,s.momentum,bool(alarm)))
        return residual,alarm
def training():
    # Midpoints of public mass range and lateral/vertical COM variants, none
    # are members of the D4 mass grid. No negative or impact labels fitted.
    return [dict(case_id=f'fit{i:02d}-{j}',kind='reference',mass=m,com=c,collision=None,pick_offset=None)
        for i,m in enumerate((.45,.95,1.65,2.6,3.25,4.55,5.45,6.85,7.65)) for j,c in enumerate(
        ((-.027,.018,-.015),(.004,-.013,.032),(.025,.003,.047)))]
def one(job):
    c,out=job
    _,log,signed=raw.execute(Probe,c)
    idx=c['case_id'];path=out/(f'{idx:03d}.npz' if isinstance(idx,int) else f'{idx}.npz')
    with path.open('xb') as f:np.savez_compressed(f,public=np.asarray(Probe.instance.rows),
        signed=signed,tool=log['f_tool'],table=log['f_table'],parent=log['detect'])
    return idx
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--group',choices=('fit','diagnose','guards'),required=True);a=p.parse_args();cs=load_cases()
    if a.group=='fit':sel=training()
    elif a.group=='diagnose':
        matched={raw.reference_key(cs[i]) for i in TARGET}
        sel=[c for c in cs if c['kind']=='reference' and raw.reference_key(c) in matched]+[cs[i] for i in TARGET]
    else:
        matched={raw.reference_key(cs[i]) for i in TARGET}
        sel=[cs[i] for i in (*range(0,189,4),*range(369,385)) if raw.reference_key(cs[i]) not in matched]
    with ProcessPoolExecutor(a.jobs) as pool:
        for i in pool.map(one,[(c,a.out) for c in sel]):print(i,flush=True)
if __name__=='__main__':main()
