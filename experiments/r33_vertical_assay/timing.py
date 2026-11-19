"""Paired wall-time sample inside actual detector.update; not a hard real-time guarantee."""
import argparse
from pathlib import Path
import time
import numpy as np
from experiments.r31_held_identity.candidate import Detector as Anchor
from experiments.r33_vertical_composition.candidate import Detector as Entrant
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r24_loaded.d4 import load_cases
from .check import level

def profile(cls,c,expected):
    class P(cls):
        def reset(self):
            super().reset();self.times=[];P.instance=self
        def update(self,*args):
            start=time.perf_counter_ns()
            result=super().update(*args)
            elapsed=time.perf_counter_ns()-start
            self.times.append(elapsed)
            return result
    rec=raw.episode_record(P,c)
    if rec['physical_sha256']!=expected:raise ValueError('changed physics')
    us=np.asarray(P.instance.times)/1000
    if us.shape!=(6000,):raise ValueError('incomplete episode')
    return dict(samples=len(us),median_us=float(np.median(us)),p99_us=float(np.quantile(us,.99)),
              maximum_us=float(us.max()),over_1ms_ticks=int(np.count_nonzero(us>1000)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if level.sha(a.reference)!=level.OLD:raise ValueError('expected R31 physics')
    old=level.get(a.reference)['rows'];cases=load_cases();rows=[]
    for i in (125,193,253):
        for name,cls in [('r31',Anchor),('r33',Entrant),('r33',Entrant),('r31',Anchor)]:
            rows.append(dict(case_id=i,kind=cases[i]['kind'],method=name,**profile(cls,cases[i],old[i]['physics_sha256'])))
    level.save(a.out,dict(schema='r33-detector-walltime-sample.v1',unit='us',order='R31 R33 R33 R31 per condition',episodes=len(rows),
        conditions=3,clock='perf_counter_ns in Python detector.update only; excludes physics and controller',
        caveat='Local shared Linux workstation with other simulations running. Neither a controlled speed benchmark nor a 1ms worst-case guarantee. No changes to the simulator timestep.',rows=rows))
if __name__=='__main__':main()
