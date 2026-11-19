"""Nominal commissioning of temporally disjoint estimates and single-joint stress characterization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import types

import numpy as np

from experiments.r23_holdout import evaluator as lab
from experiments.r24_loaded import commission
from experiments.r24_loaded.coherence import estimates,repeated
from experiments.r24_loaded.drive import departure

HERE=Path(__file__).resolve().parent


def capture(job):
    seed,variant=job
    class Recorder(commission.Recorder):
        last=None
        def reset(self):
            super().reset();Recorder.last=self
    proto=commission.capture
    run=types.FunctionType(proto.__code__,dict(proto.__globals__,Recorder=Recorder))
    row=run((seed,variant));x=estimates(Recorder.last.drive)
    if x is None:raise ValueError('insufficient split excitation')
    row['three_estimates']=x.tolist();return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('fit','check'),required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    cfg=json.loads(commission.PLAN.read_bytes());old=json.loads(lab.PLAN.read_bytes())['variants'];arms={'stock':old['stock']}
    seeds=cfg['train_noise_seeds' if a.stage=='fit' else 'check_noise_seeds']
    if a.stage=='check':
        arms['skew']=old['actuator_skew']
        # Observability is deliberately measured on both signs of individual faults, not just the known skew.
        for j in range(3):
            for delta in (-.02,.02):
                g=[1.]*6;g[j]+=delta;arms[f'axis_{j+1}_{delta:+.2f}']={'kind':'actuator_skew','joint_gain':g}
    jobs=[(s,v) for v in arms.values() for s in seeds]
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(capture,jobs))
    if a.stage=='fit':
        x=np.asarray([r['three_estimates'] for r in rows]);margin=np.maximum(.002,np.ptp(x,axis=0))
        d={'schema':'week09-r24-split-drive-model.v1','low':(x.min(0)-margin).tolist(),'high':(x.max(0)+margin).tolist(),
           'normal_only':True,'normal_records':rows,'registration_sha256':hashlib.sha256((HERE/'coherence_plan.json').read_bytes()).hexdigest()}
    else:
        m=json.loads(a.model.read_bytes());lo=np.asarray(m['low']);hi=np.asarray(m['high']);by={}
        for i,row in enumerate(rows):
            k=list(arms)[i//len(seeds)];g=np.asarray(row['three_estimates']);found=repeated(g,lo,hi) or departure(g[0],lo[0],hi[0])
            row.update(mechanics=k,departure=found);by[k]=by.get(k,0)+int(found)
        d={'classification':'known commissioning mechanism, unused noise seeds and per-axis changes; not independent collision holdout',
           'seeds_per_mechanism':len(seeds),'departures':by,'rows':rows}
    with a.out.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
    print(a.stage,d.get('departures',d.get('low')))


if __name__=='__main__':main()
