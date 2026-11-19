"""Frozen normal-only calibration and explicit acceptance or abstention on single-joint changed drives."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import types

import numpy as np
from sim import config as C
from experiments.r23_holdout import evaluator as lab
from experiments.r24_loaded import commission
from experiments.r24_loaded.dc import measurements, changed

HERE=Path(__file__).resolve().parent


def capture(job):
    class Recorder(commission.Recorder):
        last=None
        def reset(self):
            super().reset();Recorder.last=self
    source=commission.capture
    run=types.FunctionType(source.__code__,dict(source.__globals__,Recorder=Recorder))
    row=run(job);v,valid=measurements(Recorder.last.drive)
    if v is None:raise ValueError('no completed commissioning window')
    row.update(estimates=v.tolist(),eligible=valid.tolist());return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('fit','check'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args();cfg=json.loads(commission.PLAN.read_bytes());old=json.loads(lab.PLAN.read_bytes())['variants']
    seeds=list(dict.fromkeys(cfg['train_noise_seeds']+[int(C.JTS_SEED)]+[401+19*k for k in range(16)]));arms={'stock':old['stock']}
    if a.stage=='check':
        seeds=cfg['check_noise_seeds']+[733,757,787,809];arms['skew']=old['actuator_skew']
        for j in range(3):
            for delta in (-.02,.02):
                g=[1.]*6;g[j]+=delta;arms[f'axis_{j+1}_{delta:+.2f}']={'kind':'actuator_skew','joint_gain':g}
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(capture,[(s,v) for v in arms.values() for s in seeds]))
    if a.stage=='fit':
        X=np.asarray([r['estimates'] for r in rows]);eligible=np.all([r['eligible'] for r in rows],axis=0);pad=np.maximum(.002,np.ptp(X,axis=0))
        d={'schema':'week09-r24-dc-slope-calibration.v1','low':(X.min(axis=0)-pad).tolist(),'high':(X.max(axis=0)+pad).tolist(),'eligible':eligible.tolist(),
           'single_nominal_trajectory':True,'fit_records':rows,'plan_sha256':hashlib.sha256((HERE/'dc_plan.json').read_bytes()).hexdigest()}
    else:
        m=json.loads(a.model.read_bytes());lo=np.asarray(m['low']);hi=np.asarray(m['high']);good=np.asarray(m['eligible']);counts={}
        for i,r in enumerate(rows):
            k=list(arms)[i//len(seeds)];value=changed(np.asarray(r['estimates']),np.asarray(r['eligible'])&good,lo,hi)
            r.update(mechanics=k,departure=value);counts[k]=counts.get(k,0)+int(value)
        d={'schema':'week09-r24-dc-slope-diagnostic.v1','classification':'drive estimation on unloaded approach; NOT independent collision qualification',
           'noise_seeds_per_mechanism':len(seeds),'departures':counts,'rows':rows}
    with a.out.open('x') as fp:json.dump(d,fp,indent=2,allow_nan=False);fp.write('\n')
    print(a.stage,d.get('departures',d.get('eligible')))


if __name__=='__main__':main()
