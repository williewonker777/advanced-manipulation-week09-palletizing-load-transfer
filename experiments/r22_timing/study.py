"""Measure the R10 control plus detector methods on the same samples; not hard realtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from my_torque import Controller
import my_detector
from sim.runner import run_episode
from experiments.r21_yaw_impulse.coarse import Detector as Yaw
from experiments.r22_combined.core import Detector as Core
from experiments.r22_combined.candidate import Detector as Augmented

TYPES = {'r10':my_detector.Detector,'r21':Yaw,'core':Core,'augmented':Augmented}


def stats(x):
    us=np.asarray(x,dtype=float)/1000.
    return {'p50_us':float(np.percentile(us,50)),'p99_us':float(np.percentile(us,99)),
            'max_us':float(us.max()),'at_least_1ms':int(sum(us>=1000.))}


def bind(target, name, count):
    func=getattr(target,name); wall=np.zeros(count,dtype=np.int64); cpu=wall.copy(); idx=[0]
    def wrapped(*args, **kw):
        i=idx[0]
        start=time.perf_counter_ns(); thread=time.thread_time_ns()
        value=func(*args,**kw)
        cpu[i]=time.thread_time_ns()-thread;wall[i]=time.perf_counter_ns()-start;idx[0]+=1
        return value
    setattr(target,name,wrapped)
    return wall,cpu,idx


def digest(log,keys):
    h=hashlib.sha256()
    for k in keys:h.update(np.ascontiguousarray(log[k]).tobytes())
    return h.hexdigest()


def trial(case,arm,rep):
    ctrl,det=Controller(),TYPES[arm]()
    w1,c1,n1=bind(ctrl,'compute',6000);w2,c2,n2=bind(det,'update',6000)
    metric,log=run_episode(ctrl,box_mass=case['mass'],box_com=tuple(case['com']),detector=det,collision=case['collision'])
    if n1 != [6000] or n2 != [6000] or metric['diverged']:raise ValueError('incomplete timing grid')
    return {'case':case['id'],'arm':arm,'rep':rep,'cycles':n1[0],'sum_wall':stats(w1+w2),
            'sum_cpu':stats(c1+c2),'detector_wall':stats(w2),'controller_wall':stats(w1),
            'physical':digest(log,('t','q','q_d','tau','qpos','f_ext','f_tool','f_table')),
            'residual':digest(log,('r',)),'detected':digest(log,('detect',))}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--revised',action='store_true');a=p.parse_args(argv)
    plan=json.loads(Path(__file__).with_name('plan.json').read_bytes())
    if hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0))
        # Pick one of the allowed logical CPUs for this child alone, as specified in plan.
        os.sched_setaffinity(0,{cpus[-1]})
    rows=[]
    if a.revised:
        from experiments.r22_guarded.candidate import Detector as Guarded
        from experiments.r22_light.candidate import Detector as Light
        TYPES.update(guarded=Guarded,light=Light)
    order=['r10','r21','guarded','light'] if a.revised else plan['arms']
    for rep in range(2):
        for case in plan['cases']:
            for arm in (order if rep==0 else order[::-1]):rows.append(trial(case,arm,rep))
    for case in plan['cases']:
        c=[r for r in rows if r['case']==case['id']]
        if len({r['physical'] for r in c}) != 1 or len({r['residual'] for r in c}) != 1:
            raise ValueError('timing changed protected signals')
        for arm in order:
            if len({r['detected'] for r in c if r['arm']==arm})!=1:raise ValueError('nondeterministic repeat')
    summary={arm:{'worst_episode_p99_wall_us':max(r['sum_wall']['p99_us'] for r in rows if r['arm']==arm),
                  'worst_sample_wall_us':max(r['sum_wall']['max_us'] for r in rows if r['arm']==arm),
                  'samples_ge_1ms':sum(r['sum_wall']['at_least_1ms'] for r in rows if r['arm']==arm),
                  'samples':sum(r['cycles'] for r in rows if r['arm']==arm)} for arm in order}
    with a.out.open('x') as f:json.dump({'schema':'r22-cpu-profile.v1','episodes':len(rows),'realtime_proven':False,
       'isolated_machine':False,'summary':summary,'rows':rows},f,indent=2,allow_nan=False);f.write('\n')
    print(summary)


if __name__=='__main__':main()
