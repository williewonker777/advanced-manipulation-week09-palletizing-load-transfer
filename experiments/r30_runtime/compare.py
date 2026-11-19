"""Compare controller plus detection time in complete, finite, same-trajectory episodes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter_ns, thread_time_ns

import numpy as np
from my_detector import Detector as Production
from my_torque import Controller
from experiments.r29_disjoint_events.candidate import Detector as Composed
from experiments.r30_yaw_headroom.candidate import Detector as Headroom
from sim.runner import run_episode
from tools import r21_replay as shared

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name('plan.json')
ARMS = {'production':Production,'composed':Composed,'headroom':Headroom}
FIELDS = ('t','q','q_d','tau','e_mm','f_table','f_tool','qpos','f_ext')


def measure(fn, times, args):
    wall, cpu = perf_counter_ns(), thread_time_ns()
    result = fn(*args)
    dcpu = thread_time_ns()-cpu
    times.append((perf_counter_ns()-wall,dcpu))
    return result


def stats(ns):
    us = np.asarray(ns,dtype=float)*.001
    return dict(median_us=float(np.median(us)),p95_us=float(np.percentile(us,95)),p99_us=float(np.percentile(us,99)),
                max_us=float(us.max()),ge_1ms=int((us>=1000).sum()))


def episode(case, name):
    ctrl, det = [], []
    class C(Controller):
        def compute(self,*args):
            return measure(super().compute,ctrl,args)
    class D(ARMS[name]):
        def update(self,*args):
            return measure(super().update,det,args)
    metrics,log=run_episode(C(),box_mass=case['mass'],box_com=tuple(case['com']),pick_offset=case['offset'],
                            detector=D(),collision=case['collision'])
    if len(ctrl)!=6000 or len(det)!=6000 or metrics['diverged']:
        raise ValueError('incomplete or divergent episode')
    a,b=map(np.asarray,(ctrl,det)); combined=a+b
    dig=lambda k:hashlib.sha256(np.ascontiguousarray(log[k]).tobytes()).hexdigest()
    return dict(samples=6000,controller_wall=stats(a[:,0]),detector_wall=stats(b[:,0]),combined_wall=stats(combined[:,0]),
                combined_cpu=stats(combined[:,1]),physics_sha256={k:dig(k) for k in FIELDS},alarm_sha256=dig('detect'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():
        p.error('fresh destination required')
    plan=json.loads(PLAN.read_bytes()); manifest=(ROOT/shared.KIT_MANIFEST).read_bytes()
    def pins():
        return {n:shared.digest((ROOT/n).read_bytes()) for n in plan['sources']}
    if pins()!=plan['sources']:
        raise ValueError('changed method')
    kit=shared.authenticate_kit(ROOT,manifest); rows=[]
    for i,case in enumerate(plan['cases']):
        for rep in range(2):
            order=plan['order'][i%3];order=order if rep==0 else order[::-1]
            arms={name:episode(case,name) for name in order}
            if len({json.dumps(v['physics_sha256'],sort_keys=True) for v in arms.values()})!=1:
                raise ValueError('different physical trajectories')
            if rep and any(arms[n]['alarm_sha256']!=rows[-1]['arms'][n]['alarm_sha256'] for n in order):
                raise ValueError('repeat did not reproduce alarms')
            rows.append(dict(case=case['id'],repeat=rep,order=order,arms=arms))
    if pins()!=plan['sources'] or kit!=shared.authenticate_kit(ROOT,manifest):
        raise ValueError('source/kit changed')
    totals={name:dict(worst_episode_p99_us=max(row['arms'][name]['combined_wall']['p99_us'] for row in rows),
                      worst_wall_us=max(row['arms'][name]['combined_wall']['max_us'] for row in rows),
                      wall_ge_1ms=sum(row['arms'][name]['combined_wall']['ge_1ms'] for row in rows),
                      cpu_ge_1ms=sum(row['arms'][name]['combined_cpu']['ge_1ms'] for row in rows)) for name in ARMS}
    obj=dict(schema='r30-four-stock-timing.v1',plan_sha256=shared.digest(PLAN.read_bytes()),episodes=len(rows)*3,
             samples_per_arm=48000,affinity=sorted(os.sched_getaffinity(0)),rows=rows,totals=totals,
             identical_trajectories=True,repeat_alarms_identical=True,scope='controller+detector, excludes plant rendering and Python loop; local with competing processes, not WCET')
    with a.out.open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(totals,sort_keys=True))

if __name__=='__main__':
    main()
