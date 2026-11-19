"""Shared-CPU simulation timing; align controller and detector calls on the same tick."""
import argparse
import json
import os
from pathlib import Path

from experiments.r22_timing.study import bind,stats
from experiments.r26_delivery_pair import study
from experiments.r26_delivery_pair.probe import ORIGINAL,run
from experiments.r24_transfer_replay.evaluator import physical_signature


def measure():
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    p=json.loads((ORIGINAL/'plan.json').read_bytes())
    cases=[('none',None)]+[(i['id'],(i['time_s'],i['force_n']))for i in p['late_impulses'] if i['id']in ('late_x_pos','late_z_neg')]
    rows=[]
    for rep in range(2):
        for name,inp in cases:
            for arm in (['r24','pair'] if rep==0 else ['pair','r24']):
                dc,cc=study.factory(arm);det,ctrl=dc(),cc()
                dw,dt,nd=bind(det,'update',6000);cw,ct,nc=bind(ctrl,'compute',6000)
                _,_,_,log=run(p['impulse_base'],lambda:det,inp,lambda:ctrl)
                if nc!=[6000] or nd!=[6000]:raise ValueError('incomplete compute sample')
                rows.append({'repeat':rep,'case':name,'arm':arm,'physical':physical_signature(log),
                             'samples':6000,'wall':stats(cw+dw),'thread':stats(ct+dt)})
    for name,_ in cases:
        if len({x['physical']for x in rows if x['case']==name})!=1:raise ValueError('mechanics changed')
    totals={}
    for arm in ('r24','pair'):
        r=[x for x in rows if x['arm']==arm]
        totals[arm]={'samples':sum(x['samples']for x in r),'p99_worst_wall_us':max(x['wall']['p99_us']for x in r),
                     'max_wall_us':max(x['wall']['max_us']for x in r),'wall_at_least_1ms':sum(x['wall']['at_least_1ms']for x in r),
                     'p99_worst_thread_us':max(x['thread']['p99_us']for x in r)}
    return {'schema':'r26-shared-cpu-torque-placement.v1','note':'Shared machine. Per-tick controller.compute plus detector.update only, not end-to-end or hard real-time.',
            'samples_per_arm':36000,'totals':totals,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);q=p.parse_args()
    if q.out.exists() or q.out.is_symlink():p.error('fresh output required')
    j=measure()
    with q.out.open('x')as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n')
    print(j['totals'])
