"""Measure controller and detector compute/update on corresponding ticks of four R25-relevant physical twins."""
import argparse
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np
from experiments.r22_timing.study import bind, stats
from experiments.r24_holdout import evaluator as lab
from experiments.r25_fragment.study import ANCHOR,constructor
from my_torque import Controller


def run(arm):
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    plan=json.loads(lab.PLAN.read_bytes());case=next(x for x in plan['paired'] if x['id']=='p24_transfer_near_cross')
    mechanics=plan['variants']['actuator_delay_gain']
    twins={'none':(None,None),'first':(case['impacts'][0],None),'second':(None,case['impacts'][1]),'both':tuple(case['impacts'])}
    rows=[]
    for repeat in range(2):
        for name,forces in twins.items():
            for key in ([ANCHOR,arm] if repeat==0 else [arm,ANCHOR]):
                ctrl=Controller();det=constructor(key)()
                wc,cc,nc=bind(ctrl,'compute',6000);wd,cd,nd=bind(det,'update',6000);wr,cr,nr=bind(det,'reset',1)
                fn=lab.run_episode
                call=FunctionType(fn.__code__,dict(fn.__globals__,Controller=lambda:ctrl))
                _,log=call(case,mechanics,lambda:det,forces)
                if (nc,nd,nr)!=([6000],[6000],[1]) or not lab.certified.exact_force(log,forces)['passed']:raise ValueError('invalid trial')
                w=wc+wd;c=cc+cd
                rows.append({'repeat':repeat,'arm':key,'twin':name,'samples':6000,'sum_wall':stats(w),'sum_cpu':stats(c),
                             'max_wall_tick':int(np.argmax(w)),'max_cpu_tick':int(np.argmax(c)),
                             'detector_reset_wall_us':float(wr[0])/1000,'detector_reset_cpu_us':float(cr[0])/1000,
                             'physics':lab.certified.physical_hash(log),'residual':lab.certified.signal_sha(log['r']),
                             'alarm':lab.certified.signal_sha(log['detect'])})
    for tw in twins:
        m=[x for x in rows if x['twin']==tw]
        for name in ('physics','residual'):
            if any(x[name]!=m[0][name] for x in m[1:]):raise ValueError('different protected trace')
        for name in (ANCHOR,arm):
            if len({x['alarm'] for x in m if x['arm']==name})!=1:raise ValueError('repeated alarm differs')
    totals={}
    for key in (ANCHOR,arm):
        m=[x for x in rows if x['arm']==key]
        totals[key]={'samples':sum(x['samples'] for x in m),'max_detector_reset_wall_us':max(x['detector_reset_wall_us'] for x in m),
                     **{k:{'worst_episode_p99_us':max(x[k]['p99_us']for x in m),'max_us':max(x[k]['max_us']for x in m),
                           'at_least_1ms':sum(x[k]['at_least_1ms']for x in m)} for k in ('sum_wall','sum_cpu')}}
    return {'schema':'r25-shared-cpu-profile.v1','cpu_simulation_only':True,'hard_realtime_proven':False,
            'measured':'Controller.compute and detector.update on the same 6000 ticks; detector reset separate; solver, IO and end-to-end latency not included.',
            'new_episodes':len(rows),'summary':totals,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    doc=run(a.arm)
    with a.out.open('x')as f:json.dump(doc,f,allow_nan=False,indent=2);f.write('\n')
    print(doc['summary'])
