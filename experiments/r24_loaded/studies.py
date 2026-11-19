"""Evaluate disclosed R23 conditions and counterfactual newly-applied shocks near the former false rise."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path

from experiments.r23_detached.free import Detector as Free
from experiments.r23_holdout import evaluator as held
from experiments.r24_loaded.control import Projected, Sliding

HERE=Path(__file__).resolve().parent
ARMS={'projected':Projected,'sliding':Sliding}


def get_arm(key):
    if key in ARMS:return ARMS[key]
    module,symbol=key.split(':')
    if not module.startswith('experiments.r24_'):raise ValueError('expected local R24 research arm')
    return getattr(importlib.import_module(module),symbol)


def item(job):
    kind,case,variant,key,plan=job
    # The immutable R23 *scorer* reserves these two keys. Do not use its hardwired
    # default baseline here: both actual constructors are supplied explicitly.
    row=held.one((kind,case,variant,{'r22_light':Free,'r23':get_arm(key)},plan))
    row['records']={'r23_free':row['records']['r22_light'],'candidate':row['records']['r23']}
    row['r23_physical_sha256']=row.pop('r22_physics_sha256')
    return row


def reduce(rows, arm):
    sums={k:0 for k in ('singles','single_n','pair_slots','pair_n','complete','false','duplicate_slots','tool_hits','tool_n')}
    for row in rows:
        a=row['records'][arm];sums['false']+=sum(len(t['false_times_s']) for t in a.get('twins',{'none':a}).values())
        if row['kind']=='negative':
            sums['tool_hits']+=int(a['tool']['detected']);sums['tool_n']+=int(a['tool']['present'])
        else:
            name='singles' if row['kind']=='single' else 'pair_slots'
            events=a.get('events',{'single':a.get('event')}).values()
            sums[name]+=sum(e['strict'] for e in events);sums['single_n' if name=='singles' else 'pair_n']+=len(events)
            sums['duplicate_slots']+=sum(e['duplicate_rises']>0 for e in events)
            if name=='pair_slots':sums['complete']+=all(e['strict'] for e in events)
    return sums


def compare(row):
    a=row['records']['r23_free'];b=row['records']['candidate'];kind=row['kind']
    flags={'physical':b['physical_exact_vs_r22']}
    for k in b:
        if k.startswith('prefix'):flags[k]=b[k]['passed']
    for k,x in a.get('twins',{'none':a}).items():
        y=b.get('twins',{'none':b})[k]
        flags[k+':input']=all((x['force_exact']['passed'],y['force_exact']['passed']))
        flags[k+':false']=len(y['false_times_s'])<=len(x['false_times_s'])
        flags[k+':tail']=not y['stuck_tail'] or x['stuck_tail']
        flags[k+':tool']=held.tool_floor(x['tool'],y['tool'],5) and (not y['tool']['prealarm'] or x['tool']['prealarm'])
    if kind!='negative':
        for k,x in a.get('events',{'single':a.get('event')}).items():
            y=b.get('events',{'single':b.get('event')})[k]
            flags[k+':no_loss_or_delay']=not x['strict'] or (y['strict'] and y['delays_ms'][0]<=x['delays_ms'][0]+5)
            flags[k+':no_duplicate']=y['duplicate_rises']<=x['duplicate_rises']
    return [k for k,v in flags.items() if not v]


def jobs(study,key):
    p=json.loads(held.PLAN.read_bytes()); cfg=json.loads((HERE/'development.json').read_bytes())
    if study in ('disclosed','negative'):
        return [(k,c,v,key,p) for k in ('negative','single','paired') for c in p[k] for v in p['variants'].values()
                if study=='disclosed' or (c['id']=='n23_detached_heavy_interp' and v['kind']=='actuator_skew')]
    c0=next(c for c in p['negative'] if c['id']=='n23_detached_heavy_interp');n=cfg['nearby_design']
    return [('single',dict(c0,id=f'{t:.3f}:'+str(f),phase='loaded_transfer',impact=[t,f]),p['variants'][v],key,p)
            for t in n['nominal_force_times_s'] for f in n['wrenches_N'] for v in n['mechanics']]


def run(study,key,workers):
    tasks=jobs(study,key)
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(item,tasks))
    issues=[{'id':r['id'],'variant':r['variant'],'flags':compare(r)} for r in rows if compare(r)]
    return {'study':study,'candidate':key,'r23_base':'experiments.r23_detached.free:Detector','episodes':sum(r['episodes'] for r in rows),
            'plan_sha256':hashlib.sha256(held.PLAN.read_bytes()).hexdigest(),'summary':{a:reduce(rows,a) for a in ('r23_free','candidate')},
            'regressions':issues,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',choices=('disclosed','nearby','negative'),required=True)
    p.add_argument('--arm',required=True);p.add_argument('--workers',type=int,choices=(1,2,3),default=2);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();d=run(a.study,a.arm,a.workers)
    with a.out.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
    print(d['summary'],d['regressions'])
