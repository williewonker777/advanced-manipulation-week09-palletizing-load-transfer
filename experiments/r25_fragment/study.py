"""Fresh disclosed/nearby actual-plant trials, anchor and candidate, plus per-event occupancy and guard checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
from pathlib import Path
from types import FunctionType

import numpy as np
from experiments.r23_holdout import evaluator as old
from experiments.r24_holdout import evaluator as lab
from experiments.r24_composed.detector import Detector as Joint
from experiments.r24_loaded import studies as exposed

HERE=Path(__file__).resolve().parent
ANCHOR='r24_joint'


def constructor(key):
    if key==ANCHOR:return Joint
    module,member=key.split(':')
    if not module.startswith('experiments.r25_'):raise ValueError('R25 research only')
    return getattr(importlib.import_module(module),member)


def old_event(log,refs,begin,end,quiet,radius):
    e=old.event_record(log,refs,begin,end,quiet,radius)
    e['active_samples_in_slot']=int(np.asarray(log['detect'],bool)[begin:end].sum())
    return e


def old_common(log,impacts,starts,window):
    rec=old.record_common(log,impacts,starts,window)
    rec['false_active_samples']=lab.false_active_samples(log,starts,window)
    return rec


def capture(job):
    cohort,kind,c,v,keys,p=job
    auth24=cohort not in ('r23','nearby')
    anchors=('r23_free','r24') if auth24 else ('r22_light','r23')
    classes={anchors[0]:Joint,**{anchors[1] if i==0 else k:constructor(k) for i,k in enumerate(keys)}}
    if auth24:
        result=lab.one((kind,c,v,classes,p));sig=result['r23_physics_sha256']
    else:
        fn={'paired':old.paired_case,'single':old.single_case,'negative':old.negative_case}[kind]
        call=FunctionType(fn.__code__,dict(fn.__globals__,event_record=old_event,record_common=old_common))
        result=call(c,v,classes,p);sig=result['r22_physics_sha256']
    rec=result['records']
    return {'kind':kind,'id':c['id'],'variant':v['kind'],'episodes':result['episodes'],
            'physical_signature':sig,'records':{ANCHOR:rec[anchors[0]],**{
            k:rec[anchors[1] if i==0 else k] for i,k in enumerate(keys)}}}


def jobs(cohort,keys):
    if cohort in ('r23','nearby'):
        s='disclosed' if cohort=='r23' else 'nearby'
        return [(cohort,k,c,v,keys,p)for k,c,v,_,p in exposed.jobs(s,'unused')]
    p=json.loads(lab.PLAN.read_bytes())
    if cohort=='stress':
        s=json.loads((HERE/'selection_policy.json').read_bytes())['paired_stress']
        base=next(c for c in p['paired'] if c['id']=='p24_transfer_near_cross')
        return [(cohort,'paired',dict(base,id=f'cross_stress_{i}',impacts=[
            [s['first_time'],s['first_vector']],[s['first_time']+gap,vector]]),p['variants'][mech],keys,p)
            for i,(gap,vector) in enumerate(s['conditions']) for mech in s['mechanics']]
    tasks=[(cohort,k,c,v,keys,p) for k in ('negative','single','paired')
           for c in p[k] for v in p['variants'].values()]
    if cohort=='target':return [x for x in tasks if x[2]['id']=='p24_transfer_near_cross' and x[3]['kind']=='actuator_delay_gain']
    if cohort=='sample':return [x for x in tasks if x[3]['kind']=='stock' or (x[2]['id']=='p24_transfer_near_cross' and x[3]['kind']=='actuator_delay_gain')]
    if cohort!='r24':raise ValueError(cohort)
    return tasks


def events(x):
    return x.get('events',{'single':x['event']} if 'event' in x else {})


def contained(before,after):
    pool=list(before)
    for x in after:
        k=next((i for i,y in enumerate(pool) if abs(y-x)<.001000001),None)
        if k is None:return False
        pool.pop(k)
    return True


def guards(row,arm):
    a=row['records'][ANCHOR];b=row['records'][arm];g={}
    for n,x in a.get('twins',{'none':a}).items():
        y=b.get('twins',{'none':b})[n]
        g[n+':force']=x['force_exact']['passed'] and y['force_exact']['passed']
        g[n+':false']=contained(x['false_times_s'],y['false_times_s'])
        if 'false_active_samples' in x:g[n+':false_cost']=y['false_active_samples']<=x['false_active_samples']
        if n=='none' and 'loaded_transfer_normal' in x:
            g[n+':normal_cost']=y['loaded_transfer_normal']['active_samples']<=x['loaded_transfer_normal']['active_samples']
        g[n+':tool']=old.tool_floor(x['tool'],y['tool'],5) and (not y['tool']['prealarm'] or x['tool']['prealarm'])
        g[n+':tail']=not y['stuck_tail'] or x['stuck_tail']
    for n,x in events(a).items():
        y=events(b)[n]
        g[n+':time']=not x['strict'] or (y['strict'] and y['delays_ms'][0]<=x['delays_ms'][0]+5)
        g[n+':cost']=not x['strict'] or y['active_samples_in_slot']<=x['active_samples_in_slot']+5
        g[n+':duplicates']=y['duplicate_rises']<=x['duplicate_rises']
        g[n+':new_prealarm']=not x['quiet_before'] or y['quiet_before']
    for n,v in b.items():
        if n.startswith('physical_exact'):g[n]=v
        if n.startswith('prefix'):g[n]=v['passed'] and a[n]['passed']
    return [k for k,passed in g.items() if not passed]


def summarize(rows,keys):
    totals={};reg={}
    for arm in [ANCHOR,*keys]:
        recs=[r['records'][arm] for r in rows]; ev=[y for x in recs for y in events(x).values()]
        tw=[t for x in recs for t in x.get('twins',{'none':x}).values()]
        totals[arm]={'strict':sum(e['strict'] for e in ev),'slots':len(ev),'duplicate_edges':sum(e['duplicate_rises'] for e in ev),
                     'duplicate_slots':sum(e['duplicate_rises']>0 for e in ev),'false_rises':sum(len(t['false_times_s']) for t in tw),
                     'all_four':sum(all(y['strict'] for y in x['events'].values()) for x in recs if 'events' in x)}
        if arm!=ANCHOR:reg[arm]=[[r['id'],r['variant'],guards(r,arm)] for r in rows if guards(r,arm)]
    return totals,reg


def run(cohort,keys,workers):
    tasks=jobs(cohort,tuple(keys))
    with ProcessPoolExecutor(workers)as pool:rows=list(pool.map(capture,tasks))
    totals,reg=summarize(rows,keys)
    return {'schema':'r25-disclosed-original-simulator-comparison.v1','cohort':cohort,'independent':False,
            'arms':keys,'episodes':sum(r['episodes']for r in rows),'totals':totals,'regressions':reg,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',choices=('target','stress','sample','r24','r23','nearby'),required=True)
    p.add_argument('--arm',action='append',required=True);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():raise ValueError('fresh output required')
    doc=run(a.cohort,a.arm,a.workers)
    with a.out.open('x')as f:json.dump(doc,f,allow_nan=False);f.write('\n')
    print(doc['episodes'],doc['totals'],doc['regressions'])
