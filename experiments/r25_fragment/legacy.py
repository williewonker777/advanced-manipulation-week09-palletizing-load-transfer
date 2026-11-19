"""Re-evaluate the 66 older physical conditions and short protection panels relative to the actual R24 joint."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import FunctionType

import my_detector
from experiments.r22_challenge import evaluator as hist
from experiments.r23_detached import exposed
from experiments.r24_loaded import legacy
from experiments.r24_composed.detector import Detector as Joint
from experiments.r25_fragment.study import constructor


def extra(log,refs,begin,end,quiet,radius):
    ans=hist.event_record(log,refs,begin,end,quiet,radius)
    ans['active_samples_in_slot']=int(log['detect'][begin:end].astype(bool).sum())
    return ans


def case(job):
    kind,c,v,key,p=job
    fn={'negative':hist.negative_case,'single':hist.single_case,'paired':hist.paired_case}[kind]
    call=FunctionType(fn.__code__,dict(fn.__globals__,event_record=extra))
    root=call(c,v,{'r10':my_detector.Detector,'r21_yaw':Joint,'r22':constructor(key)},p)
    a,b=root['records']['r21_yaw'],root['records']['r22']
    g=exposed.compare(b,a,kind)
    g['physics_force_prefix']=all(y for k,y in root['comparison'].items() if 'prefix' in k or k=='physical_and_force_exact')
    for name,x in exposed.events(a,kind).items():
        y=exposed.events(b,kind)[name]
        g[name+':cost']=not exposed.strict(x) or y['active_samples_in_slot']<=x['active_samples_in_slot']+5
        g[name+':quiet']=not x['quiet_before'] or y['quiet_before']
    return {'id':c['id'],'kind':kind,'variant':v['kind'],'episodes':root['episodes'],'physics':root['r10_physics_sha256'],
            'anchor':a,'candidate':b,'gates':g}


def weak(job):
    fn=FunctionType(legacy.weak.__code__,dict(legacy.weak.__globals__,Free=Joint,get_arm=constructor))
    row=fn(job);row['anchor']=row.pop('r23')
    return row


def pair(job):
    fn=FunctionType(legacy.pair.__code__,dict(legacy.pair.__globals__,Free=Joint,get_arm=constructor))
    row=fn(job);row['anchor']=row.pop('r23');g=row['gates']
    for name,x in row['anchor']['slots'].items():
        y=row['candidate']['slots'][name]
        if x['credited_delays_ms']:
            g[name+':time']=bool(y['credited_delays_ms'] and y['credited_delays_ms'][0]<=x['credited_delays_ms'][0]+5)
        if row['anchor']['strict']:
            g[name+':cost']=y['active_samples_in_slot']<=x['active_samples_in_slot']+5
    return row


def run(study,arm,workers):
    root=Path(__file__).resolve().parents[1]
    if study=='legacy66':
        p=json.loads(hist.PLAN.read_bytes());tasks=[(k,c,v,arm,p) for k in ('negative','single','paired')for c in p[k]for v in p['variants'].values()]
        fn=case
    elif study=='weak':
        tasks=[(c,arm)for c in json.loads((root/'r21_weak_challenge/plan.json').read_bytes())['forced']];fn=weak
    else:
        plans=[('surface','r21_surface_pairs/plan.json','cases'),('history','r21_yaw_continuity/plan.json','fresh')]
        tasks=[(c,group,mech,arm)for group,path,field in plans for c in json.loads((root/path).read_bytes())[field]for mech in ('stock','changed')]
        fn=pair
    with ProcessPoolExecutor(workers)as pool:rows=list(pool.map(fn,tasks))
    fail=[[r['id'],r.get('variant',r.get('surface')),[k for k,v in r['gates'].items()if not v]]for r in rows if not all(r['gates'].values())]
    return {'schema':'r25-historical-protection.v1','panel':study,'arm':arm,'episodes':sum(r['episodes']for r in rows),'regressions':fail,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True);p.add_argument('--study',choices=('legacy66','weak','pairs'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,choices=(1,2),default=2);q=p.parse_args()
    if q.out.exists():raise ValueError('fresh output required')
    j=run(q.study,q.arm,q.workers)
    with q.out.open('x') as f:json.dump(j,f,allow_nan=False);f.write('\n')
    print(j['episodes'],j['regressions'])
