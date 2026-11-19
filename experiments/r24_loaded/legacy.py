"""Regression tests with original historical scorers and fully identified controller/detector pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import FunctionType

import my_detector
from experiments.r22_challenge import evaluator as r22
from experiments.r23_detached.free import Detector as Free
from experiments.r23_detached import exposed
from experiments.r24_loaded.studies import get_arm


def historical(job):
    kind,c,v,key,p=job
    row=r22.one((kind,c,v,{'r10':my_detector.Detector,'r21_yaw':Free,'r22':get_arm(key)},p))
    a,b=row['records']['r21_yaw'],row['records']['r22']
    gates=exposed.compare(b,a,kind)
    gates['physical_force_prefix']=all(val for k,val in row['comparison'].items() if 'prefix' in k or k=='physical_and_force_exact')
    return {'id':c['id'],'kind':kind,'variant':v['kind'],'episodes':row['episodes'],'physics':row['r10_physics_sha256'],
            'r23':a,'candidate':b,'gates':gates}


def weak(job):
    from experiments.r21_weak_challenge import evaluator as old
    case,key=job
    row=old.forced_row(case,{'r10':Free,'candidate':get_arm(key)},20);a,b=row['arms'].values();g=row['comparisons']['candidate']['checks']
    for name,rec in [('r23',a),('candidate',b)]:
        rec['extra']=max(0,sum(case['time']<=t<case['time']+.32 for t in rec['force']['all_rise_times'])-1)
    g['no_duplicates']=b['extra']<=a['extra']
    g['timing']=not a['force']['strict_true_positive'] or (b['force']['strict_true_positive'] and b['force']['delay_ms']<=a['force']['delay_ms']+5)
    return {'id':case['id'],'role':case['role'],'episodes':4,'r23':a,'candidate':b,'gates':g}


def pair(job):
    from experiments.r22_combined import known_pairs as old
    case,stage,mode,key=job
    # Only these two actual constructors are run, 2 * 4 twins per job.
    call=FunctionType(old.episode_group.__code__,dict(old.episode_group.__globals__,ARMS={'r10':Free,'yaw':get_arm(key)}))
    row=call((case,stage,mode,False));a,b=row['arms']['r10'],row['arms']['yaw']
    g={'no_loss':not a['strict'] or b['strict'],'no_added_safety_failures':b['preserve_genuine_quiet_false_tool_tail']['r10'],
       'input_certified':a['exact_force_and_physics'] and b['exact_force_and_physics']}
    return {'id':case['id'],'stage':stage,'surface':mode,'episodes':8,'r23':a,'candidate':b,'gates':g}


def run(study,key,n):
    root=Path(__file__).resolve().parents[1]
    if study=='legacy66':
        p=json.loads(r22.PLAN.read_bytes());jobs=[(k,c,v,key,p) for k in ('negative','single','paired') for c in p[k] for v in p['variants'].values()]
        fn=historical
    elif study=='weak':
        jobs=[(c,key) for c in json.loads((root/'r21_weak_challenge/plan.json').read_bytes())['forced']];fn=weak
    else:
        a=json.loads((root/'r21_surface_pairs/plan.json').read_bytes())['cases']
        b=json.loads((root/'r21_yaw_continuity/plan.json').read_bytes())['fresh']
        jobs=[(c,stage,mode,key) for stage,cases in [('surface',a),('history',b)] for c in cases for mode in ('stock','changed')]
        fn=pair
    with ProcessPoolExecutor(n) as pool:rows=list(pool.map(fn,jobs))
    issues=[{'id':r['id'],'mechanics':r.get('variant',r.get('surface')),'flags':[k for k,v in r['gates'].items() if not v]} for r in rows if not all(r['gates'].values())]
    return {'study':study,'unseen':False,'arm':key,'episodes':sum(r['episodes'] for r in rows),'regressions':issues,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',choices=('legacy66','weak','pairs'),required=True)
    p.add_argument('--arm',required=True);p.add_argument('--workers',type=int,choices=(1,2,3),default=2);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();doc=run(a.study,a.arm,a.workers)
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print(doc['episodes'],doc['regressions'])
