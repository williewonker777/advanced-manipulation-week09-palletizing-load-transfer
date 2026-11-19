"""Save each newly simulated R23/experimental multi-arm comparison and certify it before aggregation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r23_detached.free import Detector as Free
from experiments.r23_holdout import evaluator as lab
from experiments.r24_loaded import studies

HERE=Path(__file__).resolve().parent


def compare_with_arms(row, arm):
    return studies.compare(dict(row,records={'r23_free':row['records']['r23_free'],'candidate':row['records'][arm]}))


def capture(job):
    kind,case,variant,keys,plan=job
    classes=dict(r22_light=Free)
    for i,key in enumerate(keys):
        classes['r23' if i==0 else key]=studies.get_arm(key)
    row=lab.one((kind,case,variant,classes,plan)); rec=row['records']
    return {'kind':kind,'id':case['id'],'variant':variant['kind'],'episodes':row['episodes'],
            'physical_hash':row['r22_physics_sha256'],'records':{'r23_free':rec['r22_light'],**{
                key:rec['r23' if i==0 else key] for i,key in enumerate(keys)}}}


def run(study,keys,workers):
    jobs=[(k,c,v,tuple(keys),p) for k,c,v,_,p in studies.jobs(study,keys[0])]
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(capture,jobs))
    return {'schema':'week09-r24-observed-comparison.v1','classification':'exposed development, not independent acceptance',
            'study':study,'arms':list(keys),'incumbent':'experiments.r23_detached.free:Detector','episodes':sum(r['episodes'] for r in rows),
            'summary':{a:studies.reduce(rows,a) for a in ['r23_free',*keys]},'regressions':{
                a:[{'id':r['id'],'variant':r['variant'],'failures':compare_with_arms(r,a)} for r in rows if compare_with_arms(r,a)] for a in keys},
            'rows':rows,'disclosed_plan_sha256':hashlib.sha256(lab.PLAN.read_bytes()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',choices=('nearby','disclosed','negative'),required=True)
    p.add_argument('--arm',action='append',required=True);p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():raise ValueError('new result required')
    d=run(a.study,a.arm,a.workers)
    with a.out.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
    print(d['summary'],d['regressions'])
