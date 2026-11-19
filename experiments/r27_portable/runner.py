"""Execute known physical twins inside an authenticated overlay; never read a new holdout."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.r26_holdout import evaluator as lab
from experiments.r27_disclosed.contrasts import BASE, LABELS, acquisition_in_force, compare, inputs, resolve, sha

HERE=Path(__file__).resolve().parent


def family(job):
    case,mechanics,arm,scope,expected,plan=job
    detector=resolve(arm)
    _,zero=lab.run_episode(case,mechanics,detector,(None,None))
    normal=lab.common(zero,(None,None),[],plan)
    forced={}; vectors=inputs(case['impact'])
    for key in (LABELS if scope=='directions' else ('registered',)):
        impact=vectors[key]
        _,log=lab.run_episode(case,mechanics,detector,(impact,None))
        start,boundary=lab.actual_start(log['t'],impact)
        event=lab.event(log,[zero],start,min(6000,start+plan['event_window_ms']),plan)
        prefix=lab.certified.prefix(zero,log,start,start)
        cm=lab.common(log,(impact,None),[start],plan)
        signature=lab.certified.trace_signature({'none':zero,'single':log})
        if not(prefix['passed'] and cm['force_exact']['passed'] and normal['force_exact']['passed']):
            raise ValueError('wrong force or mismatched prefix')
        if key=='registered' and (signature!=expected['physical']
                or (arm==BASE and (event!=expected['event'] or normal!=expected['normal'] or cm!=expected['single']))):
            raise ValueError('authorized physical or archived alert differs')
        forced[key]=dict(impact=impact,boundary=boundary,physical_signature=signature,common=cm,event=event,
                         in_force=acquisition_in_force(event,boundary),prefix=prefix)
    return dict(case=case['id'],mechanics=mechanics['kind'],normal=normal,forced=forced,episodes=1+len(forced))


def execute(key,scope,jobs):
    registry=json.loads((HERE/'registry.json').read_bytes())
    archive=json.loads((HERE/'reference.json').read_bytes())
    if archive['r26_plan_sha256']!=sha(lab.PLAN) or archive['r26_first_sha256']!=registry['r26_first_sha256']:
        raise ValueError('wrong historical registration')
    plan=json.loads(lab.PLAN.read_bytes());arms=(BASE,registry['arms'][key]['entrypoint'])
    mechanics=registry['mechanics'] if scope=='directions' else ['stock']
    by_case={c['id']:c for c in plan['single']};results={}
    for arm in arms:
        tasks=[(by_case[k],plan['variants'][m],arm,scope,archive['records'][k][m],plan) for k in registry['cases'] for m in mechanics]
        with ProcessPoolExecutor(jobs)as pool:results[arm]=list(pool.map(family,tasks))
    old,new=(results[k]for k in arms)
    if scope=='directions':
        changes=compare(new,old)
    else:
        changes={'registered_in_force_gains':[]}
        for a,b in zip(old,new):
            x,y=(z['forced']['registered']for z in (a,b))
            if x['physical_signature']!=y['physical_signature']:raise ValueError('plant changed across detectors')
            if y['in_force'] and not x['in_force']:changes['registered_in_force_gains'].append([b['case'],y['event']['delays_ms'][0]])
    return {'schema':'r27-kit-authenticated-known-twins.v1','scope':scope,'research_arm':key,'entrypoint':arms[1],
            'classification':'previously disclosed conditions; neither original servo demo nor independent generalization',
            'r26_independent_first_sha256':archive['r26_first_sha256'],'default_promoted':False,'mujoco':lab.require_runtime(),
            'episodes':sum(row['episodes']for group in results.values() for row in group),'comparison':changes,'records':results}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True)
    p.add_argument('--scope',choices=('registered','directions'),required=True);p.add_argument('--jobs',type=int,choices=(1,2),default=1)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.path.lexists(a.out):p.error('new output required')
    doc=execute(a.arm,a.scope,a.jobs)
    with a.out.open('x')as f:json.dump(doc,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(doc['scope'],doc['research_arm'],doc['episodes'])


if __name__=='__main__':
    main()
