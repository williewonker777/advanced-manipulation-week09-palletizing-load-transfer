"""Run the disclosed 156-episode R26 panel; authenticate C3 against its unrevised archive."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from experiments.r25_fragment import study as floors
from experiments.r26_holdout import evaluator as lab
from experiments.r26_compare.legacy import fork
from experiments.r27_disclosed.contrasts import BASE, HERE, ROOT, resolve, sha


def one(spec):
    kind,case,mech,arm,expected,plan=spec
    new=resolve(arm); baseline=resolve(BASE)
    row=lab.one((kind,case,mech,{'r25_joint':baseline,'r26':new},plan))
    if (row['r25_physics_sha256']!=expected['r25_physics_sha256']
            or row['records']['r25_joint']!=expected['records']['r26']):
        raise ValueError('stored R26 physics or event did not reproduce: '+case['id'])
    row['records']={BASE:row['records']['r25_joint'],arm:row['records']['r26']}
    return row


def summarize(rows, arm):
    guard=fork(floors.guards,ANCHOR=BASE)
    agg=fork(floors.summarize,ANCHOR=BASE,guards=guard)
    counts,bad=agg(rows,[arm])
    for value in counts.values():
        value['all_registered_paired_slots']=value.pop('all_four')
    return counts,bad[arm]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args(argv)
    if os.path.lexists(a.out) or a.arm==BASE:p.error('distinct treatment and fresh output required')
    spec=json.loads((HERE/'policy.json').read_bytes());resolve(a.arm)
    if sha(a.archive)!=spec['prior_archive_sha256'] or sha(ROOT/'experiments/r26_yaw_acquisition/candidate.py')!=spec['baseline_sha256']:
        raise ValueError('wrong baseline or archived output')
    doc=json.loads(a.archive.read_bytes()); old={(r['kind'],r['id'],r['variant']):r for r in doc['rows']}
    plan=json.loads(lab.PLAN.read_bytes())
    if sha(lab.PLAN)!=doc['plan_sha256']:raise ValueError('wrong R26 plan')
    jobs=[(k,c,m,a.arm,old[k,c['id'],m['kind']],plan) for k in ('negative','single','paired') for c in plan[k] for m in plan['variants'].values()]
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(one,jobs))
    counts,bad=summarize(rows,a.arm)
    result={'schema':'r27-revisited-r26-real-course.v1','independent':False,'episodes':sum(r['episodes'] for r in rows),
            'baseline':BASE,'arm':a.arm,'verified_prior_archive':sha(a.archive),'counts':counts,'regressions':bad,'rows':rows}
    if len(rows)!=30 or result['episodes']!=156:raise ValueError('incomplete R26 cohort')
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(counts, 'protected failures',len(bad))


if __name__=='__main__':
    main()
