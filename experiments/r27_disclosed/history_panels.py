"""Replay the disclosed historical panels against the genuine R26 C3 records."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from experiments.r25_fragment import legacy as legacy_study
from experiments.r25_fragment import study as older
from experiments.r25_holdout import evaluator as r25
from experiments.r26_compare import study as r26
from experiments.r26_compare.legacy import fork
from experiments.r27_disclosed.contrasts import BASE, ROOT, resolve, sha
from experiments.r27_disclosed.previous import summarize

LEGACY=('legacy66','weak','pairs')
HISTORICAL=('r25','r24','r23','weak32','grip','nearby','stress',*LEGACY)


def capture(spec):
    panel,kind,case,mech,keys,plan=spec;arm=keys[0]
    if panel=='r25':
        row=r25.one((kind,case,mech,{'r24_joint':resolve(BASE),'r25':resolve(arm)},plan))
        return dict(kind=kind,id=case['id'],variant=mech['kind'],episodes=row['episodes'],
                    physical_signature=row['r24_physics_sha256'],records={BASE:row['records']['r24_joint'],arm:row['records']['r25']})
    row=fork(older.capture,Joint=resolve(BASE),constructor=resolve)(spec)
    row['records'][BASE]=row['records'].pop(older.ANCHOR)
    return row


def legacy_case(job):
    return fork(legacy_study.case,Joint=resolve(BASE),constructor=resolve)(job)


def legacy_weak(job):
    return fork(legacy_study.weak,Joint=resolve(BASE),constructor=resolve)(job)


def legacy_pair(job):
    return fork(legacy_study.pair,Joint=resolve(BASE),constructor=resolve)(job)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True)
    p.add_argument('--panel',choices=HISTORICAL,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args(argv)
    if os.path.lexists(a.out) or a.arm==BASE:p.error('distinct treatment and fresh output')
    resolve(a.arm);floor=json.loads((ROOT/'results/r26/exposed_choice.json').read_bytes())
    if sha(a.reference)!=floor['receipt_sha256']['development_'+a.panel]:raise ValueError('immutable R26 reference required')
    old=json.loads(a.reference.read_bytes())
    if a.panel in LEGACY:
        doc=fork(legacy_study.run,case=legacy_case,weak=legacy_weak,pair=legacy_pair)(a.panel,a.arm,a.workers)
        rows=doc['rows'];counts=None;bad=doc['regressions']
    else:
        tasks=r26.plan_jobs(a.panel,a.arm)
        with ProcessPoolExecutor(a.workers)as pool:rows=list(pool.map(capture,tasks))
        counts,bad=summarize(rows,a.arm)
    if len(old['rows'])!=len(rows):raise ValueError('wrong number of rows')
    for x,y in zip(old['rows'],rows):
        if x['id']!=y['id'] or x.get('variant',x.get('surface'))!=y.get('variant',y.get('surface')):
            raise ValueError('different original condition')
        if a.panel in LEGACY:
            if x['candidate']!=y['anchor'] or x.get('physics')!=y.get('physics'):raise ValueError('wrong archived baseline')
        elif x['records'][BASE]!=y['records'][BASE] or x['physical_signature']!=y['physical_signature']:
            raise ValueError('different R26 result or physics: '+x['id'])
    report=dict(schema='r27-revisited-physical-history.v1',independent=False,arm=a.arm,baseline=BASE,panel=a.panel,
                reference_sha256=sha(a.reference),episodes=sum(r['episodes']for r in rows),counts=counts,regressions=bad,rows=rows)
    with a.out.open('x')as f:json.dump(report,f,allow_nan=False);f.write('\n')
    print(a.panel,report['episodes'],'new protection failures',len(bad))


if __name__=='__main__':
    main()
