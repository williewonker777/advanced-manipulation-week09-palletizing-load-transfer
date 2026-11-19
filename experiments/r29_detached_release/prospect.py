"""Physically replay two immutable arms on previously separate detached inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from experiments.r25_fragment import study as old
from experiments.r26_compare.legacy import fork
from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r29_detached_release.candidate import Detector
from experiments.r29_force_tube.known import sha

HERE=Path(__file__).resolve().parent


def one(job):
    case,mechanics,plan=job
    row=lab.paired_case(case,mechanics,{'r25_joint':C3,'r26':Detector},plan)
    row['records']={'c3':row['records']['r25_joint'],'endpoint':row['records']['r26']}
    return row


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--plan-sha',required=True)
    cli.add_argument('--out',required=True,type=Path)
    cli.add_argument('--workers',choices=(1,2),default=2,type=int)
    a=cli.parse_args(argv)
    if a.out.exists() or a.out.is_symlink(): cli.error('fresh output required')
    p=HERE/'prospective.json'
    if sha(p)!=a.plan_sha: raise ValueError('prospective case plan differs')
    prospect=json.loads(p.read_bytes())
    if prospect['candidates']!={'c3':sha(HERE.parent/'r26_yaw_acquisition/candidate.py'),
                                'endpoint':sha(HERE/'candidate.py')}:raise ValueError('different arm')
    controls=json.loads(lab.PLAN.read_bytes())
    jobs=[(c,controls['variants'][m],controls)for c in prospect['paired']for m in prospect['mechanics']]
    with ProcessPoolExecutor(a.workers)as pool:rows=list(pool.map(one,jobs))
    guard=fork(old.guards,ANCHOR='c3');summary=fork(old.summarize,ANCHOR='c3',guards=guard)
    counts,losses=summary(rows,['endpoint'])
    paired={k:{arm:sum(r['records'][arm]['events'][k]['strict']for r in rows)for arm in ('c3','endpoint')}
            for k in ('isolated_first','isolated_second','both_first','both_second','long_single')}
    out=dict(schema='r29-prospective-detached-results.v1',episodes=sum(r['episodes']for r in rows),
             plan_sha256=a.plan_sha,positive_all=all(r['passed']for r in rows),counts=counts,
             losses=losses['endpoint'],paired_strict=paired,rows=rows)
    with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(out['episodes'],counts,'losses',len(out['losses']),paired,'qualified',out['positive_all'])


if __name__=='__main__':main()
