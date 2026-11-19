"""Run the authentic previously disclosed R26 cohort with a new explicitly pinned passive detector."""
import argparse,hashlib,importlib,json,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from experiments.r25_fragment import study as old
from experiments.r26_compare.legacy import fork
from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r27_disclosed.contrasts import BASE
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE='854459ef3292f3c20eae7bc8ba829a83eae70e6ce2b5e66f5b89abeb6836489a'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(entry,digest):
    mod,cls=entry.split(':');parts=mod.split('.')
    if (cls!='Detector' or len(parts)!=3 or parts[0]!='experiments' or not parts[1].startswith('r29_') or
        not all(v.isidentifier() for v in parts) or sha(ROOT/Path(*parts).with_suffix('.py'))!=digest):
        raise ValueError('explicit pinned study source required')
    return getattr(importlib.import_module(mod),cls)


def one(job):
    kind,c,m,entry,source,ref,plan=job
    row=lab.one((kind,c,m,{'r25_joint':C3,'r26':resolve(entry,source)},plan))
    if row['r25_physics_sha256']!=ref['r25_physics_sha256'] or row['records']['r25_joint']!=ref['records']['r26']:
        raise ValueError('actual old physics or C3 result differs')
    row['records']={BASE:row['records']['r25_joint'],entry:row['records']['r26']}
    return row


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--sha',required=True)
    p.add_argument('--archive',required=True,type=Path);p.add_argument('--kind',choices=('all','paired','single','negative'),default='all');p.add_argument('--mechanics',default='all');p.add_argument('--case')
    p.add_argument('--jobs',choices=(1,2),type=int,default=2);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if os.path.lexists(a.out) or sha(a.archive)!=ARCHIVE:p.error('prior authentic R26 archive and fresh output required')
    resolve(a.candidate,a.sha)
    prior=json.loads(a.archive.read_bytes());plan=json.loads(lab.PLAN.read_bytes())
    if prior['plan_sha256']!=sha(lab.PLAN):raise ValueError('incorrect prior plan')
    refs={(r['kind'],r['id'],r['variant']):r for r in prior['rows']}
    jobs=[(k,c,m,a.candidate,a.sha,refs[k,c['id'],m['kind']],plan) for k in ('negative','single','paired')for c in plan[k]for m in plan['variants'].values()
          if a.kind in (k,'all')and a.mechanics in (m['kind'],'all')and(not a.case or c['id']==a.case)]
    if not jobs:raise ValueError('empty selection')
    with ProcessPoolExecutor(a.jobs)as pool:rows=list(pool.map(one,jobs))
    guard=fork(old.guards,ANCHOR=BASE);summary=fork(old.summarize,ANCHOR=BASE,guards=guard)
    counts,bad=summary(rows,[a.candidate]);paired={}
    for key in ('isolated_first','isolated_second','both_first','both_second','long_single'):
        paired[key]={name:sum(row['records'][name]['events'][key]['strict']for row in rows if row['kind']=='paired') for name in (BASE,a.candidate)}
    report=dict(schema='r29-known-native-vs-course-replay.v1',candidate=a.candidate,candidate_sha256=a.sha,disclosed=True,
                authenticated_prior=ARCHIVE,episodes=sum(r['episodes']for r in rows),counts=counts,relative_losses=bad[a.candidate],paired_strict=paired,rows=rows)
    with a.out.open('x')as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print('episodes',report['episodes'],counts,'losses',len(bad[a.candidate]),'both_second',paired['both_second'])


if __name__=='__main__':main()
