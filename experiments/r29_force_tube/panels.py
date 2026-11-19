"""Unchanged disclosed R27 signed-force and held-out-normal guards with explicit R29 pins."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from experiments.r26_compare.legacy import fork
from experiments.r26_holdout import evaluator as lab
from experiments.r27_disclosed import contrasts, normal_audit
from experiments.r29_force_tube.known import ARCHIVE, BASE, C3, ROOT, resolve, sha

BASE_CONTRAST='2bfccede7dbe3d31894b19dddb9b81f95647431c1273121719b01fafffbe4f89'


def one(job):
    panel, name, digest, inputs=job
    classes={BASE:C3,name:resolve(name,digest)}
    check=normal_audit.one if panel in ('normal','additional') else contrasts.capture
    return fork(check,resolve=classes.__getitem__)(inputs)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',required=True,choices=('directions','normal','additional'));p.add_argument('--candidate',required=True)
    p.add_argument('--sha',required=True);p.add_argument('--archive',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--jobs',choices=(1,2),type=int,default=2);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    resolve(a.candidate,a.sha);plan=json.loads(lab.PLAN.read_bytes());jobs=[]
    if a.panel=='normal':
        reg=json.loads((ROOT/'experiments/r27_load_change/registration.json').read_bytes())
        jobs=[(c,plan['variants'][v],a.candidate)for c in reg['audit_normals']for v in reg['mechanics']]
    elif a.panel=='additional':
        reg=json.loads((ROOT/'experiments/r27_load_change/settled_registration.json').read_bytes())
        jobs=[(c,v,a.candidate)for c in reg['additional_future_normals']for v in plan['variants'].values()]
    else:
        if not a.reference or not a.archive or sha(a.reference)!=BASE_CONTRAST or sha(a.archive)!=ARCHIVE:
            p.error('authentic baseline and archive required')
        controls=json.loads((contrasts.HERE/'policy.json').read_bytes())['disclosed_contrasts']
        source=json.loads(a.archive.read_bytes());indices={(r['id'],r['variant']):r for r in source['rows']}
        for name in controls['cases']:
            c=next(v for v in plan['single']if v['id']==name)
            for m in controls['mechanics']:
                row=indices[name,m];jobs.append((c,plan['variants'][m],a.candidate,row['records']['r26'],row['r25_physics_sha256'],plan))
    with ProcessPoolExecutor(a.jobs)as pool:rows=list(pool.map(one,[(a.panel,a.candidate,a.sha,j)for j in jobs]))
    out=dict(schema='r29-actual-signed-and-normal-guards.v1',panel=a.panel,candidate=a.candidate,candidate_sha256=a.sha,rows=rows)
    if a.panel in ('normal','additional'):
        out.update(episodes=2*len(rows),losses=[[r['id'],r['variant']]for r in rows if not all(r['gates'].values())],
                   false_rises={n:sum(len(r['records'][n]['false_times_s'])for r in rows)for n in (BASE,a.candidate)})
    else:
        comp=contrasts.compare(rows,json.loads(a.reference.read_bytes())['rows'])
        out.update(episodes=sum(r['episodes']for r in rows),strict=sum(t['event']['strict']for r in rows for t in r['forced'].values()),
                   comparison=comp,losses=comp['regressions'],baseline_sha256=BASE_CONTRAST)
    with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(a.panel,out['episodes'],'losses',len(out['losses']),'strict',out.get('strict'),'false',out.get('false_rises'))


if __name__=='__main__':main()
