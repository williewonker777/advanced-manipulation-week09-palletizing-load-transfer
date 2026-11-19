"""Does one retired acquisition hide a genuine subsequent force? Six fixed translated-vector contrasts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r25_fragment import study
from experiments.r26_compare.legacy import fork
from experiments.r26_holdout import evaluator as lab
from experiments.r29_detached_release.candidate import Detector as Released
from experiments.r30_detached_lifetime.candidate import Detector

HERE=Path(__file__).parent


def one(job):
    c,m,p=job
    return lab.one(('paired',c,m,{'r25_joint':Released,'r26':Detector},p))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh output')
    reg=json.loads((HERE/'close_protocol.json').read_bytes()); plan=json.loads(lab.PLAN.read_bytes())
    if reg['candidate_sha256']!=hashlib.sha256((HERE/'candidate.py').read_bytes()).hexdigest():
        p.error('source changed')
    with ProcessPoolExecutor(2)as pool:
        rows=list(pool.map(one,[(c,plan['variants']['stock'],plan) for c in reg['cases']]))
    guards=fork(study.guards,ANCHOR='r25_joint')
    counts,bad=fork(study.summarize,ANCHOR='r25_joint',guards=guards)(rows,['r26'])
    both={arm:sum(r['records'][arm]['events']['both_second']['strict'] for r in rows) for arm in ('r25_joint','r26')}
    quiet=all(tw['force_exact']['passed'] for r in rows for rec in r['records'].values() for tw in rec['twins'].values())
    passes=quiet and not bad['r26'] and both['r26']>=both['r25_joint'] and both['r25_joint']>0
    j=dict(schema='r30-detached-disclosed-family-contrasts.v1',protocol_sha256=hashlib.sha256((HERE/'close_protocol.json').read_bytes()).hexdigest(),
           episodes=sum(r['episodes'] for r in rows), counts=counts, guard_losses=bad['r26'],both_second=both,
           controls_include_detectable_second=both['r25_joint']>0,passed=passes,rows=rows)
    with a.out.open('x') as f:json.dump(j,f,indent=2,sort_keys=True);f.write('\n')
    print('episodes',j['episodes'],'both_second',both,'all',counts,'losses',bad['r26'],'passed',passes)


if __name__=='__main__':main()
