"""Second normal-only audit distinct from calibration and the earlier five loads."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from experiments.r26_holdout import evaluator as lab
from experiments.r27_disclosed.contrasts import ROOT
from experiments.r27_disclosed.normal_audit import one


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if os.path.lexists(a.out):p.error('fresh path required')
    reg=json.loads((ROOT/'experiments/r27_load_change/settled_registration.json').read_bytes())
    mechanics=json.loads(lab.PLAN.read_bytes())['variants']
    jobs=[(c,m,a.arm) for c in reg['additional_future_normals'] for m in mechanics.values()]
    with ProcessPoolExecutor(a.workers)as pool:rows=list(pool.map(one,jobs))
    data=dict(schema='r27-second-unforced-guard.v1',episodes=2*len(rows),arm=a.arm,rows=rows,
              failed=[(x['id'],x['variant'])for x in rows if not all(x['gates'].values())])
    with a.out.open('x')as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(data['episodes'],data['failed'])


if __name__=='__main__':
    main()
