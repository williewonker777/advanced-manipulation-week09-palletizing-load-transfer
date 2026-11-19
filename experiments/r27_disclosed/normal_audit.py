"""Distinct non-training loaded normal cases, true contacts preserved and trajectories byte-exact."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from experiments.r26_holdout import evaluator as lab
from experiments.r27_disclosed.contrasts import BASE, ROOT, resolve


def one(spec):
    case,mech,arm=spec;out={}
    prev=None
    for key in (BASE,arm):
        _,log=lab.run_episode(case,mech,resolve(key),(None,None))
        sig=lab.certified.trace_signature({'none':log})
        if prev is not None and sig!=prev:raise ValueError('normal physics altered')
        prev=sig;out[key]=lab.common(log,(None,None),[],{'event_window_ms':270})
    old,new=out[BASE],out[arm]
    remaining=old['false_times_s'][:];new_false=[]
    for t in new['false_times_s']:
        match=next((i for i,v in enumerate(remaining) if abs(v-t)<=.001000001),None)
        if match is None:new_false.append(t)
        else:remaining.pop(match)
    gates={'force_and_physics': all(row['force_exact']['passed'] for row in out.values()),
           'no_new_false':not new_false, 'tool_floor':lab.tool_floor(old['tool'],new['tool'],5),
           'no_new_prealarm':not new['tool']['prealarm'] or old['tool']['prealarm'],
           'no_stuck':not new['stuck_tail']}
    return dict(id=case['id'],variant=mech['kind'],physical_signature=sig,records=out,new_false=new_false,gates=gates)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if os.path.lexists(a.out) or a.arm==BASE:p.error('distinct treatment and fresh output')
    reg=json.loads((ROOT/'experiments/r27_load_change/registration.json').read_bytes())
    inherited=json.loads(lab.PLAN.read_bytes())
    tasks=[(c,inherited['variants'][v],a.arm) for c in reg['audit_normals'] for v in reg['mechanics']]
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(one,tasks))
    out=dict(schema='r27-distinct-unforced-guard.v1',episodes=2*len(rows),arm=a.arm,
             failed=[(x['id'],x['variant']) for x in rows if not all(x['gates'].values())],rows=rows)
    with a.out.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('failed',out['failed'])


if __name__=='__main__':
    main()
