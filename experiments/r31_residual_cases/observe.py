"""Explain the two D4 duplicates shared by every arm; never modify an alarm."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from experiments.r22_census import check
from experiments.r31_event_replay.runner import ARMS, select


def execute(job):
    c,name=job
    class Passive(ARMS[name]):
        snapshot=None
        def reset(self):
            super().reset(); self.signals=[]; type(self).snapshot=self
        def update(self,t,th,thd,tm,ts,grip):
            old=getattr(self,'planar_notices',0); width=getattr(self,'pending_width',0)
            result=super().update(t,th,thd,tm,ts,grip)
            r=getattr(self,'r29_active_routes',None)
            self.signals.append(dict(reported=bool(result[1]),grip=bool(grip),
                routes=list(r) if r is not None else None,
                optional=bool(width or getattr(self,'planar_notices',0)>old),
                accounted=sorted(getattr(getattr(self,'native_acquisitions',None),'accounted',()))))
            return result
    metrics,log,signed=check.old.execute(Passive,c)
    if (metrics['diverged'] or len(signed)!=6000 or
            not np.array_equal(signed,check.old.expected_signed(log['t'],c['collision']))):
        raise ValueError('invalid signed inputs or incomplete')
    signals=Passive.snapshot.signals
    if not np.array_equal([s['reported'] for s in signals],log['detect']):raise ValueError('trace mismatch')
    spans=check.old.intervals(log['detect'])
    starts=[a for a,_ in spans]
    active=np.flatnonzero(np.any(signed!=0,axis=1))
    def span(nonzero):return check.old.intervals(nonzero)
    rows=[dict(tick=i, force_N=signed[i].tolist(), table_N=float(log['f_table'][i]),
               tool_N=float(log['f_tool'][i]), **signals[i]) for i in starts]
    return dict(id=c['case_id'],arm=name, intervals=spans, rises=rows,
                physical={n:check.old.array_sha(log[n]) for n in check.old.PHYSICAL},
                actual_input_ticks=None if not len(active) else [int(active[0]),int(active[-1])],
                optional_ticks=sum(s['optional'] for s in signals), tool_contact_gt_1N=span(np.asarray(log['f_tool'])>1),
                table_contact_gt_1N=span(np.asarray(log['f_table'])>1))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh output')
    cs,refs=select(check.build_cases('my_torque'),(244,304))
    with ProcessPoolExecutor(2) as pool: recs=list(pool.map(execute,[(c,n) for c in cs for n in ARMS]))
    result=[]
    for c in cs:
        group={r['arm']:r for r in recs if r['id']==c['case_id']}
        if any(r['physical']!=group['c3']['physical'] for r in group.values()):raise ValueError('different physics')
        result.append(dict(input=c,control_id=refs[check.old.reference_key(c)]['case_id'],arms=group))
    with a.out.open('x') as f:json.dump(dict(schema='r31-remaining-d4-diagnosis.v1',episodes=len(recs),rows=result),f,indent=2,sort_keys=True);f.write('\n')
    print([(c['input']['case_id'],c['arms']['identity']['rises']) for c in result])


if __name__=='__main__':main()
