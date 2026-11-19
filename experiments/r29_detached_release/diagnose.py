"""Disclosed physical end-of-event audit. True input is read only after the detector has run."""
import argparse
import json
from pathlib import Path
import numpy as np

from experiments.r22_tangent.frozen import Detector as Tangent
from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r29_detached_release.candidate import Detector


def intervals(bits):
    a=np.flatnonzero(np.diff(np.r_[False,bits,False].astype(int)))
    return a.reshape(-1,2).tolist()


def run(mechanic,case):
    trace=[]
    class Measured(Detector):
        def update(self,t,th,thd,tm,ts,grip):
            residual,reported=super().update(t,th,thd,tm,ts,grip)
            w=self.detached;e=w.evidence;v=w.vector
            trace.append((None if v is None else float(np.linalg.norm(v)),e.score,
                          int(e.quiet),self.detached_endpoint.low,bool(e.active),bool(reported)))
            return residual,reported
    impacts=tuple(case['impacts'])
    _,old=lab.run_episode(case,mechanic,C3,impacts)
    _,new=lab.run_episode(case,mechanic,Measured,impacts)
    if (lab.certified.physical_hash(new)!=lab.certified.physical_hash(old)
            or not lab.exact_force(new,impacts)['passed']):raise ValueError('physics/force mismatch')
    start,first=lab.actual_start(old['t'],impacts[0]);next_start,second=lab.actual_start(old['t'],impacts[1])
    bits=np.asarray(old['detect'],bool);fresh=np.asarray(new['detect'],bool)
    def local(arr):return [[a,b]for a,b in intervals(arr)if start<=a<next_start+200]
    delta=np.flatnonzero(bits[:next_start]&~fresh[:next_start])
    var=dict(mechanics=mechanic['kind'],physical_hash=lab.certified.physical_hash(old),
             native_intervals=local(bits),endpoint_intervals=local(fresh),first=first,second=second,
             bound=Tangent.bounds_tangent[0],release_bound=.8*Tangent.bounds_tangent[0],
             removed_first_tail=[dict(tick=int(i),instantaneous_N=trace[i][0],avg_N=trace[i][1],
                                      average_quiet_count=trace[i][2],instantaneous_quiet_count=trace[i][3])
                                 for i in delta if i>=start])
    if len(trace)!=6000:raise ValueError('incomplete witness trace')
    # Display window is applied only in this offline audit.
    ids=np.arange(start-12,next_start+120)
    curves=dict(t=ids.tolist(),native=bits[ids].astype(int).tolist(),endpoint=fresh[ids].astype(int).tolist(),
                physical_force_N=np.linalg.norm(new['f_ext_xyz_committed'][ids],axis=1).tolist(),
                mean_N=[trace[i][1]for i in ids],instantaneous_N=[trace[i][0]for i in ids])
    return var,curves


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('fresh output')
    plan=json.loads(lab.PLAN.read_bytes())
    case=next(c for c in plan['paired']if c['id']=='p26_close_detached_reverse')
    rows=[];plot=None
    for m in plan['variants'].values():
        row,curves=run(m,case);rows.append(row)
        if m['kind']=='stock':plot=curves
    out=dict(schema='r29-detached-force-mechanism.v1',actual_simulations=6,rows=rows,stock_curves=plot)
    with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print([(r['mechanics'],len(r['removed_first_tail']))for r in rows])


if __name__=='__main__':main()
