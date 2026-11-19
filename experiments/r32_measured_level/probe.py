"""Read-only physical diagnosis on disclosed inputs, before designing a replacement endpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.r22_census import check
from experiments.r26_holdout import evaluator as pair
from experiments.r31_held_identity.candidate import Detector

ROOT=Path(__file__).resolve().parents[2]
BASE='1e33b4698ef3678afe72cc4e286fcab5d27b2232e7bf087fdef7500ccc674764'
FULL='7070f5f1651e07faa8701f28421082b7eaf32a16cb866dbf8e5d9ffa07018d43'
PAIRS=('p26_close_place_cross','p26_close_transfer_oblique')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def packet(det):
    wins=det.r27_gate.windows.windows
    force=det.r27_projection.last_force
    zw=det.z_window
    if force is None or len(zw.present)!=40 or any(len(w.present)!=40 for w in wins):
        return None,None
    instant=np.column_stack(([w.present[-1] for w in wins],zw.present[-1]))
    now=np.column_stack(([w.b/40 for w in wins], zw.b/40))
    return instant,now


class Trace(Detector):
    def reset(self):
        super().reset();self.rows=[];self.tracked=None;self.tick_id=0;Trace.latest=self
    def update(self,t,th,thd,tm,ts,grip):
        notices=self.planar_notices;assoc=self.associated_native_samples
        result=super().update(t,th,thd,tm,ts,grip)
        start=self.planar_notices>notices
        if start:self.tracked=self.source;self.tick_id+=1
        tok=self.tracked
        if self.start <= round(t*1000) <= self.stop:
            instant,now=packet(self) if tok is not None else (None,None)
            def project(levels):
                if levels is None or tok is None:return None
                delta,along,off,_=tok.coordinates(levels)
                return [along.tolist(),off.tolist()]
            self.rows.append(dict(tick=round(t*1000),out=bool(result[1]),start=start,grip=bool(grip),
                     same_gate=tok is not None and tok.generation is self.r27_gate.reference,source=tok is not None and self.source is tok,
                     src_id=self.tick_id,gate=self.r27_gate.state,assoc=self.associated_native_samples>assoc,
                     routes=self.r29_active_routes,complete=self.r29_routes_complete,
                     size=None if tok is None else tok.size.tolist(), instant=project(instant),mean=project(now)))
        return result


def run(job):
    mode,c,variant,prior,dest=job
    if mode=='d4':
        lo=int(c['collision'][0]*1000);hi=lo+380
        class P(Trace):
            start=lo-5;stop=hi
        _,log,signed=check.old.execute(P,c);probe=Trace.latest
        if (not np.array_equal(signed,check.old.expected_signed(log['t'],c['collision'])) or
            prior['physics_sha256']!={n:check.old.array_sha(log[n]) for n in check.old.PHYSICAL} or
            prior['arms']['identity']['alarm_intervals']!=check.old.intervals(log['detect'])):
            raise ValueError('prior not reproduced')
        key=str(c['case_id']);target=prior['arms']
    else:
        plan=json.loads(pair.PLAN.read_bytes());mech=plan['variants']['stock'];first,second=c['impacts']
        spec={'none':(None,None),'first':(first,None),'second':(None,second),'both':(first,second),'long':(c['long'],None)}[variant]
        probe=Trace();probe.start=round(first[0]*1000)-15;probe.stop=round(second[0]*1000)+270
        _,log=pair.run_episode(c,mech,lambda:probe,spec)
        if not pair.exact_force(log,spec)['passed']:raise ValueError('signed force')
        key=c['id']+'_'+variant;target=check.old.intervals(log['detect'])
    f=dest/(key+'.json')
    with f.open('x') as fp:json.dump(dict(id=key,case=c,variant=variant,rows=probe.rows,targets=target),fp,separators=(',',':'),allow_nan=False);fp.write('\n')
    return [key,sha(f)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if (any(a.out.iterdir()) or sha(a.old)!=FULL or sha(ROOT/'experiments/r31_held_identity/candidate.py')!=BASE):p.error('pinned and fresh')
    ref=json.loads(a.old.read_bytes());jobs=[]
    cases=check.build_cases('my_torque');plan=json.loads(pair.PLAN.read_bytes())
    for r in ref['rows']:
        c=cases[r['id']]
        if c['collision'] is None:continue
        arm=r['arms']['identity'];t=arm['score']['delay_ms']
        if t is not None and any(b-s==4 and s==int(np.searchsorted(check.TIME,c['collision'][0]))+t for s,b in arm['alarm_intervals']):
            jobs.append(('d4',c,None,r,a.out))
    for c in plan['paired']:
        if c['id'] in PAIRS:
            jobs.extend([('r26',c,v,None,a.out) for v in ('none','first','second','both','long')])
    with ProcessPoolExecutor(4) as pool:idx=list(pool.map(run,jobs))
    (a.out/'index.json').write_text(json.dumps(idx,indent=2,sort_keys=True)+'\n');print(len(idx))
if __name__=='__main__':main()
