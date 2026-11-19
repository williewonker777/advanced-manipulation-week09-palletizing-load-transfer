"""Locate the actual prior activity that invalidated separation in disclosed cross stress 0."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r22_census import check
from experiments.r24_holdout import evaluator as lab
from experiments.r29_native_event_link import evaluator as cross
from experiments.r31_held_identity.candidate import Detector as Identity
from experiments.r32_measured_level.candidate import Detector as Measured
from .candidate import Detector, NoticeCompatible
from .compatibility import assert_sources


def capture(c,variant,impacts,cls):
    class Trace(cls):
        def reset(self):
            super().reset();self.samples=[];Trace.latest=self

        def update(self,*args):
            value,flag=super().update(*args)
            d=getattr(self,'decision',None)
            self.samples.append((flag if d is None else d.event_output,
                False if d is None else getattr(d,'measured_load',getattr(d,'measured_continuation',False)),
                False if d is None else getattr(d,'linked_native',False)))
            return value,flag

    metrics,log=lab.run_episode(c,variant,Trace,impacts)
    if metrics['diverged'] or not lab.certified.exact_force(log,impacts)['passed']:
        raise ValueError('not the registered two-force simulation')
    return log,np.asarray(Trace.latest.samples,bool)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    assert_sources()
    policy=json.loads((cross.ROOT/'experiments/r25_fragment/selection_policy.json').read_bytes())['paired_stress']
    original=json.loads(lab.PLAN.read_bytes())
    c=next(row.copy() for row in original['paired'] if row['id']=='p24_transfer_near_cross')
    gap,vector=policy['conditions'][0]
    first=[policy['first_time'],policy['first_vector']]
    second=[first[0]+gap,vector];c.update(id='cross_stress_0',impacts=[first,second])
    assignments={'none':(None,None),'first':(first,None),'second':(None,second),'both':(first,second)}
    scoring=json.loads(cross.PLAN.read_bytes())['scoring']['stress']
    raw={};out={}
    for arm,cls in [('identity',Identity),('a',Measured),('b',Detector),('notice', NoticeCompatible)]:
        traces={n:capture(c,original['variants']['stock'],impacts,cls) for n,impacts in assignments.items()}
        both,channels=traces['both'];tick=lab.certified.actual_start(both['t'],second)[0]
        lo=tick-scoring['quiet_ms']
        def overlaps(bits):
            return (np.flatnonzero(np.asarray(bits,bool)[lo:tick])+lo).tolist()
        def spans(bits):
            return check.old.intervals(np.asarray(bits,bool))
        owned=lab.event_record(both,[traces['first'][0]],tick,tick+scoring['event_window_ms'],
                               scoring['quiet_ms'],scoring['reference_twin_ms'])
        rec=dict(both_second=owned,quiet_window=[lo,tick],returned_spans=spans(both['detect']),
            notice_spans=spans(channels[:,0]),load_spans=spans(channels[:,1]),linked_native_spans=spans(channels[:,2]),
            contamination_ticks={'actual_return':overlaps(both['detect']),'notice':overlaps(channels[:,0]),
                                 'measured':overlaps(channels[:,1]),'linked_native':overlaps(channels[:,2])},
            physically_applied={n:spans(np.any(both[n]!=0.,axis=1)) for n in ('f_ext_xyz_native','f_ext_xyz_second')})
        raw[arm]={n:{'physics':{k:check.old.array_sha(log[k]) for k in check.old.PHYSICAL},
                     'residual':check.old.array_sha(log['r']),
                     'signed':{k:check.old.array_sha(log[k]) for k in ('f_ext_xyz_native','f_ext_xyz_second','f_ext_xyz_committed')}}
                  for n,(log,_) in traces.items()}
        out[arm]=rec
    if any(row!=raw['identity'] for row in raw.values()):raise ValueError('physics, signed force or residual differ')
    assert_sources()
    report=dict(schema='r32-cross-stress-quiet-window-diagnosis.v1',case=c,episodes=16,source='disclosed, diagnostic only',
        residual_signed_physics_exact=True,arms=out,signatures=raw['identity'])
    with a.out.open('x') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print({n:r['contamination_ticks'] for n,r in out.items()})


if __name__=='__main__':
    main()
