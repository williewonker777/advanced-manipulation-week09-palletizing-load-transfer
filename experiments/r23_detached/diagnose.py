"""Measure only the already-exposed R22 miss and its actual same-mechanics no-impact twins."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r22_challenge import evaluator as old
from experiments.r23_detached.candidate import Detector


class Probe(Detector):
    def reset(self):
        super().reset()
        self.observations = []

    def update(self,t,*args):
        result=super().update(t,*args)
        s=self.detached
        if 3.899 <= t <= 4.28:
            self.observations.append((t,None if s.vector is None else s.vector.tolist(), s.evidence.score,
                                      s.gate.armed,s.gate.accepted,bool(result[1])))
        return result


def run():
    p=json.loads(old.PLAN.read_bytes())
    case=next(c for c in p['single'] if c['id']=='s_after_release_diag');rows=[]
    for mechanic in p['variants'].values():
        obs,logs={},{}
        for label,impulse in [('none',None),('force',case['impact'])]:
            probe=Probe()
            _,log=old.run_episode(case,mechanic,lambda:probe,(impulse,None))
            obs[label]=probe.observations;logs[label]=log
            if not old.exact_force(log,(impulse,None))['passed']:raise ValueError('injected force differs')
        start,_=old.actual_start(logs['force']['t'],case['impact'])
        if not old.prefix(logs['none'],logs['force'],start,start)['passed']:raise ValueError('not a causal twin')
        a,b=obs['none'],obs['force']
        if not all(x==y for x,y in zip(a,b,strict=True) if round(x[0]*1000)<=start):
            raise ValueError('observer before force differs')
        score=lambda row: row[2] if row[2] is not None else -1.
        metric=lambda x:{'first_valid_s':next(r[0] for r in x if r[1] is not None),
                         'first_ready_s':next((r[0] for r in x if r[3]),None),
                         'max_xy_N':max(x,key=score)[2],
                         'first_added_alarm_s':next((r[0] for r in x if r[4]),None),
                         'peak_force_vector_N':max((r for r in x if r[1] is not None),key=score)[1]}
        rows.append({'mechanics':mechanic['kind'],'normal':metric(a),'applied':metric(b),
                     'effective_actual_input_start_s':float(logs['force']['t'][start]),
                     'true_peak_N':float(np.max(logs['force']['f_ext'])),
                     'twin_force_and_through_first_nonzero_exact':True})
    return {'schema':'r23-development-post-detachment-physics.v1','unseen':False,'condition':case,'rows':rows}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    with a.out.open('x') as f:json.dump(run(),f,indent=2,allow_nan=False);f.write('\n')
