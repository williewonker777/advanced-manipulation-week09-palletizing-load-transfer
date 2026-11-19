"""Read-only three-arm actual D4 focus; use only the published matrix and previous measured intervals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r24_loaded.d4 import load_cases
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r29_detached_release.candidate import Detector as Released
from experiments.r30_detached_lifetime.candidate import Detector

PLAN=Path(__file__).with_name('protocol.json')
ARMS={'c3':C3,'released':Released,'lifetime':Detector}


def one(case):
    rec={}
    for name,cls in ARMS.items():
        metrics,log,signed=raw.execute(cls,case)
        bits=np.asarray(log['detect'],bool)
        edges=np.diff(np.r_[False,bits,False].astype(np.int8))
        spans=[[int(a),int(b)] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1),strict=True)]
        force=np.flatnonzero(np.linalg.norm(signed,axis=1)>0)
        if (metrics['diverged'] or len(bits)!=6000 or
                not np.array_equal(raw.expected_signed(log['t'],case.get('collision')),signed)):
            raise ValueError('incorrect physical force')
        rec[name]=dict(intervals=spans,force=[int(force[0]),int(force[-1])] if len(force) else [],
                       physics={k:raw.array_sha(log[k]) for k in raw.PHYSICAL})
    if len({json.dumps(v['physics'],sort_keys=True) for v in rec.values()})!=1:
        raise ValueError('passive physics differ')
    return dict(id=case['case_id'],arms=rec)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(PLAN.read_bytes())
    if a.out.exists() or a.out.is_symlink() or hashlib.sha256(PLAN.with_name('candidate.py').read_bytes()).hexdigest()!=plan['candidate_sha256']:
        p.error('frozen source, fresh output')
    cases=[r for r in load_cases() if r['case_id'] in plan['scope_d4_ids']]
    if len(cases)!=len(plan['scope_d4_ids']):raise ValueError('incomplete public slice')
    with ProcessPoolExecutor(2) as pool: rows=list(pool.map(one,cases))
    target=next(r['arms'] for r in rows if r['id']==316)
    if target['c3']['intervals']!=[[3951,3989]] or target['released']['intervals']!=[[3951,3963],[3964,3982]]:
        raise ValueError('failed to reproduce the 1 ms false gap with its direct parent')
    data=dict(schema='r30-detached-lifetime-disclosed-focus.v1',episodes=3*len(rows),rows=rows,physics_and_force_exact=True,
              target_one_onset_at_original_time=[a for a,_ in target['lifetime']['intervals']]==[3951],
              other_changed_vs_released=[r['id'] for r in rows if r['id']!=316 and r['arms']['released']['intervals']!=r['arms']['lifetime']['intervals']])
    with a.out.open('x') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
    print('target',target['lifetime']['intervals'],'other_changed',data['other_changed_vs_released'],'episodes',data['episodes'])


if __name__=='__main__':main()
