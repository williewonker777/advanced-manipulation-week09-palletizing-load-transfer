"""Derived traces for the recorded R22 counterexample, not the original course demo."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r22_challenge import evaluator as e
from experiments.r23_detached.diagnose import Probe


def response():
    plan=json.loads(e.PLAN.read_bytes())
    case=next(c for c in plan['single'] if c['id']=='s_after_release_diag')
    rows=[]
    for var in plan['variants'].values():
        twins,logs={},{}
        for label,pulse in [('none',None),('force',case['impact'])]:
            det=Probe(); _,log=e.run_episode(case,var,lambda:det,(pulse,None))
            if not e.exact_force(log,(pulse,None))['passed']:raise ValueError('injection mismatch')
            log_rows=det.observations
            indices=[int(round(r[0]*1000)) for r in log_rows]
            if not np.array_equal(log['t'][indices],np.array([r[0] for r in log_rows])):raise ValueError('time alignment')
            twins[label]={'index':indices,'planar_mean_N':[r[2] for r in log_rows],
                          'alarm_indices':[int(i) for i in e.rises(log) if i in indices],
                          'applied_N':[float(log['f_ext'][i]) for i in indices]}
            logs[label]=log
        start,_=e.actual_start(logs['force']['t'],case['impact'])
        if not e.prefix(logs['none'],logs['force'],start,start)['passed']:raise ValueError('noncausal match')
        rows.append({'mechanics':var['kind'],'actual_first_nonzero_index':start,'twins':twins})
    return {'schema':'r23-post-detachment-response.v1','stock_demo':False,'unseen':False,'case':case,'episodes':6,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=response()
    with a.out.open('x') as f:json.dump(data,f,allow_nan=False,separators=(',',':'));f.write('\n')
