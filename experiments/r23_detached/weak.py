"""Re-run the exposed legacy small-pulse twins with the R22 and R23 sensors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r21_weak_challenge import evaluator as old
from experiments.r22_light.candidate import Detector as Light
from experiments.r23_detached.candidate import Detector, Free


def row(case):
    result=old.forced_row(case,{'r10':Light,'free':Free,'onset':Detector},20)
    saved={}
    for label,record in result['arms'].items():
        name='light' if label=='r10' else label
        force=record['force']
        saved[name]={'strict':force['strict_true_positive'],'delay_ms':force['delay_ms'],
                     'false':force['false_rise_times'],'none_false':record['none']['false_rise_times'],
                     'extra':max(0,sum(case['time']<=v<case['time']+.32 for v in force['all_rise_times'])-1),
                     'input_certified':all(all(record[k].values()) for k in (
                         'preimpact_prefix_exact','applied_force_exact','none_physical_vs_r10','force_physical_vs_r10'))}
    return {'id':case['id'],'role':case['role'],'arms':saved}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2);a=p.parse_args()
    plan=json.loads((Path(__file__).resolve().parents[1]/'r21_weak_challenge/plan.json').read_bytes()); cases=plan['forced']
    if len(cases)!=15:raise ValueError('incomplete small-pulse panel')
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(row,cases))
    doc={'schema':'r23-known-pulse-protection.v1','episodes':90,'unseen':False,'rows':rows}
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
