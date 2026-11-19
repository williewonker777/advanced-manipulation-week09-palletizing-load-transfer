"""Compare frozen derivatives on the known R21 single-impact pairs; not an independent holdout."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r21_weak_challenge import evaluator as e
from experiments.r21_yaw_impulse.coarse import Detector as Yaw
from experiments.r22_guarded.candidate import Detector as Guarded
from experiments.r22_light.candidate import Detector as Light

ARMS={'r10':e.STATIC['r10'],'r21':Yaw,'guarded':Guarded,'light':Light}


def pair(case):
    row=e.forced_row(case,ARMS,20)
    summary={}
    for arm,v in row['arms'].items():
        first=case['time'];s=v['force']
        summary[arm]={'strict':s['strict_true_positive'],'delay_ms':s['delay_ms'],
                      'masked':s['alarm_before'],'attributable':not s['reference_twin'],
                      'false':s['false_rise_times'],'negative_false':v['none']['false_rise_times'],
                      'raw_extra_in_window':max(0,sum(first<=x<first+.32 for x in s['all_rise_times'])-1),
                      'all_input_prefix_exact':all(v['preimpact_prefix_exact'].values()) and all(v['applied_force_exact'].values()),
                      'physical':all(v['none_physical_vs_r10'].values()) and all(v['force_physical_vs_r10'].values())}
    return {'id':case['id'],'role':case['role'],'arms':summary,'regression_checks':row['comparisons']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads((Path(__file__).resolve().parents[1]/'r21_weak_challenge/plan.json').read_bytes())
    with ProcessPoolExecutor(2) as pool:rows=list(pool.map(pair,plan['forced']))
    totals={arm:{'all':sum(r['arms'][arm]['strict'] for r in rows),
                 'early_weak':sum(r['arms'][arm]['strict'] for r in rows if r['role']=='fresh_weak_early')}
            for arm in ARMS}
    data={'schema':'r22-light-previous-weak.v1','unseen':False,'episodes':len(rows)*8,'rows':rows,'totals':totals}
    with a.out.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
    print(totals)
