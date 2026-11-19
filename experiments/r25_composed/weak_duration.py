"""Add exact in-window active samples to the historical weak-force scorer, without changing its decision."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import FunctionType
from experiments.r21_weak_challenge import evaluator as weak
from experiments.r24_composed.detector import Detector as Anchor
from experiments.r25_fragment.study import constructor


def classify(metrics, log, reference, onset, radius):
    data=weak.classify(metrics,log,reference,onset,radius)
    inside=(log['t']>=onset)&(log['t']<onset+.32)
    data['active_samples_in_slot']=int(log['detect'][inside].astype(bool).sum())
    return data


def capture(job):
    c,arm=job
    f=weak.forced_row
    call=FunctionType(f.__code__,dict(f.__globals__,classify=classify))
    row=call(c,{'r10':Anchor,'candidate':constructor(arm)},20)
    a,b=[row['arms'][k]['force'] for k in ('r10','candidate')]
    checks=row['comparisons']['candidate']['checks']
    checks['delay_and_length']=not a['strict_true_positive'] or (
        b['strict_true_positive'] and b['delay_ms']<=a['delay_ms']+5 and
        b['active_samples_in_slot']<=a['active_samples_in_slot']+5)
    return {'id':c['id'],'anchor':a,'candidate':b,'checks':checks}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',required=True);p.add_argument('--out',type=Path,required=True);q=p.parse_args()
    if q.out.exists():raise ValueError('fresh output required')
    cases=json.loads(Path(weak.__file__).with_name('plan.json').read_bytes())['forced']
    with ProcessPoolExecutor(2)as pool:rows=list(pool.map(capture,[(c,q.arm)for c in cases]))
    j={'schema':'r25-historical-weak-duration.v1','episodes':len(cases)*4,'rows':rows,'arm':q.arm,
       'regressions':[[r['id'],[g for g,v in r['checks'].items()if not v]] for r in rows if not all(r['checks'].values())]}
    with q.out.open('x')as f:json.dump(j,f,allow_nan=False);f.write('\n')
    print(j['episodes'],j['regressions'])
