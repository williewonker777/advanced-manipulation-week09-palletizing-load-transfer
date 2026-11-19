"""Replay four disclosed D4 inputs and matched no-force twins for bit identity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r31_event_channels.adapter import Detector as Adapter
from experiments.r31_held_identity.candidate import Detector as Held
from tools.d4_robustness import build_cases

HELD_SHA='1e33b4698ef3678afe72cc4e286fcab5d27b2232e7bf087fdef7500ccc674764'
CASES=(266,287,298,316);REFS=(10,25,119,47)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def intervals(values):
    a=np.asarray(values,dtype=bool);d=np.diff(np.r_[False,a,False].astype(np.int8))
    return [[int(x),int(y)] for x,y in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1),strict=True)]
def semantic_metrics_equal(a,b):
    """Ignore only measured controller wall time; compare every scientific cell."""
    keys=set(a)|set(b)
    if keys-set(('ctrl_time_us',)) != (set(a)-set(('ctrl_time_us',))):return False
    for key in keys-set(('ctrl_time_us',)):
        if key not in a or key not in b or not np.array_equal(np.asarray(a[key]),np.asarray(b[key])):return False
    return True
def capture(case):
    old_metrics,old_log,_=raw.execute(Held,case);new_metrics,new_log,_=raw.execute(Adapter,case)
    channels=Adapter.latest.channel_log
    if len(channels)!=6000:raise ValueError('incomplete telemetry')
    exact={key:bool(np.array_equal(np.asarray(old_log[key]),np.asarray(new_log[key]))) for key in raw.PHYSICAL+('r','detect')}
    cols={name:[getattr(c,name) for c in channels] for name in ('output_level','notice_level','native_level','native_source_linked')}
    rows=[]
    for i,c in enumerate(channels):
        if c.output_level or c.notice_level or c.native_level or c.native_source_linked:
            rows.append({'tick':i,'output':c.output_level,'notice':c.notice_level,'native':c.native_level,
                         'routes':list(c.native_routes),'source_linked':c.native_source_linked})
    metrics_equal=semantic_metrics_equal(old_metrics,new_metrics)
    return {'case_id':case['case_id'],'kind':case['kind'],'collision':case['collision'],'exact_arrays':exact,
            'semantic_metrics_equal':metrics_equal,'timing_observational_only':['ctrl_time_us'],'held_intervals':intervals(old_log['detect']),
            'adapter_intervals':intervals(new_log['detect']),'channels':{name:intervals(value) for name,value in cols.items()},
            'active_rows':rows,'passed':all(exact.values()) and metrics_equal and np.array_equal(old_log['detect'],new_log['detect'])}
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    root=Path(__file__).resolve().parents[2]
    if sha(root/'experiments/r31_held_identity/candidate.py')!=HELD_SHA:raise ValueError('held detector changed')
    cases=build_cases('my_torque');rows=[]
    for collision,reference in zip(CASES,REFS,strict=True):
        rows.extend((capture(cases[reference]),capture(cases[collision])))
    out={'schema':'r31-event-channels-verification.v1','classification':'read-only detector telemetry; not force truth, safety stop, or occupancy repair',
         'held_sha256':HELD_SHA,'adapter_sha256':sha(Path(__file__).with_name('adapter.py')),'episodes':16,'rows':rows,
         'passed':all(r['passed'] for r in rows),'limits':['notice intervals are report envelopes, not physical contact duration','native evidence is an internal causal detector channel, not ground-truth force']}
    with a.out.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    if not out['passed']:raise RuntimeError('adapter changed detector output')
    print('PASS episodes',out['episodes'])
if __name__=='__main__':main()
