"""Run the actual two APIs and original notice in the same disclosed placement physics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import FunctionType
from experiments.r22_census import check
from experiments.r26_holdout import evaluator as lab
from experiments.r31_event_replay.runner import select
from experiments.r32_native_handoff import compatibility as earlier
from .client import NoticeClient, SustainedAlarm

ROOT=Path(__file__).resolve().parents[2]
PINS={'experiments/r32_detached_readback/candidate.py':'a417c78fff0a2da124038921172b102ec426678e9b9a4a43a6a738fbdccbd4b1',
      'experiments/r32_joint_state/candidate.py':'633b1d49229ed07764d925bb84a5248d7db183aef2a2e4e4d835512752033d1a',
      'experiments/r32_alarm_views/client.py':'22198cf8a34ec2b0cc0c85d4f1ccedca570a70bbff2135a6ac35156e2511ad59'}
def validate():
    earlier.assert_sources()
    for n,pin in PINS.items():
        if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=pin:raise ValueError('wrong client')
def measure(job):
    f=earlier.measure
    return FunctionType(f.__code__,{**f.__globals__,'assert_sources':validate,'Detector':SustainedAlarm,
        'NoticeCompatible':NoticeClient},f.__name__,f.__defaults__,f.__closure__)(job)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh result required')
    plan=json.loads(lab.PLAN.read_bytes())
    c=next(v for v in plan['paired'] if v['id']=='p26_close_detached_reverse');m=plan['variants']['actuator_deadzone']
    f,s=c['impacts']
    jobs=[('r26',c,'detached_reverse/'+variant,(m,impacts)) for variant,impacts in [('none',(None,None)),
        ('first',(f,None)),('second',(None,s)),('both',(f,s)),('long',(c['long'],None))]]
    cases,_=select(check.build_cases('my_torque'),(266,316))
    jobs += [('d4',v,str(v['case_id']),None) for v in cases]
    with ProcessPoolExecutor(2) as pool:rows=list(pool.map(measure,jobs))
    both=next(v for v in rows if v['id']=='detached_reverse/both')
    on=both['signed_intervals']['f_ext_xyz_second'][0][0];lo=on-plan['quiet_ms']
    def active(spans):
        return [i for i in range(lo,on) if any(a<=i<b for a,b in spans)]
    summary={k:active(spans) for k,spans in both['returned_by_mode'].items()}
    result=dict(schema='r32-joint-client-example.v1',classification='Disclosed actual API compatibility, not failure regrading.',
        sources={**earlier.SOURCES,**PINS},episodes=3*len(rows),matched_ticks_per_arm=6000*len(rows),rows=rows,
        original_and_notice_equal_at_every_tick=True,all_force_residual_and_motion_samples_exact=True,
        quiet_window=[lo,on],actual_high_ticks_in_quiet_window=summary)
    with a.out.open('x') as fp:json.dump(result,fp,indent=2,sort_keys=True,allow_nan=False);fp.write('\n')
    print(result['episodes'],summary)
if __name__=='__main__':main()
