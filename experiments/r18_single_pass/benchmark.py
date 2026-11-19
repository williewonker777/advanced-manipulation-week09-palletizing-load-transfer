"""Measure identical observer computations on the same stock-physics case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import my_servo
from experiments.r16_payload_contract.controller import Controller
from experiments.r17_release_coincidence.portable_detector import Detector as Reference
from sim.runner import run_episode
from .detector import Detector


OBSERVABLES = ('prediction_armed', 'prediction_supported', 'prediction_age',
               'prediction_score', 'prediction_detected', 'prediction_clear',
               'coincidence_detected', 'coincidence_clear', 'coincidence_channels',
               'coincidence_physical_ratio', 'coincidence_mismatch_ratio',
               'prediction_features')
KEYS = ('t','q','q_d','tau','e_mm','f_table','f_tool','qpos','r','detect','f_ext')


def json_value(x):
    return x.tolist() if hasattr(x, 'tolist') else x


def summary(numbers):
    a=np.asarray(numbers, dtype=float)*1.e-3
    return {'samples':len(a), 'p50_us':float(np.median(a)), 'p99_us':float(np.percentile(a,99)),
            'max_us':float(np.max(a)), 'over_1ms':int(np.count_nonzero(a>1000.))}


def make_ctrl(cls, args):
    class Timed(cls):
        def reset(self):
            super().reset()
            inner=self.compute
            self._timer_wall=np.zeros(6000,np.int64); self._timer_thread=np.zeros(6000,np.int64)
            self._index=0
            def compute(*xs):
                k=self._index; start=time.perf_counter_ns(); cpu=time.thread_time_ns()
                v=inner(*xs)
                self._timer_thread[k]=time.thread_time_ns()-cpu
                self._timer_wall[k]=time.perf_counter_ns()-start
                self._index=k+1
                return v
            self.compute=compute
    return Timed(*args)


def make_observer(cls, model, kind):
    class Timed(cls):
        def reset(self):
            self.model_path=model
            start=time.perf_counter_ns(); cpu=time.thread_time_ns()
            super().reset()
            self.reset_thread=(time.thread_time_ns()-cpu)/1e3; self.reset_wall=(time.perf_counter_ns()-start)/1e3
            inner=self.update
            self.wall=np.zeros(6000,np.int64);self.thread=np.zeros(6000,np.int64)
            self._k=0; self.probes={key:hashlib.sha256() for key in OBSERVABLES}
            def sample(*xs):
                k=self._k; start=time.perf_counter_ns(); cpu=time.thread_time_ns()
                v=inner(*xs)
                self.thread[k]=time.thread_time_ns()-cpu; self.wall[k]=time.perf_counter_ns()-start
                self._k=k+1
                if kind=='probe' and hasattr(self._impl, 'coincidence_channels'):
                    for key, digest in self.probes.items():
                        val=getattr(self._impl,key)
                        digest.update(b'-' if val is None else np.ascontiguousarray(val).tobytes())
                return v
            self.update=sample
    return Timed()


def episode(case, height, early, model, arm, kind):
    servo=case.get('mode')=='servo'
    ctrl=make_ctrl(my_servo.Controller,()) if servo else make_ctrl(Controller,(height,early))
    obs=make_observer(Reference if arm=='reference' else Detector,model,kind)
    metrics,log=run_episode(ctrl,case['mass'],case['com'],obs,collision=case.get('collision'),pick_offset=case['offset'])
    if metrics['diverged'] or obs._k!=6000 or ctrl._index!=6000:
        raise RuntimeError('incomplete stock episode')
    sig={key:hashlib.sha256(np.ascontiguousarray(log[key]).tobytes()).hexdigest() for key in KEYS}
    sig['nonclock']={k:json_value(v) for k,v in metrics.items() if k!='ctrl_time_us'}
    checks={key:value.hexdigest() for key,value in obs.probes.items()}
    timing={};window=np.asarray(log['t'])>=3.900
    for name,v in (('observer_wall',obs.wall),('observer_thread',obs.thread),
                   ('ctrl_wall',ctrl._timer_wall),('ctrl_thread',ctrl._timer_thread),
                   ('combined_wall',obs.wall+ctrl._timer_wall),('combined_thread',obs.thread+ctrl._timer_thread)):
        timing[name]=v.copy()
    return sig,checks,timing,{'arm':arm,'kind':kind,'reset_wall_us':obs.reset_wall,'reset_thread_us':obs.reset_thread,
                             'post_release':{k:summary(v[window]) for k,v in timing.items()},
                             'whole':{k:summary(v) for k,v in timing.items()},
                             'release_tick':{k:float(v[np.flatnonzero(window)[0]]/1e3) for k,v in timing.items()}}


def study(plan,height,early,model):
    results=[]
    for i,c in enumerate(plan['cases']):
        sig0,obs0,_,_=episode(c,height,early,model,'reference','probe')
        sig1,obs1,_,_=episode(c,height,early,model,'fast','probe')
        if sig1!=sig0 or obs1!=obs0:
            raise RuntimeError(str((c['id'],'parity', {k:sig1[k]==sig0[k] for k in sig0},
                                    {k:obs1[k]==obs0[k] for k in obs0})))
        record={'id':c['id'],'all_diagnostic_streams_equal':obs1==obs0,'all_arrays_and_nonclock_equal':sig1==sig0,
                'probe_sha256':obs0,'scientific_signature':sig0,'timed_runs':[],'aggregate':{}}
        if i<2:
            ordered=('reference','fast','fast','reference') if i==0 else ('fast','reference','reference','fast')
            times={'reference':[],'fast':[]}
            for arm in ordered:
                s,_,v,row=episode(c,height,early,model,arm,'timing')
                if s!=sig0:raise RuntimeError('instrumentation changed '+c['id'])
                times[arm].append(v);record['timed_runs'].append(row)
            for arm,reps in times.items():
                allvals={k:np.concatenate([rep[k] for rep in reps]) for k in reps[0]}
                post={k:np.concatenate([rep[k][3900:] for rep in reps]) for k in reps[0]}
                record['aggregate'][arm]={'whole':{k:summary(v) for k,v in allvals.items()},
                                           'post_release':{k:summary(v) for k,v in post.items()}}
            a,b=(record['aggregate'][x] for x in ('reference','fast'))
            record['lower_post_observer_p99']=b['post_release']['observer_wall']['p99_us']<a['post_release']['observer_wall']['p99_us']
            record['all_deadlines_met']=b['whole']['combined_wall']['over_1ms']==0
        results.append(record)
        print(c['id'], 'exact', record.get('lower_post_observer_p99'), flush=True)
    return {'schema':'r18-single-pass-exact-timing.v1','cases':results,
            'same_plant':True,'independent_detector_qualification':False,
            'both_blocks_speed_up':all(r['lower_post_observer_p99'] for r in results[:2]),
            'all_combined_deadlines_met':all(r['all_deadlines_met'] for r in results[:2])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('plan','height','early','model','out'):parser.add_argument('--'+arg,required=True,type=Path)
    a=parser.parse_args()
    if a.out.exists(): parser.error('new output required')
    read=lambda p: json.loads(p.read_bytes())
    result=study(read(a.plan),read(a.height),read(a.early),a.model)
    result['input_sha256']={n:hashlib.sha256(getattr(a,n).read_bytes()).hexdigest() for n in ('plan','height','early','model')}
    with a.out.open('x') as fp:json.dump(result,fp,indent=2,allow_nan=False);fp.write('\n')
