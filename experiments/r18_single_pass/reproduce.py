"""Paired re-execution of the already exposed 16 N sentinel; no new input case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r16_payload_contract.controller import Controller
from experiments.r17_release_coincidence import replay as prior
from experiments.r18_single_pass import recovery, reference_recovery
from sim.runner import run_episode
from tools.r14_paired_events import equal_prefix, force_realization, preserve_tool
from tools.r5_metrics import event_result

ARMS = ('r10', 'raw_r15', 'D_coupled', 'R17D_coincidence', 'R18', 'R18_fast')
EXPECT = {
    'r10': ('detected', 40., []), 'raw_r15': ('missed', None, []),
    'D_coupled': ('missed', None, []), 'R17D_coincidence': ('detected', 29., [5.417,5.509]),
    'R18': ('detected', 33., []), 'R18_fast': ('detected', 33., []),
}


def pulse_signature(events):
    if set(events) != set(ARMS): return False
    for arm,(status,latency,false) in EXPECT.items():
        e=events[arm]
        if e['outcome']!=status or e['prealarm'] or e['false_alarm_times']!=false or e['false_alarms']!=len(false):return False
        if latency is None:
            if e['delay_ms'] is not None:return False
        elif e['delay_ms'] is None or abs(e['delay_ms']-latency)>1e-6:return False
    return True


def state_hash(observable):
    class Meter(observable):
        def reset(self):
            super().reset()
            self.history = hashlib.sha256()
            inner = self.update
            def update(*args):
                result = inner(*args)
                flags = (self._impl.recovery_instantaneous, self._impl.recovery_detected)
                self.history.update(bytes(flags))
                return result
            self.update = update
    return Meter


def run(height, early, model):
    c=prior.CASE
    logs={}; events={}; record={}; paired={}; physics={}; supports={}
    for arm in ARMS:
        pair={}; counts={}
        for phase,force in (('normal',None),('forced',c['collision'])):
            if arm in ('R18','R18_fast'):
                factory = reference_recovery.Detector if arm=='R18' else recovery.Detector
                det=state_hash(factory)(); det.model_path=model
                m,log=run_episode(Controller(height,early),c['mass'],tuple(c['com']),det,
                                  collision=force,pick_offset=c['offset'])
                supports.setdefault(arm,{})[phase]=det.history.hexdigest()
            else:
                m,log=prior.episode(arm,height,early,model,force)
            if m['diverged'] or not np.array_equal(log['t'],np.arange(6000)*.001):
                raise ValueError('full fixed-grid matched episodes are required')
            pair[phase] = log
            counts[phase] = {k:v for k,v in prior.compact_run(m).items() if k!='ctrl_time_us'}
            counts[phase]['rising_times']=prior.rises(log)
        n,f=pair['normal'],pair['forced']
        events[arm]=event_result(f['t'],f['detect'],c,f['f_tool'],n['detect'])
        paired[arm]={'prefix':equal_prefix(n,f,c['collision'][0]),
                     'force':force_realization(f['t'],f['f_ext'],c['collision'])}
        record[arm]=counts;logs[arm]=pair
    for arm in ('R17D_coincidence','R18','R18_fast'):
        physics[arm]={phase:{k:bool(np.array_equal(logs[arm][phase][k],logs['D_coupled'][phase][k])) for k in prior.PHYSICAL_KEYS}
                      for phase in ('normal','forced')}
    fast_pair={phase:{k:bool(np.array_equal(logs['R18'][phase][k],logs['R18_fast'][phase][k]))
                      for k in (*prior.PHYSICAL_KEYS,'r','detect')} for phase in ('normal','forced')}
    gates={'genuine_reference_signature':pulse_signature(events),
           'all_prefixes_and_realized_forces':all(y['passed'] for x in paired.values() for y in x.values()),
           'all_D_physics_identical':all(all(fields.values()) for x in physics.values() for fields in x.values()),
           'fast_exact_residual_alarm_and_state':supports['R18']==supports['R18_fast'] and all(all(x.values()) for x in fast_pair.values()),
           'preserve_tool':all(preserve_tool(events['R18'],events[k]) for k in ('r10','raw_r15','D_coupled'))}
    return {'schema':'r18-known-sentinel-live.v1','case':c,'gates':gates,'reproduced':all(gates.values()),
            'scope':'previously exposed development sentinel only; no independent generalization or release claim',
            'events':events,'paired':paired,'D_physics':physics,'fast_parity':fast_pair,'support_hashes':supports,
            'records':record,'promotion_eligible':False}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('height','early','model','out'):cli.add_argument('--'+key,required=True,type=Path)
    args=cli.parse_args()
    if args.out.exists():cli.error('new output required')
    result=run(json.loads(args.height.read_bytes()),json.loads(args.early.read_bytes()),args.model)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'gates':result['gates'],'reproduced':result['reproduced'],'promotion_eligible':False}))
    raise SystemExit(0 if result['reproduced'] else 1)
