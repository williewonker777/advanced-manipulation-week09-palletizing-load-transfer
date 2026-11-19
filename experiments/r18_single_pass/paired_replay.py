"""Independently replay the frozen four-twin cases and compare every scored arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import my_detector
import r1_torque
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r15_early_com.controller import Controller as R15
from experiments.r16_payload_contract.controller import Controller as D
from experiments.r18_paired_impulses import scoring as evaluate, runner
from experiments.r18_single_pass import reference_recovery,recovery
from tools.r14_paired_events import match_false_events,preserve_tool
from tools.r5_metrics import event_result

KINDS = ('r10','raw_r15','D_coupled','R18_recovery','single_pass')
PHYSICAL = ('t','e_mm','q','q_d','tau','f_table','f_tool','qpos','f_ext','f_ext_xyz_committed',
            'f_ext_xyz_native','f_ext_xyz_second','f_ext_combined_norm')


def signature(record):
    packed=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(packed.encode()).hexdigest()


def scalar_norm_check(trace):
    # The stock runner computes one vector norm per tick. NumPy's batched
    # row reduction need not have the same last bit. Preserve the original
    # predeclared checker intact and report the exact scalar calculation as
    # a separate diagnostic, not as retrospective permission to pass a case.
    native=np.asarray(trace['f_ext_xyz_native'],dtype=float)
    observed=np.asarray(trace['f_ext'],dtype=float)
    scalar=np.fromiter((np.linalg.norm(row) for row in native),dtype=float,count=len(native))
    batched=np.linalg.norm(native,axis=1)
    return {'stock_scalar_bit_exact':bool(np.array_equal(scalar,observed)),
            'different_batched_samples':int(np.count_nonzero(batched!=observed)),
            'max_batched_difference_N':float(np.max(np.abs(batched-observed)))}


def episode(kind, case, height, early, model, pulses):
    if kind=='r10': ctrl=r1_torque.Controller();det=my_detector.Detector()
    elif kind=='raw_r15': ctrl=R15(height,early);det=Coupled()
    else:
        ctrl=D(height,early)
        det={'D_coupled':Coupled,'R18_recovery':reference_recovery.Detector,'single_pass':recovery.Detector}[kind]()
        det.model_path=model
    metrics,log=runner.run_episode(ctrl,case['mass'],tuple(case['com']),det,pick_offset=case['offset'],impacts=pulses)
    log['r18_diverged']=metrics['diverged']
    if metrics['diverged'] or not np.array_equal(log['t'],np.arange(6000)*.001):
        raise ValueError('incomplete or divergent twin')
    return log


def normal_gate(row):
    base=row['D_coupled'];trial=row['single_pass'];ev={k:event_result(v['t'],v['detect'],{},v['f_tool']) for k,v in row.items()}
    return (all(np.array_equal(base[k],trial[k]) for k in PHYSICAL)
            and all(np.array_equal(row['R18_recovery'][k],trial[k]) for k in ('r','detect'))
            and all(match_false_events(ev['single_pass']['false_alarm_times'],ev[k]['false_alarm_times'])['passed']
                    and preserve_tool(ev['single_pass'],ev[k]) for k in KINDS[:3]))


def run(plan,ref,height,early,model):
    cases=plan['fresh_paired_cases']+plan['diagnostic_sequences']
    ids=[c['id'] for c in cases]; reference=ref['case_assessment_sha256']
    if (ref['schema']!='r18-paired-reference.v1' or len(set(ids))!=6 or set(ids)!=set(reference) or
            signature(plan)!=ref['plan_signature'] or any(set(v)!=set(KINDS[:4]) for v in reference.values())):
        raise ValueError('a complete and correctly separated six-case reference is required')
    normals={}
    for c in cases:
        normal={k:episode(k,c,height,early,model,(None,None)) for k in KINDS}
        if not normal_gate(normal):raise ValueError('normal gate mismatch: '+c['id'])
        normals[c['id']]=normal
    rows=[]
    for c in cases:
        first,second=c['impacts'];pulses={'first':(first,None),'second':(None,second),'both':(first,second)}
        logs={};hashes={};scores={}
        for k in KINDS:
            twins={'none':normals[c['id']][k]}
            twins.update({twin:episode(k,c,height,early,model,x) for twin,x in pulses.items()})
            score=evaluate.assess(c,twins,plan['quiet_ms'],plan['event_window_ms'])
            scores[k]=score;hashes[k]=signature(score);logs[k]=twins
        auth=all(hashes[k]==reference[c['id']][k] for k in KINDS[:4])
        same=hashes['single_pass']==hashes['R18_recovery'] and all(
            np.array_equal(logs['single_pass'][twin][key],logs['R18_recovery'][twin][key])
            for twin in evaluate.TWINS for key in (*PHYSICAL,'r','detect'))
        d_physics=all(np.array_equal(logs['single_pass'][twin][key],logs['D_coupled'][twin][key])
                      for twin in evaluate.TWINS for key in PHYSICAL)
        scalar_checks={tw:scalar_norm_check(logs['single_pass'][tw]) for tw in evaluate.TWINS}
        if not all(x['stock_scalar_bit_exact'] for x in scalar_checks.values()):
            raise ValueError('the scalar stock force report does not match its actual input')
        if not(auth and same and d_physics):raise ValueError(str((c['id'],'reference, fast exact, D physics',auth,same,d_physics)))
        rows.append({'id':c['id'],'role':c['claim'],'reference_matched':auth,'fast_exact_all_twins':same,
                     'same_D_physics':d_physics,'assessment_sha256':hashes,'scalar_force_checks':scalar_checks,
                     'candidate_genuine_delays_ms':scores['single_pass']['absolute_delays_ms'],
                     'candidate_twin_gates':scores['single_pass']['gates']})
    return {'schema':'r18-known-four-twin-replay.v1','reproduced':True,'promotion_eligible':False,
            'four_independent_at_first_evaluation':4,'two_previously_exposed_first_pulse_controls':2,
            'episodes':len(rows)*len(evaluate.TWINS)*len(KINDS),'rows':rows,
            'scope':'reproduction of already evaluated conditions; matching the published result is not a pass for the detector'}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for x in ('height','early','model','plan','reference','out'):cli.add_argument('--'+x,type=Path,required=True)
    a=cli.parse_args()
    if a.out.exists():cli.error('new output required')
    read=lambda p:json.loads(p.read_bytes())
    out=run(read(a.plan),read(a.reference),read(a.height),read(a.early),a.model)
    with a.out.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
