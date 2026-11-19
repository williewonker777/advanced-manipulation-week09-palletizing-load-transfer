"""Replay the disclosed close-place pair and test END/REARM causal feasibility."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r31_torque_control_candidate.candidate import Detector as Incumbent

HERE=Path(__file__).resolve().parent
INCUMBENT='experiments.r31_torque_control_candidate.candidate:Detector'
C3_ENTRY='experiments.r26_yaw_acquisition.candidate:Detector'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def causal_feasibility(first_rise,second_start,quiet_ms):
    boundary=second_start-quiet_ms
    return {'quiet_interval':[boundary,second_start], 'first_rise':first_rise,
            'first_rise_inside_quiet':boundary <= first_rise < second_start,
            'latest_compatible_first_rise':boundary-1,
            'minimum_advance_ms':max(0,first_rise-(boundary-1)),
            'end_or_rearm_after_rise_can_fix':first_rise < boundary}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.path.lexists(a.out):p.error('fresh output directory required')
    plan=json.loads(lab.PLAN.read_bytes());case=next(x for x in plan['paired'] if x['id']=='p26_close_place_cross')
    a.out.mkdir();rows=[]
    for name in ('stock','actuator_deadzone','place_surface_shift'):
        mech=plan['variants'][name]
        row=lab.one(('paired',case,mech,{'r25_joint':C3,'r26':Incumbent},plan))
        rec=row['records']['r26'];first=rec['events']['both_first'];second=rec['events']['both_second']
        first_rise=first['raw_rise_indices'][0];second_rise=second['raw_rise_indices'][0]
        second_start=second_rise-second['delays_ms'][0]
        first_start=first_rise-first['delays_ms'][0]
        proof=causal_feasibility(first_rise,second_start,plan['quiet_ms'])
        item={'mechanics':name,'episodes':row['episodes'],'first_actual_start':first_start,
              'second_actual_start':second_start,'input_gap_ms':second_start-first_start,
              'incumbent_events':rec['events'],'causal_proof':proof,
              'physical_exact':rec['physical_exact_vs_r25'],
              'all_force_exact':all(v['force_exact']['passed'] for v in rec['twins'].values()),
              'all_prefix_exact':all(v['passed'] for k,v in rec.items() if k.startswith('prefix_'))}
        tmp=a.out/(name+'.json.tmp');final=a.out/(name+'.json')
        tmp.write_text(json.dumps(item,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,final);rows.append(item)
    result={'schema':'r32-conservative-release-diagnosis.v1','episodes':sum(x['episodes'] for x in rows),
            'rows':rows,'quiet_ms':plan['quiet_ms'],'incumbent_sha256':sha(HERE.parent/'r31_torque_control_candidate/candidate.py'),
            'end_or_rearm_feasible':all(x['causal_proof']['end_or_rearm_after_rise_can_fix'] for x in rows),
            'candidate_created':False,
            'verdict':'falsified: every first rise is already inside the immutable pre-second quiet interval; causal END/REARM cannot erase emitted history',
            'limits':['Only the disclosed p26_close_place_cross three-mechanics panel was replayed.',
                      'No new detector was enrolled, so no broad protection claim is made.']}
    (a.out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'episodes':result['episodes'],'feasible':result['end_or_rearm_feasible'],'proofs':[x['causal_proof'] for x in rows]},sort_keys=True))
if __name__=='__main__':main()
