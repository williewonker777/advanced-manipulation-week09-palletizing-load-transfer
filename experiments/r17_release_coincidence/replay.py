"""Single already-exposed release sentinel negative-control replay."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import my_detector,r1_torque
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r15_early_com.controller import Controller as R15
from experiments.r16_payload_contract.controller import Controller as D
from experiments.r17_release_prediction.detector import Detector as Prediction
from experiments.r17_release_coincidence.portable_detector import Detector as Coincidence
from sim.runner import run_episode
from tools.r14_paired_events import equal_prefix,force_realization,match_false_events
from tools.r5_metrics import event_result

CASE={'id':'new_surface_normal_release','mass':5.38,'com':[.021,-.016,.011],
      'offset':[0.,0.,0.],'collision':[3.94,[0.,0.,16.]],'role':'ALREADY_EXPOSED_R16_SENTINEL'}
PHYSICAL_KEYS=('q','q_d','tau','e_mm','f_table','f_tool','qpos','f_ext')
EXPECTED_DELAYS_MS={'r10':40.0,'R17C_prediction':21.0,'R17D_coincidence':29.0}
EXPECTED_FALSE_TIMES={
 'r10':[],
 'R17C_prediction':[4.605,4.666,4.688,4.75,4.773,4.809,4.844,4.855,4.867,4.882,
  4.921,4.974,4.986,5.001,5.062,5.1080000000000005,5.136,5.15,5.184,5.214,
  5.23,5.271,5.311,5.339,5.362,5.412,5.4590000000000005,5.48,5.4990000000000006,
  5.508,5.546,5.557,5.587,5.597,5.611,5.666,5.6850000000000005,5.713,5.735,
  5.7540000000000004,5.769,5.78,5.79,5.815,5.838,5.8500000000000005,5.8740000000000006,
  5.886,5.899,5.91,5.921,5.948,5.971,5.982],
 'R17D_coincidence':[5.417,5.509]}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def detector(kind,model):
    cls={'r10':my_detector.Detector,'raw_r15':Coupled,'D_coupled':Coupled,
         'R17C_prediction':Prediction,'R17D_coincidence':Coincidence}[kind]
    obj=cls()
    if kind in ('R17C_prediction','R17D_coincidence'): obj.model_path=Path(model)
    return obj

def episode(kind,height,early,model,collision=None):
    controller={'r10':lambda:r1_torque.Controller(),'raw_r15':lambda:R15(height,early)}.get(
        kind,lambda:D(height,early))()
    return run_episode(controller,CASE['mass'],tuple(CASE['com']),detector(kind,model),
                       collision=collision,pick_offset=CASE['offset'])

def rises(log):
    d=np.asarray(log['detect'],dtype=bool); t=np.asarray(log['t'])
    return t[np.flatnonzero(d & ~np.r_[False,d[:-1]])].tolist()

def compact_run(metrics):
    return {k:metrics.get(k) for k in ('diverged','rms_mm','max_mm','saturation_pct','false_alarms','false_alarm_times','tool_contact_delay_ms') if k in metrics}

def exact_negative_signature(arms):
    timing=all(arms[k]['outcome']=='detected' and arms[k]['delay_ms'] is not None and
               abs(arms[k]['delay_ms']-expected)<=1e-6 for k,expected in EXPECTED_DELAYS_MS.items())
    false_locations=all(np.allclose(arms[k]['false_alarm_times'],expected,rtol=0.,atol=1e-12)
                        if len(arms[k]['false_alarm_times'])==len(expected) else False
                        for k,expected in EXPECTED_FALSE_TIMES.items())
    comparators=(arms['raw_r15']['outcome']=='missed' and arms['raw_r15']['false_alarms']==0 and
                 arms['D_coupled']['outcome']=='missed' and arms['D_coupled']['false_alarms']==0)
    no_prealarm=all(not event['prealarm'] for event in arms.values())
    return {'reference_delays_exact':timing,'false_locations_exact':false_locations,
            'raw_r15_and_D_miss_without_false':comparators,'no_prealarm':no_prealarm}

INTEGRITY_FIELDS = {'all_runs_complete', 'identical_D_physics', 'matched_prefixes', 'exact_forces',
                    'sentinel_R10_genuine', 'R17C_genuine', 'R17D_genuine'}
SIGNATURE_FIELDS = {'reference_delays_exact', 'false_locations_exact',
                    'raw_r15_and_D_miss_without_false', 'no_prealarm'}

def disposition(integrity,signature):
    valid=(set(integrity)==INTEGRITY_FIELDS and set(signature)==SIGNATURE_FIELDS
           and all(x is True for x in (*integrity.values(), *signature.values())))
    return valid,('NEGATIVE_CONTROL_REPRODUCED' if valid else 'INVALID_REPLAY')

def run(height,early,model):
    kinds=('r10','raw_r15','D_coupled','R17C_prediction','R17D_coincidence')
    records={}; logs={}
    for kind in kinds:
        nm,nl=episode(kind,height,early,model); fm,fl=episode(kind,height,early,model,CASE['collision'])
        records[kind]={'normal':compact_run(nm),'forced':compact_run(fm)}; logs[kind]={'normal':nl,'forced':fl}
    reference=logs['D_coupled']; d_kinds=('D_coupled','R17C_prediction','R17D_coincidence')
    physical_parity={kind:{phase:{key:bool(np.array_equal(logs[kind][phase][key],reference[phase][key])) for key in PHYSICAL_KEYS} for phase in ('normal','forced')} for kind in d_kinds}
    arms={};prefix={};force={}
    for kind in kinds:
        nl,fl=logs[kind]['normal'],logs[kind]['forced']
        arms[kind]=event_result(fl['t'],fl['detect'],{'collision':CASE['collision']},fl['f_tool'],nl['detect'])
        prefix[kind]=equal_prefix(nl,fl,CASE['collision'][0]);force[kind]=force_realization(fl['t'],fl['f_ext'],CASE['collision'])
        records[kind]['normal_rising_times_s']=rises(nl);records[kind]['forced_rising_times_s']=rises(fl)
    candidate=arms['R17D_coincidence']
    false_match={name:match_false_events(candidate['false_alarm_times'],arms[name]['false_alarm_times']) for name in ('r10','raw_r15','D_coupled')}
    integrity={'all_runs_complete':all(not records[k][p]['diverged'] for k in kinds for p in ('normal','forced')),
               'identical_D_physics':all(all(fields.values()) for value in physical_parity.values() for fields in value.values()),
               'matched_prefixes':all(x['passed'] for x in prefix.values()),
               'exact_forces':all(x['passed'] for x in force.values()),
               'sentinel_R10_genuine':arms['r10']['outcome']=='detected',
               'R17C_genuine':arms['R17C_prediction']['outcome']=='detected',
               'R17D_genuine':arms['R17D_coincidence']['outcome']=='detected'}
    failure={'R17C_added_false_rises':arms['R17C_prediction']['false_alarms'],
             'R17D_added_false_rises':arms['R17D_coincidence']['false_alarms'],
             'R17D_false_events_match_R10':false_match['r10']['passed'],
             'R17D_false_events_match_raw_R15':false_match['raw_r15']['passed'],
             'R17D_false_events_match_D':false_match['D_coupled']['passed']}
    signature=exact_negative_signature(arms);valid,status=disposition(integrity,signature)
    return {'schema':'r17-release-sentinel-negative-replay.v1','scope':'single already-exposed 5.38kg 16N sentinel; distinct R10/R15 comparators and identical trajectories for the three D-plant arms',
            'case':CASE,'model_sha256':sha(model),'arms':arms,'runs':records,'physical_parity':physical_parity,
            'prefix':prefix,'force':force,'false_event_match':false_match,'integrity':integrity,
            'expected_negative_signature':signature,'failure':failure,
            'replay_valid':valid,'status':status,
            'candidate_passed':False,'promotion_eligible':False,
            'limitations':['already exposed research sentinel','no novel external pulse','no fit from pulse','no control or deployment qualification']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--height',type=Path,required=True);parser.add_argument('--early',type=Path,required=True);parser.add_argument('--model',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink(): parser.error('fresh output required')
    result=run(json.loads(args.height.read_text()),json.loads(args.early.read_text()),args.model)
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'replay_valid':result['replay_valid'],'candidate_passed':False,
                      'delays_ms':{k:v['delay_ms'] for k,v in result['arms'].items()},
                      'false_rises':{k:v['false_alarms'] for k,v in result['arms'].items()}},sort_keys=True))
    return 0 if result['replay_valid'] else 1

if __name__=='__main__': raise SystemExit(main())
