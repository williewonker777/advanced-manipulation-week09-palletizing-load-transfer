"""Authored R17 staged qualification; run only inside the authenticated clean overlay."""
from __future__ import annotations
import argparse, hashlib, importlib, json
from pathlib import Path
import numpy as np
import my_detector, r1_torque
from sim import trajectory as T
from experiments.r14_stock_handoff.detector import Detector as R14Detector
from experiments.r14_stock_handoff.qualification import KEYS, compact
from experiments.r15_early_com.controller import Controller as R15
from experiments.r16_payload_contract.controller import Controller as DController, Detector as DDetector
from sim.runner import run_episode
from tools.r11_placement import score
from tools.r14_paired_events import equal_prefix, force_realization, match_false_events, preserve_tool
from tools.r5_metrics import event_result

CANDIDATE = CANDIDATE_DETECTOR = None
NONCLOCK_EQUIVALENT_TO_D = False
ARMS = ('r10','r15','d','candidate')
STAGE_KEYS = {'known-normal':'known_normal_cases','exposed-impact':'exposed_impact_cases',
              'fresh-normal':'fresh_normal_cases','fresh-impact':'fresh_impact_cases'}
PREVIOUS = {'exposed-impact':'known-normal','fresh-normal':'exposed-impact','fresh-impact':'fresh-normal'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make_controller(kind,height,early):
    return {'r10':lambda:r1_torque.Controller(),'r15':lambda:R15(height,early),
            'd':lambda:DController(height,early),'candidate':lambda:CANDIDATE(height,early)}[kind]()

def make_detector(kind):
    return {'r10':lambda:my_detector.Detector(),'r15':lambda:R14Detector(),
            'd':lambda:DDetector(),'candidate':lambda:CANDIDATE_DETECTOR()}[kind]()

def episode(kind,case,height,early,collision=None):
    ctrl=make_controller(kind,height,early)
    metrics,log=run_episode(ctrl,float(case['mass']),tuple(case['com']),make_detector(kind),
                            collision=collision,pick_offset=case.get('offset'))
    return compact(metrics,log,ctrl,float(case['mass'])),log

def ratio(candidate,baseline):
    return {k:float(v/max(baseline['cost'][k],1e-12)) for k,v in candidate['cost'].items()}

def tool_gate(candidate,reference):
    return reference['tool_delay_ms'] is None or (candidate['tool_delay_ms'] is not None and
            candidate['tool_delay_ms']<=reference['tool_delay_ms']+5.0+1e-9)

def no_added(candidate,reference): return match_false_events(candidate,reference)['passed']

def exact_logs(left,right,keys=KEYS): return {k:bool(np.array_equal(left[k],right[k])) for k in keys}

def nonclock(record):
    out={k:v for k,v in record.items() if k!='ctrl_p99_us'}
    if isinstance(out.get('registration'),dict):
        out['registration']={k:v for k,v in out['registration'].items() if k!='stream_timing_us'}
    return out

def saturation_min(candidate,*references):
    return candidate['saturation_pct'] <= min(x['saturation_pct'] for x in references)+1e-12

def exact_pair(lr,ll,rr,rl,keys=KEYS):
    fields=exact_logs(ll,rl,keys); metrics=nonclock(lr)==nonclock(rr)
    return {'fields':fields,'metrics':metrics,'passed':bool(metrics and all(fields.values()))}

def d_policy_equivalence(records,logs):
    if not NONCLOCK_EQUIVALENT_TO_D:
        return {'declared':False,'passed':True,'fields':{},'metrics':None}
    keys=tuple(k for k in KEYS if k!='detect')
    fields=exact_logs(logs['candidate'],logs['d'],keys)
    excluded=('ctrl_p99_us','registration','false_alarms','false_alarm_times','tool_delay_ms')
    metrics={k:v for k,v in records['candidate'].items() if k not in excluded} == {k:v for k,v in records['d'].items() if k not in excluded}
    return {'declared':True,'passed':bool(metrics and all(fields.values())),'fields':fields,'metrics':metrics,
            'note':'detector output and detector-derived metrics intentionally excluded'}

def normal_case(case,height,early):
    records,logs={},{}
    for kind in ARMS: records[kind],logs[kind]=episode(kind,case,height,early)
    c=records['candidate']; costs={k:ratio(c,records[k]) for k in ('r10','r15')}
    false={k:match_false_events(c['false_alarm_times'],records[k]['false_alarm_times']) for k in ('r10','r15','d')}
    g4=score(records['r10'],c); eq=d_policy_equivalence(records,logs)
    gates={'status':all(x['status']=='ok' for x in records.values()),'G4':bool(g4['all_four_g4']),
           'cost_r10':max(costs['r10'].values())<=1.05,'cost_r15':max(costs['r15'].values())<=1.05,
           'false_events_r10':false['r10']['passed'],'false_events_r15':false['r15']['passed'],
           'no_added_vs_d':false['d']['passed'],
           'tool_r10':tool_gate(c,records['r10']),'tool_r15':tool_gate(c,records['r15']),
           'tool_d':tool_gate(c,records['d']),
           'saturation_min':saturation_min(c,records['r10'],records['r15'],records['d']),
           'runtime':c['ctrl_p99_us']<=800.0,'declared_d_policy_equivalence':eq['passed']}
    return {'id':case['id'],'claim':case['claim'],'case':case,'arms':records,'cost_ratios':costs,
            'g4':g4,'false_event_match':false,'d_policy_equivalence':eq,'gates':gates,'passed':all(gates.values())}

def fallback_case(case,height,early):
    records,logs={},{}
    for kind in ('r10','candidate'): records[kind],logs[kind]=episode(kind,case,height,early)
    parity=exact_pair(records['candidate'],logs['candidate'],records['r10'],logs['r10'])
    gates={'status':all(x['status']=='ok' for x in records.values()),'exact_r10_nonclock_and_logs':parity['passed'],
           'runtime':records['candidate']['ctrl_p99_us']<=800.0}
    return {'id':case['id'],'case':case,'arms':records,'parity':parity,'gates':gates,'passed':all(gates.values())}

def retains(candidate,reference):
    return reference['outcome']!='detected' or (candidate['outcome']=='detected' and
           candidate['delay_ms']<=reference['delay_ms']+5.0+1e-9)


def rising_times(log):
    detect=np.asarray(log['detect'],dtype=bool); t=np.asarray(log['t'],dtype=float)
    return t[np.flatnonzero(detect & ~np.r_[False,detect[:-1]])].tolist()

def impact_case(case,height,early):
    collision=case['collision']; records={}; logs={}; events={}; prefixes={}; forces={}; statuses={}
    for kind in ARMS:
        nr,nl=episode(kind,case,height,early); fr,fl=episode(kind,case,height,early,collision)
        records[kind]={'normal':nr,'forced':fr}; logs[kind]={'normal':nl,'forced':fl}
        statuses[kind]=nr['status']==fr['status']=='ok'; prefixes[kind]=equal_prefix(nl,fl,collision[0])
        forces[kind]=force_realization(fl['t'],fl['f_ext'],collision)
        events[kind]=event_result(fl['t'],fl['detect'],{'collision':collision},fl['f_tool'],nl['detect'])
    c=events['candidate']; false={k:match_false_events(c['false_alarm_times'],events[k]['false_alarm_times']) for k in ('r10','r15','d')}
    gates={'status':all(statuses.values()),'force_realized':all(x['passed'] for x in forces.values()),
           'matched_prefix':all(x['passed'] for x in prefixes.values()),
           'retain_r10':retains(c,events['r10']),'retain_r15':retains(c,events['r15']),'retain_d':retains(c,events['d']),
           'no_candidate_prealarm':not c['prealarm'],'false_events_r10':false['r10']['passed'],
           'false_events_r15':false['r15']['passed'],'no_added_vs_d':false['d']['passed'],
           'tool_r10':preserve_tool(c,events['r10']),'tool_r15':preserve_tool(c,events['r15']),'tool_d':preserve_tool(c,events['d']),
           'forced_saturation_min':saturation_min(records['candidate']['forced'],records['r10']['forced'],records['r15']['forced'],records['d']['forced']),
           'runtime':records['candidate']['forced']['ctrl_p99_us']<=800.0}
    rise_times={kind:{phase:rising_times(logs[kind][phase]) for phase in ('normal','forced')}
                for kind in ARMS}
    return {'id':case['id'],'claim':case['claim'],'case':case,'arms':events,'summaries':records,
            'rising_times_s':rise_times,
            'prefix':prefixes,'force':forces,'false_event_match':false,'gates':gates,'passed':all(gates.values())}


def fresh_impact_evidence(rows,planned_count):
    references=('r10','r15','d')
    positives={kind:sum(row['arms'][kind]['outcome']=='detected' for row in rows)
               for kind in references}
    candidate_positives=sum(row['arms']['candidate']['outcome']=='detected' for row in rows)
    retained={kind:sum(row['arms'][kind]['outcome']=='detected' and
                       retains(row['arms']['candidate'],row['arms'][kind]) for row in rows)
              for kind in references}
    near=[]
    for row in rows:
        onset,vector=row['case']['collision']; fx,fy,fz=map(abs,vector)
        if abs(float(onset)-float(T.T_RELEASE))<=.060+1e-12 and fz>0 and fz>=max(fx,fy):
            near.append(row)
    near_reference_positive=sum(any(row['arms'][kind]['outcome']=='detected'
                                    for kind in references) for row in near)
    near_candidate_retained=sum(row['arms']['candidate']['outcome']=='detected' and
                                any(row['arms'][kind]['outcome']=='detected'
                                    for kind in references) for row in near)
    gates={'complete_fresh_matrix':len(rows)==planned_count and planned_count>=8,
           'genuine_r10_positives':positives['r10']>0,
           'genuine_r15_positives':positives['r15']>0,
           'genuine_d_positives':positives['d']>0,
           'candidate_positive_nonvacuous':candidate_positives>0,
           'retain_all_r10_positives':retained['r10']==positives['r10'],
           'retain_all_r15_positives':retained['r15']==positives['r15'],
           'retain_all_d_positives':retained['d']==positives['d'],
           'planned_near_release_normal_conditions':len(near)>=2,
           'near_release_reference_positive':near_reference_positive>0,
           'near_release_candidate_retained_response':near_candidate_retained>0}
    passed=all(gates.values())
    return {'schema':'r17-fresh-impact-evidence.v1','gates':gates,'passed':passed,
            'conclusion':'sufficient_positive_evidence' if passed else 'inconclusive_not_qualified',
            'reference_positive_counts':positives,'candidate_positive_count':candidate_positives,
            'retained_reference_positive_counts':retained,
            'near_release_normal_ids':[row['id'] for row in near],
            'near_release_reference_positive_count':near_reference_positive,
            'near_release_candidate_retained_count':near_candidate_retained}

def validate_upstream(path,stage,auth):
    d=json.loads(Path(path).read_text())
    if d.get('schema')!='r17-progressive-stage.v1' or d.get('stage')!=stage or d.get('complete') is not True or d.get('passed') is not True or d.get('auth')!=auth:
        raise ValueError('complete passing authenticated immediate upstream required')
    return d

def main(argv=None):
    global CANDIDATE,CANDIDATE_DETECTOR,NONCLOCK_EQUIVALENT_TO_D
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=tuple(STAGE_KEYS),required=True); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--height',type=Path,required=True); p.add_argument('--early',type=Path,required=True); p.add_argument('--auth',type=Path,required=True)
    p.add_argument('--upstream',type=Path); p.add_argument('--case'); p.add_argument('--out',type=Path,required=True); a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink(): p.error('fresh output required')
    try:
        plan=json.loads(a.plan.read_text()); auth=json.loads(a.auth.read_text()); height=json.loads(a.height.read_text()); early=json.loads(a.early.read_text())
        if plan.get('schema')!='r17-progressive-handoff-plan.v1' or plan.get('candidate_module')!='experiments.r17_candidate.policy' or auth.get('plan_sha256')!=sha(a.plan): raise ValueError('plan/auth mismatch')
        module=importlib.import_module('experiments.r17_candidate.policy'); CANDIDATE=module.Controller; CANDIDATE_DETECTOR=module.Detector
        NONCLOCK_EQUIVALENT_TO_D=bool(getattr(module,'NONCLOCK_EQUIVALENT_TO_D',False))
        prev=PREVIOUS.get(a.stage)
        if prev is None:
            if a.upstream is not None: raise ValueError('first stage accepts no upstream')
            upstream_sha=None
        else:
            if a.upstream is None: raise ValueError('stage requires immediate upstream')
            validate_upstream(a.upstream,prev,auth); upstream_sha=sha(a.upstream)
        cases=plan[STAGE_KEYS[a.stage]]; selected=cases if a.case is None else [x for x in cases if x['id']==a.case]
        if not selected: raise ValueError('unknown case id')
        rows=[(impact_case if a.stage.endswith('impact') else normal_case)(x,height,early) for x in selected]
        fallbacks=[fallback_case(x,height,early) for x in plan['fallback_cases']] if a.stage=='known-normal' and a.case is None else []
        complete=a.case is None and len(rows)==len(cases) and (a.stage!='known-normal' or len(fallbacks)==len(plan['fallback_cases']))
        aggregate=(fresh_impact_evidence(rows,len(cases)) if a.stage=='fresh-impact' and a.case is None else None)
        passed=bool(complete and all(x['passed'] for x in rows+fallbacks) and
                    (aggregate is None or aggregate['passed']))
        failed=[{'id':x['id'],'gates':[k for k,v in x['gates'].items() if not v]} for x in rows+fallbacks if not x['passed']]
        if aggregate is not None and not aggregate['passed']:
            failed.append({'id':'aggregate_fresh_impact_evidence',
                           'gates':[k for k,v in aggregate['gates'].items() if not v]})
        result={'schema':'r17-progressive-stage.v1','stage':a.stage,'auth':auth,'complete':complete,'diagnostic_case':a.case,
                'rows':rows,'fallbacks':fallbacks,'aggregate_evidence':aggregate,
                'passed':passed,'failed':failed,'upstream_sha256':upstream_sha,
                'timing_note':'controller p99 is measured simulation timing, not hard real-time proof'}
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as e: p.error(str(e))
    with a.out.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'stage':a.stage,'complete':complete,'passed':passed,'failed':failed},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
