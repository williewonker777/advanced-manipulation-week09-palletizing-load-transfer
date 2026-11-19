"""Stage-gated independent evaluation of the separately frozen impulse detector."""
from __future__ import annotations
import argparse,hashlib,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.r16_payload_contract.controller import Controller as D
from experiments.r18_paired_impulses import runner,scoring
from experiments.r18_single_pass import paired_replay as replay
from experiments.r19_close_pairs import assay
from experiments.r19_event_state.physical_detector import Detector as V2
from experiments.r19_impulse_end.detector import Detector as Candidate
from tools.r14_paired_events import match_false_events,preserve_tool
from tools.r5_metrics import event_result

ROOT=Path(__file__).resolve().parents[2];TWINS=scoring.TWINS;BASE=('r10','raw_r15','D_coupled','single_pass')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(x):return replay.signature(x)
def write(p,x):
 p=Path(p)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def episode(arm,case,h,e,model,impacts):
 if arm in BASE:return replay.episode(arm,case,h,e,model,impacts)
 det=(V2 if arm=='event' else Candidate)();det.model_path=model
 m,l=runner.run_episode(D(h,e),case['mass'],tuple(case['com']),det,pick_offset=case.get('offset'),impacts=impacts);l['r18_diverged']=m['diverged']
 if not scoring.finite_grid(l):raise ValueError('incomplete '+case['id']+'/'+arm)
 return l

def assignments(case):
 first,second=case['impacts'];return {'none':(None,None),'first':(first,None),'second':(None,second),'both':(first,second)}

def characterize_arm(arm,case,h,e,model,quiet,window):
 twins={k:episode(arm,case,h,e,model,v) for k,v in assignments(case).items()};record,raw=assay.characterize(case,twins,quiet,window);return twins,record,raw

def valid_force(record):
 names=('complete_finite_strict_grid','force_free_prefixes','all_required_prefixes_exact','distinct_non_noop_impacts','no_sign_inversion')
 return record['signed_and_scalar_exact'] and all(record['r18_gates'][x] for x in names)

def duplicate_not_above(candidate,v2):
 # Crediting can legitimately increase when a carried-high V2 alarm is ended
 # and the same one raw response becomes quiet/attributable.  Duplicate control
 # therefore compares raw rises; credited must remain a subset of candidate raw.
 return {slot:(len(candidate['slots'][slot]['raw_delays_ms'])<=len(v2['slots'][slot]['raw_delays_ms']) and len(candidate['slots'][slot]['credited_delays_ms'])<=len(candidate['slots'][slot]['raw_delays_ms'])) for slot in assay.SLOTS}

def known_case(index,case,h,e,model,quiet,window,reference):
 r18,candidate={},{};r18,rr,raw_r=characterize_arm('single_pass',case,h,e,model,quiet,window);candidate,cc,raw_c=characterize_arm('candidate',case,h,e,model,quiet,window)
 ref=reference[case['id']];ref_r18,ref_v2=ref['arms']['single_pass'],ref['arms']['event'];cmp18=assay.compare(cc,ref_r18);cmpv2=assay.compare(cc,ref_v2)
 physics={tw:all(np.array_equal(candidate[tw][n],r18[tw][n]) for n in replay.PHYSICAL+('r',)) for tw in TWINS}
 actual_r18_signature=signature(raw_r)
 gates={'published_R18_score_exact':actual_r18_signature==ref['assessment_signatures']['single_pass'],'published_R18_character_exact':rr==ref_r18,'published_reference_authenticated':True,'candidate_force_prefix':valid_force(cc),'R18_residual_and_physics_exact':all(physics.values()),'retain_no_false_tool_tail_R18':assay.all_comparisons([cmp18]),'retain_no_false_tool_tail_V2':assay.all_comparisons([cmpv2]),'duplicates_not_above_V2':all(duplicate_not_above(cc,ref_v2).values())}
 if case['id']=='pair_return_opposed':gates['known_5_71_plus33_once']=cc['slots']['both_second']['credited_delays_ms']==cc['slots']['both_second']['raw_delays_ms']==[33]
 if case['id'].startswith('diag_known_'):gates['known_anchor_preserved']=assay.one_per_pulse(cc)
 if case['id']=='gap80_weak':gates['revive_plus24_quiet_unsplit']=(cc['slots']['both_second']['credited_delays_ms']==cc['slots']['both_second']['raw_delays_ms']==[24] and cc['slots']['both_second']['quiet_before_onset'] and len(cc['slots']['both_first']['raw_delays_ms'])==len(cc['slots']['both_first']['credited_delays_ms'])==1)
 return {'id':case['id'],'case_role':'previously_exposed_regression_only','episodes':8,'published_baseline_signatures':ref['assessment_signatures'],'actual_R18_assessment_sha256':actual_r18_signature,'compact_reference_hashes':{k:signature(ref['arms'][k]) for k in ('r10','raw_r15','D_coupled','single_pass','event')},'candidate_assessment_sha256':signature(raw_c),'candidate':cc,'comparisons':{'R18':cmp18,'V2':cmpv2},'duplicate_not_above_V2':duplicate_not_above(cc,ref_v2),'physics':physics,'gates':gates,'passed':all(gates.values())}

def normal_case(index,case,h,e,model):
 logs={k:episode(k,case,h,e,model,(None,None)) for k in (*BASE,'candidate')};records={k:event_result(v['t'],v['detect'],{},v['f_tool']) for k,v in logs.items()};trial,parent=logs['candidate'],logs['single_pass']
 gates={'complete_finite':all(scoring.finite_grid(v) for v in logs.values()),'no_force':all(not np.any(v['f_ext_xyz_committed']) for v in logs.values()),'D_R18_candidate_physics':all(np.array_equal(trial[n],logs[k][n]) for k in ('D_coupled','single_pass') for n in replay.PHYSICAL),'R18_residual_parity':np.array_equal(trial['r'],parent['r']),'R18_bits_equal':np.array_equal(trial['detect'],parent['detect']),'no_added_false':all(match_false_events(records['candidate']['false_alarm_times'],records[k]['false_alarm_times'])['passed'] for k in BASE),'no_tool_loss':all(preserve_tool(records['candidate'],records[k]) for k in BASE)}
 return {'id':case['id'],'episodes':5,'events':records,'gates':gates,'passed':all(gates.values())}

def novel_case(index,case,h,e,model,quiet,window):
 arms={};raws={};twins={}
 for kind in (*BASE,'candidate'):
  twins[kind],arms[kind],raws[kind]=characterize_arm(kind,case,h,e,model,quiet,window)
 comparisons={k:assay.compare(arms['candidate'],arms[k]) for k in BASE};physics={k:{tw:all(np.array_equal(twins['candidate'][tw][n],twins[k][tw][n]) for n in replay.PHYSICAL+(('r',) if k=='single_pass' else ())) for tw in TWINS} for k in ('D_coupled','single_pass')}
 normal_logs={k:v['none'] for k,v in twins.items()};records={k:event_result(v['t'],v['detect'],{},v['f_tool']) for k,v in normal_logs.items()}
 normal={'no_added_false':all(match_false_events(records['candidate']['false_alarm_times'],records[k]['false_alarm_times'])['passed'] for k in BASE),'no_tool_loss':all(preserve_tool(records['candidate'],records[k]) for k in BASE),'R18_normal_bits_residual':np.array_equal(normal_logs['candidate']['detect'],normal_logs['single_pass']['detect']) and np.array_equal(normal_logs['candidate']['r'],normal_logs['single_pass']['r'])}
 gates={'candidate_force_prefix':valid_force(arms['candidate']),'all_actual_baseline_force_prefix':all(valid_force(arms[k]) for k in BASE),'exactly_one_raw_and_credited_per_slot':assay.one_per_pulse(arms['candidate']),'all_reference_retention_false_tool_tail':assay.all_comparisons(comparisons.values()),'D_R18_physics_and_residual':all(all(x.values()) for x in physics.values()),'normal_gate_inside_twins':all(normal.values())}
 return {'id':case['id'],'episodes':20,'case':case,'assessment_signatures':{k:signature(v) for k,v in raws.items()},'arms':arms,'comparisons':comparisons,'physics':physics,'normal':normal,'gates':gates,'passed':all(gates.values())}

def parallel(stage,cases,out,func,args,jobs=2):
 folder=Path(out)/stage;folder.mkdir();rows={}
 with ProcessPoolExecutor(max_workers=jobs) as pool:
  futures={pool.submit(func,i,c,*args):c for i,c in enumerate(cases)}
  for f in as_completed(futures):
   row=f.result();write(folder/(row['id']+'.json'),row);rows[row['id']]=row;print(stage,row['id'],row['passed'],flush=True)
 return [rows[c['id']] for c in cases]

def pinset(a):
 paths={'plan':a.plan,'candidate_detector':ROOT/'experiments/r19_impulse_end/detector.py','candidate_state':ROOT/'experiments/r19_impulse_end/state.py','close_assay':ROOT/'experiments/r19_close_pairs/assay.py','paired_runner':ROOT/'experiments/r18_paired_impulses/runner.py','paired_scorer':ROOT/'experiments/r18_paired_impulses/scoring.py','r17_model':a.model,'height':a.height,'early':a.early,'published_stock':a.stock}
 return {k:sha(v) for k,v in paths.items()}

def main(argv=None):
 q=argparse.ArgumentParser()
 for n in ('plan','stock','prior-plan','close-plan','height','early','model','out'):q.add_argument('--'+n,type=Path,required=True)
 q.add_argument('--jobs',type=int,default=2,choices=(1,2));a=q.parse_args()
 if a.out.exists():q.error('fresh output directory required')
 a.out.mkdir()
 plan=read(a.plan);stock=read(a.stock);published=stock['result'];expected=plan['published_reference']['sha256']
 if sha(a.stock)!=expected or not published['stages']['known']['passed'] or not published['stages']['normal']['passed']:raise ValueError('untrusted published baseline')
 prior=read(a.prior_plan);close=read(a.close_plan);known=prior['fresh_paired_cases']+prior['diagnostic_sequences']+close['new'];fresh=plan['new']
 if len(known)!=12 or len(fresh)!=6 or len({x['id'] for x in known+fresh})!=18:raise ValueError('case matrix mismatch')
 for c in fresh:
  if round(1000*(c['impacts'][1][0]-c['impacts'][0][0]))!=120+c['gap_ms']:raise ValueError('declared gap mismatch')
 refs={r['id']:r for stage in ('known','new') for r in published['stages'][stage]['rows']};h,e=read(a.height),read(a.early);pins=pinset(a);write(a.out/'INPUT_PINS.json',pins)
 def verify():
  if pinset(a)!=pins:raise ValueError('input/source drift')
 stages={};rows=parallel('known',known,a.out,known_case,(h,e,a.model,plan['quiet_ms'],plan['window_ms'],refs),a.jobs);verify();stages['known']={'rows':rows,'passed':all(x['passed'] for x in rows)}
 if stages['known']['passed']:
  rows=parallel('normal',fresh,a.out,normal_case,(h,e,a.model),a.jobs);verify();stages['normal']={'rows':rows,'passed':all(x['passed'] for x in rows)}
 if stages.get('normal',{}).get('passed'):
  rows=parallel('novel',fresh,a.out,novel_case,(h,e,a.model,plan['quiet_ms'],plan['window_ms']),a.jobs);verify();stages['novel']={'rows':rows,'strict_merit_count':sum(x['passed'] for x in rows),'passed':all(x['passed'] for x in rows)}
 report={'schema':'r19-impulse-gate-result.v1','stages':stages,'pins':pins,'published_reference_sha256':sha(a.stock),'novel_forces_opened':len(stages.get('novel',{}).get('rows',[])),'aggregate_success':len(stages)==3 and stages['novel']['passed'],'promotion_eligible':False,'episodes':sum(x['episodes'] for s in stages.values() for x in s['rows'])};write(a.out/'report.json',report);print(json.dumps({k:{'passed':v['passed'],'rows':len(v['rows'])} for k,v in stages.items()}));return 0

if __name__=='__main__':raise SystemExit(main())
