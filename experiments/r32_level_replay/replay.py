"""Replay disclosed D4 anchors and the published R26 close-place force twins."""
import argparse,hashlib,json
from pathlib import Path
from types import FunctionType
import numpy as np
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r22_census import check
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r29_native_event_link import evaluator as paired
from experiments.r31_held_identity.candidate import Detector as Identity
from experiments.r32_measured_level.candidate import Detector as Level
from experiments.r32_native_handoff.candidate import Detector as Handoff
from experiments.r32_level_replay.channels import IdentityChannels,LevelChannels,HandoffChannels

CASES=(266,287,298,316);ARMS={'c3':C3,'identity':Identity,'level_a':Level,'handoff_c2':Handoff}
def spans(bits):
 a=np.asarray(bits,bool);d=np.diff(np.r_[False,a,False].astype(np.int8));return [[int(x),int(y)] for x,y in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1),strict=True)]
def captured_record(case, payload):
 # Score the exact saved run; never integrate a second forced episode just to construct its record.
 def from_capture(_detector,expected_case):
  if expected_case is not case:raise ValueError('different captured input')
  return payload
 f=raw.episode_record
 return FunctionType(f.__code__,{**f.__globals__,'execute':from_capture},f.__name__,f.__defaults__,f.__closure__)(None,case)

def d4():
 cases=check.build_cases('my_torque')
 if hashlib.sha256(check.old.canonical_cases(cases)).hexdigest()!=check.MATRIX:raise ValueError('wrong physical input matrix')
 calls=0;refs={check.old.reference_key(c):c for c in cases if c['kind']=='reference'};rows=[]
 for cid in CASES:
  c=cases[cid];control=refs[check.old.reference_key(c)];arms={}
  for name,base,wrapped in (('c3',C3,None),('identity',Identity,IdentityChannels),('level_a',Level,LevelChannels),('handoff_c2',Handoff,HandoffChannels)):
   normal=raw.episode_record(base,control);calls+=1
   if wrapped is None:metrics,log,signed=raw.execute(base,c);extra={}
   else:
    metrics,log,signed=raw.execute(wrapped,c);channel=wrapped.latest.channel_log
    extra={k:spans([x[k] for x in channel]) for k in channel[0] if k!='routes'}
    if name=='identity':extra['route_spans']=[{'span':s,'routes':list(channel[s[0]]['routes'])} for s in extra['native']]
   calls+=1;record=captured_record(c,(metrics,log,signed));score=check.score(c,record,normal)
   if not all(score['valid'].values()) or wrapped and extra['output']!=record['detect_intervals']:raise ValueError('invalid captured force, prefix or public output')
   force=np.linalg.norm(signed,axis=1)>0;alarm=spans(log['detect'])
   arms[name]={'returned_boolean':alarm,'score':score,'signed_force_first':int(np.flatnonzero(force)[0]),
               'signed_force_last':int(np.flatnonzero(force)[-1]),'returned_overlap_samples':[int(np.count_nonzero(force[a:b])) for a,b in alarm],**extra}
  rows.append({'id':cid,'control_id':control['case_id'],'input':c,'arms':arms})
 return rows,calls
def place():
 plan=json.loads(paired.PLAN.read_bytes());table={'c3':C3,'identity':IdentityChannels,'level_a':LevelChannels,'handoff_c2':HandoffChannels}
 calls=0
 def detector_run(run,case,variant,detector_type,impacts):
  nonlocal calls
  calls+=1
  detector=detector_type();metrics,log=run(case,variant,lambda:detector,impacts)
  force=np.linalg.norm(np.asarray(log['f_ext_xyz_committed']),axis=1)>0
  diag={'returned_boolean':spans(log['detect']),'actual_signed_force':spans(force)}
  if getattr(detector,'channel_log',None):
   for key in detector.channel_log[0]:
    if key!='routes':diag[key]=spans([r[key] for r in detector.channel_log])
  return metrics,log,diag
 row=FunctionType(paired.p26_case.__code__,{**paired.p26_case.__globals__,'ARMS':table,'detector_run':detector_run},paired.p26_case.__name__,paired.p26_case.__defaults__,paired.p26_case.__closure__)(plan)
 if row['episodes']!=calls:raise ValueError('wrong physical pair integration count')
 for name,record in row['records'].items():
  event=record['events']['both_second'];both=record['diagnostics']['both']
  record['both_second_registered_check']={'quiet_before':event['quiet_before'],'high_before':event['high_before'],
      'strict':event['strict'],'returned_boolean':both['returned_boolean'],'actual_signed_force':both['actual_signed_force']}
 return row
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('scope',choices=('examples','place','both'));p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
 rows,d4_episodes=d4() if a.scope in ('examples','both') else ([],0);pair=place() if a.scope in ('place','both') else None
 pair_episodes=0 if pair is None else pair['episodes'];episodes=d4_episodes+pair_episodes
 if d4_episodes!=2*len(rows)*len(ARMS):raise ValueError('wrong matched D4 integration count')
 out={'schema':'r32-level-replay.v2','execution_counts':{'d4_forced':d4_episodes//2,'d4_matched_calm':d4_episodes//2,'paired_twins':pair_episodes},'scope':a.scope,'arms':list(ARMS),'episodes':episodes,'d4':rows,'close_place':pair,
      'course_original':False,'default_changed':False,'candidate_a_enrollable':False,
      'interpretation':['returned_boolean is the actual course output','identity notice is a separate optional event envelope; returned output may also contain native activity','measured continuation/load/native linkage is detector telemetry, not physical force truth','C2 has disclosed strict-separation and broad historical losses and is not claimed superior']}
 with a.out.open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(a.scope,episodes)
if __name__=='__main__':main()
