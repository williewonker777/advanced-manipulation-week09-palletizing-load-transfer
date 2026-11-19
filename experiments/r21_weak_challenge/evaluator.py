"""Run fixed weak-collision twins with a restricted detector-arm registry."""
from __future__ import annotations
import argparse,hashlib,importlib,json,os
from pathlib import Path
import numpy as np
import my_detector
from my_torque import Controller
from experiments.r18_paired_impulses import runner
from experiments.r20_horizontal import candidate_detector
from sim.runner import COLLISION_HOLD,COLLISION_RAMP

ROOT=Path(__file__).resolve().parents[2]
PHYSICAL=('t','q','q_d','tau','e_mm','f_table','f_tool','qpos','f_ext',
          'f_ext_xyz_native','f_ext_xyz_second','f_ext_xyz_committed')
PREFIX=PHYSICAL+('r','detect')
STATIC={'r10':my_detector.Detector,'r20':candidate_detector.Detector}
CANDIDATE={'r21_pick':'experiments.r21_pick_observer.detector'}
REQUIRED_PLAN={'schema','status','selection','dt_s','reference_twin_ms','true_window_tail_ms',
               'source_pins','allowed_arms','forced','negative_only','required_trials','gates','interpretation'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rises(log):
 bits=np.asarray(log['detect'],bool);return np.asarray(log['t'])[np.flatnonzero(bits&~np.r_[False,bits[:-1]])].tolist()
def tool(log):
 f=np.asarray(log['f_tool'])>1.;d=np.asarray(log['detect'],bool)
 if not f.any():return {'present':False,'detected':False,'prealarm':False,'peak_N':float(np.max(log['f_tool']))}
 a=int(np.argmax(f));b=min(len(f),int(np.flatnonzero(f)[-1])+300)
 return {'present':True,'detected':bool(d[a:b].any()),'prealarm':bool(d[:a].any()),'peak_N':float(np.max(log['f_tool']))}
def twin_at(reference,hit,radius):
 if hit is None:return False
 t=np.asarray(reference['t']);k=int(np.searchsorted(t,hit))
 return bool(np.asarray(reference['detect'],bool)[max(0,k-radius):min(len(t),k+radius+1)].any())
def classify(metrics,log,reference,onset,radius):
 delay=metrics['detect_delay_ms'];hit=None if delay is None else onset+delay/1000.
 twin=twin_at(reference,hit,radius)
 official=('alarm_before' if metrics['alarm_before_collision'] else 'missed' if delay is None
           else 'indistinguishable' if twin else 'detected')
 stop=onset+2*COLLISION_RAMP+COLLISION_HOLD+.2
 return {'official':official,'strict_true_positive':bool(hit is not None and not metrics['alarm_before_collision']
          and not twin and onset<=hit<stop),'delay_ms':delay,'alarm_before':bool(metrics['alarm_before_collision']),
          'reference_twin':twin,'false_rise_times':metrics['false_alarm_times'],'all_rise_times':rises(log)}
def exact_map(a,b,keys):return {k:bool(np.array_equal(a[k],b[k])) for k in keys}
def prefix_map(a,b,keys,stop):return {k:bool(np.array_equal(a[k][:stop],b[k][:stop])) for k in keys}
def unmatched(a,b,tol=.0011):return [float(x) for x in a if not any(abs(float(x)-float(y))<=tol for y in b)]
def expected_force(log,collision):
 expected=runner.expected_signed(log['t'],(collision,None))
 scalar=np.fromiter((np.linalg.norm(row) for row in expected),dtype=float,count=len(expected))
 return {'native':bool(np.array_equal(log['f_ext_xyz_native'],expected)),
         'second_zero':bool(np.array_equal(log['f_ext_xyz_second'],np.zeros_like(expected))),
         'committed':bool(np.array_equal(log['f_ext_xyz_committed'],expected)),
         'stock_scalar':bool(np.array_equal(log['f_ext'],scalar))}
def validate(plan):
 if set(plan)!=REQUIRED_PLAN or plan['schema']!='r21-weak-horizontal-challenge.v1' or plan['status']!='FROZEN_BEFORE_ANY_EPISODE_OUTCOME':raise ValueError('invalid plan')
 ids=[x['id'] for x in plan['forced']+plan['negative_only']]
 if len(ids)!=len(set(ids)) or len(plan['forced'])!=15 or len(plan['negative_only'])!=3:raise ValueError('incomplete fixed cohort')
 for rel,pin in plan['source_pins'].items():
  if sha(ROOT/rel)!=pin:raise ValueError('source pin differs: '+rel)
def resolve(names,candidate_pin):
 out={}
 for name in names:
  if name in STATIC:out[name]=STATIC[name]
  elif name in CANDIDATE:
   if not candidate_pin:raise ValueError('candidate pin required')
   module=importlib.import_module(CANDIDATE[name]);source=Path(module.__file__)
   if sha(source)!=candidate_pin:raise ValueError('candidate pin differs')
   out[name]=module.Detector
  else:raise ValueError('unknown arm: '+name)
 if 'r10' not in out:raise ValueError('r10 comparator required')
 return out
def episode(detector,case,collision):
 metrics,log=runner.run_episode(Controller(),box_mass=case['mass'],box_com=tuple(case['com']),
   detector=detector(),collision=collision,pick_offset=case.get('pick_offset'),impacts=None)
 return metrics,log
def forced_row(case,arms,radius):
 traces={}
 for name,det in arms.items():
  traces[name]={}
  traces[name]['none']=episode(det,case,None)
  traces[name]['force']=episode(det,case,(case['time'],case['force']))
 ref=traces['r10'];result={}
 for name,pair in traces.items():
  nm,nl=pair['none'];fm,fl=pair['force'];stop=int(np.searchsorted(nl['t'],case['time']))
  result[name]={
   'none':{'false_rise_times':nm['false_alarm_times'],'all_rise_times':rises(nl),'tool':tool(nl),
           'stuck_tail':bool(nl['detect'][-1])},
   'force':{**classify(fm,fl,nl,case['time'],radius),'tool':tool(fl),'stuck_tail':bool(fl['detect'][-1])},
   'preimpact_prefix_exact':prefix_map(nl,fl,PREFIX,stop),
   'applied_force_exact':expected_force(fl,(case['time'],case['force'])),
   'none_physical_vs_r10':exact_map(ref['none'][1],nl,PHYSICAL),
   'force_physical_vs_r10':exact_map(ref['force'][1],fl,PHYSICAL)}
 base=result['r10']
 comparisons={}
 for name,row in result.items():
  if name=='r10':continue
  checks={'no_added_no_force_false':not unmatched(row['none']['false_rise_times'],base['none']['false_rise_times']),
   'no_added_force_false':not unmatched(row['force']['false_rise_times'],base['force']['false_rise_times']),
   'no_new_prealarm':not row['force']['alarm_before'] or base['force']['alarm_before'],
   'preserve_tool':not base['force']['tool']['detected'] or row['force']['tool']['detected'],
   'no_new_stuck_tail':not row['force']['stuck_tail'] or base['force']['stuck_tail'],
   'no_strict_r10_loss':not base['force']['strict_true_positive'] or row['force']['strict_true_positive'],
   'preimpact_prefix_exact':all(row['preimpact_prefix_exact'].values()),
   'applied_force_exact':all(row['applied_force_exact'].values()),
   'physical_equal':all(row['none_physical_vs_r10'].values()) and all(row['force_physical_vs_r10'].values())}
  comparisons[name]={'checks':checks,'passed':all(checks.values())}
 return {'id':case['id'],'role':case['role'],'case':case,'episodes':2*len(arms),'arms':result,'comparisons':comparisons,
         'passed':all(x['passed'] for x in comparisons.values())}
def negative_row(case,arms):
 traces={n:episode(d,case,None) for n,d in arms.items()};base=traces['r10'];rows={}
 for n,(m,l) in traces.items():
  rows[n]={'false_rise_times':m['false_alarm_times'],'all_rise_times':rises(l),'tool':tool(l),
           'stuck_tail':bool(l['detect'][-1]),'physical_vs_r10':exact_map(base[1],l,PHYSICAL)}
 comps={}
 for n,r in rows.items():
  if n=='r10':continue
  c={'no_added_false':not unmatched(r['false_rise_times'],rows['r10']['false_rise_times']),
     'preserve_tool':not rows['r10']['tool']['detected'] or r['tool']['detected'],
     'no_new_prealarm':not r['tool']['prealarm'] or rows['r10']['tool']['prealarm'],
     'no_new_stuck_tail':not r['stuck_tail'] or rows['r10']['stuck_tail'],
     'physical_equal':all(r['physical_vs_r10'].values())}
  comps[n]={'checks':c,'passed':all(c.values())}
 return {'id':case['id'],'role':case['role'],'case':case,'episodes':len(arms),'arms':rows,'comparisons':comps,
         'passed':all(x['passed'] for x in comps.values())}
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--arms',default='r10,r20')
 p.add_argument('--candidate-pin');a=p.parse_args(argv)
 if os.path.lexists(a.out):p.error('new output path required')
 raw=a.plan.read_bytes();plan=json.loads(raw);validate(plan)
 names=a.arms.split(',')
 if len(names)!=len(set(names)) or any(n not in plan['allowed_arms'] for n in names):p.error('invalid arm set')
 arms=resolve(names,a.candidate_pin);radius=int(plan['reference_twin_ms'])
 before={rel:sha(ROOT/rel) for rel in plan['source_pins']}
 forced=[forced_row(c,arms,radius) for c in plan['forced']]
 negatives=[negative_row(c,arms) for c in plan['negative_only']]
 if any(sha(ROOT/rel)!=v for rel,v in before.items()):raise ValueError('source changed during run')
 doc={'schema':'r21-weak-horizontal-result.v1','plan_sha256':hashlib.sha256(raw).hexdigest(),
      'arms':names,'candidate_pin':a.candidate_pin,'episodes':sum(r['episodes'] for r in forced+negatives),
      'forced':forced,'negative_only':negatives,
      'passed':all(r['passed'] for r in forced+negatives)}
 with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'episodes':doc['episodes'],'forced':len(forced),'negative_only':len(negatives),'passed':doc['passed']}))
 return 0
if __name__=='__main__':raise SystemExit(main())
