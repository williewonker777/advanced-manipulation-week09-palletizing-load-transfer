"""Frozen four-trial post-fit assay for the R21 observer."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from experiments.r20_horizontal.candidate_detector import Detector as R20
from experiments.r21_pick_observer.candidate import Detector as R21,MODEL
from my_torque import Controller
from sim.runner import COLLISION_HOLD,COLLISION_RAMP,run_episode
KEYS=('t','q','q_d','tau','e_mm','f_table','f_tool','f_ext','qpos')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rises(log):
 d=np.asarray(log['detect'],bool);return np.asarray(log['t'])[np.flatnonzero(d&~np.r_[False,d[:-1]])].tolist()
def run(cls,c,force):return run_episode(Controller(),box_mass=c['mass'],box_com=c['com'],detector=cls(),collision=force)
def classify(result,log,ref,c):
 delay=result['detect_delay_ms'];hit=None if delay is None else c['time']+delay/1000
 t=np.asarray(ref['t']);k=0 if hit is None else int(np.searchsorted(t,hit)); twin=False if hit is None else bool(np.asarray(ref['detect'],bool)[max(0,k-20):k+21].any())
 strict=bool(hit is not None and not result['alarm_before_collision'] and not twin and hit<c['time']+2*COLLISION_RAMP+COLLISION_HOLD+.2)
 return {'delay_ms':delay,'strict':strict,'alarm_before':result['alarm_before_collision'],'reference_twin':twin,'false_times':result['false_alarm_times'],'rises':rises(log)}
def main(plan_path,out):
 raw=Path(plan_path).read_bytes();p=json.loads(raw)
 if p['status']!='FROZEN_BEFORE_ANY_FRESH_OUTCOME' or sha(Path(__file__).with_name('candidate.py'))!=p['candidate_sha256'] or sha(MODEL)!=p['model_sha256']:raise ValueError('frozen source/plan required')
 rows=[]
 for c in p['cases']:
  trials={}
  for arm,cls in [('r20',R20),('r21',R21)]:
   nr,nl=run(cls,c,None);fr,fl=run(cls,c,(c['time'],c['force']));trials[arm]=(nr,nl,fr,fl)
  a,b=trials['r20'],trials['r21'];none_eq={k:bool(np.array_equal(a[1][k],b[1][k])) for k in KEYS};force_eq={k:bool(np.array_equal(a[3][k],b[3][k])) for k in KEYS}
  ca,cb=classify(a[2],a[3],a[1],c),classify(b[2],b[3],b[1],c)
  row={'id':c['id'],'case':c,'r20':ca,'r21':cb,'candidate_only':cb['strict'] and not ca['strict'],
       'none_false':{'r20':a[0]['false_alarm_times'],'r21':b[0]['false_alarm_times']},'none_equal':none_eq,'force_equal':force_eq}
  row['gates']={'physical_force_exact':all(none_eq.values()) and all(force_eq.values()),'no_added_none_false':row['none_false']['r21']==row['none_false']['r20'],'no_added_force_false':cb['false_times']==ca['false_times'],'no_baseline_loss':not ca['strict'] or cb['strict']};row['passed']=all(row['gates'].values());rows.append(row)
 result={'schema':'r21-pick-observer-fresh-result.v1','plan_sha256':hashlib.sha256(raw).hexdigest(),'episodes':4*len(rows),'rows':rows,'gates':{'all_rows':all(r['passed'] for r in rows),'candidate_only_gain':any(r['candidate_only'] for r in rows)},'candidate_only':sum(r['candidate_only'] for r in rows)}
 Path(out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'episodes':result['episodes'],'candidate_only':result['candidate_only'],'all_rows':result['gates']['all_rows']}))
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--plan',required=True);q.add_argument('--out',required=True);a=q.parse_args();main(a.plan,a.out)
