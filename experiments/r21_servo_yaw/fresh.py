import argparse,hashlib,json
from pathlib import Path
import numpy as np
import my_detector
from experiments.r21_servo_yaw.candidate import Detector as Candidate
from my_servo import Controller
from sim.runner import COLLISION_HOLD,COLLISION_RAMP,run_episode
KEYS=('t','q','q_d','tau','e_mm','f_table','f_tool','f_ext','qpos')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cls,c,col):return run_episode(Controller(),box_mass=c['mass'],box_com=c['com'],detector=cls(),collision=col)
def classify(result,log,ref,c):
 delay=result['detect_delay_ms'];hit=None if delay is None else c['time']+delay/1000;t=np.asarray(ref['t']);k=0 if hit is None else int(np.searchsorted(t,hit));twin=False if hit is None else bool(np.asarray(ref['detect'])[max(0,k-20):k+21].any());strict=bool(hit is not None and not result['alarm_before_collision'] and not twin and hit<c['time']+2*COLLISION_RAMP+COLLISION_HOLD+.2);return {'delay_ms':delay,'strict':strict,'alarm_before':result['alarm_before_collision'],'reference_twin':twin,'false':result['false_alarm_times']}
def main(plan,out):
 raw=Path(plan).read_bytes();p=json.loads(raw);root=Path(__file__).parent
 if p['status']!='FROZEN_BEFORE_ANY_FRESH_OUTCOME' or sha(root/'candidate.py')!=p['candidate_sha256'] or sha(root/'normal_bins.json')!=p['normal_model_sha256']:raise ValueError('frozen inputs')
 rows=[]
 for c in p['cases']:
  a=(run(my_detector.Detector,c,None),run(my_detector.Detector,c,(c['time'],c['force'])));b=(run(Candidate,c,None),run(Candidate,c,(c['time'],c['force'])));ca=classify(a[1][0],a[1][1],a[0][1],c);cb=classify(b[1][0],b[1][1],b[0][1],c);eq=[{k:bool(np.array_equal(a[i][1][k],b[i][1][k])) for k in KEYS} for i in (0,1)];g={'exact_physics_force':all(v for x in eq for v in x.values()),'no_added_none_false':b[0][0]['false_alarm_times']==a[0][0]['false_alarm_times'],'no_added_force_false':cb['false']==ca['false'],'no_baseline_loss':not ca['strict'] or cb['strict']};rows.append({'id':c['id'],'force_N':float(np.linalg.norm(c['force'])),'r10':ca,'candidate':cb,'candidate_only':cb['strict'] and not ca['strict'],'physical_equal':eq,'gates':g,'passed':all(g.values())})
 doc={'schema':'r21-servo-yaw-fresh-result.v1','plan_sha256':hashlib.sha256(raw).hexdigest(),'episodes':4*len(rows),'rows':rows,'passed':all(r['passed'] for r in rows),'weak':{'r10':sum(r['r10']['strict'] for r in rows if r['force_N']==10),'candidate':sum(r['candidate']['strict'] for r in rows if r['force_N']==10),'candidate_only':sum(r['candidate_only'] for r in rows if r['force_N']==10)}};Path(out).write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n');print(json.dumps({'episodes':doc['episodes'],'passed':doc['passed'],'weak':doc['weak']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.plan,a.out)
