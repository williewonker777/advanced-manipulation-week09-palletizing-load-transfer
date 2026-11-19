import argparse,json,hashlib
from pathlib import Path
import numpy as np
from experiments.r20_horizontal.candidate_detector import Detector as R20
from experiments.r21_pick_observer.candidate import Detector as R21
from my_torque import Controller
from sim.runner import run_episode
KEYS=('t','q','q_d','tau','e_mm','f_table','f_tool','f_ext','qpos')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cls,c):return run_episode(Controller(),box_mass=c['mass'],box_com=c['com'],detector=cls(),pick_offset=c['offset'])
def outcome(log):
 tool=np.asarray(log['f_tool'])>1;det=np.asarray(log['detect'],bool);first=None if not tool.any() else int(np.argmax(tool));rise=np.flatnonzero(det&~np.r_[False,det[:-1]])
 return {'tool':bool(tool.any()),'tool_first_s':None if first is None else float(log['t'][first]),'prealarm':False if first is None else bool(det[:first].any()),'detect_after_tool':False if first is None else bool(det[first:].any()),'rises_s':np.asarray(log['t'])[rise].tolist()}
def main(plan,out):
 raw=Path(plan).read_bytes();p=json.loads(raw)
 if sha(Path(__file__).with_name('candidate.py'))!=p['candidate_sha256']:raise ValueError('pin')
 rows=[]
 for c in p['cases']:
  a,al=run(R20,c);b,bl=run(R21,c);oa,ob=outcome(al),outcome(bl);eq={k:bool(np.array_equal(al[k],bl[k])) for k in KEYS};passed=all(eq.values()) and not ob['prealarm'] and (not oa['detect_after_tool'] or ob['detect_after_tool'])
  rows.append({'id':c['id'],'r20':oa,'r21':ob,'r20_false':a['false_alarm_times'],'r21_false':b['false_alarm_times'],'physical_equal':eq,'passed':passed})
 doc={'schema':'r21-pick-tool-control-result.v1','plan_sha256':hashlib.sha256(raw).hexdigest(),'episodes':2*len(rows),'rows':rows,'passed':all(r['passed'] for r in rows)};Path(out).write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps({'episodes':doc['episodes'],'passed':doc['passed'],'rows':[(r['id'],r['r21']['tool'],r['r21']['prealarm'],r['r21']['detect_after_tool']) for r in rows]}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.plan,a.out)
