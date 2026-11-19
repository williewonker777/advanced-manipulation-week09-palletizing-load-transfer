"""One-process evaluation inside an authenticated JTS-seed-only fixture."""
import argparse,json
import numpy as np
from experiments.r20_horizontal.candidate_detector import Detector as R20
from experiments.r21_pick_observer.candidate import Detector as R21
from my_torque import Controller
from sim.runner import run_episode
KEYS=('t','q','q_d','tau','e_mm','f_table','f_tool','f_ext','qpos')
def run(cls,mass,com,col):return run_episode(Controller(),box_mass=mass,box_com=com,detector=cls(),collision=col)
def main(seed,masses,com,collision,out):
 rows=[]
 for mass in masses:
  arms={n:(run(c,mass,com,None),run(c,mass,com,collision)) for n,c in [('r20',R20),('r21',R21)]};a,b=arms['r20'],arms['r21']
  equal=[{k:bool(np.array_equal(a[i][1][k],b[i][1][k])) for k in KEYS} for i in (0,1)]
  row={'seed':seed,'mass':mass,'r20_delay':a[1][0]['detect_delay_ms'],'r21_delay':b[1][0]['detect_delay_ms'],'r20_none_false':a[0][0]['false_alarm_times'],'r21_none_false':b[0][0]['false_alarm_times'],'r20_force_false':a[1][0]['false_alarm_times'],'r21_force_false':b[1][0]['false_alarm_times'],'physical_equal':equal}
  row['passed']=all(v for e in equal for v in e.values()) and row['r21_none_false']==row['r20_none_false'] and row['r21_force_false']==row['r20_force_false'] and (row['r20_delay'] is None or row['r21_delay'] is not None);rows.append(row)
 open(out,'w').write(json.dumps({'seed':seed,'episodes':4*len(rows),'rows':rows,'passed':all(r['passed'] for r in rows)},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--masses',required=True);p.add_argument('--com',required=True);p.add_argument('--time',type=float,required=True);p.add_argument('--force',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.seed,[float(x) for x in a.masses.split(',')],[float(x) for x in a.com.split(',')],(a.time,[float(x) for x in a.force.split(',')]),a.out)
