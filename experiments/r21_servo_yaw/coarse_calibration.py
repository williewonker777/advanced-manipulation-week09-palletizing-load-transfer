"""Servo normal-only bins with the exact frozen torque-study fit procedure."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from experiments.r21_servo_yaw.detector import Detector
from experiments.r21_yaw_impulse.coarse_calibration import BINS,bin_index,fit
from my_servo import Controller
from sim.runner import run_episode
ROOT=Path(__file__).resolve().parent
class Probe(Detector):
 def reset(self):super().reset();self.cells=np.full((2,BINS),-1.)
 def update(self,t,*args):
  out=super().update(t,*args)
  if t>=.08 and self.yaw_value is not None:
   idx=(int(bool(args[-1])),bin_index(t));self.cells[idx]=max(self.cells[idx],abs(self.yaw_value))
  return out
def run(case):
 m,c=case;d=Probe();r,_=run_episode(Controller(),m,tuple(c),detector=d)
 if r['diverged']:raise ValueError('normal diverged')
 return d.cells
def main(out,jobs):
 plan=json.loads((ROOT/'plan.json').read_bytes());cases=[(m,c) for m in plan['normal_masses'] for c in plan['normal_coms']]
 with ProcessPoolExecutor(jobs) as pool:data=np.stack(list(pool.map(run,cases)))
 groups=np.array([plan['normal_coms'].index(c) for _,c in cases]);bounds,reserve=fit(data,groups)
 doc={'schema':'r21-servo-yaw-normal-bins.v1','normal_episodes':len(cases),'threshold_Nm':bounds.tolist(),'held_com_reserve_Nm':reserve.tolist(),'full_normal_max_Nm':data.max(axis=(0,2)).tolist()};Path(out).write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n');print(json.dumps({'episodes':len(cases),'reserve':reserve.tolist(),'at_1_42':bounds[1,14].tolist()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--jobs',type=int,choices=(1,2),default=1);a=p.parse_args();main(a.out,a.jobs)
