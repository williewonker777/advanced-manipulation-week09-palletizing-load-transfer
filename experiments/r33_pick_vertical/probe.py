"""D4 diagnostic, not a detector. Record public signals and labels in separate arrays."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import numpy as np
from experiments.r15_delayed_wrench.observer import KERNEL, N, NormalForce
from experiments.r24_loaded.d4 import load_cases
from experiments.r21_full_protection import runner_v2 as raw
from experiments.r31_held_identity.candidate import Detector as R31
from sim import config as C
import mujoco

TARGET=(193,194,203,204,213,214)
class Probe(R31):
    def reset(self):
        super().reset()
        # Stay in the empty-arm coordinate frame across closure. A change of
        # commanded grip is not a step in the sensor or in the body's momentum.
        self.empty_sensor=NormalForce(self.nom,self.nom,dt=C.DT,viscous_spring=C.D,jts_hz=C.JTS_CUTOFF)
        self.rows=[];self.gripped=False;self.jp=np.zeros((3,6));self.jr=self.jp.copy()
        self.tip=self.nom.m.site('tcp').id
        Probe.instance=self
    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,alarm=super().update(t,th,thd,tau_m,tau_s,grip)
        if self.gripped and not grip:return residual,alarm
        self.gripped |= bool(grip)
        # This range is only for offline I/O, never an input to detection.
        if t>2.:return residual,alarm
        obs=self.empty_sensor; obs.update(t,th,tau_s,False,project=False)
        packet=self._shared
        empty=np.full(6,np.nan); xyz=np.full(3,np.nan)
        if len(obs.midpos)==N and obs.bias is not None:
            m=self.nom; m.d.qpos[:]=obs.midpos[N//2]
            mujoco.mj_kinematics(m.m,m.d);mujoco.mj_comPos(m.m,m.d)
            mujoco.mj_jacSite(m.m,m.d,self.jp,self.jr,self.tip)
            jac=np.vstack((self.jp,self.jr));xyz=m.d.site_xpos[self.tip].copy()
            if np.linalg.cond(jac)<1e4:
                empty=np.linalg.solve(jac.T,KERNEL@np.asarray(obs.residual)-obs.bias)
        self.rows.append(np.r_[t,grip,alarm,xyz,empty,packet.last_wrench,packet.last_residual_wrench,
            packet.last_fast_innovation,packet.load_fraction,packet.last_table_allowed,th,thd,tau_s])
        return residual,alarm
def one(job):
    idx,out=job;c=load_cases()[idx];_,log,signed=raw.execute(Probe,c)
    data=np.asarray(Probe.instance.rows);path=out/f'{idx:03d}.npz'
    with path.open('xb') as f:np.savez_compressed(f,public=data,signed=signed[:len(data)],table=log['f_table'][:len(data)],tool=log['f_tool'][:len(data)])
    return idx
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--group',choices=('primary','normal','extra'),required=True);a=p.parse_args();cs=load_cases()
    matched={raw.reference_key(cs[i]) for i in TARGET}
    primary=[c['case_id'] for c in cs if c['kind']=='reference' and raw.reference_key(c) in matched]+list(TARGET)
    if a.group=='primary': ids=primary
    elif a.group=='normal':ids=[i for i in (*range(0,189,5),*range(369,385)) if i not in primary]
    else:ids=[i for i in range(189,249) if i not in primary]
    with ProcessPoolExecutor(a.jobs) as pool:
        for i in pool.map(one,[(i,a.out) for i in ids]):print(i,flush=True)
if __name__=='__main__':main()
