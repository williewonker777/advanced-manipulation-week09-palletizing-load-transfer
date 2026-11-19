"""Save real sensor-only streams from unchanged C3. External clocks and forces are used only for scoring."""
import argparse,os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pathlib import Path
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r26_holdout import evaluator as lab
from experiments.r29_force_tube.sources import inputs
from experiments.r29_force_tube.stream import Streams


class Trace(C3):
    def reset(self):
        super().reset()
        self.s=Streams(self.box_mass,self.box_com)
        self.observations=np.full((6000,16),np.nan)
        self.phase=np.full(6000,-1,dtype=np.int8);self.tick=0

    def update(self,t,th,thd,motor,jts,grip):
        r,hit=super().update(t,th,thd,motor,jts,grip)
        phase,data=self.s.read(self,t,th,thd,jts,grip)
        self.phase[self.tick]=phase
        if data is not None:self.observations[self.tick]=data
        self.tick+=1
        return r,hit


def work(spec):
    out,split,c,m,twin,imp=spec
    name='__'.join((split,c['id'],m['kind'],twin))+'.npz';path=out/name
    if os.path.lexists(path):raise ValueError('fresh capture required: '+name)
    det=Trace();_,log=lab.run_episode(c,m,lambda:det,imp)
    if det.tick!=6000 or not lab.exact_force(log,imp)['passed']:raise ValueError('episode integrity')
    with path.open('xb') as f:np.savez_compressed(f,data=det.observations,phase=det.phase,native=log['detect'],physical=lab.certified.trace_signature({'trace':log}),
                        force1=log['f_ext_xyz_native'],force2=log['f_ext_xyz_second'],t=log['t'])
    return name


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--split',required=True)
    p.add_argument('--mechanics');p.add_argument('--jobs',type=int,choices=(1,2),default=2)
    a=p.parse_args();dest=a.out;dest.mkdir(parents=True,exist_ok=True)
    jobs=[(dest,*s)for s in inputs()if s[0]in a.split.split(',')and(not a.mechanics or s[2]['kind']in a.mechanics.split(','))]
    if not jobs:raise ValueError('empty case list')
    with ProcessPoolExecutor(a.jobs)as pool:
        for n in pool.map(work,jobs,chunksize=1):print(n,flush=True)
