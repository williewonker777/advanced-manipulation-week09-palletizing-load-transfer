"""Plot actual alarm levels separately from the inferred ongoing signed yaw event."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def collect():
    import mujoco
    if mujoco.__version__!='3.17.0':raise ValueError('course runtime required')
    from experiments.r25_holdout import evaluator as lab
    from experiments.r25_composed.detector import Detector as R25
    from experiments.r26_yaw_acquisition.candidate import Detector as Acquisition

    class Telemetry(Acquisition):
        def reset(self):
            super().reset();self.trace=[]
        def update(self,*args):
            r,hit=super().update(*args)
            self.trace.append([self.yaw_acquisition_episode_active,self.yaw_reacquisition_consumed])
            return r,hit
    p=json.loads(lab.PLAN.read_bytes())
    c=next(c for c in p['single'] if c['id']=='s25_long_transfer_signed');v=p['variants']['stock'];inp=(c['impact'],None)
    det=Telemetry()
    _,old=lab.run_episode(c,v,R25,inp);_,new=lab.run_episode(c,v,lambda:det,inp)
    assert lab.certified.physical_hash(old)==lab.certified.physical_hash(new)
    assert lab.exact_force(old,inp)['passed'] and lab.exact_force(new,inp)['passed']
    k,bd=lab.actual_start(new['t'],inp[0]);sl=slice(k-25,k+255)
    A=np.column_stack(((np.arange(len(new['t']))-k),new['f_ext_xyz_committed'][:,0],old['detect'],new['detect'],np.asarray(det.trace)))
    return A[sl],bd


def draw(a,p):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(7.5,4.6),sharex=True,gridspec_kw={'height_ratios':[1.2,1,1,1]})
    fig.suptitle('One continuous signed force | disclosed torque simulation')
    for i,(tag,col) in enumerate((('Applied X (N)','#425166'),('R25 alarm','#a64532'),('New onset alarm','#087461'),('Estimated event*','#52689e'))):
        ax=axs[i];y=a[:,i+1];ax.step(a[:,0],y,where='post',color=col,lw=1.5)
        ax.fill_between(a[:,0],0,y,step='post',color=col,alpha=.17);ax.set_ylabel(tag,fontsize=9,rotation=0,ha='right',va='center')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18)
        if i:ax.set_ylim(-.08,1.25);ax.set_yticks([0,1])
    axs[-1].set_xlabel('ms after first committed nonzero force sample')
    fig.text(.975,.008,'* Observer state, not a Boolean alarm or ground truth.',ha='right',fontsize=8,color='#455469')
    fig.tight_layout(rect=(0,0.028,1,1))
    fig.savefig(p,dpi=150,metadata={'Software':'R26 authored measurement chart'});plt.close(fig)


if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--png',type=Path,required=True);q.add_argument('--trace',type=Path,required=True);args=q.parse_args()
    if any(p.exists() or p.is_symlink()for p in (args.png,args.trace)):q.error('fresh outputs required')
    a,b=collect();np.savez_compressed(args.trace,values=a)
    assert (np.flatnonzero(np.diff(a[:,2],prepend=0)==1)-25).tolist()==[60,84,154] and (np.flatnonzero(np.diff(a[:,3],prepend=0)==1)-25).tolist()==[60]
    draw(a,args.png);print(b,hashlib.sha256(args.png.read_bytes()).hexdigest())
