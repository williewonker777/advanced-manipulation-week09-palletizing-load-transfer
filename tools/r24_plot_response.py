"""Plot acquired R23/R24 yaw data on same-motion twins, not an illustration or a course demo."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(data,out):
    if out.exists() or out.is_symlink():raise ValueError('fresh graph filename required')
    expected=['t_ms','grip','raw_yaw_Nm','used_yaw_Nm','original_bound_Nm','estimated_positive_support_N','reported','true_table_N','applied_N']
    if data['schema']!='week09-r24-near-landing-twins.v1' or data['episodes']!=8 or data['fields']!=expected:raise ValueError('expected measured matched pulses')
    rec=data['mechanics']['actuator_skew']; fig,axs=plt.subplots(2,1,figsize=(8.2,5.6),sharex=True,layout='constrained')
    colors={'raw':'#6d7781','corrected':'#006e68','alarm':'#ba401c'}
    for ax,tw,label in zip(axs,('none','impact'),('A   Same motion, no applied impact','B   Same motion, +14 N external impact'),strict=True):
        base,mod=rec['r23'][tw],rec['candidate'][tw]
        if base['physical']!=mod['physical']:raise ValueError('comparison uses different physical trajectories')
        x=np.array([r[0] for r in mod['samples']],int);y=mod['samples']
        if len(x)!=len(set(x)) or not np.all(np.diff(x)==1):raise ValueError('missing sample')
        cut=(x>=3815)&(x<=3898)
        def values(index):
            return np.array([np.nan if r[index] is None else float(r[index]) for r in y])[cut]
        ax.plot(x[cut]/1000,np.abs(values(2)),color=colors['raw'],lw=1.7,label='R23 observed yaw')
        ax.plot(x[cut]/1000,np.abs(values(3)),color=colors['corrected'],lw=1.8,label='R24 unexplained yaw')
        ax.plot(x[cut]/1000,values(4),'--',color='#394855',lw=1.25,label='Original alarm threshold')
        ax.set_title(label,loc='left',fontsize=12);ax.set_ylabel('|Yaw torque| (N·m)');ax.grid(alpha=.2)
        for name,arm,c in [('R23',base,'#8b6b54'),('R24',mod,colors['alarm'])]:
            for t in arm['rises_ms']:
                if x[cut][0]<=t<=x[cut][-1]:
                    ax.axvline(t/1000,color=c,ls=':',alpha=.85)
                    ax.annotate(f'{name} alarm',(t/1000,.96 if name=='R23' else .78),xycoords=('data','axes fraction'),
                                xytext=(3,0),textcoords='offset points',color=c,fontsize=8,va='top')
        if tw=='impact':
            t=int(x[np.flatnonzero([r[8]>0 for r in y])[0]])
            ax.axvspan(t/1000,x[cut][-1]/1000,color='#f4c168',alpha=.12)
            ax.annotate('First nonzero external force',(t/1000,.06),xycoords=('data','axes fraction'),
                        xytext=(3,0),textcoords='offset points',fontsize=8,color='#78570f')
    axs[0].legend(loc='upper left',bbox_to_anchor=(0.0,.85),framealpha=.93,fontsize=8);axs[1].set_xlabel('Simulation time (s)')
    fig.suptitle('Yaw during load transfer with changed actuator gains\nMeasured research simulation — not the course demo',fontsize=12.2,weight='bold')
    fig.savefig(out,dpi=160,metadata={'Software':'Week09 R24 measured research plot'});plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(json.loads(a.input.read_bytes()),a.out)
