"""Plot the measured R24/R25 twins from the research simulator; no synthesized force or alarms."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source, target):
    doc = json.loads(source.read_bytes())
    if doc['schema'] != 'r25-fragmentation-observed-twins.v1' or doc['episodes']!=8:
        raise ValueError('expected measured paired and isolated research twins')
    data=doc['records'];baseline=data['r24_joint'];candidate=data[doc['candidate']]
    for tw in ('none','first','second','both'):
        if candidate[tw]['physics_sha256']!=baseline[tw]['physics_sha256']:
            raise ValueError('mismatched physical twin')
    fig,axes=plt.subplots(3,1,figsize=(9,5.8),sharex=True,gridspec_kw={'height_ratios':[1.5,1,1]})
    fig.subplots_adjust(top=.83,bottom=.13,left=.145,right=.97,hspace=.52)
    fig.suptitle('Two real inputs, two alerts',x=.145,y=.97,ha='left',weight='bold',fontsize=16)
    fig.text(.145,.9,'Measured research simulation · 5.64 kg · changed motor response · not the course demo',
             fontsize=9,color='#334155')
    t0=doc['force_xy_first_sample_index'];v=np.asarray(baseline['both']['force_xy_samples_1ms_N']);t=(np.arange(len(v))+t0)/1000
    axes[0].plot(t,v[:,0],color='#1f69a5',label='Applied X',lw=2)
    axes[0].plot(t,v[:,1],color='#ac5427',label='Applied Y',lw=2)
    axes[0].set_ylabel('Force (N)');axes[0].set_yticks([-8,0,8]);axes[0].legend(loc='upper left',ncol=2,fontsize=8,frameon=False)
    for ax,tw,title in zip(axes[1:],('second','both'),('Second input in isolation','Both inputs on the same movement'),strict=True):
        for lo,hi in baseline[tw]['actual_force_half_open_ms']:
            ax.axvspan(lo/1000,hi/1000,facecolor='#d9e4ea',alpha=.55,zorder=0)
        for y,color,rec in [(1,'#a95c34',baseline),(0,'#196a70',candidate)]:
            for lo,hi in rec[tw]['alarms_ms']:
                ax.broken_barh([(lo/1000,(hi-lo)/1000)],(y-.16,.32),facecolors=color)
        ax.set_ylim(-.45,1.45);ax.set_yticks([0,1],['R25','R24']);ax.set_title(title,loc='left',fontsize=10,weight='bold',pad=4)
    for ax in axes:
        ax.set_xlim(t[0],t[-1]);ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Simulation time (s)');fig.text(.145,.025,'Shading = measured nonzero applied force. Bars = actual alarm level (1 ms samples).',fontsize=8,color='#334155')
    fig.savefig(target,dpi=145);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('new output required')
    plot(a.input,a.out)
