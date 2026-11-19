"""Plot measured input and detector samples from the disclosed same-control twins."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r26_delivery_pair.candidate import Detector
from experiments.r26_delivery_pair.probe import ORIGINAL,run
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r24_transfer_replay.evaluator import physical_signature


def capture(which, p):
    spec=next(x for x in p['late_impulses']if x['id']==which)
    inp=(spec['time_s'],spec['force_n'])
    old=run(p['impulse_base'],Coupled,inp)[-1];new=run(p['impulse_base'],Detector,inp)[-1]
    if physical_signature(old)!=physical_signature(new):raise ValueError('physical logs differ')
    component=int(np.argmax(np.abs(inp[1])))
    g=(old['t']>=inp[0]-.015)&(old['t']<inp[0]+.22)
    return np.column_stack(((old['t'][g]-inp[0])*1000.,old['signed_force'][g,component],
                            old['detect'][g],new['detect'][g]))


def draw(data, target):
    import matplotlib.pyplot as plt
    fig, axs=plt.subplots(3,2,figsize=(8.9,4.2),sharex=True,
                         gridspec_kw={'height_ratios':[1.7,1,1]})
    for col,(key,title) in enumerate((('late_x_pos','+X: previously missed'),('late_z_neg','-Z: previously late'))):
        a=data[key];t=a[:,0]
        for j,(label,color) in enumerate((('Applied force (N)','#405364'),('Existing alarm','#a64532'),('C1 alarm','#087461'))):
            ax=axs[j,col];ax.axvline(0,color='#b7b9bb',lw=.7);ax.grid(axis='x',alpha=.16)
            if j==0:
                ax.plot(t,a[:,1],color=color,lw=2);ax.set_title(title,fontsize=11)
            else:
                ax.step(t,a[:,j+1],where='post',color=color,lw=1.5);ax.fill_between(t,0,a[:,j+1],step='post',color=color,alpha=.26)
                ax.set_yticks([0,1]);ax.set_ylim(-.1,1.7)
                edges=np.flatnonzero((a[:,j+1]>.5)&~np.r_[False,a[:-1,j+1]>.5])
                label_edge='not detected' if not len(edges) else f'first +{t[edges[0]]:.0f} ms'
                ax.text(.02,.79,label_edge,transform=ax.transAxes,fontsize=8,color=color)
            if col==0:ax.set_ylabel(label,fontsize=9)
            ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
        axs[-1,col].set_xlabel('Time from scheduled input (ms)',fontsize=9)
    fig.suptitle('Measured landing simulation | same R24 torque controller, different detector',fontsize=12)
    fig.tight_layout()
    fig.savefig(target,dpi=150,metadata={'Software':'R26 authored measurement chart'});plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--png',type=Path,required=True);a=p.parse_args()
    if a.trace.exists() or a.png.exists():p.error('fresh trace and diagram required')
    plan=json.loads((ORIGINAL/'plan.json').read_bytes())
    values={k:capture(k,plan)for k in ('late_x_pos','late_z_neg')}
    np.savez_compressed(a.trace,**values);draw(values,a.png)
