"""Draw the disclosed research counterexample from its measured force/normal twins."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(data,out):
    if out.exists() or out.is_symlink():raise ValueError('use a new PNG filename')
    if (data.get('schema')!='r23-post-detachment-response.v1' or data.get('stock_demo') is not False
            or data.get('episodes')!=6 or {r['mechanics'] for r in data['rows']}!={'stock','contact_suction','actuator_gain'}):
        raise ValueError('six original, disclosed force and normal trials required')
    fig,axs=plt.subplots(3,1,figsize=(8,6),sharex=True,sharey=True,layout='constrained')
    names={'stock':'Original mechanics','contact_suction':'Contact / suction variation','actuator_gain':'Actuator gain variation'}
    for ax,r in zip(axs,data['rows'],strict=True):
        n,f=r['twins']['none'],r['twins']['force'];origin=r['actual_first_nonzero_index']
        if n['index']!=f['index'] or not f['alarm_indices']:raise ValueError('unmatched sample grids or missing alarm')
        x=np.asarray(f['index'])-origin
        ax.plot(x,f['applied_N'],color='#9e7745',lw=1.3,ls=':',label='Applied force (scoring only)')
        ax.plot(x,n['planar_mean_N'],color='#66757e',lw=1.4,label='No-impact estimate')
        ax.plot(x,f['planar_mean_N'],color='#116a7b',lw=2,label='Impact estimate')
        ax.axhline(4.60253623135071,color='#52427d',lw=1.2,ls='--',label='Frozen empty-arm limit')
        for hit in f['alarm_indices']:
            if hit in f['index']:
                ax.scatter(hit-origin,f['planar_mean_N'][f['index'].index(hit)],s=36,color='#b34338',zorder=4,label='Reported rise')
        ax.axvline(0,color='#9e7745',lw=.65,alpha=.7)
        ax.set_title(names[r['mechanics']],loc='left',fontsize=10)
        ax.set_xlim(-28,205);ax.set_ylim(-.25,12);ax.set_ylabel('Force (N)');ax.grid(axis='y',alpha=.19)
        ax.spines[['right','top']].set_visible(False)
    axs[-1].set_xlabel('Time from first nonzero applied force (ms)')
    fig.suptitle('After detachment: an exposed 10.47 N diagonal impact\nResearch simulation, not the course demonstration',fontsize=13)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,fontsize=8,frameon=False)
    with out.open('xb') as stream:fig.savefig(stream,format='png',dpi=145,metadata={'Software':'R23 measured force response'})
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    obj=json.loads(a.input.read_bytes());plot(obj.get('data',obj),a.out)
