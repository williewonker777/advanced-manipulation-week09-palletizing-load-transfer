"""Draw derived timelines of actual checked impulses and recorded alarm intervals only."""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def draw(src, out):
    doc=json.loads(src.read_bytes())
    rows=doc['rows']
    if len(rows)!=2 or not all(x['signed_and_scalar_exact_all_arms'] for x in rows):
        raise ValueError('need the two force-authenticated comparisons')
    # English labels keep the figure readable on machines with no Korean typeface.
    mpl.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'font.size':11,'axes.edgecolor':'#526174','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(10,6.45),layout='constrained')
    colors={'single_pass':'#be6725','event':'#166b77','impulse':'#5b4e91'}
    for ax,row,limits,label in zip(axes,rows,[(4170,4310),(3930,4220)],
            ['A   One physical pulse, three R18 rises','B   Two separated pulses: impulse rule rearms before pulse 2']):
        ax.set_title(label,loc='left',fontweight='bold',fontsize=14)
        imp=row['measured_exact_stock_force_stimuli']
        t=np.arange(limits[0],limits[1]+1,dtype=float);stim=np.zeros_like(t)
        for onset,xyz in imp:
            age=t-1000*onset
            weight=np.maximum(0.,np.minimum(np.minimum(age/doc['stock_ramp_ms'],1.),
                        (2*doc['stock_ramp_ms']+doc['stock_plateau_ms']-age)/doc['stock_ramp_ms']))
            stim+=xyz[2]*weight
        top=3.1+stim/max(max(abs(stim)),1)*.40
        ax.fill_between(t,3.1,top,color='#506c9c',alpha=.31,step='mid')
        ax.plot(t,top,color='#435e8a',linewidth=1.4)
        for y,k in [(2,'single_pass'),(1,'event'),(0,'impulse')]:
            seq=row['on_intervals_ms'][k]
            ax.broken_barh([(a,b-a) for a,b in seq],(y-.14,.28),facecolors=colors[k])
            for a,b in seq:
                if limits[0]<=a<=limits[1]:ax.plot(a,y,marker='|',ms=14,mew=1.4,color=colors[k])
        ax.axvline(imp[1][0]*1000,linestyle=':',color='#2c3f52',lw=1.2)
        ax.set_yticks((0,1,2,3.1),('8-ms impulse','R19 V2','R18','Applied Fz'))
        ax.set_ylim(-.55,3.84);ax.set_xlim(limits);ax.grid(axis='x',color='#bac4cc',linewidth=.55)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].annotate('3 rises → 1',xy=(4233,1),xytext=(4257,1.48),ha='center',color='#185d68',
                     arrowprops={'arrowstyle':'->','color':'#185d68','lw':.9})
    axes[1].annotate('pulse 2 starts',xy=(4066,3.14),xytext=(4026,3.68),ha='center',
                     fontsize=9,color='#334a69',arrowprops={'arrowstyle':'->','color':'#334a69','lw':.9})
    axes[1].set_xlabel('Time from simulation start (ms)  ·  bars = alarm high')
    fig.savefig(out,dpi=155,facecolor='white',metadata={'Software':'Matplotlib'});plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('source',type=Path);cli.add_argument('out',type=Path)
    args=cli.parse_args()
    if args.out.exists():cli.error('fresh output path required')
    draw(args.source,args.out)
