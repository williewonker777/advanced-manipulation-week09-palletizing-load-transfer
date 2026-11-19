"""Render measured time intervals only, not a simulation frame or a force waveform."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def render(out):
    r=ROOT/'results/r31'
    examples=json.loads((r/'representative.json').read_bytes())['examples']
    remaining=json.loads((r/'remaining-vertical.json').read_bytes())['rows']
    h=next(row for row in examples if row['id']==266)
    v=next(row for row in remaining if row['input']['case_id']==304)
    base1=round(h['input']['collision'][0]*1000)
    base2=round(v['input']['collision'][0]*1000)
    first,last=h['actual_force_ticks_inclusive']
    left=[('External force',[(first,last+1)],'#537797'),('R30 alerts',h['arms']['headroom']['alarm_intervals'],'#c66b32'),
          ('R31 alerts',h['arms']['identity']['alarm_intervals'],'#18776c')]
    first,last=v['arms']['identity']['actual_input_ticks']
    right=[('External force',[(first,last+1)],'#537797'),
           ('Table contact >1 N',v['arms']['identity']['table_contact_gt_1N'],'#756293'),
           ('R31 alerts',v['arms']['identity']['intervals'],'#18776c')]
    fig,axes=plt.subplots(2,1,figsize=(7.6,4.7),constrained_layout=True)
    fig.suptitle('Measured events in the research simulation (not the course demo)',fontsize=13,weight='bold')
    for ax,items,base,limit,title in zip(axes,(left,right),(base1,base2),(150,350),
              ('A. Carry: one diagonal input (10 N), no later alert in R31',
               'B. Carry: one downward input (100 N), later table contact still reported')):
        for index,(label,spans,color) in enumerate(items):
            for a,b in spans:
                if b>base-8 and a<base+limit:
                    ax.broken_barh([(a-base,b-a)],(index-.18,.36),facecolors=color,zorder=3)
        ax.set(xlim=(-8,limit),ylim=(2.5,-.5),yticks=[0,1,2],yticklabels=[i[0] for i in items])
        ax.set_title(title,fontsize=10,loc='left')
        ax.spines[['right','left','top']].set_visible(False)
        ax.grid(axis='x',color='#e1e6e8');ax.tick_params(axis='y',length=0)
    axes[-1].set_xlabel('ms from commanded external-force onset; bars show active intervals')
    fig.savefig(out,dpi=150,metadata={'Software':'r31-interval-plot'});plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh output required')
    render(a.out)
