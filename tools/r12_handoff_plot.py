"""Plot actual handoff and release decisions from a saved paired R12 report."""
import argparse
import json
from pathlib import Path

import numpy as np


def draw(record, out):
    import matplotlib.pyplot as plt
    if record.get('schema')!='r12-paired-outcomes.v1' or out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError('paired R12 records and a fresh PNG path required')
    rows=record['physics']
    impacts=record['impacts']
    expected=[(i,k) for i in ('ce0','ce1','ce2') for k in ('normal','offset')]
    if ([(r['case'],r['kind']) for r in rows]!=expected or [r['id'] for r in impacts]!=['ce0','ce1','ce2']
            or any(r['counter_outcome']!='detected' for r in impacts)):
        raise ValueError('a complete matched study with reported release outcomes is required')
    colors={'candidate':'#12796e','ref':'#718195','bad':'#a33f2d','ink':'#213344'}
    with plt.rc_context({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,
                         'font.family':'DejaVu Sans','axes.labelcolor':colors['ink'],
                         'text.color':colors['ink'],'savefig.facecolor':'white'}):
        fig,(left,right)=plt.subplots(1,2,figsize=(11.2,4.7),gridspec_kw={'width_ratios':[1.3,1.]})
        fig.subplots_adjust(top=.74,bottom=.26,wspace=.38,left=.075,right=.975)
        fig.suptitle('Can the table carry the box without hiding a collision?',x=.075,y=.975,
                     ha='left',weight='bold',fontsize=18)
        fig.text(.075,.88,'Torque simulation  /  6 normal or pickup-offset trials  /  3 separate release pulses',
                 color='#526575',fontsize=11)
        indices=np.array([0,.8,2.,2.8,4.,4.8])
        reference=[100*r['support']['R1']['mean_mg'] for r in rows]
        mean=[100*r['support']['counter']['mean_mg'] for r in rows]
        minimum=[100*r['support']['counter']['min_mg'] for r in rows]
        left.bar(indices-.18,reference,width=.32,color=colors['ref'],label='R1 mean')
        left.bar(indices+.16,mean,width=.32,color=colors['candidate'],label='Hybrid mean')
        left.scatter(indices+.16,minimum,marker='D',s=22,color='#162e3a',label='Hybrid minimum',zorder=4)
        left.axhline(50,color='#687781',ls=':',lw=1)
        left.set_ylim(0,108);left.set_xticks(indices,['A','A*','B','B*','C','C*'])
        left.set_ylabel('% of box weight');left.set_title('Support in last 20 ms',loc='left',fontweight='bold')
        left.legend(loc='upper right',frameon=False,fontsize=8,ncol=2,handlelength=1.3)
        left.grid(axis='y',alpha=.16);left.set_axisbelow(True)
        for idx,row in zip(indices,rows):
            if row['g4']['all_four_g4'] is False:
                left.text(idx+.14,mean[rows.index(row)]+3,'fail',ha='center',color=colors['bad'],fontsize=8)
        ix=np.array([0.,1.,2.])
        for i,row in enumerate(impacts):
            x=row['R1_delay_ms'];y=row['counter_delay_ms']
            if x is not None:
                right.plot([x,y],[i,i],color='#bec9d0',lw=3,zorder=1)
                right.scatter(x,i,marker='o',facecolor='white',edgecolor=colors['ref'],s=60,zorder=3,
                              label='R1 + original detector' if i==0 else None)
            else:
                right.text(30,i,'R1 missed',va='center',fontsize=9,color=colors['bad'])
            right.scatter(y,i,marker='D',c=colors['candidate'],s=45,zorder=4,
                          label='Hybrid + corrected detector' if i==0 else None)
            right.text(y,i+.19,f'{y:.0f} ms',ha='center',va='center',weight='bold',fontsize=8)
        right.set_xlim(28,73);right.set_ylim(2.55,-.7);right.set_yticks(ix,['horizontal','downward','oblique'])
        right.set_xlabel('Time to distinguishable alarm (ms)')
        right.set_title('Separate release pulses',loc='left',fontweight='bold')
        right.grid(axis='x',alpha=.16);right.set_axisbelow(True)
        right.legend(loc='upper left',fontsize=7.8,frameon=False,ncol=1)
        fig.text(.075,.09,'* Pickup offset. Both offset failures remain visible; B* also exceeded the peak-force limit (276%).',
                 color='#455968',fontsize=9)
        fig.text(.075,.045,'The raw hybrid alarm coincided with normal release in all 3 pulse comparisons. No tool-contact validation.',
                 color='#455968',fontsize=9)
        try:
            fig.savefig(out,dpi=150,format='png',metadata={'Software':'R12 paired handoff plot'})
        finally:
            plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.out.suffix.lower()!='.png':parser.error('a new .png is required')
    draw(json.loads(a.input.read_text()),a.out)
