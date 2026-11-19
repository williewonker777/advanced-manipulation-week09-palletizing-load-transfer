"""Plot a measured placement result; labels distinguish force from geometry."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(result, output):
    by_mode = result['modes']
    if set(by_mode) != {'servo','torque'}:
        raise ValueError('two measured modes required')
    entries = [(mode, r) for mode in ('servo','torque') for r in by_mode[mode]['candidate']['rows']]
    labels = [m.capitalize()+'  |  '+r['id'].replace('_',' ') for m,r in entries]
    y = np.arange(len(entries))
    bg, ink, subtle = '#fafaf8', '#162f3b', '#516874'
    fig, (ax, vel) = plt.subplots(1, 2, figsize=(10.6,4.7), sharey=True,
                                  gridspec_kw={'width_ratios':[.84,1.2]})
    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(left=.19,right=.97,bottom=.23,top=.80,wspace=.18)
    for a in (ax,vel):
        a.set_facecolor(bg); a.spines[['right','top','left']].set_visible(False)
        a.spines['bottom'].set_color('#acb9b9'); a.tick_params(length=0,colors=subtle)
        a.grid(axis='x',color='#dbe1df',lw=.7); a.set_axisbelow(True)
        a.axhline(2.5,color='#b6c2c3',lw=.8)
    valid=np.array([r.get('status','ok') == 'ok' for _,r in entries],dtype=bool)
    means=[r['support']['mean_mg'] for (_,r),ok in zip(entries,valid) if ok]
    mins=[r['support']['min_mg'] for (_,r),ok in zip(entries,valid) if ok]
    ax.barh(y[valid],means,height=.42,color='#089981',label='Mean',zorder=2)
    ax.scatter(mins,y[valid],color='#133849',s=20,zorder=3,label='Minimum')
    for i in np.flatnonzero(~valid):
        ax.text(.015,i,entries[i][1]['status'].replace('_',' '),va='center',color=subtle,fontsize=8)
    ax.axvline(.50,ls='--',color='#a84937',lw=1.15)
    ax.axvline(.25,ls=':',color='#a84937',lw=1.0)
    ax.set_xlim(-.025,max(.56,max(means,default=0.)+.05)); ax.set_xlabel('Fraction of box weight',color=ink)
    ax.set_yticks(y,labels=labels);ax.invert_yaxis()
    ax.set_title('A   Table support',loc='left',color=ink,fontweight='bold')
    ax.legend(loc='upper left',bbox_to_anchor=(0.,-.20),frameon=False,ncol=2,fontsize=9)
    for field,label,color,symbol,dy in [('taught_rigid_corner_speed_mm_s','Taught','#88989c','x',-.15),
       ('actual_link_rigid_corner_speed_mm_s','Arm (rigid FK)','#ce7750','s',0.),
       ('first_to_last_lowest_corner_speed_mm_s','Free box','#176381','o',.15)]:
        speeds=[r['last20_geometry'][field] for (_,r),ok in zip(entries,valid) if ok]
        vel.scatter(speeds,y[valid]+dy,color=color,marker=symbol,s=29,zorder=3,label=label)
    vel.axvline(0,color='#4a626a',lw=.9)
    vel.set_xlabel('Vertical speed (mm/s); positive = upward',color=ink)
    vel.set_title('B   Lowest corner motion',loc='left',color=ink,fontweight='bold')
    vel.legend(loc='upper left',bbox_to_anchor=(0.,-.20),frameon=False,ncol=3,fontsize=8)
    fig.suptitle('A box can touch the table without transferring its weight',
                 x=.05,ha='left',y=.965,color=ink,fontsize=15,fontweight='bold')
    fig.text(.05,.883,'Measured over the final 20 ms before release. Dashed: mean target (0.50); dotted: minimum target (0.25).',
             color=subtle,fontsize=9.3)
    fig.savefig(output,dpi=150,facecolor=bg,metadata={'Software':'Week09 placement assessment'})
    plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input',required=True,type=Path);cli.add_argument('--out',required=True,type=Path)
    args=cli.parse_args()
    if args.out.exists() or args.out.suffix.lower()!='.png':
        cli.error('choose a new .png file')
    draw(json.loads(args.input.read_text()),args.out)
