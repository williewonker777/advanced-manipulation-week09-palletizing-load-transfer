"""Make a figure from actual extracted replay results; points are notice onsets, not a force plot."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=(('c3','C3 reference'),('composed','R29 combined'),('rising','R29 early'),('geometry','R30 direction'),('headroom','R30 yaw reserve'))


def load(folder,scope):
    d=json.loads((folder/'result.json').read_bytes())
    if d['schema']!='r30-disclosed-event-replay.v1' or d['scope']!=scope or d['course_original'] or d['default_changed']:
        raise ValueError('wrong research replay')
    return d['rows']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--signed',type=Path,required=True)
    p.add_argument('--paired',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():
        p.error('fresh output required')
    signed=load(a.signed,'signed-stock');paired=load(a.paired,'paired-directions')
    def single(name):
        row=next(r for r in signed[name] if r['case']=='s26_long_carry_signed')
        return row['forced']['perpendicular_xy']['event']['delays_ms']
    opposite=next(r for r in paired if r['id']=='cross_stress_1')
    points=[(single,'A   One sideways carry force'),
            (lambda n:opposite['records'][n]['events']['isolated_second']['delays_ms'],'B   Opposite pulse applied alone')]
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharey=True,layout='constrained')
    for axis,(records,title) in zip(axes,points):
        for y,(name,label) in enumerate(ARMS):
            values=records(name)
            if not values:
                axis.text(5,y,'no notice',va='center',fontsize=8,color='#59636d')
                continue
            for j,value in enumerate(values):
                color='#c05036' if j else '#176278'
                axis.plot(value,y,'o',ms=7,color=color,zorder=3)
                axis.annotate(str(value),(value,y),xytext=(0,7),textcoords='offset points',
                              ha='center',color=color,fontsize=8)
        axis.set_title(title,loc='left',fontsize=10,fontweight='bold')
        axis.axvline(0,color='#7b828d',ls='--',lw=1)
        axis.set_xlim(-1,55);axis.set_xticks(range(0,51,10));axis.grid(axis='y',color='#e4e6e8',lw=.8)
        axis.set_xlabel('ms after first nonzero force sample',fontsize=9)
        for side in ('top','right'):
            axis.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(ARMS)),[v for _,v in ARMS]);axes[0].set_ylim(len(ARMS)-.5,-.65)
    fig.suptitle('One physical pulse per panel | first notice in blue, additional notice in red',fontsize=11)
    fig.savefig(a.out,dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
