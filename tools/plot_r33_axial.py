"""Plot three physically recorded signed-force trajectories, never the original course demo."""
import argparse
import json
from pathlib import Path

def plot(source, target):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    d=json.loads(Path(source).read_bytes());labels={194:'Pickup · −5 N',253:'Carry · +5 N',313:'After release · +5 N'}
    if d.get('schema')!='r33-signed-force-traces.v1':raise ValueError('expected signed physical trace')
    fig,axes=plt.subplots(3,1,figsize=(7.2,5.4),sharex=True)
    for ax,r in zip(axes,[next(x for x in d['rows'] if x['id']==i) for i in labels],strict=True):
        x=np.array(r['t_ms'])-r['onset'];y=np.array(r['actual_world_z_N'])
        ax.axhline(0,color='#9ca9b2',lw=.7)
        ax.fill_between(x,y,0,color='#bed6e5',alpha=.7);ax.plot(x,y,color='#235d80',lw=1.7)
        ax.set_xlim(-15,155);amp=max(1,np.max(np.abs(y)));ax.set_ylim(-1.25*amp,1.35*amp)
        for lo,hi in r['r33']:
            if -15<=lo-r['onset']<=155:
                ax.axvspan(lo-r['onset'],hi-r['onset'],color='#cf5719',alpha=.32)
                ax.axvline(lo-r['onset'],color='#b54713',lw=1.25,ls='--')
        rises=[a-r['onset'] for a,b in r['r33'] if 0<=a-r['onset']<=155]
        note=f'R33 +{rises[0]} ms' if rises else 'No R33 notice'
        ax.text(.985,.77,note,transform=ax.transAxes,ha='right',fontsize=10,color='#963e12' if rises else '#5d5350')
        ax.text(.018,.79 if np.min(y)<0 else .16,f"#{r['id']}  {labels[r['id']]}",transform=ax.transAxes,fontsize=10,fontweight='bold')
        ax.spines[['top','right']].set_visible(False);ax.set_ylabel('N');ax.grid(axis='x',alpha=.18)
    axes[-1].set_xlabel('Milliseconds from commanded onset')
    fig.suptitle('Research replay: weak vertical impacts',fontsize=15,fontweight='bold',x=.12,ha='left')
    fig.text(.12,.91,'All three were missed by R31. Curves show logged force, not a detector estimate.',fontsize=9,color='#4b555e')
    fig.legend(handles=[Line2D([],[],color='#235d80',label='Actual applied Z force'),Line2D([],[],color='#b54713',ls='--',label='First R33 notice')],loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(top=.855,hspace=.37,bottom=.16,left=.105,right=.97)
    fig.savefig(target,dpi=160,metadata={'Software':'R33 physical replay'});plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():p.error('new output required')
    plot(a.source,a.out)
