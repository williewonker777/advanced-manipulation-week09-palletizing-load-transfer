"""Draw only measured first/last tool force and alarm timestamps from the R30 results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CASES=(('q30_s1_low','weak offset'),('q30_s2_low','already detected'),
       ('q30_s3_low','weak offset'),('q30_later_oblique','known weak pickup'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh output')
    result=json.loads(a.qualification.read_bytes())
    if result['schema']!='r30-g4-unload-notice-qualification.v1' or result['episodes']!=22:
        raise ValueError('complete measured G4 qualification required')
    rows={r['id']:r for r in result['rows']};fig,ax=plt.subplots(figsize=(8.4,3.5),layout='constrained')
    for y,(key,label) in enumerate(CASES):
        row=rows[key];c=row['contact'];base=c['raw_nonzero_first_tick']
        ax.plot([0,c['raw_nonzero_last_tick']-base],[y,y],lw=8,solid_capstyle='butt',color='#b6d6e0')
        ax.plot([c['registered_first_tick']-base,c['registered_last_tick']-base],[y,y],lw=8,
                solid_capstyle='butt',color='#247488')
        if row['notice']:
            ticks=[n['tick'] for n in row['notice']];marker='D';color='#c05036'
        else:
            ticks=[t for t in row['candidate_rises'] if base<=t<=c['raw_nonzero_last_tick']];marker='o';color='#332d44'
        if len(ticks)!=1:raise ValueError('expected one matched notice')
        delay=ticks[0]-base;ax.scatter([delay],[y],marker=marker,c=color,s=41,zorder=4)
        ax.annotate('+'+str(delay),(delay,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set_yticks(range(len(CASES)),[f"{rows[k]['case']['mass']:.1f} kg | {s}" for k,s in CASES]);ax.set_ylim(len(CASES)-.45,-.68)
    ax.axvline(0,ls=':',c='#576371',lw=.7);ax.set_xlim(-1,63);ax.set_xlabel('ms since first nonzero tool contact')
    ax.set_title('Pickup notices relative to measured contact bounds (not a force waveform)',loc='left',fontsize=11)
    handles=[Line2D([],[],c='#247488',lw=6,label='first–last >1 N'),
             Line2D([],[],c='#b6d6e0',lw=6,label='first–last >0 N'),
             Line2D([],[],c='#c05036',marker='D',ls='',label='new notice'),
             Line2D([],[],c='#332d44',marker='o',ls='',label='preserved notice')]
    ax.legend(handles=handles,ncol=2,frameon=False,loc='upper left',bbox_to_anchor=(0,-.26),fontsize=8)
    for side in ('right','top','left'):ax.spines[side].set_visible(False)
    fig.savefig(a.out,dpi=160);plt.close(fig)


if __name__=='__main__':main()
