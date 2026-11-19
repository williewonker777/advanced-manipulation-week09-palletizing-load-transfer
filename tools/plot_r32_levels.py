"""Plot sample-exact alarm and observer spans; no course artwork or simulated video."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def render(data, out):
    if data.get('schema') != 'r32-client-compatibility.v1' or not data.get('actual_notice_equal_at_every_tick'):
        raise ValueError('authenticated client comparison required')
    figure, axes = plt.subplots(2,1,figsize=(8.2,5.8),constrained_layout=True)
    figure.suptitle('Is the load still measured, or is the detector still responding?',
                    weight='bold',fontsize=14)
    rows={row['id']:row for row in data['rows']}
    for ax, index, name, title in zip(axes,range(2),('307','p26_close_place_cross/both'),(
            'A. Research simulation: one oblique input during carry (100 N)',
            'B. Two separate inputs during placement')):
        r=rows[name]; ext=r['signed_intervals']; returned=r['returned_by_mode']
        fields=[('External input',ext['external'],'#266398')] if index==0 else [
            ('First input',ext['f_ext_xyz_native'],'#266398'),
            ('Second input',ext['f_ext_xyz_second'],'#7954a6')]
        base=fields[0][1][0][0]
        fields.extend([('R31 event notice',returned['historical_r31'],'#627481'),
                       ('Signed load evidence',r['measured_load'],'#12866a'),
                       ('Native response linked',r['linked_native'],'#bd7724'),
                       ('R32 Boolean output',returned['sustained_level'],'#273e54')])
        ax.set_title(title,loc='left',weight='bold',fontsize=10,pad=8)
        xmax=max(b for _,intervals,_ in fields for a,b in intervals if base-10<a<base+250)-base+14
        for k,(_,spans,color) in enumerate(fields):
            bars=[(a-base,b-a) for a,b in spans if b>base-10 and a<base+xmax]
            if any(width<=0 for _,width in bars):raise ValueError('invalid half-open span')
            ax.broken_barh(bars,(k-.23,.46),facecolors=color,zorder=4)
        ax.set(xlim=(-7,xmax),ylim=(len(fields)-.45,-.65),yticks=range(len(fields)),
               yticklabels=[f[0] for f in fields])
        ax.axvline(0,linewidth=.65,color='#74818c')
        ax.xaxis.set_major_locator(MultipleLocator(25));ax.grid(axis='x',color='#e3e8e9',linewidth=.7)
        ax.tick_params(axis='y',length=0);ax.spines[['left','right','top']].set_visible(False)
    axes[-1].set_xlabel('Milliseconds after the first nonzero applied-force sample (1 sample = 1 ms)')
    figure.savefig(out,dpi=160,metadata={'Software':'r32-signed-level-intervals'})
    plt.close(figure)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() or args.out.is_symlink():parser.error('new file required')
    render(json.loads(args.records.read_bytes()),args.out)
