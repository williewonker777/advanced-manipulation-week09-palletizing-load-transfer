"""Render the R14 registered-vs-unregistered measurement comparison (derived results only)."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'results/r14/registered_height.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error('new output path required')
    result = json.loads(args.input.read_text())
    if result.get('schema') != 'r14-airborne-validation.v1':
        parser.error('expected derived R14 results')
    fig, axes = plt.subplots(1,2,figsize=(9.6,4.9),sharex=True,sharey=True)
    colors = {'before':'#ab604c','after':'#006f68'}
    for ax,mode in zip(axes,('servo','torque')):
        record=result['by_arm']['passive'][mode]
        selected=[r for r in record['rows'] if r['variant'] in ('up','down')]
        for row in selected:
            truth=row['truth_mm'];raw=row['com_residual_height_mm'];corrected=row['extra_height_mm']
            ax.plot([truth,truth],[raw,corrected],color='#abb8bf',lw=1,zorder=1)
            ax.scatter(truth,raw,color=colors['before'],marker='x',s=41,zorder=3)
            ax.scatter(truth,corrected,facecolors=colors['after'] if row['accept'] else 'white',
                       edgecolors=colors['after'],linewidths=1.6,s=37,zorder=4)
        ax.plot([-7,7],[-7,7],ls='--',color='#61697a',lw=1)
        ax.axhline(0,color='#bdc3ce',lw=.7);ax.axvline(0,color='#bdc3ce',lw=.7)
        ax.set(xlim=(-7.4,3.5),ylim=(-7.4,7.3),xlabel='Measured bottom-height change (mm)')
        ax.grid(alpha=.15)
        m=record['metrics']
        ax.set_title(f"{mode.title()}  ·  {m['vertical_point_rmse_mm']:.2f} mm RMSE",fontsize=12,loc='left',weight='bold')
        ax.text(.03,.96,f"{m['large_signed_accurate_action']}/{m['large_cases']} material shifts eligible",
                transform=ax.transAxes,va='top',fontsize=9,color='#243344')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Estimated change (mm)')
    fig.suptitle('The grasp frame moves; the table frame does not',x=.08,y=.98,ha='left',fontsize=15,weight='bold')
    fig.legend(handles=[Line2D([],[],marker='x',ls='',color=colors['before'],label='COM residual alone'),
                        Line2D([],[],marker='o',ls='',mec=colors['after'],mfc='white',label='Registered: abstain'),
                        Line2D([],[],marker='o',ls='',color=colors['after'],label='Registered: eligible')],
               ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,.015))
    fig.subplots_adjust(left=.085,right=.985,bottom=.2,top=.83,wspace=.17)
    fig.savefig(args.out,dpi=150,metadata={'Software':'Week09 result renderer'});plt.close(fig)


if __name__=='__main__':
    main()
