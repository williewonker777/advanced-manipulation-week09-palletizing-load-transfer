"""Plot the measured stock paired-reverse case; never label this research figure as the course demo."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or a.out.is_symlink():p.error('fresh figure required')
    j=json.loads(a.input.read_bytes())
    if j.get('schema')!='r29-detached-force-mechanism.v1':raise ValueError('registered real-mechanism trace required')
    z=j['stock_curves']; row=next(v for v in j['rows']if v['mechanics']=='stock');t=np.asarray(z['t']);start=row['second']['first_nonzero_index']
    fig,ax=plt.subplots(2,1,figsize=(8.2,4.4),sharex=True,gridspec_kw=dict(height_ratios=[1.4,1]))
    fig.subplots_adjust(left=.105,right=.98,bottom=.15,top=.79,hspace=.12)
    fig.suptitle('Two real forces after release | disclosed torque study',fontweight='bold',fontsize=14,x=.105,ha='left',y=.99)
    ax[0].plot(t,z['physical_force_N'],color='#4c535b',linewidth=1.3,label='Applied force (offline)')
    ax[0].plot(t,z['mean_N'],color='#286c9d',linewidth=1.6,label='Measured planar force (12-sample mean)')
    ax[0].axhline(row['release_bound'],linewidth=1,linestyle='--',color='#28745a',label='Native release bound')
    ax[0].set_ylabel('Force [N]');ax[0].legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,fontsize=7.7,frameon=False,borderaxespad=0)
    for index,(name,key,color) in enumerate((('R29, return confirmed','endpoint','#007f66'),('C3, previous','native','#b4522f'))):
        width=np.r_[t,t[-1]+1]
        values=np.r_[z[key],z[key][-1]]
        ax[1].fill_between(width,index+.12,index+.12+.52*values,step='post',color=color)
    ax[1].set_yticks([.37,1.37],['R29','C3']);ax[1].set_ylim(-.10,2.3);ax[1].set_ylabel('Alarm');ax[1].set_xlabel('Simulation time [ms]')
    for k in ax:
        k.axvspan(start-22,start,color='#9488b3',alpha=.18)
        k.axvline(start,color='#625879',linewidth=.9,linestyle=':')
        k.spines[['right','top']].set_visible(False);k.grid(axis='x',color='#c8ced0',alpha=.35)
    ax[1].text(start-11,1.91,'quiet check',ha='center',fontsize=8,color='#55466f')
    for r in row['native_intervals'][:1]:ax[1].annotate(str(r[1]),(r[1],1.48),xytext=(r[1]+4,1.7),fontsize=7.7)
    for r in row['endpoint_intervals'][:1]:ax[1].annotate(str(r[1]),(r[1],.43),xytext=(r[1]-8,-.03),fontsize=7.7)
    ax[1].set_xlim(t[0],t[-1]);fig.savefig(a.out,dpi=150,metadata={'Software':'Week09 student research'});plt.close(fig)


if __name__=='__main__':main()
