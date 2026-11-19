"""Fit from no-force external training motions only, with one vote per trajectory per bin."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .lift_probe import training
from .signal import Sample
from .state import CENTERS,Evidence

def main():
    p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    traces=[];pins={}
    for c in training():
        f=a.traces/f"{c['case_id']}.npz";pins[c['case_id']]=hashlib.sha256(f.read_bytes()).hexdigest()
        with np.load(f) as d:
            if np.any(d['signed']):raise ValueError('forced training is not permitted')
            traces.append(d['public'])
    cells=[[] for _ in CENTERS]
    for arr in traces:
        for j,z in enumerate(CENTERS):
            which=arr[np.abs(arr[:,1]-z)<.002,2:4]
            if len(which):cells[j].append(np.median(which,axis=0))
    if min(map(len,cells))<len(traces)-1:raise ValueError('training bin undercovered')
    meds=np.array([np.median(v,axis=0) for v in cells]);signed=[];jitter=[]
    for arr in traces:
        ev=Evidence(meds);last=None
        for tick,h,inv,mom,native in arr:
            score=ev.feed(Sample(h,inv,mom))
            if score is not None:
                signed.append(score)
                if last is not None:jitter.append(abs(score-last))
            last=score
    margin=max(.75,3*np.median(jitter))
    result=dict(schema='r33-pick-normal-geometry.v1',traces=pins,centers=CENTERS.tolist(),medians=meds.tolist(),margin_N=margin,
      negative_limit_N=-min(signed)+margin,positive_limit_N=max(signed)+margin,bin_trajectory_counts=list(map(len,cells)),normal_samples=len(signed))
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(result['negative_limit_N'],result['positive_limit_N'],margin,len(signed))
if __name__=='__main__':main()
