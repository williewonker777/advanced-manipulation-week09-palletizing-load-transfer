"""Build the pose-conditioned normal set with separate normal-only calibration motions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as Baseline
from experiments.r27_load_change.event import DIRECTIONS
from experiments.r27_load_change.manifold_event import Coordinates
from experiments.r27_load_change.projection import Projection
from experiments.r27_load_change.settled_event import PairedWindows, channels

HERE=Path(__file__).resolve().parent


class Capture(Baseline):
    def reset(self):
        super().reset()
        self.r27_projection=Projection(self.box_mass,self.box_com)
        self.r27_windows=PairedWindows()
        self.r27_coords=Coordinates()
        self.rows=[]

    def update(self,t,th,thd,tau_m,tau_s,grip):
        residual,native=super().update(t,th,thd,tau_m,tau_s,grip)
        pair=channels(self,grip)
        coords=self.r27_coords.feed(th,pair is not None)
        means=self.r27_windows.feed(pair)
        if coords is not None and means is not None:
            self.rows.append(np.r_[coords,means[1].reshape(-1)])
        return residual,native


def collect(spec):
    split,case,m=spec;c=Capture()
    _,log=lab.run_episode(case,m,lambda:c,(None,None))
    if not lab.exact_force(log,(None,None))['passed'] or not len(c.rows):raise ValueError('expected no force and complete loaded observations')
    return (split,case['id'],m['kind'],lab.certified.trace_signature({'none':log}),np.asarray(c.rows))


def cover(x,k):
    center=np.median(x,axis=0)
    lo,hi=np.quantile(x,[.05,.95],axis=0)
    scale=np.maximum(hi-lo,np.r_[[.02]*6,[.05]*6])
    z=(x-center)/scale
    cloud=z[::8]
    nodes=[cloud[np.argmin(np.sum((cloud-np.median(cloud,axis=0))**2,axis=1))]]
    dist=np.full(len(cloud),np.inf)
    while len(nodes)<k:
        dist=np.minimum(dist,np.sum((cloud-nodes[-1])**2,axis=1))
        nodes.append(cloud[np.argmax(dist)])
    return center,scale,np.array(nodes)


def envelopes(x,center,scale,nodes):
    n=len(nodes);bounds=np.full((n,2,8),-np.inf)
    radius=np.zeros(n);support=np.zeros(n,dtype=int)
    for batch in np.array_split(x,max(1,len(x)//500)):
        coords=(batch[:,:12]-center)/scale;levels=batch[:,12:].reshape(-1,2,2)@DIRECTIONS.T
        dist=np.sum((coords[:,None,:]-nodes[None,:,:])**2,axis=2)
        nearest=np.argsort(dist,axis=1,kind='stable')[:,:2]
        for i in range(n):
            rows=np.any(nearest==i,axis=1)
            if np.any(rows):
                bounds[i]=np.maximum(bounds[i],np.max(levels[rows],axis=0))
                radius[i]=max(radius[i],float(np.max(dist[rows,i])))
                support[i]+=int(rows.sum())
    if np.any(support==0):raise ValueError('uncovered training/calibration region')
    return bounds,radius,support


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args();reg=json.loads((HERE/'manifold_registration.json').read_bytes());original=json.loads((HERE/'registration.json').read_bytes())
    mechanics=json.loads(lab.PLAN.read_bytes())['variants']
    tasks=[(split,c,mechanics[m]) for split,cs in (('fit',original['fit_normals']),('cal',reg['extra_cal_normals']))
           for c in cs for m in original['mechanics']]
    with ProcessPoolExecutor(a.workers)as pool: data=list(pool.map(collect,tasks))
    train,cal=(np.concatenate([x[-1]for x in data if x[0]==split])for split in ('fit','cal'))
    center,scale,nodes=cover(train[:,:12],24)
    bf,rf,sf=envelopes(train,center,scale,nodes);bc,rc,sc=envelopes(cal,center,scale,nodes)
    prior=json.loads((HERE/'model.json').read_bytes());anchors={(x['id'],x['variant']):x['physics']for x in prior['normals']}
    if not all(anchors[c,m]==signature for split,c,m,signature,_ in data if split=='fit'):raise ValueError('wrong 27 normal physics')
    upper=np.maximum(np.maximum(bf,bc)+np.maximum(0.,bc-bf),0.)
    j=dict(schema='r27-pose-velocity-normal-set.v1',registration_sha256=hashlib.sha256((HERE/'manifold_registration.json').read_bytes()).hexdigest(),
           location=center.tolist(),scale=scale.tolist(),nodes=nodes.tolist(),upper_bounds_N=upper.tolist(),
           radius_squared=np.maximum(rf,rc).tolist(),fit_support=sf.tolist(),calibration_support=sc.tolist(),
           source=[dict(split=k,id=c,mechanics=m,physics=signature,eligible_ticks=len(rows))for k,c,m,signature,rows in data])
    with a.out.open('x')as f:json.dump(j,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('normal episodes',len(data),'kinematic cells',len(nodes),'smallest fit/cal coverage',min(sf),min(sc))


if __name__=='__main__':
    main()
