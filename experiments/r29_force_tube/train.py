"""Train from the registered normal splits only; force columns are never used as features."""
import argparse,hashlib,json,os
from pathlib import Path

import numpy as np
from experiments.r29_force_tube.sources import inputs
from experiments.r29_force_tube.tube import DIRECTIONS,FIELDS,FLOOR,batch_features,loads,near
HERE=Path(__file__).resolve().parent


def medoids(z):
    cloud=z[::8,:12];nodes=[cloud[np.argmin(np.sum((cloud-np.median(cloud,axis=0))**2,axis=1))]]
    dist=np.full(len(cloud),np.inf)
    for _ in range(11):
        dist=np.minimum(dist,np.sum((cloud-nodes[-1])**2,axis=1));nodes.append(cloud[np.argmax(dist)])
    return np.array(nodes)


def design(z):return np.column_stack((np.ones(len(z)),z))


def fit_region(train,cal):
    x,y=train
    center=np.median(x,axis=0);lo,hi=np.quantile(x,[.05,.95],axis=0);scale=np.maximum(hi-lo,FLOOR)
    z=(x-center)/scale;nodes=medoids(z);zc=(cal[0]-center)/scale
    codes=[];distances=[]
    for g in (z,zc):
        ds=near(g,nodes);distances.append(ds);codes.append(np.argsort(ds,axis=1,kind='stable')[:,:2])
    coef=[];upper=[];radius=[];support=[]
    penalty=np.diag([0.]+[.01]*FIELDS)
    for i in range(len(nodes)):
        u=np.any(codes[0]==i,axis=1);v=np.any(codes[1]==i,axis=1)
        if not u.any() or not v.any():raise ValueError('unsupported calibration cell')
        a=design(z[u]);b=design(zc[v]);coef_i=np.linalg.solve(a.T@a/len(a)+penalty,a.T@y[u]/len(a))
        def limit(target,mat):
            proj=(target-mat@coef_i).reshape(len(mat),2,2,2)@DIRECTIONS.T
            return np.max(proj,axis=0)
        f=limit(y[u],a);c=limit(cal[1][v],b)
        coef.append(coef_i);upper.append(np.maximum(0.,np.maximum(f,c)+np.maximum(0.,c-f)))
        radius.append(max(distances[0][u,i].max(),distances[1][v,i].max()))
        support.append([int(u.sum()),int(v.sum())])
    return {k:np.asarray(v).tolist()for k,v in dict(center=center,scale=scale,nodes=nodes,coef=coef,limits=upper,radius=radius,support=support).items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--scans',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if os.path.lexists(a.out):p.error('fresh model required')
    store={k:[]for k in ('fit','cal')};rows=[]
    for split,c,m,twin,imp in inputs():
        if split not in store:continue
        assert imp==(None,None) and twin=='none'
        file=a.scans/('__'.join((split,c['id'],m['kind'],twin))+'.npz')
        with np.load(file,allow_pickle=False)as raw:
            if np.any(raw['force1'])or np.any(raw['force2']):raise ValueError('unforced only')
            phase=raw['phase'];ticks,x,y=batch_features(raw['data'],phase,loads(c['mass'],c['com']))
            store[split].append((phase[ticks],x,y))
            rows.append(dict(split=split,id=c['id'],mechanics=m['kind'],physical=str(raw['physical']),samples=len(ticks)))
    if len(store['fit'])!=27 or len(store['cal'])!=15:raise ValueError('incomplete historical normal split')
    tables={}
    for phase in range(3):
        def collect(values):
            return tuple(np.concatenate([v[n][v[0]==phase]for v in values])for n in (1,2))
        tables[str(phase)]=fit_region(collect(store['fit']),collect(store['cal']))
    s=dict(schema='r29-calibrated-two-wrench-tube.v1',protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),normals=rows,phases=tables)
    with a.out.open('x')as f:json.dump(s,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('normal-only',len(rows),'regions',sum(len(t['nodes'])for t in tables.values()))


if __name__=='__main__':main()
