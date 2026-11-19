"""Causal two-window estimates and calibrated, local normal wrench feasibility."""
import numpy as np

WIDTH=8
PERSIST=4
DIRECTIONS=np.stack((np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)),axis=-1)
FLOOR=np.array([.02]*6+[.05]*6+[.5,.01,.01,.01])
FIELDS=16


def loads(mass,com):
    return np.r_[mass,float(mass)*np.asarray(com,dtype=float)]


def batch_features(data,phase,weight):
    """Exactly the same two disjoint half-open sample blocks used by Windows."""
    data=np.asarray(data,dtype=float)
    good=np.isfinite(data).all(axis=1);safe=np.where(good[:,None],data,0.)
    start=np.r_[0,np.cumsum(~good)]
    prefix=np.vstack((np.zeros(FIELDS),np.cumsum(safe,axis=0)))
    q=np.arange(2*WIDTH-1,len(data));end=q+1
    ok=(start[end]==start[end-2*WIDTH]) & (phase[q]==phase[end-2*WIDTH])
    q=q[ok];end=end[ok];now=(prefix[end]-prefix[end-WIDTH])/WIDTH
    before=(prefix[end-WIDTH,12:]-prefix[end-2*WIDTH,12:])/WIDTH
    features=np.column_stack((now[:,:12],np.tile(weight,(len(q),1))))
    return q,features,np.column_stack((now[:,12:],now[:,12:]-before))


class Windows:
    def __init__(self):
        self.buffer=np.zeros((2*WIDTH,FIELDS));self.phase=None;self.n=0
    def clear(self):
        self.n=0;self.phase=None
    def feed(self,phase,data,weight):
        if data is None or np.shape(data)!=(FIELDS,) or not np.isfinite(data).all():
            self.clear()
            return None
        if phase!=self.phase:self.n=0;self.phase=phase
        self.buffer[self.n%(2*WIDTH)]=data
        self.n+=1
        if self.n<2*WIDTH:return None
        idx=self.n%(2*WIDTH)
        # Small bounded stencil. Summing actual samples prevents unbounded integration drift.
        before=self.buffer[(np.arange(WIDTH)+idx)%(2*WIDTH),12:].mean(axis=0)
        now=self.buffer[(np.arange(WIDTH)+idx+WIDTH)%(2*WIDTH)].mean(axis=0)
        return np.r_[now[:12],weight],np.r_[now[12:],now[12:]-before]


def near(z,nodes):
    return np.sum((z[...,:12][:,None,:]-nodes[None,:,:])**2,axis=2)


class Region:
    def __init__(self,spec):
        for key in ('center','scale','nodes','coef','limits','radius'):
            setattr(self,key,np.asarray(spec[key],dtype=float))
        k=len(self.nodes)
        if (self.center.shape!=(FIELDS,) or self.scale.shape!=(FIELDS,) or (self.scale<=0).any() or
            self.nodes.shape!=(k,12) or k<2 or self.coef.shape!=(k,FIELDS+1,8) or
            self.limits.shape!=(k,2,2,8) or self.radius.shape!=(k,) or
            not all(np.isfinite(getattr(self,k)).all()for k in ('center','scale','nodes','coef','limits','radius'))):
            raise ValueError('invalid region model')
    def margin(self,features,measurement):
        z=(features-self.center)/self.scale
        distances=np.sum((self.nodes-z[:12])**2,axis=1)
        close=np.argsort(distances,kind='stable')[:2]
        if np.any(distances[close]>self.radius[close]+1e-10):return None
        pred=np.r_[1.,z] @ self.coef[close]
        residual=(measurement-pred).reshape(2,2,2,2) @ DIRECTIONS.T
        # Both neighboring cells and both physical estimators must exclude normal.
        return (residual-self.limits[close]).min(axis=(0,2))


class Notice:
    def __init__(self):
        self.reset()
    def reset(self):
        self.direction=None;self.pending=None;self.count=0;self.left=0
        self.seen_native=False;self.notified=False
    def feed(self,margin,native):
        self.notified=False
        if margin is None:
            self.reset()
            return bool(native)
        level,change=margin
        if self.direction is not None:
            # The jointly certified disturbance no longer persists in its acquired direction.
            if level[self.direction]<=0:self.reset()
            else:
                if native:self.left=0
                optional=self.left>0;self.left=max(0,self.left-1)
                return bool(native or optional)
        score=np.minimum(level,change);idx=int(np.argmax(score))
        if score[idx]<=0:
            self.pending=None;self.count=0;self.seen_native=False
        else:
            angular=None if self.pending is None else min((idx-self.pending)%8,(self.pending-idx)%8)
            if angular is None or angular>1:self.count=0;self.seen_native=False
            self.pending=idx;self.count+=1;self.seen_native|=bool(native)
            if self.count>=PERSIST:
                self.direction=idx;self.pending=None;self.count=0
                if not self.seen_native:self.notified=True;self.left=PERSIST
        optional=self.left>0;self.left=max(0,self.left-1)
        return bool(native or optional)
