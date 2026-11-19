"""Reference in measured distance, plus signed agreement of force-only and full-wrench fits."""
from collections import deque
import numpy as np

RAYS=np.column_stack((np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)))
SIDE=np.column_stack((-RAYS[:,1],RAYS[:,0]))
BLOCK=8

def coordinates(mode,progress,z):
    return z if mode==2 else progress

def row_vector(s):
    return np.r_[s.inverse,s.point,s.momentum,s.free_moment]

class Bank:
    def __init__(self,meta):
        self.knots={int(k):np.asarray(v['knots']) for k,v in meta.items()}
        self.curves={int(k):np.asarray(v['baseline']) for k,v in meta.items()}
    def center(self,mode,x):
        k=self.knots[mode]
        if x<k[0]-.001 or x>k[-1]+.001:return None
        j=np.clip(np.searchsorted(k,x)-1,0,len(k)-2)
        alpha=np.clip((x-k[j])/(k[j+1]-k[j]),0,1)
        return (1.-alpha)*self.curves[mode][j]+alpha*self.curves[mode][j+1]

class Windows:
    def __init__(self):
        self.v=deque(maxlen=2*BLOCK);self.mode=0
    def clear(self):
        self.v.clear();self.mode=0
    def feed(self,mode,v):
        if self.mode!=mode:self.v.clear();self.mode=mode
        self.v.append(v)
        if len(self.v)<self.v.maxlen:return None
        a=np.asarray(self.v)
        return a[:BLOCK].mean(0),a[BLOCK:].mean(0)

def scores(a,b,reach=.16):
    f1=a[:9].reshape(3,3);f2=b[:9].reshape(3,3)
    first=f1[:,:2]@RAYS.T;second=f2[:,:2]@RAYS.T
    # Force-only inverse and unrestricted inverse may disagree where a
    # separate moment is acting. The moment check is not a claim of contact localization.
    score=np.minimum(first.min(0),second.min(0))
    moment=np.maximum(abs(a[9:11]@SIDE.T),abs(b[9:11]@SIDE.T))/reach
    return np.where(score>moment,score,0.),first,second
