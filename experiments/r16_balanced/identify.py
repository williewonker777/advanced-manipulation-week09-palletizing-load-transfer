"""Small bounded max-min pressure fit. No external numerical dependency or sim import."""
from itertools import combinations
import numpy as np


def solve(gradients, slacks, lower=(0., 0.), upper=(1., 1.)):
    """Choose a,b and maximal common force margin; constraints g*x+m<=slack."""
    gradients=np.asarray(gradients,dtype=float);slacks=np.asarray(slacks,dtype=float)
    low=np.asarray(lower,float);high=np.asarray(upper,float)
    if (gradients.ndim!=2 or gradients.shape[1]!=2 or slacks.shape!=(len(gradients),)
            or not len(slacks) or np.any(low>high) or not np.isfinite(np.r_[gradients.ravel(),slacks,low,high]).all()):
        raise ValueError('invalid force constraints')
    bounds=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,1.,0.]])
    A=np.vstack((np.c_[gradients,np.ones(len(slacks))],bounds))
    b=np.r_[slacks, -low[0],high[0],-low[1],high[1]]
    candidates=[]
    for i in combinations(range(len(b)),3):
        idx=list(i)
        try:v=np.linalg.solve(A[idx],b[idx])
        except np.linalg.LinAlgError:continue
        if np.isfinite(v).all() and np.all(A@v<=b+1e-10):candidates.append(v)
    if not candidates:raise ValueError('no bounded data-supported solution')
    return max(candidates,key=lambda v:(round(float(v[2]),11),-float(v[:2]@v[:2]),-v[0]))


def features(f, clock):
    f=np.asarray(f,float); t=np.rint(1000*np.asarray(clock,float)).astype(int)
    def window(lo,hi):
        sel=(lo<=t)&(t<hi)
        if list(t[sel])!=list(range(lo,hi)):raise ValueError('incomplete release time grid')
        return f[sel]
    final=window(3880,3900)
    return np.array([window(3600,3900).max(),window(3900,4100).max(),-final.min(),-final.mean()])


def refit_from_table(rows):
    """Recreate the two bounded coefficients directly from publishable scores.

    Each arm has the same four signed criteria (upper bounds): before/after
    maxima, negative minimum and negative mean. No raw kit data is needed.
    """
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('distinct training cases required')
    directions, slack = [], []
    for row in rows:
        arms = np.asarray(row['observed_arms'], dtype=float)
        bound = np.asarray(row['constraint_bounds'], dtype=float)
        if arms.shape != (3, 4) or bound.shape != (4,):
            raise ValueError('R15, release-only and phase-only arm scores required')
        directions.extend((arms[1:] - arms[0]).T)
        slack.extend(bound - arms[0])
    return solve(directions, slack)
