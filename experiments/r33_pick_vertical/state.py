"""Height-conditioned, signed two-sensor evidence for the first cleared pickup surface."""
from collections import deque
import numpy as np

CENTERS=np.arange(.005,.035,.004)
def expected(height,medians):
    return np.array([np.interp(height,CENTERS,medians[:,axis]) for axis in (0,1)])

class Evidence:
    def __init__(self, medians):
        self.medians=np.asarray(medians,float)
        self.buffer=deque(maxlen=8)
    def feed(self,s):
        if s is None or not .003<=s.height<=.035:
            self.buffer.clear();return None
        delta=np.array((s.inverse,s.momentum))-expected(s.height,self.medians)
        if not np.isfinite(delta).all():
            self.buffer.clear();return None
        self.buffer.append(delta)
        if len(self.buffer)!=self.buffer.maxlen:return None
        a,b=np.mean(self.buffer,axis=0)
        return float(np.sign(a)*min(abs(a),abs(b))) if a*b>0 else 0.
