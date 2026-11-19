"""Set-valued normal force certificate in observable pose/velocity, not elapsed task time."""
from collections import deque

import numpy as np

from experiments.r27_load_change.event import DIRECTIONS, WIDTH


class Coordinates:
    def __init__(self):
        self.x=deque(maxlen=WIDTH+1)

    def feed(self,encoder,eligible):
        if not eligible or not np.isfinite(encoder).all():
            self.x.clear()
            return None
        self.x.append(np.asarray(encoder,float).copy())
        if len(self.x)!=WIDTH+1:return None
        return np.r_[(self.x[0]+self.x[-1])*.5,(self.x[-1]-self.x[0])/(.001*WIDTH)]


class FeasibleNormals:
    def __init__(self,model):
        self.center=np.asarray(model['location'],float)
        self.scale=np.asarray(model['scale'],float)
        self.nodes=np.asarray(model['nodes'],float)
        self.limit=np.asarray(model['upper_bounds_N'],float)
        self.radius=np.asarray(model['radius_squared'],float)
        if (self.nodes.ndim!=2 or self.nodes.shape[1]!=12 or self.limit.shape!=(len(self.nodes),2,8)
                or self.radius.shape!=(len(self.nodes),) or self.scale.shape!=(12,) or self.center.shape!=(12,)
                or np.any(self.scale<=0) or np.any(self.radius<0)
                or not all(np.isfinite(a).all() for a in (self.center,self.scale,self.nodes,self.limit,self.radius))):
            raise ValueError('invalid force/kinematics envelope')

    def unexpected(self,coords,levels,axis):
        if coords is None or levels is None or not np.isfinite(coords).all() or not np.isfinite(levels).all():
            return False
        distances=np.sum((self.nodes-(coords-self.center)/self.scale)**2,axis=1)
        neighbors=np.argsort(distances,kind='stable')[:2]
        if np.any(distances[neighbors]>self.radius[neighbors]):
            return False  # Unknown geometry/motion is not positive evidence.
        bound=self.limit[neighbors,:,axis].max(axis=0)
        return bool(np.all(np.asarray(levels) @ DIRECTIONS[axis]>bound))
