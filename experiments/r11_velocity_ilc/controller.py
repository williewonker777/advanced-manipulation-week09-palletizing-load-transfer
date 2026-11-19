"""Normal-only learned, compact Cartesian-vertical reference correction."""
import numpy as np
from sim import trajectory as T
import r1_servo
import r1_torque
from experiments.r11_datum.controller import frame


def basis(t, start, end):
    """C2 local polynomial bumps and exact time derivatives."""
    duration = end-start
    if duration <= 0:
        raise ValueError('invalid taught descent')
    centers = np.array([.51,.71,.91,1.09])
    width = .26*duration
    x = (np.asarray(t)[...,None]-(start+duration*centers))/width
    keep = np.abs(x) < 1.
    x = np.clip(x,-1.,1.)
    h = 1.-x*x
    return np.stack((h**3, -6.*x*h*h/width, (-6.+36.*x*x-30.*x**4)/(width*width)),axis=-2)*keep[...,None,:]


class _NormalVelocity:
    amplitudes_mm = (0.,0.,0.,0.)

    def reset(self):
        super().reset()
        end = T.T_RELEASE
        start = max(row[0] for row in T.WAYPOINTS if row[0] < end)
        self._first = int(np.floor(start/self.dt))
        stop = int(np.ceil((start+(end-start)*1.36)/self.dt))
        ts = self.dt*np.arange(self._first,stop+1)
        amps = np.asarray(self.amplitudes_mm, dtype=float)
        if amps.shape != (4,) or not np.isfinite(amps).all():
            raise ValueError('four finite normal coefficients required')
        self._profile = (basis(ts,start,end) @ amps) * (1e-3*np.clip(self.box_mass/2.,0.,1.))
        _, _, j = frame(self.nom,T.reference(end)[0],True)
        self._vertical = np.linalg.solve(j,[0.,0.,1.,0.,0.,0.])

    def _reference(self,t,*args):
        command = super()._reference(t,*args)
        k = round(t/self.dt)-self._first
        if not 0 <= k < len(self._profile):
            return command
        return tuple(u + gain*self._vertical for u,gain in zip(command,self._profile[k]))


class Servo(_NormalVelocity,r1_servo.Controller):
    pass


class Torque(_NormalVelocity,r1_torque.Controller):
    pass
