"""Two causal planar wrench estimates and measured kinematics; no force schedule or twin inputs."""
import numpy as np
from experiments.r15_delayed_wrench.observer import NormalForce
from sim import config
from sim.nominal import Nominal


class Streams:
    def __init__(self, mass, com):
        empty,loaded=Nominal(),Nominal()
        loaded.set_payload(float(mass),tuple(com))
        self.observer=NormalForce(empty,loaded,dt=config.DT,viscous_spring=config.D,jts_hz=config.JTS_CUTOFF)
        self.had_grip=False

    def read(self, detector, t, th, motor_velocity, jts, grip):
        g=bool(grip);phase=1 if g else (2 if self.had_grip else 0)
        self.had_grip |= g
        sample=self.observer.update(t,th,jts,g)
        packet=detector._shared
        momentum=getattr(packet,'last_residual_wrench',None)
        if (sample is None or momentum is None or np.shape(momentum)!=(6,) or
                not np.isfinite(momentum).all() or not np.isfinite(sample.force_n).all()):
            return phase, None
        return phase,np.r_[np.asarray(th,float),np.asarray(motor_velocity,float),np.asarray(momentum[:2]),sample.force_n[:2]]
