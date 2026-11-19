"""Read the incumbent's sensor integration window; project it using private nominal scratch."""
import mujoco
import numpy as np

from experiments.r15_delayed_wrench.observer import KERNEL,N
from sim.nominal import Nominal


class Projection:
    def __init__(self,mass,com):
        self.nom=Nominal()
        self.nom.set_payload(float(mass),tuple(com))
        self.jp=np.empty((3,6));self.jw=np.empty((3,6))
        self.site=self.nom.m.site('tcp').id

    def from_integrated(self, detector):
        # C3 intentionally skips its own Jacobian projection when no native signal is active.
        witness=detector.witness
        if (witness is None or witness.bias is None or len(witness.residual)!=N
                or len(witness.midpos)!=N or not witness.prev_grip):
            return None
        self.nom.d.qpos[:]=witness.midpos[N//2]
        mujoco.mj_kinematics(self.nom.m,self.nom.d);mujoco.mj_comPos(self.nom.m,self.nom.d)
        mujoco.mj_jacSite(self.nom.m,self.nom.d,self.jp,self.jw,self.site)
        jac=np.vstack((self.jp,self.jw))
        if np.linalg.cond(jac)>1e4:return None
        return np.linalg.solve(jac.T,KERNEL@np.asarray(witness.residual)-witness.bias)[:2]
