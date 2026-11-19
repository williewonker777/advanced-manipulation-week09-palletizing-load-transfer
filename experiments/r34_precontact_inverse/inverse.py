"""Full six-axis causal inverse wrench at the empty flange center of mass."""
from dataclasses import dataclass

import mujoco
import numpy as np

from experiments.r15_delayed_wrench.observer import KERNEL, LAG, N, NormalForce
from sim import config as C


@dataclass(frozen=True)
class FlangeSample:
    time: float
    effective_time: float
    wrench: np.ndarray
    bottom_z: float


class FlangeInverse:
    """Reuse the 31-sample encoder/JTS integration, changing only projection."""

    def __init__(self, empty, loaded):
        self.observer = NormalForce(empty, loaded, dt=C.DT, viscous_spring=C.D,
                                    jts_hz=C.JTS_CUTOFF)
        self.bottom_offset = np.array(
            [0.0, 0.0, C.BOX_POS[2] + 0.5 * C.BOX_SIZE[2]], dtype=float)

    def update(self, t, encoder, jts, grip):
        state = self.observer
        state.update(t, encoder, jts, grip, project=False)
        if state.bias is None or len(state.residual) != N or len(state.midpos) != N:
            return None
        model = state.loaded if grip else state.empty
        model.d.qpos[:] = state.midpos[N // 2]
        mujoco.mj_kinematics(model.m, model.d)
        mujoco.mj_comPos(model.m, model.d)
        body = model.m.body("link_6").id
        jp, jw = np.zeros((3, 6)), np.zeros((3, 6))
        mujoco.mj_jacBodyCom(model.m, model.d, jp, jw, body)
        jacobian = np.vstack((jp, jw))
        if np.linalg.cond(jacobian) > 1e4:
            return None
        wrench = np.linalg.solve(jacobian.T, KERNEL @ np.asarray(state.residual) - state.bias)
        rotation = model.d.xmat[body].reshape(3, 3)
        bottom_z = float((model.d.xpos[body] + rotation @ self.bottom_offset)[2])
        return FlangeSample(float(t), float(t - LAG * state.dt), wrench.copy(), bottom_z)
