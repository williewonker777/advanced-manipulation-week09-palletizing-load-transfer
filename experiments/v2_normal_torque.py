"""C_normal wrapper around the frozen v1 torque controller."""

from sim import trajectory as T
from v1_torque import Controller as V1Controller

from experiments.v2_normal_core import NormalReferenceMixin


class Controller(NormalReferenceMixin, V1Controller):
    """Retain v1 torque control and add the preregistered normal correction."""

    NORMAL_TRAJECTORY = T

    def _reference(self, t):
        return self._normal_reference(*super()._reference(t))

    def compute(self, t, th, thd, tau_s, grip, q_d, qd_d, qdd_d):
        self._normal_update(t, th, thd, tau_s, grip)
        return super().compute(t, th, thd, tau_s, grip, q_d, qd_d, qdd_d)
