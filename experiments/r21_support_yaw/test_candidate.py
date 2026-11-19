import math

import numpy as np

from experiments.r21_support_yaw.support import admissible_yaw, unexplained_yaw


class Packet:
    last_table_allowed = True
    last_table_wrench = np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0])
    inv_k = np.zeros(6)
    half_x = 0.1
    half_y = 0.2
    friction_mu = 0.08
    torsion_radius = 0.02

    @staticmethod
    def _point_jacobian(q):
        return None, np.array([0.3, 0.4, 0.0]), None


def test_physical_bound():
    expected = 50.0 * (0.08 * (0.5 + math.hypot(0.1, 0.2)) + 0.02)
    assert admissible_yaw(Packet(), np.zeros(6), np.zeros(6), True) == expected


def test_bound_requires_admissible_gripped_support():
    packet = Packet()
    packet.last_table_allowed = False
    assert admissible_yaw(packet, np.zeros(6), np.zeros(6), True) == 0.0
    packet.last_table_allowed = True
    assert admissible_yaw(packet, np.zeros(6), np.zeros(6), False) == 0.0


def test_only_explainable_interval_is_removed():
    assert unexplained_yaw(7.0, 3.0) == 4.0
    assert unexplained_yaw(-7.0, 3.0) == -4.0
    assert unexplained_yaw(2.0, 3.0) == 0.0
