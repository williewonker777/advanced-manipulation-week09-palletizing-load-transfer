import unittest
import numpy as np
from tools.r3_signal_analysis import innovations


class TestR3SignalAnalysis(unittest.TestCase):
    def test_matched_normal_subtraction_and_force_window(self):
        t = np.arange(500) * .001
        normal = dict(obs_t=t, joint_residual=np.ones((500, 6)),
                      estimated_wrench=np.ones((500, 6)), active=np.ones(500, bool))
        collision = {key: value.copy() for key, value in normal.items()}
        collision['joint_residual'][210, 0] += 3
        collision['estimated_wrench'][210, 1] += 4
        collision['estimated_wrench'][400, 1] += 100
        value = innovations(collision, normal, .2)
        self.assertEqual(value['joint_residual_innovation_peak_nm'], 3)
        self.assertEqual(value['horizontal_wrench_innovation_peak_n'], 4)
        self.assertEqual(value['force_on_samples'], 120)
