import unittest
import numpy as np

from experiments.r11_datum.controller import frame, normal_datum, smooth
from sim.nominal import Nominal
from sim import trajectory as T


class DatumTests(unittest.TestCase):
    def test_c2_boundaries_and_time_derivatives(self):
        np.testing.assert_array_equal(smooth(-1., 1., 2.), [0., 0., 0.])
        np.testing.assert_array_equal(smooth(2., 1., 2.), [1., 0., 0.])
        dt = 1e-6
        for t in [1.005, 1.2, 1.99]:
            num = (smooth(t+dt, 1., 2.)[:2] - smooth(t-dt, 1., 2.)[:2]) / (2*dt)
            np.testing.assert_allclose(num, smooth(t, 1., 2.)[1:], atol=1e-8)

    def test_horizontal_error_and_yaw_do_not_press_or_tip(self):
        i = np.eye(3); z = np.zeros(3)
        yaw = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        delta, scale = normal_datum((z, i), (np.array([.002, -.001, 0.]), yaw), i)
        np.testing.assert_array_equal(delta, np.zeros(6))
        self.assertEqual(scale, 1.)

    def test_caps_and_nominal_jacobian_give_prescribed_normal_direction(self):
        i, z = np.eye(3), np.zeros(3)
        a = .02
        rx = np.array([[1.,0.,0.],[0.,np.cos(a),-np.sin(a)],[0.,np.sin(a),np.cos(a)]])
        delta, scale = normal_datum((z, i), (np.array([.1, 0., .01]), rx), i)
        self.assertAlmostEqual(scale, .15)
        self.assertAlmostEqual(delta[2], .0015)
        self.assertLess(np.linalg.norm(delta[3:]), np.deg2rad(.4))
        n = Nominal()
        q = T.reference(T.T_RELEASE)[0]
        p, _, j = frame(n, q, True)
        dq = np.linalg.solve(j, delta)
        pp, _ = frame(n, q + dq / 100)
        np.testing.assert_allclose((pp - p) * 100, delta[:3], atol=2e-7)
