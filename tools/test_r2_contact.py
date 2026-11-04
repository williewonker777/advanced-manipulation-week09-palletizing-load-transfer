"""Pure geometry, derivative and R1 elastic-tracker regression tests."""
import unittest
import importlib
import numpy as np
from sim.nominal import Nominal
from sim import trajectory as T
from sim import config as C
from experiments.r2_contact import (
    profile, so3_exp, so3_log, se3_log, dexp_minus, lift_command,
    make_controller, START, EARLY, SMALL,
    normal_lie_error,
)


def setup(cls):
    c = cls()
    c.nom, c.nom_box = Nominal(), Nominal()
    c.nom_box.set_payload(5.3, (.01, -.012, .02))
    c.box_mass, c.box_com = 5.3, (.01, -.012, .02)
    c.reset()
    return c


class ContactTests(unittest.TestCase):
    def test_profiles(self):
        for name in ('quintic', 'seventh_joint', 'S_curve_joint'):
            np.testing.assert_allclose(profile(0., name), [0., 0., 0.], atol=1e-12)
            np.testing.assert_allclose(profile(1., name), [1., 0., 0.], atol=1e-12)
            for s in (.13, .37, .61, .91):
                h = 1e-5
                p, v, a = profile(s, name)
                pm, vm, _ = profile(s-h, name)
                pp, vp, _ = profile(s+h, name)
                self.assertAlmostEqual((pp-pm)/(2*h), v, places=6)
                self.assertAlmostEqual((vp-vm)/(2*h), a, places=5)

    def test_lie_translation_and_rotation(self):
        p = np.array([.001, -.002, .0007])
        np.testing.assert_allclose(se3_log(np.eye(3), p), np.r_[p, np.zeros(3)])
        normal = np.array([0., 0., 1., 0., 0., 0.])
        np.testing.assert_allclose(dexp_minus(np.r_[p, np.zeros(3)])@normal, normal)
        for w in (np.array([.1, -.2, .07]), np.array([1e-9, 0., 0.])):
            np.testing.assert_allclose(so3_log(so3_exp(w)), w, atol=1e-12)

    def test_frozen_tracker_parity(self):
        for mode in ('servo', 'torque'):
            base = importlib.import_module('r1_'+mode).Controller
            a = setup(base)
            b = setup(make_controller(mode, dict(family='C6')))
            for t in (.1, .101, .102, .103):
                q, v, acc = T.reference(t)
                th, thd, ts = q+.0001, v+.0002, a.nom.gravity(q)
                out_a = a.compute(t, th, thd, ts, 0, q, v, acc)
                out_b = b.compute(t, th, thd, ts, 0, q, v, acc)
                np.testing.assert_array_equal(out_a, out_b)

    def test_wrench_duality_and_moment(self):
        c = setup(make_controller('servo', dict(family='C1')))
        q = T.reference(3.8)[0]
        J, p, R = c.shared_estimator._point_jacobian(q)
        force = 20.
        f = np.array([0., 0., force])
        com_from_bottom = np.asarray(c.box_com)-np.array([0., 0., C.BOX_SIZE[2]*.5])
        cop = np.r_[np.linalg.solve(R[:2, :2], (R@com_from_bottom)[:2]), 0.]
        moment = np.cross(R@cop, f)
        np.testing.assert_allclose(c._support_wrench(q, force), J.T@np.r_[f, moment], atol=1e-12)

    def test_C6_derivatives_and_fixed_release(self):
        for variant in ('seventh_joint', 'S_curve_joint', 'SO3_R3_seventh'):
            c = setup(make_controller('torque', dict(family='C6', variant=variant)))
            for t in (START+.02, START+.07, EARLY+.01):
                h = 1e-5
                q, v, a = c._interpolation_reference(t, T.reference(t))
                qm, vm, _ = c._interpolation_reference(t-h, T.reference(t-h))
                qp, vp, _ = c._interpolation_reference(t+h, T.reference(t+h))
                np.testing.assert_allclose((qp-qm)/(2*h), v, atol=3e-7)
                np.testing.assert_allclose((vp-vm)/(2*h), a, atol=3e-5)
            q, v, a = c._interpolation_reference(EARLY+.01, T.reference(EARLY+.01))
            np.testing.assert_allclose(q, c.qf, atol=1e-11)
            np.testing.assert_allclose(v, 0., atol=1e-11)

    def test_admittance_ZOH(self):
        c = setup(make_controller('servo', dict(family='C2', error_scale=SMALL)))
        equilibrium = np.array([.0001, 0.])
        np.testing.assert_allclose(c.admit_A@equilibrium+c.admit_B*c.normal_k*.0001,
                                   equilibrium, atol=1e-15)
        self.assertLess(max(abs(np.linalg.eigvals(c.admit_A))), 1.)

    def test_lie_virtual_work_nonidentity(self):
        R, Rr = so3_exp(np.array([.1, -.12, .07])), so3_exp(np.array([-.03, .02, .04]))
        p, pr = np.array([.4, -.2, .5]), np.array([.41, -.205, .501])
        e, ed, dual = normal_lie_error(R, p, Rr, pr, np.zeros(6), np.zeros(6))
        h = 1e-6
        for j in range(6):
            d = np.eye(6)[j]
            ep = normal_lie_error(R@so3_exp(h*d[3:]), p+h*R@d[:3], Rr, pr, np.zeros(6), np.zeros(6))[0]
            em = normal_lie_error(R@so3_exp(-h*d[3:]), p-h*R@d[:3], Rr, pr, np.zeros(6), np.zeros(6))[0]
            actual_twist_world = np.r_[R@d[:3], R@d[3:]]
            self.assertAlmostEqual(-(ep-em)/(2*h), float(dual@actual_twist_world), places=8)

    def test_lie_moving_frame_error_rate(self):
        R, Rr = so3_exp(np.array([.1, -.12, .07])), so3_exp(np.array([-.03, .02, .04]))
        p, pr = np.array([.4, -.2, .5]), np.array([.41, -.205, .501])
        V, Vr = np.array([.04, -.01, -.02, .1, -.2, .1]), np.array([.02, .01, -.03, -.1, .05, .2])
        error, rate, dual = normal_lie_error(R, p, Rr, pr, V, Vr)
        h = 1e-6
        def at(dt):
            return normal_lie_error(so3_exp(dt*V[3:])@R, p+dt*V[:3],
                                    so3_exp(dt*Vr[3:])@Rr, pr+dt*Vr[:3], V, Vr)[0]
        self.assertAlmostEqual((at(h)-at(-h))/(2*h), rate, places=8)


if __name__ == '__main__':
    unittest.main()
