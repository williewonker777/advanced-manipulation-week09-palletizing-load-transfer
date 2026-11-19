"""Numerical contract tests for the frozen public R13 grasp estimator."""
import unittest

import numpy as np

from tools import r13_grasp_estimator as estimator


def manufactured_feature(mass, height_mm=0.0):
    """A full-rank public-signal feature with a known vertical displacement."""
    time = ((estimator.CENTRES - estimator.CENTRES.mean()) /
            np.ptp(estimator.CENTRES))
    n, joints = len(time), 6
    jacobian = np.empty((n, joints, 3))
    for joint in range(joints):
        jacobian[:, joint, 0] = .010 * (joint + 1) * (1. + .2 * time)
        jacobian[:, joint, 1] = .013 * (-1) ** joint * (1. - .3 * time ** 2)
        jacobian[:, joint, 2] = (.040 * np.sin((joint + 1) * np.pi * (time + 1.) / 2.)
                                 + .012 * (joint + 1) * time)
    stiffness = np.stack([.03 * np.cos((joint + 1) * np.pi * time)
                          for joint in range(joints)], axis=1)
    bias = np.stack([.02 * (joint + 1) + .004 * (mass - 5.) * np.cos(2 * np.pi * time)
                     for joint in range(joints)], axis=1)
    return {
        'mass': np.array(mass),
        'y': bias + height_mm * jacobian[:, :, 2],
        'jac': jacobian,
        'stiffness': stiffness,
        'prior_bottom_mm': np.array(.2),
        'prior_com_mm': np.zeros(3),
        'vertical': np.array([0., 0., 1.]),
    }


class PublicInputContractTest(unittest.TestCase):
    @staticmethod
    def valid_input():
        count = 3551
        t = np.arange(count) * estimator.DT
        theta = np.zeros((count, 6))
        sensed = np.zeros_like(theta)
        grip = np.zeros(count, dtype=bool)
        grip[1000:] = True
        return t, theta, sensed, grip

    def test_frozen_stencil_never_uses_samples_after_decision(self):
        self.assertEqual(estimator.DECISION_S, 3.55)
        self.assertLessEqual(estimator.CENTRES.max() + estimator.HALF + 1,
                             round(estimator.DECISION_S / estimator.DT))
        self.assertAlmostEqual(estimator.WEIGHT.sum(), 1.0)
        np.testing.assert_allclose(estimator.WEIGHT, estimator.WEIGHT[::-1])

    def test_filtered_input_accepts_exact_public_grid_and_one_prior_grasp(self):
        values = self.valid_input()
        encoder, displacement, torque, latch = estimator.filtered_input(
            *values, estimator.DT, np.full(6, 1000.), np.full(6, 10.),
            np.array([True, False, True, False, True, False]), 300.)
        self.assertEqual(latch, 1000)
        self.assertEqual(encoder.shape, (3549, 6))
        self.assertEqual(displacement.shape, torque.shape)
        np.testing.assert_array_equal(displacement, 0.)

    def test_filtered_input_rejects_irregular_future_and_nonbinary_data(self):
        t, theta, sensed, grip = self.valid_input()
        bad = []
        irregular = t.copy(); irregular[200] += 1e-8
        bad.append((irregular, theta, sensed, grip))
        bad.append((np.r_[t, t[-1] + estimator.DT],
                    np.vstack((theta, theta[-1])), np.vstack((sensed, sensed[-1])),
                    np.r_[grip, True]))
        nonbinary = grip.astype(float); nonbinary[2000] = .5
        bad.append((t, theta, sensed, nonbinary))
        no_carry = grip.copy(); no_carry[estimator.CENTRES[4]] = False
        bad.append((t, theta, sensed, no_carry))
        for values in bad:
            with self.subTest(length=len(values[0])), self.assertRaises(ValueError):
                estimator.filtered_input(*values, estimator.DT, np.full(6, 1000.),
                                         np.full(6, 10.), np.ones(6, bool), 300.)


class ManufacturedIdentificationTest(unittest.TestCase):
    def setUp(self):
        self.masses = (3., 4., 5., 6., 7.)
        self.features = [manufactured_feature(mass) for mass in self.masses]
        self.model, self.guard, self.loo = estimator.calibration(
            self.features, np.full(6, .05))

    def test_calibration_uses_normal_runs_and_recovers_signed_height(self):
        self.assertLess(self.guard, 1e-10)
        self.assertEqual(len(self.loo), 5)
        for height in (-5., -3., 3., 5.):
            with self.subTest(height=height):
                result = estimator.estimate(manufactured_feature(5.5, height),
                                            self.model, self.guard)
                self.assertAlmostEqual(result['extra_height_mm'], height, places=4)
                self.assertEqual(np.sign(result['extra_height_mm']), np.sign(height))
                self.assertTrue(result['accept'])
                self.assertLess(result['half_width_mm'], 1.0)

    def test_zero_signal_abstains_and_declared_interval_covers_zero(self):
        result = estimator.estimate(manufactured_feature(5.5), self.model, self.guard)
        self.assertFalse(result['accept'])
        self.assertLessEqual(result['interval_mm'][0], 0.0)
        self.assertGreaterEqual(result['interval_mm'][1], 0.0)
        self.assertGreaterEqual(result['half_width_mm'], .75)
        self.assertEqual(len(result['block_delete_height_mm']), 5)

    def test_mass_extrapolation_never_accepts_even_strong_signal(self):
        result = estimator.estimate(manufactured_feature(8., 5.), self.model, self.guard)
        self.assertFalse(result['accept'])
        self.assertFalse(result['fitted_displacement_within_domain'])

    def test_duplicate_or_too_few_calibration_masses_are_rejected(self):
        with self.assertRaises(ValueError):
            estimator.fit_normal(self.features[:2], np.full(6, .05))
        duplicate = self.features[:3] + [manufactured_feature(5.)]
        with self.assertRaises(ValueError):
            estimator.fit_normal(duplicate, np.full(6, .05))


if __name__ == '__main__':
    unittest.main()
