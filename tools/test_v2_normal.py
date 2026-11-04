#!/usr/bin/env python3
"""Dependency-free unit tests for the preregistered C_normal core."""

import unittest

import numpy as np

from experiments.v2_normal_core import (
    NormalAdmittance,
    _evaluate_polynomial,
    _quintic_boundary,
    normal_joint_direction,
)


class NormalAdmittanceTest(unittest.TestCase):
    def make_core(self):
        return NormalAdmittance(dt=0.001, release_time=1.0, mass=8.0)

    def test_integrator_is_bounded(self):
        core = self.make_core()
        for index in range(120):
            displacement, velocity, acceleration = core.step(
                0.88 + index * 0.001, estimated_force=-1000.0
            )
            self.assertLessEqual(abs(displacement), 0.0015 + 1e-15)
            self.assertLessEqual(abs(velocity), 0.01 + 1e-15)
            self.assertLessEqual(abs(acceleration), 0.5 + 1e-12)

    def test_reset_clears_all_dynamic_state(self):
        core = self.make_core()
        core.step(0.90, estimated_force=0.0)
        core.step(0.91, estimated_force=0.0)
        core.reset()
        self.assertEqual((core.displacement, core.velocity, core.acceleration), (0.0, 0.0, 0.0))
        self.assertIsNone(core.last_time)
        self.assertIsNone(core.initial_force)
        self.assertIsNone(core.return_coefficients)

    def test_alarm_freezes_extra_downward_motion(self):
        core = self.make_core()
        core.step(0.88, estimated_force=0.0)
        core.step(0.94, estimated_force=0.0)
        before = core.displacement
        displacement, velocity, _ = core.step(0.941, estimated_force=0.0, alarm=True)
        self.assertGreaterEqual(displacement, before)
        self.assertGreaterEqual(velocity, 0.0)

    def test_release_return_is_derivative_consistent(self):
        core = self.make_core()
        for index in range(100):
            core.step(0.88 + index * 0.001, estimated_force=0.0)
        release_state = np.array([core.displacement, core.velocity, core.acceleration])
        at_release = np.array(core.step(1.0, estimated_force=0.0))
        np.testing.assert_allclose(at_release, release_state, atol=1e-12)

        h = 1e-5
        xm = core.step(1.05 - h, estimated_force=0.0)[0]
        x0, v0, a0 = core.step(1.05, estimated_force=0.0)
        xp = core.step(1.05 + h, estimated_force=0.0)[0]
        self.assertAlmostEqual((xp - xm) / (2.0 * h), v0, places=7)
        self.assertAlmostEqual((xp - 2.0 * x0 + xm) / h**2, a0, places=5)
        np.testing.assert_allclose(core.step(1.10, estimated_force=0.0), 0.0, atol=1e-15)

    def test_registered_extreme_exposes_fixed_return_bound_conflict(self):
        """Characterize why independent global bounds plus fixed 0.10 s conflict."""
        coefficients = _quintic_boundary(
            -0.0014, -0.009, -0.4, 0.0, 0.0, 0.0, 0.10
        )
        states = np.array(
            [
                _evaluate_polynomial(coefficients, time)
                for time in np.linspace(0.0, 0.10, 10001)
            ]
        )
        maxima = np.max(np.abs(states), axis=0)
        self.assertGreater(maxima[0], 0.0015)
        self.assertGreater(maxima[1], 0.01)
        self.assertGreater(maxima[2], 0.5)


class NormalDirectionTest(unittest.TestCase):
    def test_jacobian_virtual_power(self):
        jacobian = np.array(
            [
                [1.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.2, 0.0, 0.0, 0.0],
                [0.1, 0.0, 1.1, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.3, 0.1],
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.8],
            ]
        )
        direction = normal_joint_direction(jacobian)
        normal_twist = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(jacobian @ direction, normal_twist, atol=1e-12)

        wrench = np.array([3.0, -2.0, 7.0, 0.4, -0.3, 0.2])
        joint_torque = jacobian.T @ wrench
        self.assertAlmostEqual(
            float(joint_torque @ direction), float(wrench @ normal_twist), places=12
        )


if __name__ == "__main__":
    unittest.main()
