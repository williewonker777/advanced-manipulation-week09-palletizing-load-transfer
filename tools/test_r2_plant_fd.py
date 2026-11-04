"""Tiny owned-model tests for the read-only MuJoCo FD linearizer."""

import unittest

import mujoco
import numpy as np

from tools.r2_plant_fd import exact_zoh, linearize_mujoco_acceleration


OSCILLATOR_XML = """
<mujoco model="fd_oscillator">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="arm">
      <joint name="hinge" type="hinge" axis="0 0 1" stiffness="4" damping="0.6"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""


class MuJoCoFiniteDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(OSCILLATOR_XML)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0] = 0.13
        self.data.qvel[0] = -0.2
        self.data.qfrc_applied[0] = 123.0  # must be ignored, not mutated
        mujoco.mj_forward(self.model, self.data)

    def test_known_hinge_oscillator_and_read_only_source(self):
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        applied_before = self.data.qfrc_applied.copy()
        option_before = (self.model.opt.integrator, self.model.opt.solver)
        result = linearize_mujoco_acceleration(
            self.model,
            self.data,
            torque=np.array([0.7]),
            tau_indices=np.array([0]),
            viscous=np.array([0.4]),  # explicitly extra to model-native 0.6
            coulomb=np.array([0.0]),
        )

        full_mass = np.empty((1, 1))
        mujoco.mj_fullM(self.model, self.data, full_mass)
        inertia = full_mass[0, 0]
        self.assertAlmostEqual(result.dacc_dq[0, 0], -4.0 / inertia, places=7)
        self.assertAlmostEqual(result.dacc_dv[0, 0], -1.0 / inertia, places=7)
        self.assertAlmostEqual(result.dacc_du[0, 0], 1.0 / inertia, places=7)
        np.testing.assert_allclose(result.A, [[0.0, 1.0], [-4.0 / inertia, -1.0 / inertia]], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(result.B, [[0.0], [1.0 / inertia]], rtol=1e-7, atol=1e-9)
        self.assertTrue(result.diagnostics["step_converged"])
        self.assertLess(result.diagnostics["max_relative_step_change"], 1e-6)

        np.testing.assert_array_equal(self.data.qpos, qpos_before)
        np.testing.assert_array_equal(self.data.qvel, qvel_before)
        np.testing.assert_array_equal(self.data.qfrc_applied, applied_before)
        self.assertEqual((self.model.opt.integrator, self.model.opt.solver), option_before)

    def test_smooth_coulomb_derivative_and_exact_zoh_export(self):
        self.data.qvel[0] = 0.0
        mujoco.mj_forward(self.model, self.data)
        result = linearize_mujoco_acceleration(
            self.model,
            self.data,
            torque=np.array([0.0]),
            tau_indices=[0],
            coulomb=np.array([0.5]),
            friction_eps=0.25,
        )
        full_mass = np.empty((1, 1))
        mujoco.mj_fullM(self.model, self.data, full_mass)
        inertia = full_mass[0, 0]
        expected_damping = 0.6 + 0.5 / 0.25
        self.assertAlmostEqual(
            result.dacc_dv[0, 0], -expected_damping / inertia, places=6
        )
        Ad, Bd = exact_zoh(result.A, result.B, 0.001)
        self.assertEqual(Ad.shape, (2, 2))
        self.assertEqual(Bd.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(Ad)))


if __name__ == "__main__":
    unittest.main()
