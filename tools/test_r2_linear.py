"""Synthetic, simulation-free tests for tools.r2_linear."""

import unittest

import numpy as np

from tools import r2_linear as R


class ExactZohTests(unittest.TestCase):
    def test_harmonic_oscillator(self):
        w, dt = 7.0, 0.013
        A = np.array([[0.0, 1.0], [-w * w, 0.0]])
        Ad, _ = R.exact_zoh(A, np.zeros((2, 0)), dt)
        expected = np.array(
            [[np.cos(w * dt), np.sin(w * dt) / w],
             [-w * np.sin(w * dt), np.cos(w * dt)]]
        )
        np.testing.assert_allclose(Ad, expected, atol=2e-14, rtol=2e-14)

    def test_semigroup(self):
        A = np.array([[0.2, 2.0], [-3.0, -0.4]])
        B = np.array([[0.3], [1.1]])
        Ad, Bd = R.exact_zoh(A, B, 0.02)
        Ah, Bh = R.exact_zoh(A, B, 0.01)
        np.testing.assert_allclose(Ad, Ah @ Ah, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(Bd, Ah @ Bh + Bh, atol=2e-14, rtol=2e-14)

    def test_constant_input_known_solution_and_refinement(self):
        a, b, dt = -2.5, 3.0, 0.2
        Ad, Bd = R.exact_zoh(np.array([[a]]), np.array([[b]]), dt)
        self.assertAlmostEqual(Ad[0, 0], np.exp(a * dt), places=14)
        self.assertAlmostEqual(Bd[0, 0], b * np.expm1(a * dt) / a, places=14)
        x = np.array([0.7])
        u = np.array([1.2])
        fine_A, fine_B = R.exact_zoh(np.array([[a]]), np.array([[b]]), dt / 100)
        refined = x.copy()
        for _ in range(100):
            refined = fine_A @ refined + fine_B @ u
        np.testing.assert_allclose(Ad @ x + Bd @ u, refined, atol=2e-14, rtol=2e-14)


class PlantTests(unittest.TestCase):
    def test_coordinate_matrices_and_no_elastic_armature_in_link_mass(self):
        M = np.diag(np.arange(1.0, 7.0))
        Bm = np.arange(0.1, 0.7, 0.1)
        plant = R.build_elastic_plant(
            M, Bm, [100.0, 200.0, 300.0], [1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(plant.H[:3, :3], np.diag(Bm[:3]))
        np.testing.assert_allclose(
            plant.H[3:, 3:], M + np.diag([0.0, 0.0, 0.0, *Bm[3:]])
        )
        np.testing.assert_allclose(plant.L @ np.r_[np.ones(3), np.ones(6)], 0.0)
        self.assertEqual(plant.G.shape, (9, 6))
        self.assertEqual(plant.A.shape, (18, 18))
        self.assertEqual(plant.B.shape, (18, 6))

    def test_stiffness_gravity_contact_and_friction_derivatives(self):
        M = np.eye(6)
        J_link = np.zeros((1, 6))
        J_link[0, 1] = 2.0
        J = np.hstack((np.zeros((1, 3)), J_link))
        gravity = np.diag(np.arange(1.0, 7.0))
        bias_velocity = np.diag(np.arange(0.1, 0.7, 0.1))
        plant = R.build_elastic_plant(
            M,
            np.ones(6),
            np.zeros(6),
            np.zeros(6),
            viscous=np.ones(6),
            coulomb=np.ones(6) * 2.0,
            velocity=np.zeros(6),
            friction_eps=0.5,
            gravity_derivative=gravity,
            bias_velocity_derivative=bias_velocity,
            contact_jacobian=J_link,
            contact_stiffness=3.0,
            contact_damping=4.0,
        )
        self.assertAlmostEqual(plant.K[4, 4], gravity[1, 1] + 12.0)
        expected_friction = plant.G @ np.diag(np.full(6, 5.0)) @ plant.G.T
        expected_damping = expected_friction + J.T @ np.array([[4.0]]) @ J
        expected_damping[3:, 3:] += bias_velocity
        np.testing.assert_allclose(plant.D, expected_damping)

    def test_true_jts_elastic_and_rigid_definitions(self):
        plant = R.build_elastic_plant(
            np.eye(6), np.ones(6), [10.0, 20.0, 30.0], [1.0, 2.0, 3.0]
        )
        x = np.zeros(18)
        x[:3] = [0.2, 0.3, 0.4]
        x[3:6] = [0.1, 0.1, 0.1]
        x[9:12] = [0.05, 0.06, 0.07]
        x[12:15] = [0.01, 0.02, 0.03]
        u = np.arange(1.0, 7.0)
        tau = R.true_joint_torque(
            plant, x, u,
            stiffness=[10.0, 20.0, 30.0], joint_damping=[1.0, 2.0, 3.0],
            motor_inertia=np.ones(6), viscous=np.zeros(6), coulomb=np.zeros(6),
        )
        np.testing.assert_allclose(tau[:3], [1.04, 4.08, 9.12])
        acceleration = plant.A @ x + plant.B @ u
        np.testing.assert_allclose(tau[3:], u[3:] - acceleration[15:18])


class OrderingAndControllerTests(unittest.TestCase):
    def test_jts_filter_then_delay(self):
        measurement, filt, delayed = R.jts_sensor_step(
            np.ones(6), np.zeros(6), np.full(6, 7.0), cutoff=10.0, dt=0.01
        )
        np.testing.assert_allclose(measurement, 7.0)
        expected = 1.0 - np.exp(-2.0 * np.pi * 10.0 * 0.01)
        np.testing.assert_allclose(filt, expected)
        np.testing.assert_allclose(delayed, filt)

    def test_drive_uses_updated_filter_and_integral_vel_row_unused(self):
        cmd = np.zeros((4, 6))
        cmd[0] = 1.0
        cmd[1] = 999.0
        torque, vf, integ = R.servo_drive_step(
            cmd, np.zeros(6), np.ones(6), np.zeros(6), np.zeros(6),
            kp=2.0, kv=np.ones(6) * 3.0, ki=np.ones(6) * 5.0,
            filter_coefficient=0.2, dt=0.1,
        )
        np.testing.assert_allclose(vf, 0.2)
        np.testing.assert_allclose(integ, 0.18)
        np.testing.assert_allclose(torque, 3.0 * 1.8 + 5.0 * 0.18)

    def test_torque_runner_applies_old_pending_command(self):
        plant = R.build_elastic_plant(
            np.eye(6), np.ones(6), np.zeros(3), np.zeros(3)
        )
        Ad, Bd = R.exact_zoh(plant.A, plant.B, 0.001)
        old = np.arange(1.0, 7.0)
        new = np.arange(11.0, 17.0)
        state = R.RunnerState(
            plant=np.zeros(18), controller=np.array([2.0]), pending_command=old,
            drive_velocity_filter=np.zeros(6), drive_integral=np.zeros(6),
            jts_filter=np.zeros(6), jts_delay=np.zeros(6), applied_torque=np.zeros(6),
            encoder_previous=np.zeros(6),
        )

        def controller(xc, observation, reference, parameters):
            self.assertEqual(observation.shape, (18,))
            return xc + 1.0, new

        next_state = R.exact_runner_step(
            state, controller, plant, Ad=Ad, Bd=Bd, mode="torque",
            stiffness=np.zeros(3), joint_damping=np.zeros(3),
            motor_inertia=np.ones(6), viscous=np.zeros(6), coulomb=np.zeros(6),
        )
        np.testing.assert_allclose(next_state.plant, Bd @ old)
        np.testing.assert_allclose(next_state.pending_command, new)
        np.testing.assert_allclose(next_state.applied_torque, old)
        np.testing.assert_allclose(next_state.controller, 3.0)

    def test_servo_runner_uses_current_encoder_with_old_command_only(self):
        plant = R.build_elastic_plant(
            np.eye(6), np.ones(6), np.zeros(3), np.zeros(3)
        )
        Ad, Bd = R.exact_zoh(plant.A, plant.B, 0.001)
        old = np.zeros((4, 6))
        old[0] = 1.0
        new = np.zeros((4, 6))
        new[0] = 9.0
        x = np.zeros(18)
        # Current encoder is zero; previous=-1 mrad, so sampled thd=1 rad/s.
        state = R.RunnerState(
            plant=x, controller=np.zeros(0), pending_command=old,
            drive_velocity_filter=np.zeros(6), drive_integral=np.zeros(6),
            jts_filter=np.zeros(6), jts_delay=np.zeros(6), applied_torque=np.zeros(6),
            encoder_previous=np.full(6, -0.001),
        )

        def controller(xc, observation, reference, parameters):
            np.testing.assert_allclose(observation[6:12], 1.0)
            return xc, new

        next_state = R.exact_runner_step(
            state, controller, plant, Ad=Ad, Bd=Bd, mode="servo",
            stiffness=np.zeros(3), joint_damping=np.zeros(3),
            motor_inertia=np.ones(6), viscous=np.zeros(6), coulomb=np.zeros(6),
            drive_kp=2.0, drive_kv=np.full(6, 3.0), drive_ki=np.zeros(6),
        )
        # vf_next=0.2*sampled_thd=0.2, so torque=3*(2*(1-0)-0.2)=5.4.
        np.testing.assert_allclose(next_state.applied_torque, 5.4)
        np.testing.assert_allclose(next_state.plant, Bd @ np.full(6, 5.4))
        np.testing.assert_allclose(next_state.pending_command, new)
        np.testing.assert_allclose(next_state.encoder_previous, 0.0)

    def test_open_plant_measurement_and_step_apis(self):
        plant = R.build_elastic_plant(
            np.eye(6), np.ones(6), np.zeros(3), np.zeros(3)
        )
        Ad, Bd = R.exact_zoh(plant.A, plant.B)
        x, u = np.zeros(18), np.arange(1.0, 7.0)
        observation, filt, delay, ideal = R.plant_measurement_step(
            plant, x, np.zeros(6), np.zeros(6), np.full(6, 8.0),
            stiffness=np.zeros(3), joint_damping=np.zeros(3),
            motor_inertia=np.ones(6), viscous=np.zeros(6), coulomb=np.zeros(6),
        )
        np.testing.assert_allclose(observation[12:], 8.0)
        np.testing.assert_allclose(filt, delay)
        self.assertEqual(ideal.shape, (6,))
        np.testing.assert_allclose(R.plant_step(Ad, Bd, x, u), Bd @ u)
        contract = R.controller_adapter_contract(5, "servo")
        self.assertEqual(contract["controller_state_size"], 5)
        self.assertIn("current sampled encoder", contract["queue_ordering"])

    def test_state_schema_round_trip(self):
        state = R.RunnerState(
            plant=np.arange(18.0), controller=np.arange(4.0),
            pending_command=np.arange(24.0).reshape(4, 6),
            drive_velocity_filter=np.arange(6.0), drive_integral=np.arange(6.0) + 1,
            jts_filter=np.arange(6.0) + 2, jts_delay=np.arange(6.0) + 3,
            applied_torque=np.arange(6.0) + 4,
            encoder_previous=np.arange(6.0) + 5,
        )
        flat = R.pack_runner_state(state, "servo")
        recovered = R.unpack_runner_state(flat, 4, "servo")
        np.testing.assert_allclose(recovered.pending_command, state.pending_command)
        np.testing.assert_allclose(recovered.controller, state.controller)
        np.testing.assert_allclose(recovered.encoder_previous, state.encoder_previous)
        self.assertEqual(flat.size, R.runner_state_sizes(4, "servo")["size"])


class LinearizationAndAnalysisTests(unittest.TestCase):
    def test_central_difference_full_controller_state(self):
        A = np.array([[0.8, 0.2], [-0.1, 0.7]])
        B = np.array([[1.0], [2.0]])
        result = R.linearize_discrete_map(
            lambda x, u: A @ x + B @ u,
            np.array([0.4, -0.3]), np.array([0.2]), state_step=1e-5, input_step=1e-5,
        )
        np.testing.assert_allclose(result.A, A, atol=1e-10)
        np.testing.assert_allclose(result.B, B, atol=1e-10)
        self.assertLess(result.diagnostics["A_relative_step_change"], 1e-10)

    def test_delayed_feedback_state_space(self):
        Ad, Bd = np.array([[0.9]]), np.array([[1.0]])
        feedback = R.FeedbackStateSpace(
            Ac=np.array([[0.5]]), Bc=np.array([[0.2]]),
            Cc=np.array([[0.3]]), Dc=np.array([[-0.4]]),
        )
        Acl = R.build_delayed_feedback_closed_loop(
            Ad, Bd, np.array([[2.0]]), np.array([[0.1]]), feedback
        )
        expected = np.array(
            [[0.9, 0.0, 1.0, 0.0],
             [0.4, 0.5, 0.0, 0.02],
             [-0.8, 0.3, 0.0, -0.04],
             [0.0, 0.0, 1.0, 0.0]]
        )
        np.testing.assert_allclose(Acl, expected)

    def test_poles_and_structured_sampled_margins(self):
        def builder(scales):
            return np.diag(0.5 + 0.1 * np.asarray(scales))

        report = R.analyze_closed_loop(builder(np.ones(2)), margin_builder=builder, axes=2)
        self.assertTrue(report["stable"])
        self.assertAlmostEqual(report["spectral_radius"], 0.6)
        self.assertEqual(report["gain_margins"]["scalar"]["upper"]["status"], "sampled_boundary")
        self.assertIn("N/A", report["feedforward_margin"])
        self.assertIn("not a proof", report["gain_margins"]["qualification"])


if __name__ == "__main__":
    unittest.main()
