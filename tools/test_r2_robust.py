"""Pure tests for R2 C3/C4 controller mechanics; no simulator episodes."""
from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from sim import config as C
from sim.nominal import Nominal
from sim import trajectory as T
from experiments.r2_robust import (
    POSITION_SCALE,
    TORQUE_RESERVE,
    _nominal_derivative,
    dob_observer_step,
    implicit_midpoint_step,
    make_controller,
)


class LinearNominal:
    def mass_matrix(self, q):
        del q
        return np.eye(6)

    def bias(self, q, qd):
        del q, qd
        return np.zeros(6)


class TestR2Robust(unittest.TestCase):
    def test_protocol_grid_is_enforced(self):
        with self.assertRaises(ValueError):
            make_controller("torque", {"family": "C3", "law": "DOB", "bandwidth": 7})
        with self.assertRaises(ValueError):
            make_controller("servo", {"family": "C4", "law": "P-IDC",
                                      "rate": 9, "damping_scale": 0.7})

    def test_implicit_midpoint_rest_is_stationary_without_gravity(self):
        state = np.zeros(18)
        out = implicit_midpoint_step(state, np.zeros(6), LinearNominal())
        np.testing.assert_allclose(out, state, atol=1e-15)

    def test_implicit_midpoint_elastic_energy_is_bounded(self):
        state = np.zeros(18)
        state[0] = 1e-3
        initial = state.copy()
        nominal = LinearNominal()
        for _ in range(200):
            state = implicit_midpoint_step(state, np.zeros(6), nominal)
        self.assertTrue(np.all(np.isfinite(state)))
        # Physical damping/friction may remove energy; the implicit update must
        # not have forward-Euler growth by orders of magnitude.
        self.assertLess(np.linalg.norm(state), 20.0 * np.linalg.norm(initial))

    def test_implicit_midpoint_residual(self):
        state = np.zeros(18)
        state[0] = 1e-3
        torque = np.zeros(6)
        out = implicit_midpoint_step(state, torque, LinearNominal())
        residual = out - state - C.DT * _nominal_derivative(
            0.5 * (state + out), torque, LinearNominal()
        )
        self.assertLess(np.linalg.norm(residual, ord=np.inf), 1e-9)

    def test_dob_zero_equilibrium_and_constant_disturbance_convergence(self):
        z = np.zeros(6)
        z, estimate = dob_observer_step(
            z, np.zeros(6), np.zeros(6), np.zeros(6), 18.0
        )
        np.testing.assert_allclose(z, 0.0)
        np.testing.assert_allclose(estimate, 0.0)
        disturbance = np.full(6, 2.0)
        # With u-tau-friction=-d, d_hat tends to d.
        for _ in range(1000):
            z, estimate = dob_observer_step(
                z, np.zeros(6), np.zeros(6), disturbance, 18.0
            )
        np.testing.assert_allclose(estimate, disturbance, atol=1e-7)

    def test_factory_declares_only_output_affecting_state(self):
        expected = {
            ("C3", "NRIC"): "r2_nominal_state",
            ("C3", "DOB"): "r2_dob_z",
            ("C4", "P-IDC"): "r2_reference_integral",
        }
        for (family, law), field in expected.items():
            params = {"family": family, "law": law}
            if law == "NRIC":
                params["gamma"] = 2.0
            elif law == "DOB":
                params["bandwidth"] = 9.0
            else:
                params.update(rate=9.0, damping_scale=1.0)
            cls = make_controller("torque", params)
            self.assertIn(field, cls.R2_STABILITY_STATE)
        apd = make_controller("torque", {
            "family": "C4", "law": "P-APD", "rate": 9.0,
            "damping_scale": 1.0,
        })
        self.assertEqual(apd.R2_STABILITY_STATE, ())

    def test_c3_compute_uses_only_public_controller_inputs(self):
        for mode, params in (
            ("servo", {"family": "C3", "law": "NRIC", "gamma": 2.0}),
            ("torque", {"family": "C3", "law": "DOB", "bandwidth": 9.0}),
        ):
            cls = make_controller(mode, params)
            ctrl = cls()
            ctrl.nom, ctrl.nom_box = Nominal(), Nominal()
            ctrl.nom_box.set_payload(2.3, (0.01, -0.01, 0.02))
            ctrl.box_mass, ctrl.box_com = 2.3, (0.01, -0.01, 0.02)
            ctrl.reset()
            q, qd, qdd = T.reference(2.0)
            shape = (4, 6) if mode == "servo" else (6,)
            for k in range(3):
                out = np.asarray(ctrl.compute(2.0 + k * C.DT, q, qd,
                                              np.zeros(6), 1, q, qd, qdd))
                self.assertEqual(out.shape, shape)
                self.assertTrue(np.all(np.isfinite(out)))

    def test_servo_drive_replay_starts_from_original_q0_pending(self):
        cls = make_controller("servo", {
            "family": "C3", "law": "DOB", "bandwidth": 9.0,
        })
        ctrl = cls()
        ctrl.nom, ctrl.nom_box = Nominal(), Nominal()
        ctrl.box_mass, ctrl.box_com = 0.0, (0.0, 0.0, 0.0)
        ctrl.reset()
        q0 = T.reference(0.0)[0]
        np.testing.assert_allclose(ctrl.r2_pending[0], q0)
        np.testing.assert_allclose(ctrl._new_applied_torque(q0, np.zeros(6)), 0.0,
                                   atol=1e-15)

    def test_nric_nominal_queue_excludes_actual_correction(self):
        for mode in ("servo", "torque"):
            cls = make_controller(mode, {
                "family": "C3", "law": "NRIC", "gamma": 1.0,
            })
            ctrl = cls()
            ctrl.nom, ctrl.nom_box = Nominal(), Nominal()
            ctrl.nom_box.set_payload(2.3, (0.01, -0.01, 0.02))
            ctrl.box_mass, ctrl.box_com = 2.3, (0.01, -0.01, 0.02)
            ctrl.reset()
            q, qd, qdd = T.reference(2.0)
            ctrl.compute(2.0, q, qd, np.zeros(6), 1, q, qd, qdd)
            # Create a nominal-vs-actual deviation before the next active
            # transport sample.  The actual command must get a correction,
            # while nominal_pending remains the uncompensated outer command.
            ctrl.r2_nominal_state[3] += 0.01
            out = np.asarray(ctrl.compute(
                2.001, q, qd, np.zeros(6), 1, q, qd, qdd
            ))
            nominal = np.asarray(ctrl.r2_nominal_pending)
            if mode == "servo":
                np.testing.assert_allclose(out[:3], nominal[:3])
                self.assertGreater(np.max(np.abs(out[3] - nominal[3])), 1e-6)
            else:
                self.assertGreater(np.max(np.abs(out - nominal)), 1e-6)

    def test_c4_uses_prne_action_not_bias_with_design_velocity(self):
        calls = []

        class FakePRNE:
            def __init__(self, nom):
                self.nom = nom

            def action(self, q, actual_velocity, design_velocity):
                calls.append((actual_velocity.copy(), design_velocity.copy()))
                return np.zeros(6)

        old = sys.modules.get("experiments.r2_prne")
        sys.modules["experiments.r2_prne"] = types.SimpleNamespace(PRNE=FakePRNE)
        try:
            cls = make_controller("torque", {
                "family": "C4", "law": "P-IDC", "rate": 4.5,
                "damping_scale": 0.5,
            })
            ctrl = cls()
            ctrl.nom, ctrl.nom_box = Nominal(), Nominal()
            ctrl.nom_box.set_payload(2.3, (0.01, -0.01, 0.02))
            ctrl.box_mass, ctrl.box_com = 2.3, (0.01, -0.01, 0.02)
            ctrl.reset()
            q, qd, qdd = T.reference(2.0)
            out = ctrl.compute(2.0, q + 1e-3, qd, np.zeros(6), 1, q, qd, qdd)
            self.assertEqual(np.asarray(out).shape, (6,))
            self.assertEqual(len(calls), 2)
            self.assertTrue(any(not np.allclose(v, u) for v, u in calls))
        finally:
            if old is None:
                sys.modules.pop("experiments.r2_prne", None)
            else:
                sys.modules["experiments.r2_prne"] = old

    def test_c4_real_prne_and_shared_elastic_lift_are_finite(self):
        # Import the real parallel-lane pRNE implementation and exercise both
        # runner output contracts without advancing a Plant episode.
        for mode in ("servo", "torque"):
            cls = make_controller(mode, {
                "family": "C4", "law": "P-APD", "rate": 4.5,
                "damping_scale": 0.5,
            })
            ctrl = cls()
            ctrl.nom, ctrl.nom_box = Nominal(), Nominal()
            ctrl.nom_box.set_payload(2.3, (0.01, -0.01, 0.02))
            ctrl.box_mass, ctrl.box_com = 2.3, (0.01, -0.01, 0.02)
            ctrl.reset()
            q, qd, qdd = T.reference(2.0)
            out = np.asarray(ctrl.compute(
                2.0, q + 1e-3, qd, np.zeros(6), 1, q, qd, qdd
            ))
            self.assertEqual(out.shape, (4, 6) if mode == "servo" else (6,))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_p0_constants_are_finite_and_positive(self):
        for mode in ("servo", "torque"):
            self.assertTrue(np.all(TORQUE_RESERVE[mode] > 0.0))
            self.assertTrue(np.all(POSITION_SCALE[mode] > 0.0))


if __name__ == "__main__":
    unittest.main()
