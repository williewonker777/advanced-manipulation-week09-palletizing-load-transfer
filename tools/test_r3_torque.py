#!/usr/bin/env python3
"""Pure unit and mathematical tests for the self-contained R3 torque module."""
import ast
import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import r1_torque
from experiments import r2_robust
from experiments.r2_prne import PRNE
from experiments import r3_torque
from sim import config as C
from sim.nominal import Nominal
from sim import trajectory as T
from tools.production_audit import audit_text


def _controller(cls, mass=5.4):
    controller = cls()
    controller.nom = Nominal()
    controller.nom_box = Nominal()
    controller.box_mass = float(mass)
    controller.box_com = (0.011, -0.008, 0.027)
    controller.nom_box.set_payload(controller.box_mass, controller.box_com)
    controller.reset()
    return controller


def _parameters(schedule):
    return {
        "family": "R3_B", "architecture": "torque",
        "rate": 4.5, "damping_scale": 1.0, "schedule": schedule,
        "mass_start_kg": 3.15, "mass_full_kg": 5.65,
        "handoff_start_s": 3.60, "handoff_end_s": 3.84,
    }


class R3TorqueTests(unittest.TestCase):
    def test_source_is_r1_prefix_and_shared_estimator_ast_is_identical(self):
        r1_text = (ROOT / "r1_torque.py").read_text()
        r3_text = (ROOT / "experiments/r3_torque.py").read_text()
        self.assertTrue(r3_text.startswith(r1_text))
        r1_tree, r3_tree = ast.parse(r1_text), ast.parse(r3_text)
        r1_shared = next(node for node in r1_tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "SharedEstimator")
        r3_shared = next(node for node in r3_tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == "SharedEstimator")
        self.assertEqual(ast.dump(r1_shared, include_attributes=False),
                         ast.dump(r3_shared, include_attributes=False))

    def test_production_audit_and_allowed_imports(self):
        text = (ROOT / "experiments/r3_torque.py").read_text()
        result = audit_text(text, "experiments/r3_torque.py")
        self.assertEqual(result["status"], "pass", result["errors"])
        allowed = {"numpy", "mujoco", "sim"}
        imports = {node.module.split(".")[0]
                   for node in ast.walk(ast.parse(text))
                   if isinstance(node, ast.ImportFrom) and node.module}
        imports.update(alias.name.split(".")[0]
                       for node in ast.walk(ast.parse(text))
                       if isinstance(node, ast.Import) for alias in node.names)
        self.assertLessEqual(imports, allowed)

    def test_import_does_not_replace_unrelated_r1_controller(self):
        original = r1_torque.Controller
        module = importlib.reload(r3_torque)
        self.assertIs(r1_torque.Controller, original)
        self.assertIs(module.Controller.R3_PARAMETERS,
                      module.R3_DEFAULT_PARAMETERS)

    def test_factory_validation_reset_and_stability_state(self):
        for schedule in ("phase", "mass", "mass_phase"):
            cls = r3_torque.make_controller("torque", _parameters(schedule))
            ctrl = _controller(cls)
            self.assertEqual(ctrl.R2_STABILITY_STATE,
                             ("r3_reference_integral", "r3_last_grip"))
            self.assertEqual(ctrl.r3_reference_integral.shape, (6,))
            self.assertIs(type(ctrl.r3_last_grip), bool)
            self.assertTrue(np.all(np.isfinite(ctrl.r3_reserve)))
        with self.assertRaises(ValueError):
            r3_torque.make_controller("servo", _parameters("phase"))
        bad = _parameters("phase")
        bad["rate"] = np.nan
        with self.assertRaises(ValueError):
            r3_torque.make_controller("torque", bad)
        bad = _parameters("unknown")
        with self.assertRaises(ValueError):
            r3_torque.make_controller("torque", bad)

    def test_blended_prne_matches_two_endpoint_actions_and_no_alias(self):
        empty, full = Nominal(), Nominal()
        full.set_payload(6.2, (0.014, -0.012, 0.031))
        optimized = r3_torque._R3BlendedPRNE(empty, full)
        endpoint_empty, endpoint_full = PRNE(empty), PRNE(full)
        rng = np.random.default_rng(369802)
        maximum = 0.0
        saved = None
        for _ in range(8):
            q = rng.uniform(-1.2, 1.2, 6)
            velocity = rng.uniform(-0.9, 0.9, 6)
            design = rng.uniform(-1.0, 1.0, 6)
            a0 = endpoint_empty.action(q, velocity, design)
            a1 = endpoint_full.action(q, velocity, design)
            for load in (0.0, 0.17, 0.53, 0.91, 1.0):
                actual = optimized.action(q, velocity, design, load)
                expected = a0 + load * (a1 - a0)
                maximum = max(maximum,
                              float(np.max(np.abs(actual - expected))))
                if saved is None:
                    saved = actual
                else:
                    snapshot = saved.copy()
                    optimized.action(q + 1e-5, velocity, design, load)
                    np.testing.assert_array_equal(saved, snapshot)
        self.assertLess(maximum, 2e-13)

    def test_endpoint_inverse_dynamics_matches_mass_gravity_formula(self):
        rng = np.random.default_rng(369803)
        maximum = 0.0
        for mass in (0.0, 6.2):
            nominal = Nominal()
            nominal.set_payload(mass, (0.014, -0.012, 0.031))
            for _ in range(10):
                q = rng.uniform(-1.0, 1.0, 6)
                acceleration = rng.uniform(-3.0, 3.0, 6)
                expected = (nominal.mass_matrix(q) @ acceleration
                            + nominal.gravity(q))
                actual = nominal.inverse_dynamics(q, np.zeros(6), acceleration)
                maximum = max(maximum,
                              float(np.max(np.abs(actual - expected))))
        self.assertLess(maximum, 2e-12)

    def test_alpha_one_complete_link_law_matches_r2_pidc(self):
        r2_cls = r2_robust.make_controller("torque", {
            "family": "C4", "law": "P-IDC", "rate": 4.5,
            "damping_scale": 1.0,
        })
        r3_cls = r3_torque.make_controller("torque", _parameters("phase"))
        old, new = _controller(r2_cls), _controller(r3_cls)
        maximum = 0.0
        for k in range(12):
            t = 2.20 + k * C.DT
            q, qd, qdd = T.reference(t)
            th = q + np.linspace(-8e-4, 9e-4, 6)
            thd = qd + np.linspace(4e-3, -3e-3, 6)
            tau_s = np.linspace(-0.4, 0.6, 6)
            old_out = old.compute(t, th, thd, tau_s, True, q, qd, qdd)
            new_out = new.compute(t, th, thd, tau_s, True, q, qd, qdd)
            maximum = max(maximum,
                          float(np.max(np.abs(old_out - new_out))))
        self.assertLess(maximum, 2e-11)

    def test_alpha_zero_is_exact_r1_same_state_fast_path(self):
        r1 = _controller(r1_torque.Controller, mass=5.4)
        cls = r3_torque.make_controller("torque", _parameters("phase"))
        r3 = _controller(cls, mass=5.4)
        for k in range(8):
            t = 4.10 + k * C.DT
            q, qd, qdd = T.reference(t)
            th = q + np.linspace(-5e-4, 4e-4, 6)
            thd = qd + np.linspace(2e-3, -2e-3, 6)
            tau_s = np.linspace(-0.2, 0.3, 6)
            expected = r1.compute(t, th, thd, tau_s, False, q, qd, qdd)
            actual = r3.compute(t, th, thd, tau_s, False, q, qd, qdd)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(r3.tau_link_prev, r1.tau_link_prev)
            np.testing.assert_array_equal(r3.tau_dot, r1.tau_dot)
            np.testing.assert_array_equal(r3.shared_estimator.q_prev,
                                          r1.shared_estimator.q_prev)
            np.testing.assert_array_equal(r3.shared_estimator.v,
                                          r1.shared_estimator.v)
        np.testing.assert_array_equal(r3.r3_reference_integral, np.zeros(6))
        self.assertFalse(r3.r3_last_grip)

    def test_integral_reset_freeze_and_clip(self):
        cls = r3_torque.make_controller("torque", _parameters("mass"))
        ctrl = _controller(cls, mass=5.65)
        q = np.zeros(6)
        baseline = np.zeros(6)
        # Active transport integrates and clips componentwise.
        for _ in range(200):
            ctrl._r3_pidc_link(
                q, q, np.ones(6), q, q, 0.5, baseline, 2.5, True)
        limit = 10.0 * ctrl.r3_position_scale / 4.5
        np.testing.assert_allclose(ctrl.r3_reference_integral, limit)
        # A grip transition resets before the frozen decay, so no stale state.
        ctrl._r3_pidc_link(
            q, q, np.ones(6), q, q, 0.0, baseline, 4.2, False)
        np.testing.assert_array_equal(ctrl.r3_reference_integral, np.zeros(6))
        self.assertFalse(ctrl.r3_last_grip)

    def test_phase_mass_and_product_schedules_have_smooth_knots(self):
        phase_cls = r3_torque.make_controller("torque", _parameters("phase"))
        mass_cls = r3_torque.make_controller("torque", _parameters("mass"))
        product_cls = r3_torque.make_controller("torque", _parameters("mass_phase"))
        phase = _controller(phase_cls, mass=4.4)
        self.assertEqual(phase._r3_alpha(3.59), 1.0)
        self.assertAlmostEqual(phase._r3_alpha(3.72), 0.5, places=14)
        self.assertEqual(phase._r3_alpha(3.85), 0.0)
        h = 1e-5
        for knot in (3.60, 3.84):
            left = (phase._r3_alpha(knot) - phase._r3_alpha(knot - h)) / h
            right = (phase._r3_alpha(knot + h) - phase._r3_alpha(knot)) / h
            self.assertLess(abs(left), 2e-6)
            self.assertLess(abs(right), 2e-6)
        values = []
        for mass in (3.15, 4.4, 5.65):
            ctrl = _controller(mass_cls, mass=mass)
            values.append(ctrl._r3_alpha(2.0))
        np.testing.assert_allclose(values, [0.0, 0.5, 1.0], atol=1e-14)
        for knot in (3.15, 5.65):
            low = _controller(mass_cls, mass=knot - h)._r3_alpha(2.0)
            mid = _controller(mass_cls, mass=knot)._r3_alpha(2.0)
            high = _controller(mass_cls, mass=knot + h)._r3_alpha(2.0)
            self.assertLess(abs((mid - low) / h), 2e-6)
            self.assertLess(abs((high - mid) / h), 2e-6)
        product = _controller(product_cls, mass=4.4)
        self.assertAlmostEqual(product._r3_alpha(3.72), 0.25, places=14)

    def test_default_nominal_compute_is_finite(self):
        ctrl = _controller(r3_torque.Controller)
        q, qd, qdd = T.reference(2.5)
        output = ctrl.compute(2.5, q, qd, np.zeros(6), True,
                              q, qd, qdd)
        self.assertEqual(np.asarray(output).shape, (6,))
        self.assertTrue(np.all(np.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
