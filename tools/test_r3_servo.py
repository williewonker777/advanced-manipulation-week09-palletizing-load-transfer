"""Pure R3-A servo/DOB regressions; no physics episodes."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from sim import config as C
from sim.drive import Drive
from sim.nominal import Nominal
from sim import trajectory as T
from tools.production_audit import audit_text
from experiments import r3_servo as R3
from experiments.r2_robust import dob_observer_step as r2_dob_observer_step
import r1_servo as R1


ROOT = Path(__file__).resolve().parents[1]


def _models(controller, mass=2.3):
    controller.nom = Nominal()
    controller.nom_box = Nominal()
    controller.box_mass = float(mass)
    controller.box_com = (0.01, -0.01, 0.02)
    controller.nom_box.set_payload(controller.box_mass, controller.box_com)


def _isolated_module():
    path = ROOT / "experiments" / "r3_servo.py"
    spec = importlib.util.spec_from_file_location("unrelated_r3_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestR3Servo(unittest.TestCase):
    def test_source_is_frozen_r1_prefix_and_shared_estimator_ast_matches(self):
        r1_text = (ROOT / "r1_servo.py").read_text(encoding="utf-8")
        r3_text = (ROOT / "experiments" / "r3_servo.py").read_text(
            encoding="utf-8"
        )
        self.assertTrue(r3_text.startswith(r1_text))
        r1_tree, r3_tree = ast.parse(r1_text), ast.parse(r3_text)

        def shared(tree):
            return next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "SharedEstimator"
            )

        self.assertEqual(
            ast.dump(shared(r3_tree), include_attributes=False),
            ast.dump(shared(r1_tree), include_attributes=False),
        )
        compute = next(
            node
            for node in ast.walk(r3_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "compute"
            and node.lineno > len(r1_text.splitlines())
        )
        super_compute_calls = [
            node
            for node in ast.walk(compute)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compute"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ]
        self.assertEqual(len(super_compute_calls), 1)

    def test_whole_module_passes_production_audit(self):
        path = ROOT / "experiments" / "r3_servo.py"
        result = audit_text(path.read_text(encoding="utf-8"), str(path))
        self.assertEqual(result["status"], "pass", result["errors"])

    def test_default_parameters_and_factory_contract(self):
        self.assertEqual(R3.MODE, "servo")
        self.assertIs(R3.Controller.R3_PARAMETERS, R3.R3_DEFAULT_PARAMETERS)
        self.assertEqual(R3.R3_DEFAULT_PARAMETERS["id"], "A_handoff_long")
        for params in (R3.R3_DEFAULT_PARAMETERS, R3.R3_SHORT_PARAMETERS):
            cls = R3.make_controller("servo", params)
            self.assertEqual(cls.R3_PARAMETERS["fade_start_s"], params["fade_start_s"])
        redesign = dict(R3.R3_DEFAULT_PARAMETERS)
        redesign.update(id="A_redesign", fade_start_s=3.20, fade_end_s=3.58)
        self.assertEqual(
            R3.make_controller("servo", redesign).R3_PARAMETERS["id"], "A_redesign"
        )
        with self.assertRaises(ValueError):
            R3.make_controller("torque", R3.R3_DEFAULT_PARAMETERS)
        invalid = dict(redesign, fade_start_s=T.T_GRASP)
        with self.assertRaises(ValueError):
            R3.make_controller("servo", invalid)

    def test_quintic_gain_endpoints_and_c2_join(self):
        start, end = 3.10, 3.60
        self.assertEqual(R3._r3_handoff_gain(start - 1.0, start, end), 1.0)
        self.assertEqual(R3._r3_handoff_gain(start, start, end), 1.0)
        self.assertEqual(R3._r3_handoff_gain(end, start, end), 0.0)
        self.assertEqual(R3._r3_handoff_gain(end + 1.0, start, end), 0.0)
        h = 1e-5
        for knot, outside in ((start, 1.0), (end, 0.0)):
            values = np.array(
                [R3._r3_handoff_gain(knot + k * h, start, end) for k in (-2, -1, 0, 1, 2)]
            )
            first = (values[3] - values[1]) / (2.0 * h)
            second = (values[3] - 2.0 * values[2] + values[1]) / h**2
            self.assertAlmostEqual(values[2], outside, places=14)
            self.assertLess(abs(first), 1e-7)
            self.assertLess(abs(second), 1e-3)

    def test_exact_dob_update_matches_frozen_r2_math(self):
        z = np.array([0.3, -0.2, 0.1, -0.4, 0.5, -0.6])
        velocity = np.array([0.2, -0.1, 0.3, -0.2, 0.15, -0.05])
        applied = np.array([1.0, 2.0, -1.0, 0.5, -0.4, 0.2])
        sensed = np.array([0.5, 1.4, -0.7, 0.1, -0.2, 0.3])
        rate = 4.5
        p = rate * C.B * velocity
        friction = C.VISCOUS * velocity + C.COULOMB * np.tanh(
            velocity / C.FRICTION_EPS
        )
        decay = np.exp(-rate * C.DT)
        expected_z = decay * z - (1.0 - decay) * (
            p + applied - sensed - friction
        )
        actual_z, disturbance = R3.dob_observer_step(
            z, velocity, applied, sensed, rate
        )
        r2_z, r2_disturbance = r2_dob_observer_step(
            z, velocity, applied, sensed, rate
        )
        np.testing.assert_allclose(actual_z, expected_z, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(disturbance, expected_z + p, rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(actual_z, r2_z)
        np.testing.assert_array_equal(disturbance, r2_disturbance)

    def test_mirrored_drive_queue_matches_public_drive(self):
        controller = R3.Controller()
        _models(controller)
        controller.reset()
        command = np.array(
            [
                [0.1, -0.2, 0.05, 0.02, -0.03, 0.04],
                np.zeros(6),
                [0.03, -0.02, 0.01, 0.0, 0.02, -0.01],
                [0.4, -0.3, 0.2, 0.1, -0.2, 0.3],
            ]
        )
        th = np.array([0.09, -0.19, 0.04, 0.01, -0.02, 0.03])
        thd = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.0])
        velocity = np.array([0.01, 0.0, -0.01, 0.02, -0.02, 0.01])
        integral = np.array([0.001, -0.002, 0.003, 0.0, -0.001, 0.002])
        controller.r3_pending = command.copy()
        controller.r3_drive_velocity = velocity.copy()
        controller.r3_drive_integral = integral.copy()
        drive = Drive()
        drive.v = velocity.copy()
        drive.I = integral.copy()
        expected = drive.torque(command, th, thd)
        actual = controller._r3_new_applied_torque(th, thd)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(controller.r3_drive_velocity, drive.v)
        np.testing.assert_allclose(controller.r3_drive_integral, drive.I)

    def test_reset_finite_no_external_detector_and_unrelated_import(self):
        module = _isolated_module()
        controller = module.Controller()
        _models(controller)
        controller.reset()
        expected_fields = (
            "r3_dob_z",
            "r3_pending",
            "r3_applied_previous",
            "r3_drive_velocity",
            "r3_drive_integral",
        )
        self.assertEqual(controller.R2_STABILITY_STATE, expected_fields)
        for name in expected_fields:
            self.assertTrue(np.all(np.isfinite(getattr(controller, name))))
        q, qd, qdd = T.reference(2.0)
        output = np.asarray(
            controller.compute(2.0, q, qd, np.zeros(6), True, q, qd, qdd)
        )
        self.assertEqual(output.shape, (4, 6))
        self.assertTrue(np.all(np.isfinite(output)))
        controller.reset()
        np.testing.assert_allclose(controller.r3_dob_z, 0.0)

    def test_observer_continues_during_fade_then_original_gate_tracks_p(self):
        baseline, controller = R1.Controller(), R3.Controller()
        _models(baseline)
        _models(controller)
        baseline.reset()
        controller.reset()
        q, qd, qdd = T.reference(3.40)
        controller.r3_dob_z = np.linspace(-0.3, 0.3, 6)
        before = controller.r3_dob_z.copy()
        expected, _ = R3.dob_observer_step(
            before,
            qd,
            controller.r3_applied_previous,
            np.zeros(6),
            4.5,
        )
        base_output = np.asarray(
            baseline.compute(3.40, q, qd, np.zeros(6), True, q, qd, qdd)
        )
        output = np.asarray(
            controller.compute(3.40, q, qd, np.zeros(6), True, q, qd, qdd)
        )
        np.testing.assert_allclose(controller.r3_dob_z, expected, atol=1e-15)
        disturbance = expected + 4.5 * C.B * qd
        expected_correction = R3._r3_handoff_gain(3.40, 3.10, 3.60) * np.clip(
            -disturbance, -R3._R3_TORQUE_RESERVE, R3._R3_TORQUE_RESERVE
        )
        np.testing.assert_allclose(
            output[3] - base_output[3], expected_correction, atol=1e-15
        )
        q, qd, qdd = T.reference(3.60)
        controller.compute(3.60, q, qd, np.zeros(6), True, q, qd, qdd)
        np.testing.assert_allclose(controller.r3_dob_z, -4.5 * C.B * qd, atol=1e-15)

    def test_zero_gain_same_state_matches_r1_output(self):
        baseline, candidate = R1.Controller(), R3.Controller()
        _models(baseline)
        _models(candidate)
        baseline.reset()
        candidate.reset()
        q, qd, qdd = T.reference(3.60)
        args = (3.60, q, qd, np.zeros(6), True, q, qd, qdd)
        expected = np.asarray(baseline.compute(*args))
        actual = np.asarray(candidate.compute(*args))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(candidate.tau_link_prev, baseline.tau_link_prev)
        np.testing.assert_allclose(candidate.tau_dot, baseline.tau_dot)

    def test_phase_specific_stability_state(self):
        controller = R3.Controller()
        _models(controller)
        controller.reset()
        self.assertEqual(
            controller.r2_stability_state(3.20, True),
            controller.R2_STABILITY_STATE,
        )
        self.assertEqual(controller.r2_stability_state(3.60, True), ())
        self.assertEqual(controller.r2_stability_state(2.50, False), ())
        controller.shared_estimator.detected = True
        self.assertEqual(controller.r2_stability_state(3.20, True), ())


if __name__ == "__main__":
    unittest.main()
