"""Focused regression tests for the self-contained D3 detector."""

import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity
check_integrity(ROOT)

import my_detector
from sim.nominal import Nominal


def detector():
    d = my_detector.Detector()
    d.nom = Nominal()
    d.nom_box = Nominal()
    d.nom_box.set_payload(8.0, (0.02, -0.01, 0.03))
    d.box_mass, d.box_com = 8.0, (0.02, -0.01, 0.03)
    d.reset()
    return d


def test_production_imports_and_forbidden_names():
    path = Path(my_detector.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert roots <= {"numpy", "mujoco", "sim"}
    text = path.read_text(encoding="utf-8")
    for forbidden in ("JTS_OFFSET", "JTS_DRIFT", "JTS_SEED", "Plant", "sim.runner", "table_force"):
        assert forbidden not in text


def test_shared_core_template_matches_production_ast():
    production = ast.parse(Path(my_detector.__file__).read_text(encoding="utf-8"))
    template = ast.parse((ROOT / "tools" / "shared_estimator_core.py").read_text(encoding="utf-8"))
    production_core = next(
        node for node in production.body if isinstance(node, ast.ClassDef) and node.name == "SharedEstimator"
    )
    template_core = next(
        node for node in template.body if isinstance(node, ast.ClassDef) and node.name == "SharedEstimator"
    )
    assert ast.dump(production_core, include_attributes=False) == ast.dump(
        template_core, include_attributes=False
    )


def test_shared_core_is_timestamp_idempotent_and_adapter_reuses_it():
    nom, nom_box = Nominal(), Nominal()
    nom_box.set_payload(4.0, (0.02, -0.01, 0.03))
    core = my_detector.SharedEstimator(nom, nom_box, 4.0, (0.02, -0.01, 0.03))
    q = np.zeros(6)
    packet = core.update(0.1, q, q, q, 0)
    residual = packet.r_joint.copy()
    integral = packet.integral.copy()
    assert core.update(0.1, np.ones(6), np.ones(6), np.ones(6), 1) is packet
    assert np.array_equal(core.r_joint, residual)
    assert np.array_equal(core.integral, integral)

    d = my_detector.Detector()
    d.nom, d.nom_box, d.box_mass = nom, nom_box, 4.0
    d.ctrl = type("ControllerWithSharedCore", (), {"shared_estimator": core})()
    d.reset()
    returned, detected = d.update(0.1, q, q, q, q, 0)
    assert np.array_equal(returned, core.r_joint)
    assert detected == core.detected
    assert d.last_update_t == core.last_update_t


def test_gate_requires_grip_and_taught_height():
    d = detector()
    z = float(d.table_heights[0])
    assert not d._table_allowed(False, z)
    assert d._table_allowed(True, z)
    d.box_mass = 0.0
    assert not d._table_allowed(True, z)
    d.box_mass = 8.0
    assert not d._table_allowed(True, float(np.max(d.table_heights)) + d.height_band + 0.01)


def test_world_unilateral_projection_and_rotated_footprint():
    d = detector()
    yaw = 0.7
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    rx_pi = np.diag([1.0, -1.0, -1.0])
    yaw_rotation = rz @ rx_pi
    tilt = 0.18
    angle = np.pi + tilt
    tilt_rotation = rz @ np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )
    for rotation in (yaw_rotation, tilt_rotation):
        assert np.isclose(np.linalg.det(rotation), 1.0)
        assert np.allclose(rotation.T @ rotation, np.eye(3))
        wrench = np.array([20.0, -10.0, 100.0, 30.0, -30.0, 9.0])
        projected = d._project_table(wrench, rotation)
        assert projected[2] == 100.0
        assert np.linalg.norm(projected[:2]) <= d.friction_mu * projected[2] + 1e-12
        f = projected[:3]
        skew = np.array([[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]])
        mapping = -skew @ rotation[:, :2]
        # Remove the allowed world-normal torsion before recovering the COP.
        best = None
        for torsion in np.linspace(-d.torsion_radius * f[2], d.torsion_radius * f[2], 401):
            cop = np.linalg.lstsq(mapping, projected[3:] - [0, 0, torsion], rcond=1e-8)[0]
            err = np.linalg.norm(mapping @ cop + [0, 0, torsion] - projected[3:])
            if best is None or err < best[0]:
                best = err, cop
        assert best[0] < 1e-2
        assert abs(best[1][0]) <= d.half_x + 1e-3
        assert abs(best[1][1]) <= d.half_y + 1e-3
        negative = d._project_table(np.array([0, 0, -20, 0, 0, 0.0]), rotation)
        assert np.allclose(negative, 0.0)


def test_alarm_persistence_and_release_are_stable():
    class ScriptedAlarmDetector(my_detector.Detector):
        scripted_over = False

        def _observer(self, q, v, tau, nom, grip):
            self.r_joint[:] = 0.0
            if self.integral is None:
                self.integral = np.zeros(6)

        def _point_jacobian(self, q):
            return np.eye(6), np.array([0.0, 0.0, 10.0]), np.eye(3)

        def _directional_alarm(self, residual, raw_wrench, table_allowed):
            return self.scripted_over, 2.0 if self.scripted_over else 0.0

    d = ScriptedAlarmDetector()
    d.nom = Nominal(); d.nom_box = Nominal(); d.reset()
    zero = np.zeros(6)
    d.scripted_over = True
    for k in range(d.persistence - 1):
        assert not d.update(0.1 + k * 0.001, zero, zero, zero, zero, 0)[1]
    assert d.update(0.2, zero, zero, zero, zero, 0)[1]
    d.scripted_over = False
    for k in range(d.release_persistence - 1):
        assert d.update(0.3 + k * 0.001, zero, zero, zero, zero, 0)[1]
    assert not d.update(0.4, zero, zero, zero, zero, 0)[1]


def test_fast_channel_is_grip_off_only_and_uses_production_persistence():
    class ScriptedFastDetector(my_detector.Detector):
        scripted_wrench = np.zeros(6)

        def reset(self):
            super().reset()
            self.fast_force_threshold[:] = 1.0
            self.fast_moment_threshold[:] = 1.0
            self.fast_positive_fz_threshold = 1.0
            self.fast_persistence = 3

        def _observer(self, q, v, tau, nom, grip):
            self.r_joint[:] = self.scripted_wrench
            self.last_grip = bool(grip)
            if self.integral is None:
                self.integral = np.zeros(6)

        def _point_jacobian(self, q):
            return np.eye(6), np.array([0.0, 0.0, 10.0]), np.eye(3)

        def _directional_alarm(self, residual, raw_wrench, table_allowed):
            del residual, raw_wrench, table_allowed
            return False, 0.0

    d = ScriptedFastDetector()
    d.nom = Nominal(); d.nom_box = Nominal(); d.box_mass = 8.0; d.reset()
    zero = np.zeros(6)
    assert not d.update(0.1, zero, zero, zero, zero, 0)[1]
    d.scripted_wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert not d.update(0.2, zero, zero, zero, zero, 0)[1]
    assert not d.update(0.3, zero, zero, zero, zero, 0)[1]
    assert d.update(0.4, zero, zero, zero, zero, 0)[1]

    d.reset()
    d.scripted_wrench[:] = 0.0
    assert not d.update(0.5, zero, zero, zero, zero, 1)[1]
    d.scripted_wrench[0] = 10.0
    for k in range(5):
        assert not d.update(0.6 + 0.01 * k, zero, zero, zero, zero, 1)[1]
        assert d.fast_alarm_count == 0


def test_bias_tracking_freezes_for_grip_suspicion_and_alarm():
    d = detector()
    d.integral = np.zeros(6)
    d.r_joint[:] = 1.0
    original = d.bias.copy()
    d._track_bias(True, False)
    assert np.array_equal(d.bias, original)
    d._track_bias(False, True)
    assert np.array_equal(d.bias, original)
    d.detected = True
    d._track_bias(False, False)
    assert np.array_equal(d.bias, original)
    d.detected = False
    d._track_bias(False, False)
    assert np.all(d.bias < original)


def load_tests(loader, standard_tests, pattern):
    """Include the dependency-free functional regressions in unittest discovery."""
    del loader, pattern
    suite = unittest.TestSuite(standard_tests)
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
