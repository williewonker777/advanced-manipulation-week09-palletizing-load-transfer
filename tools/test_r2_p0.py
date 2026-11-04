import unittest

import numpy as np

import r1_servo
import r1_torque
from sim import trajectory as T
from sim.drive import Drive
from sim.nominal import Nominal
from tools.r2_p0 import aggregate_boundaries, first_crossing, force_displacement_curve, longest_true_run, mirrored_servo_drive, numeric_state, probe_controller, quintic_shift, spectrum


class R2P0PureTests(unittest.TestCase):
    @staticmethod
    def configured(controller, servo=False):
        controller.nom, controller.nom_box = Nominal(), Nominal()
        controller.nom_box.set_payload(2.6, (0.013, -0.009, 0.019))
        controller.box_mass, controller.box_com = 2.6, (0.013, -0.009, 0.019)
        if servo:
            controller.drive = Drive()
        controller.reset()
        return controller

    def test_quintic_shift_has_fixed_endpoints(self):
        self.assertEqual(quintic_shift(0.9, 1.0, 2.0, 0.003), 0.0)
        self.assertAlmostEqual(quintic_shift(1.5, 1.0, 2.0, 0.003), 0.0015)
        self.assertEqual(quintic_shift(2.1, 1.0, 2.0, 0.003), 0.003)

    def test_first_crossing_is_strict_and_pre_release(self):
        t = np.arange(3.6, 3.906, 0.001)
        force = np.zeros_like(t)
        force[200:] = 2.0
        row = first_crossing(t, force, 1.0)
        self.assertAlmostEqual(row["time_s"], t[200])
        self.assertAlmostEqual(row["remaining_to_release_s"], 3.9 - t[200])

    def test_force_curve_preserves_nonlinear_range(self):
        displacement = np.linspace(0.0, 0.002, 101)
        force = 2e6 * displacement**2
        rows = force_displacement_curve(force, displacement, np.ones(101, bool), bins=5)
        self.assertEqual(len(rows), 5)
        self.assertGreater(rows[-1]["force_mean_n"], 3 * rows[2]["force_mean_n"])
        self.assertEqual(sum(row["samples"] for row in rows), 101)

    def test_spectrum_finds_known_tone(self):
        t = np.arange(2000) * 0.001
        values = np.sin(2 * np.pi * 25 * t)[:, None]
        self.assertAlmostEqual(spectrum(values)["peak_frequency_hz"][0], 25.0)

    def test_boundary_aggregation_counts_measured_rate_exceedance(self):
        rows = []
        for architecture, speeds in (("servo", (0.004, 0.006)), ("torque", (0.009, 0.012))):
            for index, speed in enumerate(speeds):
                rows.append({"architecture": architecture,
                             "contact": {"force_gt_1n": {"remaining_to_release_s": 0.05 + index * 0.01},
                                         "force_50pct_mg_crossing": {
                                             "tcp_displacement_from_1n_contact_m": 0.0002,
                                             "box_bottom_displacement_from_1n_contact_m": 0.00019,
                                             "required_average_tcp_speed_m_s": speed,
                                             "required_average_box_bottom_speed_m_s": speed - 0.0005}}})
        result = aggregate_boundaries(rows)["by_architecture"]
        self.assertEqual(result["servo"]["episodes_above_r1_0p01m_s_tcp_rate"], 0)
        self.assertEqual(result["torque"]["episodes_above_r1_0p01m_s_tcp_rate"], 1)

    def test_boundary_aggregation_preserves_missing_contact_and_crossing(self):
        missing = {"architecture": "servo", "contact": {
            "force_gt_1n": {"remaining_to_release_s": None},
            "force_50pct_mg_crossing": {"time_s": None}}}
        crossing_without_contact = {"architecture": "torque", "contact": {
            "force_gt_1n": {"remaining_to_release_s": None},
            "force_50pct_mg_crossing": {"time_s": 3.88,
                                         "reference_contact_available": False,
                                         "tcp_displacement_from_1n_contact_m": None,
                                         "box_bottom_displacement_from_1n_contact_m": None,
                                         "required_average_tcp_speed_m_s": None,
                                         "required_average_box_bottom_speed_m_s": None}}}
        result = aggregate_boundaries([missing, crossing_without_contact])["by_architecture"]
        self.assertEqual(result["servo"]["contact_remaining_to_release_s_range"], [None, None])
        self.assertEqual(result["servo"]["episodes_without_50pct_mg_crossing"], 1)
        self.assertEqual(result["torque"]["episodes_with_50pct_crossing_but_no_1n_reference"], 1)
        self.assertEqual(result["torque"]["required_average_tcp_speed_to_50pct_mg_m_s_range"],
                         [None, None])

    def test_numeric_state_excludes_runtime_objects(self):
        class Example:
            def __init__(self):
                self.nom = object()
                self.drive = object()
                self.vector = np.array([1.0, 2.0])
                self.counter = 3
                self.other = object()
        state = numeric_state(Example())
        self.assertEqual(state, {"vector": [1.0, 2.0], "counter": 3})

    def test_longest_true_run_reports_contiguous_duration(self):
        t = np.arange(8) * 0.001
        row = longest_true_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool), t)
        self.assertEqual(row, {"duration_s": 0.003, "start_s": 0.004, "end_s": 0.007})

    def test_zero_probe_is_exact_frozen_r1_reference_for_both_architectures(self):
        config = {"start_s": 3.75, "end_s": 3.89, "amplitude_m": 0.0}
        servo_base = self.configured(r1_servo.Controller(), servo=True)
        servo_probe = self.configured(probe_controller(r1_servo.Controller, "servo", config)(), servo=True)
        torque_base = self.configured(r1_torque.Controller())
        torque_probe = self.configured(probe_controller(r1_torque.Controller, "torque", config)())
        for t in (3.75, 3.87, 3.894, 3.9):
            desired = T.reference(t)
            for expected, actual in zip(
                    servo_base._reference(t, *desired), servo_probe._reference(t, *desired), strict=True):
                self.assertTrue(np.array_equal(expected, actual), f"servo mismatch at {t}")
            for expected, actual in zip(
                    torque_base._reference(t), torque_probe._reference(t), strict=True):
                self.assertTrue(np.array_equal(expected, actual), f"torque mismatch at {t}")

    def test_nonzero_probe_adds_finite_perturbation_to_r1_reference(self):
        config = {"start_s": 3.75, "end_s": 3.89, "amplitude_m": 0.0004}
        base = self.configured(r1_servo.Controller(), servo=True)
        probe = self.configured(probe_controller(r1_servo.Controller, "servo", config)(), servo=True)
        desired = T.reference(3.87)
        expected = base._reference(3.87, *desired)
        actual = probe._reference(3.87, *desired)
        self.assertTrue(all(np.all(np.isfinite(value)) for value in actual))
        self.assertGreater(np.linalg.norm(actual[0] - expected[0]), 0.0)
        self.assertGreater(np.linalg.norm(actual[1] - expected[1]), 0.0)

    def test_mirrored_servo_drive_uses_caller_owned_state(self):
        command = np.vstack((np.ones(6) * 0.2, np.zeros(6), np.ones(6) * 0.1,
                             np.ones(6) * 0.3))
        torque, velocity, integral = mirrored_servo_drive(
            command, np.ones(6) * 0.1, np.ones(6) * 0.05,
            np.zeros(6), np.zeros(6))
        self.assertTrue(np.allclose(velocity, 0.01))
        expected_error = 0.1 + 12.0 * 0.1 - 0.01
        self.assertTrue(np.allclose(integral, expected_error * 0.001))
        self.assertTrue(np.allclose(torque, 0.3 + np.array([120, 120, 80, 24, 24, 12]) *
                                    expected_error + np.array([600, 600, 400, 120, 120, 60]) *
                                    expected_error * 0.001))


if __name__ == "__main__":
    unittest.main()
