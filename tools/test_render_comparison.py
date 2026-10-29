import unittest

import numpy as np

from tools.render_comparison import SCENARIOS, frame_times, phase, residual_to_wrench, timewarp_segments
from my_detector import SharedEstimator
from sim.nominal import Nominal


class RenderComparisonTests(unittest.TestCase):
    def test_scenario_names_are_stable_and_portable(self):
        self.assertEqual([s["stem"] for s in SCENARIOS], [
            "01_servo_no_collision", "02_torque_no_collision",
            "03_servo_place_collision", "04_servo_transport_collision",
        ])

    def test_timewarp_covers_episode_without_gaps(self):
        for collision in (None, 3.0, 3.86):
            segments = timewarp_segments(collision)
            self.assertEqual(segments[0]["simulation_start_s"], 0.0)
            self.assertEqual(segments[-1]["simulation_end_s"], 6.0)
            self.assertTrue(all(a["simulation_end_s"] == b["simulation_start_s"]
                                for a, b in zip(segments, segments[1:])))
            times = frame_times(collision)
            self.assertTrue((times[1:] > times[:-1]).all())
            self.assertGreaterEqual(times[0], 0.0)
            self.assertLess(times[-1], 6.0)

    def test_pick_place_are_slow_and_transport_collision_is_slow(self):
        segments = timewarp_segments(3.0)
        def speed_at(t):
            return next(x["playback_speed"] for x in segments
                        if x["simulation_start_s"] <= t < x["simulation_end_s"])
        self.assertEqual(speed_at(1.4), 0.35)
        self.assertEqual(speed_at(3.0), 0.35)
        self.assertEqual(speed_at(3.9), 0.35)
        self.assertEqual(speed_at(5.0), 1.30)

    def test_phase_labels(self):
        self.assertEqual(phase(1.0), "접근")
        self.assertEqual(phase(1.5), "집기·상승")
        self.assertEqual(phase(3.0), "운반")
        self.assertEqual(phase(3.8), "놓기 접근")
        self.assertEqual(phase(4.1), "하중 이양·복귀")

    def test_offline_wrench_uses_detector_bottom_centre_jacobian(self):
        detector = SharedEstimator()
        detector.nom = Nominal()
        detector.nom_box = Nominal()
        detector.box_mass = 8.0
        detector.box_com = (0.02, -0.01, 0.03)
        detector.reset()
        qs = np.array([[0.0, -0.4, 0.8, 0.0, 0.5, 0.0],
                       [0.2, -0.6, 1.0, -0.1, 0.3, 0.2]])
        residual = np.array([[1.0, -2.0, 0.5, 0.2, 0.3, -0.1],
                             [-0.5, 1.0, 2.0, -0.2, 0.1, 0.4]])
        got = residual_to_wrench(residual, qs)
        expected = []
        for r, q in zip(residual, qs):
            jacobian, _, _ = detector._point_jacobian(q)
            expected.append(np.linalg.solve(jacobian.T, r))
        np.testing.assert_allclose(got, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
