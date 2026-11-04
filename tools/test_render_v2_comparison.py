import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import render_v2_comparison as R


class RenderV2ComparisonTests(unittest.TestCase):
    def test_scenarios_and_combinations_are_stable(self):
        self.assertEqual([row["stem"] for row in R.SCENARIOS], [
            "05_v2_servo_transport_collision",
            "06_v2_torque_transport_collision",
        ])
        self.assertEqual([row["mass"] for row in R.SCENARIOS], [3.7, 3.7])
        self.assertEqual(R.combinations("servo"), {
            "baseline": ("servo_controller", "collision_detector"),
            "v1": ("v1_servo", "v1_detector"),
            "v2": ("my_servo", "my_detector"),
        })
        self.assertEqual(R.combinations("torque"), {
            "baseline": ("torque_controller", "collision_detector"),
            "v1": ("v1_torque", "v1_detector"),
            "v2": ("my_torque", "my_detector"),
        })

    def test_v1_and_v2_controllers_are_frozen_identical_sources(self):
        self.assertEqual(R.sha256(R.ROOT / "v1_servo.py"),
                         R.sha256(R.ROOT / "my_servo.py"))
        self.assertEqual(R.sha256(R.ROOT / "v1_torque.py"),
                         R.sha256(R.ROOT / "my_torque.py"))
        self.assertEqual(len(R.sha256(R.ROOT / "v1_detector.py")), 64)
        self.assertEqual(len(R.sha256(R.ROOT / "my_detector.py")), 64)

    def test_source_hashes_are_relative_complete_and_content_based(self):
        hashes = R.source_hashes()
        expected = {
            "servo_controller.py", "torque_controller.py", "collision_detector.py",
            "v1_servo.py", "v1_torque.py", "v1_detector.py",
            "my_servo.py", "my_torque.py", "my_detector.py",
            "sim/runner.py", "tools/render_v2_comparison.py",
        }
        self.assertEqual(set(hashes), expected)
        self.assertTrue(all(not Path(name).is_absolute() for name in hashes))
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            path.write_bytes(b"abc")
            self.assertEqual(R.sha256(path), hashlib.sha256(b"abc").hexdigest())

    def test_collision_and_timewarp_contract(self):
        self.assertEqual(R.COLLISION, (3.0, (0.0, 15.0, 0.0)))
        pieces = R.timewarp_segments(R.COLLISION[0])
        speed = next(x["playback_speed"] for x in pieces
                     if x["simulation_start_s"] <= 3.0 < x["simulation_end_s"])
        self.assertLess(speed, 1.0)
        times = R.frame_times(R.COLLISION[0])
        self.assertTrue((times[1:] > times[:-1]).all())
        self.assertEqual(R.FPS, 15)
        self.assertEqual(R.GIF_FPS, 8)

    def test_outcome_text_handles_all_official_states(self):
        self.assertEqual(R.outcome_text({"alarm_before_collision": True}), "충돌 전 경보")
        self.assertEqual(R.outcome_text({"alarm_before_collision": False,
                                         "detect_delay_ms": None}), "놓침")
        self.assertEqual(R.outcome_text({"alarm_before_collision": False,
                                         "detect_delay_ms": 42.4}), "감지 42 ms")


if __name__ == "__main__":
    unittest.main()
