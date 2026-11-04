import hashlib
import unittest

import numpy as np

from tools import render_r2_comparison as R


class R2MediaToolTests(unittest.TestCase):
    def test_frozen_four_way_modules_and_multidirectional_scenarios(self):
        self.assertEqual(tuple(R.combinations("servo")), R.VARIANT_ORDER)
        self.assertEqual(R.combinations("servo")["r1"], ("r1_servo", "r1_detector"))
        self.assertEqual(R.combinations("servo")["r2"], ("my_servo", "my_detector"))
        self.assertEqual([row["stem"] for row in R.SCENARIOS],
                         ["07_r2_servo_comparison", "08_r2_torque_comparison"])
        np.testing.assert_allclose(R.SCENARIOS[0]["collision"][1], [0, 15, 0])
        np.testing.assert_allclose(R.SCENARIOS[1]["collision"][1], [-15, 0, 0])

    def test_identical_alias_and_label_are_hash_driven(self):
        digest = hashlib.sha256(b"same").hexdigest()
        hashes = {"my_servo.py": digest, "r1_servo.py": digest,
                  "my_torque.py": "1" * 64, "r1_torque.py": "2" * 64,
                  "my_detector.py": digest, "r1_detector.py": digest}
        self.assertEqual(R.identical_source_aliases(hashes),
                         {"my_servo.py": "r1_servo.py", "my_detector.py": "r1_detector.py"})
        self.assertIn("R2=R1", R.display_labels("servo", hashes)["r2"])
        self.assertEqual(R.display_labels("torque", hashes)["r2"], "R2")

    def test_four_way_plot_contains_g4_support_trace(self):
        t = np.linspace(0.0, 6.0, 16)
        logs = {name: {"t": t, "e_mm": np.full(t.shape, index + 1.0),
                       "f_ext": np.zeros_like(t), "support_normalized": t / 6.0,
                       "detect": np.zeros(t.shape, dtype=bool)}
                for index, name in enumerate(R.VARIANT_ORDER)}
        image = R.plot_panel(logs, 3.0, R.LABELS, "+y 15 N")
        self.assertEqual(image.shape, (330, 1280, 3))


if __name__ == "__main__":
    unittest.main()
