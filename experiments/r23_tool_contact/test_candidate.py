import json
from pathlib import Path
import unittest

import numpy as np

from experiments.r23_tool_contact import candidate


class ToolWitnessTest(unittest.TestCase):
    def test_model_is_normal_only_and_well_formed(self):
        model = json.loads(Path(candidate.MODEL).read_bytes())
        self.assertEqual(len(model["fit_row_keys"]), 12)
        self.assertTrue(all(key.startswith("calibration_normal__") for key in model["fit_row_keys"]))
        self.assertEqual(np.asarray(model["center_joint_Nm"]).shape, (200, 6))
        self.assertEqual(np.asarray(model["bound_joint_Nm"]).shape, (200, 6))

    def test_candidate_has_no_ground_truth_or_offset_input(self):
        source = Path(candidate.__file__).read_text()
        self.assertNotIn("f_tool", source)
        self.assertNotIn("pick_offset", source)
        self.assertNotIn("collision", source)


if __name__ == "__main__":
    unittest.main()
