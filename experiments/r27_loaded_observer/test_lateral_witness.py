import unittest

import numpy as np
import json
from pathlib import Path

from experiments.r27_loaded_observer.candidate import MODEL, Detector
from experiments.r27_loaded_observer.lateral_witness import (
    LoadedLateralOnset,
    lateral_residual,
)


def model():
    return {
        "schema": "r27-loaded-lateral-model.v1",
        "onset_xy_from_z": [[0.0, 0.0], [0.0, 0.0]],
        "sustain_xy_from_z": [[0.0, 0.0], [0.0, 0.0]],
        "grip_stability_samples": 1,
        "onset_lag_samples": 1,
        "sustain_lag_samples": 3,
        "onset_persistence_samples": 2,
        "pending_samples": 2,
        "pulse_samples": 3,
        "rearm_quiet_samples": 2,
        "onset_threshold_N": 1.0,
        "sustain_threshold_N": 2.0,
        "ownership_release_N": 1.0,
    }


class LateralWitnessTests(unittest.TestCase):
    def test_embedded_candidate_model_matches_frozen_asset(self):
        path = Path(__file__).with_name("model_v3.json")
        self.assertEqual(MODEL, json.loads(path.read_bytes()))
        self.assertTrue(issubclass(Detector, object))

    def test_vertical_normal_coupling_is_removed_with_sign(self):
        coefficients = [[0.0, 0.0], [0.5, -0.25]]
        result = lateral_residual([2.0, -1.0, 4.0], [0.0, 0.0, 0.0], coefficients)
        np.testing.assert_allclose(result, [0.0, 0.0])
        opposite = lateral_residual([-2.0, 1.0, -4.0], [0.0, 0.0, 0.0], coefficients)
        np.testing.assert_allclose(opposite, [0.0, 0.0])

    def test_unexplained_lateral_ramp_emits_one_bounded_pulse(self):
        state = LoadedLateralOnset(model())
        values = [0, 0, 0, 0, 2, 4, 6, 6, 6, 6, 6, 6]
        bits = [state.feed(tick, [value, 0, 0], tick + 1, False)
                for tick, value in enumerate(values)]
        self.assertEqual([index for index, bit in enumerate(bits) if bit], [8, 9, 10])
        self.assertEqual(state.onset_count, 1)
        self.assertTrue(state.owner)

    def test_pending_is_cancelled_by_native_r26(self):
        state = LoadedLateralOnset(model())
        bits = []
        for tick, value in enumerate([0, 0, 0, 0, 2, 4, 6, 6, 6]):
            bits.append(state.feed(tick, [value, 0, 0], tick + 1, tick == 7))
        self.assertFalse(any(bits))
        self.assertEqual(state.onset_count, 0)


if __name__ == "__main__":
    unittest.main()
