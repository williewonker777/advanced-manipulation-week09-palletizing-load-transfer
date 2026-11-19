import unittest

from experiments.r27_loaded_observer.witness import LoadedForceOnset


def model():
    return {
        "schema": "r27-loaded-observer-model.v1",
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


class WitnessTests(unittest.TestCase):
    def test_bounded_pulse_and_telemetry_do_not_hold_output(self):
        state = LoadedForceOnset(model())
        values = [0, 0, 0, 0, 2, 4, 6, 6, 6, 6, 6, 6]
        bits = [state.feed(tick, [value, 0, 0], tick + 1, False)
                for tick, value in enumerate(values)]
        self.assertEqual([index for index, bit in enumerate(bits) if bit], [8, 9, 10])
        self.assertTrue(state.owner)
        self.assertEqual(state.onset_count, 1)

    def test_incumbent_cancels_pending_and_is_never_extended(self):
        state = LoadedForceOnset(model())
        bits = []
        for tick, value in enumerate([0, 0, 0, 0, 2, 4, 6, 6, 6]):
            bits.append(state.feed(tick, [value, 0, 0], tick + 1, tick == 7))
        self.assertFalse(any(bits))
        self.assertEqual(state.onset_count, 0)

    def test_live_incumbent_suppresses_only_extra_pulse(self):
        state = LoadedForceOnset(model())
        for tick, value in enumerate([0, 0, 0, 0, 2, 4, 6, 6]):
            self.assertFalse(state.feed(tick, [value, 0, 0], tick + 1, False))
        self.assertTrue(state.feed(8, [6, 0, 0], 9, False))
        self.assertFalse(state.feed(9, [6, 0, 0], 10, True))
        self.assertTrue(state.owner)

    def test_grip_instability_resets_history(self):
        state = LoadedForceOnset(model())
        for tick in range(6):
            state.feed(tick, [3 * tick, 0, 0], tick + 1, False)
        state.feed(6, [18, 0, 0], 0, False)
        self.assertFalse(state.owner)
        self.assertEqual(state.onset_count, 0)


if __name__ == "__main__":
    unittest.main()
