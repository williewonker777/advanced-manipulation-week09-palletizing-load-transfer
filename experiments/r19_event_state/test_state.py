import unittest

from experiments.r19_event_state.state import EventState


class EventStateTest(unittest.TestCase):
    def test_onset_is_immediate_and_fragments_remain_one_event(self):
        state = EventState()
        output = [state.update(onset, unloaded) for onset, unloaded in (
            [(False, True)] * 12
            + [(True, False)]
            + [(False, False)] * 5
            + [(False, True)] * 3
            + [(True, False)]
            + [(False, True)] * 8
        )]
        self.assertFalse(any(output[:12]))
        self.assertTrue(all(output[12:-1]))
        self.assertFalse(output[-1])

    def test_second_event_rearms_after_observable_unloading(self):
        state = EventState()
        first = state.update(True, False)
        for _ in range(8):
            released = state.update(False, True)
        second = state.update(True, False)
        self.assertTrue(first)
        self.assertFalse(released)
        self.assertTrue(second)

    def test_no_time_based_release_and_disarm_resets(self):
        state = EventState()
        state.update(True, False)
        self.assertTrue(all(state.update(False, False) for _ in range(100)))
        self.assertFalse(state.update(False, False, armed=False))
        self.assertEqual(state.unload_samples, 0)


if __name__ == "__main__":
    unittest.main()
