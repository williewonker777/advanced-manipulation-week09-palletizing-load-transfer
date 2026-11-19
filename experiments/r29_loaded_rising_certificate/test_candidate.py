import unittest
from types import SimpleNamespace

import numpy as np

from experiments.r27_load_change.event import DIRECTIONS, WIDTH
from experiments.r29_loaded_rising_certificate.candidate import (
    Detector,
    rising_certificate,
    rising_means,
)


class Window:
    def __init__(self, before, now):
        self.past = [None] * WIDTH
        self.present = [None] * WIDTH
        self.a = WIDTH * np.asarray(before, float)
        self.b = WIDTH * np.asarray(now, float)


class Normals:
    def __init__(self, answer=True):
        self.answer = answer
        self.calls = []

    def unexpected(self, coords, levels, axis):
        self.calls.append((coords, levels, axis))
        return self.answer


def gate(before=((0.0, 0.0), (0.0, 0.0)), now=((2.0, 0.0), (3.0, 0.0))):
    return SimpleNamespace(
        state="rising",
        axis=0,
        native_seen=False,
        windows=SimpleNamespace(windows=[Window(before[0], now[0]), Window(before[1], now[1])]),
    )


class RisingCertificateTest(unittest.TestCase):
    def test_reads_existing_adjacent_means(self):
        before, now = rising_means(gate())
        np.testing.assert_array_equal(before, np.zeros((2, 2)))
        np.testing.assert_array_equal(now, np.array(((2.0, 0.0), (3.0, 0.0))))

    def test_requires_both_positive_signed_changes_and_frozen_certificate(self):
        normals = Normals()
        levels = np.array(((3.0, 0.0), (4.0, 0.0)))
        self.assertTrue(rising_certificate(gate(), normals, np.zeros(12), levels,
                                           True, False, True))
        self.assertEqual(normals.calls[0][2], 0)
        falling = gate(now=((2.0, 0.0), (-1.0, 0.0)))
        self.assertFalse(rising_certificate(falling, normals, np.zeros(12), levels,
                                            True, False, True))

    def test_fails_closed_on_native_stale_unknown_or_nonfinite_inputs(self):
        levels = np.array(((3.0, 0.0), (4.0, 0.0)))
        for changed in (
            dict(native_active=True),
            dict(routes_complete=False),
            dict(grip=False),
        ):
            args = dict(grip=True, native_active=False, routes_complete=True)
            args.update(changed)
            self.assertFalse(rising_certificate(gate(), Normals(), np.zeros(12), levels, **args))
        stale = gate()
        stale.native_seen = True
        self.assertFalse(rising_certificate(stale, Normals(), np.zeros(12), levels,
                                            True, False, True))
        self.assertFalse(rising_certificate(gate(), Normals(False), np.zeros(12), levels,
                                            True, False, True))
        self.assertFalse(rising_certificate(gate(), Normals(), np.zeros(12), None,
                                            True, False, True))

    def test_composes_without_a_second_native_detector_branch(self):
        names = [cls.__module__ + "." + cls.__name__ for cls in Detector.mro()]
        self.assertIn("experiments.r29_disjoint_events.candidate.Detector", names)
        self.assertEqual(names.count("experiments.r26_yaw_acquisition.candidate.Detector"), 1)


if __name__ == "__main__":
    unittest.main()
