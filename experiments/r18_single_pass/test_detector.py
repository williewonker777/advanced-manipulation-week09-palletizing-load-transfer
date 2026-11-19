"""Finite numerical and boundary contracts for the single-pass evaluator."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r17_release_coincidence.portable_detector import _LoadedCoincidence
from .detector import _SinglePass, mass_corrections, update_hold


class ObserverTests(unittest.TestCase):
    def test_full_mass_table_is_the_same_contraction(self):
        rng = np.random.default_rng(17)
        coefficients = rng.standard_normal((128, 6, 2))
        for mass in (0., 2.86, 5.38, 7.90, 8.):
            per_tick = np.stack([np.einsum('fc,c->f', row, [1., mass]) for row in coefficients])
            np.testing.assert_array_equal(mass_corrections(coefficients, mass), per_tick)

    def test_hold_and_rearm(self):
        state, cleared = True, 0
        for step in range(1, 9):
            state, cleared = update_hold(False, True, state, cleared)
            self.assertEqual((state, cleared), (step < 8, step))
        self.assertEqual(update_hold(True, True, state, cleared), (True, 0))
        self.assertEqual(update_hold(False, False, True, 3), (False, 0))

    def test_two_edges_short_history_and_exhausted_model(self):
        # Full public update on both observers, except the upstream Coupled
        # sensor packet, whose signed channel values are supplied by this test.
        from collections import deque
        from types import SimpleNamespace
        rng = np.random.default_rng(43)
        def base_init(obj):
            obj.fast_coef = np.zeros((250, 6, 2))
            obj.box_mass = 5.
            obj.prediction_coefficients = rng_fixed.copy()
            obj.prediction_envelope = np.ones((4, 6))
            obj.prediction_min = np.full((6, 5), -100.)
            obj.prediction_max = -obj.prediction_min
            obj.prediction_center = np.zeros((6, 5)); obj.prediction_scale = np.ones((6, 5))
            obj.prediction_mass_range = [2., 8.]
            obj.prediction_last_grip = obj.prediction_seen_grip = obj.prediction_armed = False
            obj.prediction_supported = obj.prediction_detected = obj.coincidence_detected = False
            obj.prediction_clear = obj.coincidence_clear = obj.prediction_age = 0
            obj.prediction_score = 0.
            obj.prediction_features = None
            obj.prediction_corrected_history, obj.prediction_command_history = [], []
            obj.coincidence_channels = np.zeros(6, bool)
            obj.coincidence_mismatch_ratio = np.zeros(6); obj.coincidence_physical_ratio = np.zeros(6)
            obj._shared = SimpleNamespace(last_fast_innovation=np.zeros(6))
            obj.normal_model = SimpleNamespace(detected=False)
            obj.cf_detected = False
        rng_fixed = rng.normal(size=(4, 6, 6))
        old = _LoadedCoincidence(); new = _SinglePass()
        for obj in (old,new): base_init(obj)
        new._commands, new._history, new._basis = deque(maxlen=32),deque(maxlen=32),None
        new._bias=mass_corrections(new.fast_coef,new.box_mass)
        grip = [False]*3+[True]*7+[False]*11+[True]*36+[False]*15+[True]*36+[False]*60
        # Include signed over/below the six distinct physical floors.
        with patch.object(Coupled,'update',lambda self,*a:(np.zeros(6),False)):
            for i,g in enumerate(grip):
                signal = rng.normal(size=6)*8.; command = rng.normal(size=6)
                for obj in (old,new): obj._shared.last_fast_innovation = signal
                a,b=(obj.update(i*.001, np.zeros(6),np.zeros(6),command,np.zeros(6),g) for obj in (old,new))
                np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
                for field in ('prediction_armed','prediction_supported','prediction_age','prediction_score',
                              'prediction_detected','prediction_clear','coincidence_channels',
                              'coincidence_physical_ratio','coincidence_mismatch_ratio','coincidence_detected','coincidence_clear'):
                    np.testing.assert_array_equal(getattr(old,field),getattr(new,field),err_msg=field)
