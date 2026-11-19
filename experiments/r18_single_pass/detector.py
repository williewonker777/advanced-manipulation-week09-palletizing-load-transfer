"""Single-pass evaluation of the frozen prediction/physical-coincidence observer."""
from __future__ import annotations

from collections import deque

import numpy as np
import my_detector
from experiments.r14_stock_handoff.detector import Detector as Coupled
from experiments.r16_payload_contract.controller import attach
from experiments.r17_release_coincidence.portable_detector import (
    Detector as Original, _LoadedCoincidence, POSITIVE_FLOOR, NEGATIVE_FLOOR,
)
from experiments.r17_release_prediction.detector import CLEAR_SAMPLES, MARGIN, PRE_SAMPLES


def mass_corrections(coefficients, mass):
    """Known-mass regression, no measured state. Keep the original contraction order."""
    return np.einsum('kfc,c->kf', coefficients, np.array([1., float(mass)]))


def update_hold(breach, armed, current, clear):
    if breach:
        return True, 0
    if armed:
        clear += 1
        return current and clear < CLEAR_SAMPLES, clear
    return False, 0


class _SinglePass(_LoadedCoincidence):
    def reset(self):
        super().reset()
        self._bias = mass_corrections(self.fast_coef, self.box_mass)
        self._commands = deque(maxlen=PRE_SAMPLES)
        self._history = deque(maxlen=PRE_SAMPLES)
        self._basis = None

    def _release(self):
        self.prediction_armed = True
        self.prediction_age = self.prediction_clear = 0
        self.prediction_detected = False
        self.prediction_supported = False
        # In the original unsupported short-history case the previous features,
        # if any, are not consulted. Preserve its public diagnostic attributes.
        if len(self._history) != PRE_SAMPLES:
            return
        history, commands = np.asarray(self._history), np.asarray(self._commands)
        raw = np.stack((np.full(6, float(self.box_mass)), history.mean(0),
                        history[-1], commands.mean(0), commands[-1]), axis=1)
        z = (raw - self.prediction_center) / self.prediction_scale
        lo, hi = self.prediction_mass_range
        self.prediction_supported = bool(lo <= self.box_mass <= hi
                                         and np.all(z >= self.prediction_min - 1e-12)
                                         and np.all(z <= self.prediction_max + 1e-12)
                                         and np.isfinite(z).all())
        self.prediction_features = z if self.prediction_supported else None
        self._basis = np.c_[np.ones(6), z] if self.prediction_supported else None

    def update(self, t: float, th, thd, tau_m, tau_s, grip) -> tuple[np.ndarray, bool]:
        # Bypass only the two redundant prediction layers. Coupled owns the
        # unaltered baseline and continues consuming every sensor sample.
        residual, coupled = Coupled.update(self, t, th, thd, tau_m, tau_s, grip)
        grip = bool(grip)
        if grip:
            self.prediction_armed = self.prediction_supported = False
            self.prediction_seen_grip = True
        elif self.prediction_last_grip and self.prediction_seen_grip:
            self._release()
        self.prediction_last_grip = grip

        usable = self.prediction_armed and self.prediction_supported
        self.coincidence_channels.fill(False)
        self.coincidence_physical_ratio.fill(0.)
        self.coincidence_mismatch_ratio.fill(0.)
        self.prediction_score = 0.
        over = False
        if grip or usable:
            tick = min(round(float(t) * 1000.0), len(self._bias) - 1)
            corrected = np.asarray(self._shared.last_fast_innovation, dtype=float) - self._bias[tick]
            if grip:
                self._history.append(corrected)
                self._commands.append(np.array(tau_m, dtype=float, copy=True))
            else:
                horizon = len(self.prediction_coefficients)
                age = self.prediction_age
                # C stops announcing new predictions when the model is exhausted;
                # D's existing behavior after exhaustion uses its final envelope.
                idx = min(age, horizon - 1)
                pred = np.einsum('fc,fc->f', self.prediction_coefficients[idx], self._basis)
                mismatch = corrected - pred
                self.coincidence_mismatch_ratio = np.abs(mismatch) / self.prediction_envelope[idx]
                if age < horizon:
                    self.prediction_score = float(np.max(self.coincidence_mismatch_ratio))
                    self.prediction_age = age + 1
                floors = np.where(corrected >= 0., POSITIVE_FLOOR, NEGATIVE_FLOOR)
                self.coincidence_physical_ratio = np.abs(corrected) / floors
                self.coincidence_channels = ((self.coincidence_mismatch_ratio > MARGIN)
                                              & (self.coincidence_physical_ratio > 1.)
                                              & (mismatch * corrected > 0.))
                over = bool(np.any(self.coincidence_channels))
        self.prediction_detected, self.prediction_clear = update_hold(
            self.prediction_score > MARGIN, self.prediction_armed, self.prediction_detected, self.prediction_clear)
        self.coincidence_detected, self.coincidence_clear = update_hold(
            over, self.prediction_armed, self.coincidence_detected, self.coincidence_clear)
        return residual, bool(coupled or self.coincidence_detected)


class Detector(Original):
    """Explicit opt-in; empty or servo paths use the exact baseline."""
    def reset(self):
        actual = getattr(self.ctrl, '_active', self.ctrl)
        if actual is None:
            raise ValueError('controller must be reset before detector')
        loaded = float(self.box_mass) > 0. and getattr(actual, 'MODE', None) == 'torque'
        impl = _SinglePass() if loaded else my_detector.Detector()
        if loaded:
            impl.model_path = self.model_path
        attach(self, impl)
        impl.ctrl = actual
        impl.reset()
        self._impl = impl
        self.update = impl.update
