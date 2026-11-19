"""Same-direction physical-floor coincidence over the frozen R17C predictor."""
from __future__ import annotations

import numpy as np

import my_detector
from experiments.r17_release_prediction.detector import (
    CLEAR_SAMPLES, MARGIN, MODEL, _LoadedPrediction,
)


POSITIVE_FLOOR = np.array([11.141377260806653, 11.141377260806653,
                           2.812605901696073, 4.885650078933188,
                           4.885650078933188, 0.4058946982950356])
NEGATIVE_FLOOR = np.array([11.141377260806653, 11.141377260806653,
                           27.842061547424592, 4.885650078933188,
                           4.885650078933188, 0.4058946982950356])


class _LoadedCoincidence(_LoadedPrediction):
    def reset(self):
        super().reset()
        self.coincidence_detected = False
        self.coincidence_clear = 0
        self.coincidence_channels = np.zeros(6, dtype=bool)
        self.coincidence_physical_ratio = np.zeros(6)
        self.coincidence_mismatch_ratio = np.zeros(6)

    def update(self, t: float, th, thd, tau_m, tau_s, grip) -> tuple[np.ndarray, bool]:
        residual, _ = super().update(t, th, thd, tau_m, tau_s, grip)
        packet = self._shared
        k = min(round(float(t) * 1000.0), len(self.fast_coef) - 1)
        correction = np.einsum("fc,c->f", self.fast_coef[k],
                               np.array([1.0, float(self.box_mass)]))
        corrected = np.asarray(packet.last_fast_innovation, dtype=float) - correction
        over = False
        self.coincidence_channels[:] = False
        self.coincidence_physical_ratio[:] = 0.0
        self.coincidence_mismatch_ratio[:] = 0.0
        if (self.prediction_armed and self.prediction_supported
                and 0 < self.prediction_age <= len(self.prediction_coefficients)):
            age = self.prediction_age - 1
            z = np.c_[np.ones(6), self.prediction_features]
            predicted = np.einsum("fc,fc->f", self.prediction_coefficients[age], z)
            mismatch = corrected - predicted
            envelope = self.prediction_envelope[age]
            floors = np.where(corrected >= 0.0, POSITIVE_FLOOR, NEGATIVE_FLOOR)
            self.coincidence_mismatch_ratio = np.abs(mismatch) / envelope
            self.coincidence_physical_ratio = np.abs(corrected) / floors
            same_direction = mismatch * corrected > 0.0
            self.coincidence_channels = (self.coincidence_mismatch_ratio > MARGIN
                                         ) & (self.coincidence_physical_ratio > 1.0
                                              ) & same_direction
            over = bool(np.any(self.coincidence_channels))
        if not self.prediction_armed:
            self.coincidence_detected = False
            self.coincidence_clear = 0
        elif over:
            self.coincidence_detected = True
            self.coincidence_clear = 0
        else:
            self.coincidence_clear += 1
            if self.coincidence_clear >= CLEAR_SAMPLES:
                self.coincidence_detected = False
        learned = bool(getattr(self.normal_model, "detected", False)
                       or getattr(self, "r5_extra", False)
                       or getattr(self, "r8_extra", False))
        coupled = bool(self.cf_detected or learned)
        return residual, bool(coupled or self.coincidence_detected)


class Detector:
    """Runner adapter accepting unwrapped C or the already-reset D wrapper."""
    model_path = MODEL

    def reset(self):
        actual = getattr(self.ctrl, "_active", self.ctrl)
        if actual is None:
            raise ValueError("controller must be reset before R17D detector")
        loaded_torque = float(self.box_mass) > 0.0 and getattr(actual, "MODE", None) == "torque"
        implementation = _LoadedCoincidence() if loaded_torque else my_detector.Detector()
        if loaded_torque:
            implementation.model_path = self.model_path
        for name in ("nom", "nom_box", "box_mass", "box_com"):
            setattr(implementation, name, getattr(self, name))
        implementation.ctrl = actual
        implementation.reset()
        self._impl = implementation
        self.update = implementation.update

    def __getattr__(self, name):
        implementation = vars(self).get("_impl")
        if implementation is None:
            raise AttributeError(name)
        return getattr(implementation, name)
