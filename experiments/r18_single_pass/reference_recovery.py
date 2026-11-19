"""Rolling support observer over the frozen R17D innovation coincidence.

This is simulation research, not the original demonstration.  The 4-of-8
operational design was informed by already exposed R17D pairs.  It does not
change the frozen normal-only predictor, its envelopes, or the physical floors.
"""
from __future__ import annotations

from collections import deque

import my_detector
from experiments.r17_release_coincidence.portable_detector import (
    MODEL,
    _LoadedCoincidence,
)


SUPPORT_SAMPLES = 4
WINDOW_SAMPLES = 8


class RecoveryState:
    """Continuously rearming finite-support state with no refractory period."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._history = deque(maxlen=WINDOW_SAMPLES)
        self.detected = False

    def update(self, instantaneous, armed=True):
        if not armed:
            self.reset()
            return False
        self._history.append(bool(instantaneous))
        self.detected = sum(self._history) >= SUPPORT_SAMPLES
        return self.detected


class _LoadedRecovery(_LoadedCoincidence):
    def reset(self):
        super().reset()
        self.recovery = RecoveryState()
        self.recovery_instantaneous = False
        self.recovery_detected = False

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, _ = super().update(t, th, thd, tau_m, tau_s, grip)
        self.recovery_instantaneous = bool(self.coincidence_channels.any())
        self.recovery_detected = self.recovery.update(
            self.recovery_instantaneous, armed=self.prediction_armed
        )
        learned = bool(
            getattr(self.normal_model, "detected", False)
            or getattr(self, "r5_extra", False)
            or getattr(self, "r8_extra", False)
        )
        coupled = bool(self.cf_detected or learned)
        return residual, bool(coupled or self.recovery_detected)


class Detector:
    """Runner adapter accepting unwrapped C or the already-reset D wrapper."""

    model_path = MODEL

    def reset(self):
        actual = getattr(self.ctrl, "_active", self.ctrl)
        if actual is None:
            raise ValueError("controller must be reset before R18 detector")
        loaded_torque = (
            float(self.box_mass) > 0.0
            and getattr(actual, "MODE", None) == "torque"
        )
        implementation = _LoadedRecovery() if loaded_torque else my_detector.Detector()
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
