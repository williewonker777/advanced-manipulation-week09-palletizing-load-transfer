"""Cross-fit normal release predictor OR'ed with unchanged Coupled detection.

The auxiliary decision is armed by an observed grip falling edge, predicts all
six corrected innovation directions from known mass and bounded pre-edge sensor
and applied-command summaries, and abstains outside its normal-data support.
It has no absolute clock, collision, case, or evaluator input.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import my_detector
from experiments.r14_stock_handoff.detector import Detector as Coupled


HERE = Path(__file__).resolve().parent
MODEL = HERE / "model.json"
SCHEMA = "r17-release-prediction-normal-model.v1"
PRE_SAMPLES = 32
MARGIN = 1.25
CLEAR_SAMPLES = 8


def _array(document, name, shape):
    value = np.asarray(document[name], dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("invalid R17C array: " + name)
    return value


def _model(path):
    document = json.loads(Path(path).read_text())
    expected = {"schema", "plan_sha256", "coupled_coefficient_sha256", "pre_samples",
                "ridge", "margin", "age_count", "mass_range", "feature_center",
                "feature_scale", "feature_min", "feature_max", "coefficients",
                "envelope", "mass_groups", "duplicate_groups", "crossfit_summary"}
    if set(document) != expected or document.get("schema") != SCHEMA:
        raise ValueError("invalid R17C model schema")
    age = int(document["age_count"])
    mass_range = _array(document, "mass_range", (2,))
    center = _array(document, "feature_center", (6, 5))
    scale = _array(document, "feature_scale", (6, 5))
    lower = _array(document, "feature_min", (6, 5))
    upper = _array(document, "feature_max", (6, 5))
    coefficients = _array(document, "coefficients", (age, 6, 6))
    envelope = _array(document, "envelope", (age, 6))
    if (document["pre_samples"] != PRE_SAMPLES or document["ridge"] != 0.001
            or document["margin"] != MARGIN or age <= 0
            or not 0.0 < mass_range[0] < mass_range[1]
            or np.any(scale <= 0.0) or np.any(lower > upper) or np.any(envelope <= 0.0)):
        raise ValueError("invalid R17C bounds")
    return document, mass_range, center, scale, lower, upper, coefficients, envelope


class _LoadedPrediction(Coupled):
    model_path = MODEL

    def reset(self):
        super().reset()
        (self.prediction_model, self.prediction_mass_range, self.prediction_center,
         self.prediction_scale, self.prediction_min, self.prediction_max,
         self.prediction_coefficients, self.prediction_envelope) = _model(self.model_path)
        if self.prediction_model["coupled_coefficient_sha256"] != self.coefficient_sha256:
            raise ValueError("R17C/Coupled coefficient provenance mismatch")
        self.prediction_corrected_history = []
        self.prediction_command_history = []
        self.prediction_seen_grip = False
        self.prediction_last_grip = False
        self.prediction_armed = False
        self.prediction_supported = False
        self.prediction_features = None
        self.prediction_age = 0
        self.prediction_score = 0.0
        self.prediction_detected = False
        self.prediction_clear = 0

    def update(self, t, th, thd, tau_m, tau_s, grip):
        tau_m = np.asarray(tau_m, dtype=float)
        residual, coupled = super().update(t, th, thd, tau_m, tau_s, grip)
        packet = self._shared
        k = min(round(float(t) * 1000.0), len(self.fast_coef) - 1)
        correction = np.einsum("fc,c->f", self.fast_coef[k],
                               np.array([1.0, float(self.box_mass)]))
        corrected = np.asarray(packet.last_fast_innovation, dtype=float) - correction
        grip = bool(grip)
        if grip:
            self.prediction_seen_grip = True
            self.prediction_armed = False
            self.prediction_supported = False
            self.prediction_detected = False
            self.prediction_clear = 0
            self.prediction_corrected_history.append(corrected.copy())
            self.prediction_command_history.append(tau_m.copy())
            self.prediction_corrected_history = self.prediction_corrected_history[-PRE_SAMPLES:]
            self.prediction_command_history = self.prediction_command_history[-PRE_SAMPLES:]
        elif self.prediction_last_grip and self.prediction_seen_grip:
            self.prediction_armed = True
            self.prediction_age = 0
            self.prediction_detected = False
            self.prediction_clear = 0
            if (len(self.prediction_corrected_history) == PRE_SAMPLES
                    and len(self.prediction_command_history) == PRE_SAMPLES):
                history = np.asarray(self.prediction_corrected_history)
                command = np.asarray(self.prediction_command_history)
                raw = np.stack((np.full(6, float(self.box_mass)), history.mean(0),
                                history[-1], command.mean(0), command[-1]), axis=1)
                z = (raw - self.prediction_center) / self.prediction_scale
                inside_mass = (self.prediction_mass_range[0] <= float(self.box_mass)
                               <= self.prediction_mass_range[1])
                inside_feature = bool(np.all(z >= self.prediction_min - 1e-12)
                                      and np.all(z <= self.prediction_max + 1e-12))
                self.prediction_supported = bool(inside_mass and inside_feature
                                                 and np.isfinite(z).all())
                self.prediction_features = z if self.prediction_supported else None
            else:
                self.prediction_supported = False
                self.prediction_features = None
        self.prediction_last_grip = grip

        over = False
        if (self.prediction_armed and self.prediction_supported
                and self.prediction_age < len(self.prediction_coefficients)):
            z = np.c_[np.ones(6), self.prediction_features]
            predicted = np.einsum("fc,fc->f",
                                  self.prediction_coefficients[self.prediction_age], z)
            normalized = np.abs(corrected - predicted) / self.prediction_envelope[self.prediction_age]
            self.prediction_score = float(np.max(normalized))
            over = self.prediction_score > MARGIN
            self.prediction_age += 1
        else:
            self.prediction_score = 0.0
        if over:
            self.prediction_detected = True
            self.prediction_clear = 0
        elif self.prediction_armed:
            self.prediction_clear += 1
            if self.prediction_clear >= CLEAR_SAMPLES:
                self.prediction_detected = False
        return residual, bool(coupled or self.prediction_detected)


class Detector:
    """Runner adapter accepting unwrapped C or the already-reset D wrapper."""
    model_path = MODEL

    def reset(self):
        actual = getattr(self.ctrl, "_active", self.ctrl)
        if actual is None:
            raise ValueError("controller must be reset before R17C detector")
        loaded_torque = float(self.box_mass) > 0.0 and getattr(actual, "MODE", None) == "torque"
        implementation = _LoadedPrediction() if loaded_torque else my_detector.Detector()
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
