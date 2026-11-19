"""Torque-only scheduled-approach tool-contact witness composed with R22 LIGHT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.r22_light.candidate import Detector as Light


MODEL = Path(__file__).with_name("model.json")


class Detector(Light):
    def reset(self):
        super().reset()
        model = json.loads(MODEL.read_bytes())
        self._tool_center = np.asarray(model["center_joint_Nm"], dtype=float)
        self._tool_bound = np.asarray(model["bound_joint_Nm"], dtype=float)
        self._tool_start = int(model["start_index"])
        self._tool_stop = int(model["stop_index"])
        self._tool_threshold = float(model["score_threshold"])
        self._tool_persistence = int(model["persistence_samples"])
        self._tool_latch_samples = int(model["latch_samples"])
        expected = (self._tool_stop - self._tool_start, 6)
        if (model.get("schema") != "r23-normal-approach-torque-envelope.v1"
                or self._tool_center.shape != expected or self._tool_bound.shape != expected
                or not np.isfinite(self._tool_center).all() or not np.isfinite(self._tool_bound).all()
                or not (self._tool_bound > 0).all() or self._tool_threshold <= 0
                or self._tool_persistence <= 0 or self._tool_latch_samples <= 0):
            raise ValueError("invalid normal-only tool-contact envelope")
        self._tool_enabled = getattr(self.ctrl, "MODE", None) == "torque"
        self._tool_streak = self._tool_latch = 0
        self.tool_score = 0.

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, base = super().update(t, th, thd, tau_m, tau_s, grip)
        tick = int(round(float(t) / .001))
        active = self._tool_enabled and not bool(grip) and self._tool_start <= tick < self._tool_stop
        if active:
            slot = tick - self._tool_start
            self.tool_score = float(np.max(np.abs(np.asarray(tau_s) - self._tool_center[slot]) /
                                           self._tool_bound[slot]))
            self._tool_streak = self._tool_streak + 1 if self.tool_score > self._tool_threshold else 0
            if self._tool_streak >= self._tool_persistence:
                self._tool_latch = self._tool_latch_samples
        else:
            self.tool_score = 0.
            self._tool_streak = 0
        witness = self._tool_latch > 0
        self._tool_latch = max(0, self._tool_latch - 1)
        return residual, bool(base or witness)
