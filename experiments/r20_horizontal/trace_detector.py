"""Passive signal recorder around the unchanged R10 detector.

This module is diagnostic only: it returns exactly the baseline decision and
records quantities computed from the documented detector inputs/public model.
"""
import numpy as np

import my_detector


class Detector(my_detector.Detector):
    def reset(self):
        super().reset()
        self.trace = []

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, detected = super().update(t, th, thd, tau_m, tau_s, grip)
        p = self._shared
        self.trace.append({
            "t": float(t),
            "grip": bool(grip),
            "detected": bool(detected),
            "r_joint": p.r_joint.copy(),
            "wrench": p.last_wrench.copy(),
            "residual": p.last_residual_wrench.copy(),
            "innovation": p.last_fast_innovation.copy(),
            "score": float(p.last_score),
            "fast_score": float(p.last_fast_score),
            "th": np.asarray(th, float).copy(),
            "thd": np.asarray(thd, float).copy(),
            "tau_m": np.asarray(tau_m, float).copy(),
            "tau_s": np.asarray(tau_s, float).copy(),
        })
        return residual, detected
