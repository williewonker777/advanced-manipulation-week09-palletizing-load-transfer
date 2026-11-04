"""R1 observer with a ridge fitted on the matched R2 normal observations.

This is a diagnostic attribution ablation, not a new control method.  The
factory accepts already-loaded parameters so the runtime path performs no file
I/O and continues to use :mod:`r1_detector`'s geometric gate and update law.
"""
from __future__ import annotations

import r1_detector


def make_detector(model_mapping):
    """Return an R1 detector class using explicit matched-ridge parameters."""
    mapping = model_mapping.get("models", model_mapping)
    if set(mapping) != {"servo", "torque"}:
        raise ValueError("matched ridge requires exactly servo and torque models")

    class Detector(r1_detector.Detector):
        R2_VARIANT = "ridge_only_matched"

        def reset(self):
            super().reset()
            mode = getattr(getattr(self, "ctrl", None), "MODE", None)
            self.normal_model = (
                r1_detector.NormalResidualModel(**mapping[mode])
                if mode in mapping else None
            )

    Detector.__name__ = "R2MatchedRidgeDetector"
    return Detector


__all__ = ["make_detector"]
