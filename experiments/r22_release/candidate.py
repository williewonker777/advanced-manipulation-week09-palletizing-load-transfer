"""R21 yaw witness with a causal payload-release state transition."""
import json
from pathlib import Path

import numpy as np

from experiments.r21_yaw_impulse.detector import Detector as YawDetector
from experiments.r21_yaw_impulse.coarse_calibration import bin_index
from experiments.r22_release.state import ReleaseBalance


class Detector(YawDetector):
    """Composable detector base; later observers may subclass this class."""

    def reset(self):
        super().reset()
        self.yaw = ReleaseBalance(self.box_mass)
        source = json.loads(Path(__file__).with_name("model.json").read_bytes())
        self._release_bounds = np.asarray(source["threshold_Nm"], dtype=float).T
        if (source.get("schema") != "r21-release-momentum-bounds.v1"
                or self._release_bounds.shape != (60, 2)
                or not np.isfinite(self._release_bounds).all()
                or not (self._release_bounds > 0).all()):
            raise ValueError("expected normal-only release momentum bounds")

    def update(self, t, *args):
        self.bounds = self._release_bounds[bin_index(t)]
        return super().update(t, *args)
