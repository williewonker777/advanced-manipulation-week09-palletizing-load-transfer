"""Compose the isolated-spike correction and release momentum continuity without two R10 updates."""
import json
from pathlib import Path

import numpy as np

from experiments.r22_normal.candidate import Detector as Median
from experiments.r22_release.state import ReleaseBalance

BOUNDS = Path(__file__).resolve().parents[1] / 'r22_release/model.json'


class Detector(Median):
    def reset(self):
        super().reset()
        model = json.loads(BOUNDS.read_bytes())
        table = np.asarray(model['threshold_Nm'], dtype=float)
        if (model['schema'] != 'r21-release-momentum-bounds.v1' or table.shape != (2, 60)
                or not np.isfinite(table).all() or not (table > 0).all()):
            raise ValueError('invalid frozen release-continuity bounds')
        self._by_tick = table.T
        self.yaw = ReleaseBalance(self.box_mass)
