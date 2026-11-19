from pathlib import Path
import json

import numpy as np

from experiments.r21_yaw_impulse.detector import Detector as BalanceDetector
from experiments.r21_yaw_impulse.coarse_calibration import bin_index
from experiments.r21_yaw_continuity.state import ReleaseBalance


class Detector(BalanceDetector):
    def reset(self):
        super().reset()
        self.yaw = ReleaseBalance(self.box_mass)
        src = json.loads(Path(__file__).with_name('model.json').read_bytes())
        self._bins = np.asarray(src['threshold_Nm'], float).T
        if src['schema'] != 'r21-release-momentum-bounds.v1' or self._bins.shape != (60, 2) or not (
                np.isfinite(self._bins).all() and (self._bins > 0).all()):
            raise ValueError('expected normal-only release balance calibration')

    def update(self, t, *args):
        self.bounds = self._bins[bin_index(t)]
        return super().update(t, *args)
