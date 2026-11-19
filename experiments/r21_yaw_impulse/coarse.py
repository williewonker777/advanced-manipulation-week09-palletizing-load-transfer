"""Coarse normal amplitude correction, no per-tick signed signal prediction."""
import json
from pathlib import Path

import numpy as np

from experiments.r21_yaw_impulse.frozen import Detector as Global


class Detector(Global):
    def reset(self):
        super().reset()
        src = json.loads(Path(__file__).with_name('normal_bins.json').read_bytes())
        table = np.array(src['threshold_Nm'], dtype=float)
        if src['schema'] != 'r21-yaw-normal-bins.v1' or table.shape != (2, 60) or not (
                np.all(np.isfinite(table)) and np.all(table > 0)):
            raise ValueError('expected normal-only yaw bounds')
        self._by_tick = table.T

    def update(self, t, th, thd, tau_m, tau_s, grip):
        idx = int(round(1000 * float(t))) // 100
        self.bounds = self._by_tick[idx] if 0 <= idx < 60 else Global.bounds
        return super().update(t, th, thd, tau_m, tau_s, grip)
