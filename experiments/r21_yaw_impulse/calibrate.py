import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from experiments.r21_yaw_impulse.detector import Detector
from my_torque import Controller
from sim.runner import run_episode

ROOT = Path(__file__).resolve().parent


class BoundsProbe(Detector):
    def reset(self):
        super().reset()
        self.maxima = [0., 0.]
        self.peaks = [None, None]

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, flag = super().update(t, th, thd, tau_m, tau_s, grip)
        g = int(bool(grip))
        if t >= .08 and self.yaw_value is not None:
            if not np.isfinite(self.yaw_value):
                raise ValueError('nonfinite observation')
            if abs(self.yaw_value) > self.maxima[g]:
                self.maxima[g] = abs(self.yaw_value)
                self.peaks[g] = t
        return residual, flag


def normal(case):
    m, com = case
    probe = BoundsProbe()
    metrics, _ = run_episode(Controller(), m, tuple(com), detector=probe)
    if metrics['diverged']:
        raise ValueError('diverged normal calibration')
    return {'mass': m, 'com': com, 'max_abs_Nm': probe.maxima, 'peak_s': probe.peaks,
            'normal_false': metrics['false_alarms']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    if a.out.exists():
        p.error('new output required')
    plan = json.loads((ROOT / 'plan.json').read_bytes())
    cases = [(m, c) for m in plan['normal_masses'] for c in plan['normal_coms']]
    with ProcessPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(normal, cases))
    bounds = (1.05 * np.max([r['max_abs_Nm'] for r in rows], axis=0)).tolist()
    with a.out.open('x') as f:
        json.dump({'schema': 'r21-yaw-balance-calibration.v1', 'rows': rows, 'bounds_Nm': bounds},
                  f, indent=2, allow_nan=False)
        f.write('\n')
    print(bounds)


if __name__ == '__main__':
    main()
