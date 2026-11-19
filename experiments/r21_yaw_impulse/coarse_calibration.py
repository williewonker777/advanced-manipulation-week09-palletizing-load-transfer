"""Coarse amplitude-only envelope and out-of-group reserve on no-force motions."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.r21_yaw_impulse.calibrate import BoundsProbe, ROOT
from my_torque import Controller
from sim.runner import run_episode

BINS = 60


def bin_index(t):
    return min(BINS - 1, max(0, int(round(1000 * float(t))) // 100))


class Probe(BoundsProbe):
    def reset(self):
        super().reset()
        self.cells = np.full((2, BINS), -1.)

    def update(self, t, th, thd, tau_m, tau_s, grip):
        result = super().update(t, th, thd, tau_m, tau_s, grip)
        if t >= .08 and self.yaw_value is not None:
            idx = (int(bool(grip)), bin_index(t))
            self.cells[idx] = max(self.cells[idx], abs(self.yaw_value))
        return result


def run(case):
    mass, com = case
    d = Probe()
    result, _ = run_episode(Controller(), mass, tuple(com), detector=d)
    if result['diverged']:
        raise ValueError('diverged normal')
    return d.cells


def neighborhoods(maxima):
    return np.stack([maxima[:, max(0, k - 1):min(BINS, k + 2)].max(axis=1) for k in range(BINS)], axis=1)


def fit(data, groups):
    reserve = np.zeros(2)
    for g in np.unique(groups):
        valid = groups == g
        learned = neighborhoods(data[~valid].max(axis=0))
        observed = data[valid].max(axis=0)
        deficit = np.where(observed >= 0, observed - learned, 0)
        reserve = np.maximum(reserve, deficit.max(axis=1))
    envelope = neighborhoods(data.max(axis=0))
    glb = data.max(axis=(0, 2))
    if np.any(glb <= 0) or not np.isfinite(data).all():
        raise ValueError('nonfinite or empty yaw calibration')
    values = np.where(envelope >= 0, envelope + reserve[:, None], glb[:, None])
    return 1.05 * values, reserve


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    if a.out.exists():
        p.error('fresh output required')
    plan = json.loads((ROOT / 'plan.json').read_bytes())
    cases = [(m, c) for m in plan['normal_masses'] for c in plan['normal_coms']]
    with ProcessPoolExecutor(a.jobs) as pool:
        data = np.stack(list(pool.map(run, cases)))
    groups = np.array([plan['normal_coms'].index(com) for _, com in cases])
    bounds, reserve = fit(data, groups)
    doc = {'schema': 'r21-yaw-normal-bins.v1', 'normal_episodes': len(cases), 'threshold_Nm': bounds.tolist(),
           'held_com_reserve_Nm': reserve.tolist(), 'full_normal_max_Nm': data.max(axis=(0, 2)).tolist()}
    with a.out.open('x') as fp:
        json.dump(doc, fp, indent=2, allow_nan=False)
        fp.write('\n')
    print('OOF reserve:', reserve, 'threshold at 1.42:', bounds[1, 14], 'at 3.86:', bounds[1, 38])


if __name__ == '__main__':
    main()
