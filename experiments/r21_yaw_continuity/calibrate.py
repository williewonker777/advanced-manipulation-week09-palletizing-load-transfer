import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from experiments.r21_yaw_impulse.coarse_calibration import Probe as Original, fit
from experiments.r21_yaw_continuity.state import ReleaseBalance
from my_torque import Controller
from sim.runner import run_episode


class Probe(Original):
    def reset(self):
        super().reset()
        self.yaw = ReleaseBalance(self.box_mass)


def run(case):
    mass, com = case
    det = Probe()
    metrics, _ = run_episode(Controller(), mass, tuple(com), detector=det)
    if metrics['diverged']:
        raise ValueError('nonfinite calibration')
    return det.cells


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('new output required')
    ref = Path(__file__).resolve().parents[1] / 'r21_yaw_impulse/plan.json'
    plan = json.loads(ref.read_bytes())
    cases = [(m, c) for m in plan['normal_masses'] for c in plan['normal_coms']]
    with ProcessPoolExecutor(2) as pool:
        samples = np.stack(list(pool.map(run, cases)))
    bounds, reserve = fit(samples, np.array([plan['normal_coms'].index(c) for _, c in cases]))
    with a.out.open('x') as fp:
        json.dump({'schema': 'r21-release-momentum-bounds.v1', 'normal_episodes': len(cases),
                   'threshold_Nm': bounds.tolist(), 'held_com_reserve_Nm': reserve.tolist()}, fp, indent=2)
        fp.write('\n')
