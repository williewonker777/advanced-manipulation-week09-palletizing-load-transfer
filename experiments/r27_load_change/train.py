"""Calibrate on registered normal-only episodes; no forced cases available to the fitter."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as Baseline
from experiments.r27_load_change.event import Changes, DIRECTIONS, sensor

HERE = Path(__file__).resolve().parent


class NormalScan(Baseline):
    def reset(self):
        super().reset()
        self.windows = Changes()
        self.maxima = np.zeros(8)
        self.active_n = 0

    def update(self, t, th, thd, tau_m, tau_s, grip):
        residual, hit = super().update(t, th, thd, tau_m, tau_s, grip)
        means = self.windows.feed(sensor(self, grip))
        if means is not None:
            self.active_n += 1
            self.maxima = np.maximum(self.maxima, DIRECTIONS @ (means[1] - means[0]))
        return residual, hit


def episode(job):
    case, variant = job
    detector = NormalScan()
    _, log = lab.run_episode(case, variant, lambda: detector, (None, None))
    if not lab.exact_force(log, (None, None))['passed']:
        raise ValueError('no-force training only')
    return dict(id=case['id'], variant=variant['kind'], usable=detector.active_n,
                maxima_N=detector.maxima.tolist(), tool=lab.certified.tool_record(log),
                physics=lab.certified.trace_signature({'none': log}))


def fit(reg, workers):
    plan = json.loads(lab.PLAN.read_bytes())
    tasks = [(c, plan['variants'][m]) for c in reg['fit_normals'] for m in reg['mechanics']]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(episode, tasks))
    if any(x['usable'] <= 0 for x in rows):
        raise ValueError('no eligible loaded history')
    maxima = np.array([np.max([row['maxima_N'] for row in rows if row['variant']==m], axis=0)
                       for m in reg['mechanics']])
    # An explicit mechanics spread in addition to worst observed normal, not a threshold fitted on hits.
    bounds = maxima.max(axis=0) + np.ptp(maxima, axis=0)
    return {'schema':'r27-normal-only-direction-change.v1',
            'bounds_N':bounds.tolist(),'training_episodes':len(tasks),
            'registration_sha256':hashlib.sha256((HERE/'registration.json').read_bytes()).hexdigest(),
            'normals':rows}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2);a=p.parse_args()
    if a.out.is_symlink() or a.out.exists():p.error('fresh result required')
    data=fit(json.loads((HERE/'registration.json').read_bytes()),a.workers)
    with a.out.open('x') as f: json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(data['training_episodes'],data['bounds_N'])
