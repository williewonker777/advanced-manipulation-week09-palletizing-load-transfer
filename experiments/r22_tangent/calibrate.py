import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.r22_tangent.detector import Detector
from my_torque import Controller
from sim.runner import run_episode


class Probe(Detector):
    def reset(self):
        super().reset()
        self.peaks = [None] * 4

    def update(self, t, *args):
        result = super().update(t, *args)
        g, score = self.tangent.group, self.tangent.score
        if g is not None and score is not None and (self.peaks[g] is None or self.peaks[g][0] < score):
            self.peaks[g] = [score, float(t)]
        return result


def normal(case):
    mass, com = case
    probe = Probe()
    m, _ = run_episode(Controller(), box_mass=mass, box_com=com, detector=probe)
    if m['diverged']:
        raise ValueError('diverged normal')
    return {'mass':mass, 'com':com, 'max_and_time':probe.peaks, 'baseline_false':m['false_alarm_times']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); a=p.parse_args()
    plan = json.loads(Path(__file__).with_name('plan.json').read_bytes())
    with ProcessPoolExecutor(2) as pool:
        rows = list(pool.map(normal, ((m,c) for m in plan['masses'] for c in plan['coms'])))
    bounds = [1.05 * max(r['max_and_time'][g][0] for r in rows if r['max_and_time'][g] is not None) for g in range(4)]
    with a.out.open('x') as f:
        json.dump({'schema':'r22-tangent-normal-fit.v1','episodes':len(rows),'bounds_N':bounds,'rows':rows},f,indent=2,allow_nan=False)
        f.write('\n')
    print('class bounds N:', bounds)
