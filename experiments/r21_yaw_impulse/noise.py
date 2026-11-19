"""Post-freeze sensor-seed stress check with a clean reference for both impacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.r20_horizontal.candidate_detector import Detector as Horizontal
from experiments.r21_weak_challenge import evaluator as weak
from experiments.r21_yaw_impulse.coarse import Detector as Yaw
from sim import config as C


def run(plan, seed):
    if C.JTS_SEED != seed or seed not in plan['seeds']:
        raise ValueError('wrong test fixture')
    rows = []
    for mass in plan['masses']:
        case = {'mass': mass, 'com': plan['com']}
        arms = {'r20': Horizontal, 'yaw': Yaw}
        normals = {key: weak.episode(det, case, None) for key, det in arms.items()}
        for t in plan['time']:
            hit = (t, plan['force'])
            forced = {key: weak.episode(det, case, hit) for key, det in arms.items()}
            scores = {}
            for key in arms:
                normal_metrics, normal = normals[key]
                metrics, log = forced[key]
                prefix = int(np.searchsorted(normal['t'], t))
                scores[key] = weak.classify(metrics, log, normal, t, plan['reference_radius_ms'])
                window = t + .2 + 2 * weak.COLLISION_RAMP + weak.COLLISION_HOLD
                scores[key].update(normal_false=normal_metrics['false_alarm_times'],
                                   window_rises=sum(t <= h < window for h in weak.rises(log)))
                if (not all(weak.expected_force(log, hit).values())
                        or not all(weak.prefix_map(log, normal, weak.PREFIX, prefix).values())):
                    raise ValueError('force or preimpact twin changed')
            physical_equal = all(np.array_equal(pair['r20'][1][k], pair['yaw'][1][k])
                                 for pair in (normals, forced) for k in weak.PHYSICAL)
            if not physical_equal:
                raise ValueError('different physics between observers')
            a, b = scores['r20'], scores['yaw']
            rows.append({'seed': seed, 'mass': mass, 'time': t, 'scores': scores,
                         'physics_force_and_prefix_exact': True,
                         'added_normal_false': weak.unmatched(b['normal_false'], a['normal_false']),
                         'added_force_false': weak.unmatched(b['false_rise_times'], a['false_rise_times'])})
    return {'episodes': 2 * len(plan['masses']) * (1 + len(plan['time'])), 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(__file__).with_name('noise_plan.json').read_bytes())
    with args.out.open('x') as fp:
        json.dump(run(plan, args.seed), fp, indent=2, allow_nan=False)
