"""Profile the selected research detector in the observed no-force and impact twins, not a deadline test."""
import argparse
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np

from experiments.r22_timing.study import bind, digest, stats
from experiments.r23_detached.free import Detector as Incumbent
from experiments.r23_holdout import evaluator as lab
from experiments.r24_composed.detector import Detector as Joint
from my_torque import Controller


def trial(case, variant, force, arm, rep):
    control, detector = Controller(), {'r23': Incumbent, 'joint': Joint}[arm]()
    cw, cc, nc = bind(control, 'compute', 6000)
    dw, dc, nd = bind(detector, 'update', 6000)
    rw, rc, nr = bind(detector, 'reset', 1)
    original = lab.certified.stock.run_episode
    runner = FunctionType(original.__code__,
                          dict(original.__globals__, Plant=lab.local_plant(None, variant)),
                          original.__name__, original.__defaults__, original.__closure__)
    values, log = runner(control, box_mass=case['mass'], box_com=tuple(case['com']),
                         pick_offset=case['offset'], detector=detector, collision=force)
    if values['diverged'] or nc != [6000] or nd != [6000] or nr != [1]:
        raise ValueError('incomplete measured run')
    return {'mechanics': variant['kind'], 'twin': 'impact' if force else 'none', 'arm': arm,
            'repeat': rep, 'cycles': nc[0], 'physical_and_residual_sha256':
            digest(log, ('t', 'q', 'q_d', 'tau', 'qpos', 'f_ext', 'f_tool', 'f_table', 'r')),
            'alarm_sha256': digest(log, ('detect',)), 'wall': stats(cw + dw),
            'thread_cpu': stats(cc + dc), 'max_wall_index': int(np.argmax(cw + dw)),
            'max_cpu_index': int(np.argmax(cc + dc)),
            'reset_wall_us': float(rw[0]) / 1000, 'reset_cpu_us': float(rc[0]) / 1000}


def run():
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    plan = json.loads(lab.PLAN.read_bytes())
    case = next(row for row in plan['negative'] if row['id'] == 'n23_detached_heavy_interp')
    arms = ('r23', 'joint')
    rows = [trial(case, plan['variants'][variant], force, arm, rep)
            for rep in range(2) for variant in ('stock', 'actuator_skew')
            for force in (None, (3.837, (14., 0., 0.))) for arm in (arms if rep == 0 else arms[::-1])]
    for variant in ('stock', 'actuator_skew'):
        for twin in ('none', 'impact'):
            local = [r for r in rows if r['mechanics'] == variant and r['twin'] == twin]
            if len({r['physical_and_residual_sha256'] for r in local}) != 1 or any(
                len({r['alarm_sha256'] for r in local if r['arm'] == a}) != 1 for a in arms):
                raise ValueError('protected trace or repeated detection changed')
    sums = {}
    for a in arms:
        matches = [r for r in rows if r['arm'] == a]
        sums[a] = {'cycles': sum(r['cycles'] for r in matches),
                   'max_reset_wall_us': max(r['reset_wall_us'] for r in matches)}
        for t in ('wall', 'thread_cpu'):
            sums[a][t] = {'worst_episode_p99_us': max(r[t]['p99_us'] for r in matches),
                          'worst_sample_us': max(r[t]['max_us'] for r in matches),
                          'samples_ge_1ms': sum(r[t]['at_least_1ms'] for r in matches)}
    return {'schema': 'week09-r24-cpu-sample.v1', 'episodes': len(rows),
            'scope': 'Measured on one shared Linux computer. Control.compute and detector.update only, paired by tick. Detector reset measured separately; import, other solver/plant work, IO, and scheduling guarantees excluded.',
            'hard_realtime_proven': False, 'summary': sums, 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(result['summary'])
