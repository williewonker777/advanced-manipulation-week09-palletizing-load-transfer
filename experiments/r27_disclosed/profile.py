"""Paired controller and detector timing on the two disclosed stock loaded impacts."""
import argparse
import json
import os
from pathlib import Path
from types import FunctionType

import mujoco
from experiments.r22_timing.study import bind, digest, stats
from experiments.r26_holdout import evaluator as lab
from experiments.r27_disclosed.contrasts import BASE, HERE, resolve
from my_torque import Controller

CANDIDATE = 'experiments.r27_load_change.manifold:Detector'


def measure():
    if mujoco.__version__ != '3.17.0':
        raise ValueError('requires the original course environment')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    cases = json.loads((HERE / 'policy.json').read_bytes())['disclosed_contrasts']['cases']
    plan = json.loads(lab.PLAN.read_bytes())
    series = []
    for rep in range(2):
        for name in cases:
            case = next(c for c in plan['single'] if c['id'] == name)
            for entry in ((BASE, CANDIDATE) if rep == 0 else (CANDIDATE, BASE)):
                ctrl, det = Controller(), resolve(entry)()
                cw, ct, cn = bind(ctrl, 'compute', 6000)
                dw, dt, dn = bind(det, 'update', 6000)
                native = lab.run_episode
                run = FunctionType(native.__code__, {**native.__globals__, 'Controller': lambda: ctrl},
                                   native.__name__, native.__defaults__, native.__closure__)
                _, log = run(case, plan['variants']['stock'], lambda: det, (case['impact'], None))
                if cn != [6000] or dn != [6000]:
                    raise ValueError('missing per-tick measurement')
                start, _ = lab.actual_start(log['t'], case['impact'])
                cm = lab.common(log, (case['impact'], None), [start], plan)
                if not cm['force_exact']['passed']:
                    raise ValueError('incorrect applied force')
                series.append(dict(repeat=rep, case=name, entry=entry,
                                   physical_sha256=lab.certified.trace_signature({'single':log}), alarm_sha256=digest(log,['detect']),
                                   samples=6000, wall=stats(cw+dw), thread=stats(ct+dt),
                                   detector_thread=stats(dt)))
    for name in cases:
        subset = [r for r in series if r['case'] == name]
        if len({r['physical_sha256'] for r in subset}) != 1 or any(
                len({r['alarm_sha256'] for r in subset if r['entry']==entry}) != 1 for entry in (BASE,CANDIDATE)):
            raise ValueError('physics or repeated alarm stream changed')
    totals = {}
    for entry in (BASE,CANDIDATE):
        rows = [row for row in series if row['entry']==entry]
        totals[entry] = dict(samples=sum(r['samples'] for r in rows),
            worst_episode_p99_wall_us=max(r['wall']['p99_us'] for r in rows),
            worst_episode_p99_thread_us=max(r['thread']['p99_us'] for r in rows),
            max_wall_us=max(r['wall']['max_us'] for r in rows),
            wall_at_least_1ms=sum(r['wall']['at_least_1ms'] for r in rows))
    return dict(schema='r27-known-loaded-shared-timing.v1',episodes=8,simulation_only=True,
                note='Controller.compute + Detector.update on each 1 ms tick. Two actual force cases, two repeats, reversed arm order. Shared CPU with affinity, no end-to-end or hard real-time guarantee.',
                sources=[BASE,CANDIDATE],rows=series,totals=totals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.path.lexists(args.out):
        parser.error('fresh output file required')
    result = measure()
    with args.out.open('x') as file:
        json.dump(result,file,sort_keys=True,indent=2,allow_nan=False)
        file.write('\n')
    print(result['totals'])
