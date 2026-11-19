"""Measure controller+detector deadlines in the kit with a bounded recorder.

Does not change GC, scheduling or affinity. Reports GC that actually started
inside either call; overlap alone does not assign the entire delay to GC.
"""
import argparse
import gc
import hashlib
import os
import platform
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity
from tools.r9_timing import module


def observe(mode, detector, repeat, probe_lists=False):
    from sim.runner import run_episode

    misses = []
    start_count = [0, 0, 0]

    def on_gc(phase, info):
        if phase == 'start':
            start_count[info['generation']] += 1

    controller = module(ROOT / ('my_' + mode + '.py')).Controller
    detector_cls = module(detector).Detector
    rows = []
    for episode in range(repeat):
        times = np.zeros((2, 6000, 2), dtype=np.int64)
        collections = np.zeros((2, 6000, 3), dtype=np.uint8)
        # Optional control reproduces the growing *observer* bookkeeping of the
        # earlier paired run, not overhead from the submitted algorithm itself.
        kept = [] if probe_lists else None

        def measured(side, method, args):
            k = round(float(args[0]) * 1000)
            before = tuple(start_count)
            w, c = time.perf_counter_ns(), time.thread_time_ns()
            out = method(*args)
            cpu, wall = time.thread_time_ns() - c, time.perf_counter_ns() - w
            times[side, k] = wall, cpu
            if before != tuple(start_count):
                collections[side, k] = [a-b for a, b in zip(start_count, before)]
            if kept is not None:
                kept.append((wall, cpu, k))
            return out

        class Controller(controller):
            def compute(self, *args):
                return measured(0, super().compute, args)

        class Detector(detector_cls):
            def update(self, *args):
                return measured(1, super().update, args)

        gc.callbacks.append(on_gc)
        try:
            metrics, _ = run_episode(Controller(), 3.037, (-.006, .012, -.014),
                                     detector=Detector())
        finally:
            gc.callbacks.remove(on_gc)
        if metrics['diverged'] or (times == 0).any():
            raise ValueError('incomplete cycle series')
        wc = times.sum(axis=0) * .001
        for k in np.flatnonzero(np.any(wc >= 1000, axis=1)):
            misses.append(dict(episode=episode, cycle=int(k), wall_us=wc[k, 0], cpu_us=wc[k, 1],
                               gc_starts=collections[:, k].tolist()))
        rows.append(dict(p99_wall_us=float(np.percentile(wc[:, 0], 99)),
                         max_wall_us=float(wc[:, 0].max()), max_cpu_us=float(wc[:, 1].max()),
                         wall_ge_1ms=int(np.count_nonzero(wc[:, 0] >= 1000)),
                         cpu_ge_1ms=int(np.count_nonzero(wc[:, 1] >= 1000)),
                         gc_starts_during_calls=collections.sum(axis=(0, 1)).tolist()))
    return dict(mode=mode, cycles=repeat*6000, gc_policy_unchanged=True,
                growing_recorder_control=probe_lists, episodes=rows, outliers=misses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', default=ROOT / 'my_detector.py', type=Path)
    parser.add_argument('--mode', choices=('servo', 'torque'), default='servo')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--growing-recorder', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    check_integrity(ROOT)
    result = observe(args.mode, args.detector, args.episodes, args.growing_recorder)
    import mujoco
    result['sources'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
        ('detector', args.detector), ('controller', ROOT / f'my_{args.mode}.py'),
        ('kit_manifest', ROOT / 'tools/kit_manifest.sha256'), ('probe', Path(__file__)))}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
                                 mujoco=mujoco.__version__, kernel=platform.release(),
                                 arch=platform.machine(), gc_threshold=gc.get_threshold(),
                                 affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                                 scheduler=os.sched_getscheduler(0) if hasattr(os, 'sched_getscheduler') else None,
                                 blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                                 omp_threads=os.environ.get('OMP_NUM_THREADS'))
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
