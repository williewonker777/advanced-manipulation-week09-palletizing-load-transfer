"""Paired C1 timing/bit-parity on the unmodified 6 s kit runner, R1 control.

Requires the licensed local kit, not raw historical logs. This is a numerical
regression and a finite latency measurement, not a new detection holdout.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.check_integrity import check_integrity


def module(path):
    spec = importlib.util.spec_from_file_location("timing_" + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


# Development pilot; separate fixed acceptance cases include an empty hand,
# payload/COM variation, offset/tool contact, weak contact, and a normal cycle.
PILOT = [("servo", 6.1, (.014, -.009, .016), None, None)]
FAST1_CASES = [
    ("servo", 0., (0., 0., 0.), (1.448, (0., -13., 0.)), None),
    ("servo", 2.37, (-.017, .013, .021), None, None),
    ("servo", 7.41, (.011, -.018, -.008), (3.839, (12., 0., 0.)), None),
    ("servo", 5.63, (-.014, -.012, .017), None, (.016, -.007, .003)),
    ("torque", 3.19, (-.016, .019, -.012), (3.027, (0., 0., -14.)), None),
    ("torque", 7.23, (.012, -.014, .006), None, None),
]


def hexdigest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def run_one(case, detector_cls):
    from sim.runner import run_episode

    mode, mass, com, collision, offset = case
    controller_cls = module(ROOT / f'r1_{mode}.py').Controller

    class Controller(controller_cls):
        def compute(self, *args):
            w, c = time.perf_counter_ns(), time.thread_time_ns()
            out = super().compute(*args)
            cpu, wall = time.thread_time_ns() - c, time.perf_counter_ns() - w
            ctrl_times.append((wall, cpu))
            return out

    class Detector(detector_cls):
        def reset(self):
            w = time.perf_counter_ns()
            super().reset()
            self.reset_ms = (time.perf_counter_ns() - w) * 1e-6

        def update(self, *args):
            w, c = time.perf_counter_ns(), time.thread_time_ns()
            out = super().update(*args)
            cpu, wall = time.thread_time_ns() - c, time.perf_counter_ns() - w
            det_times.append((wall, cpu))
            branches.append((self.r5_base_detect, self.r5_extra, self.r8_base_detect, self.r8_extra))
            return out

    ctrl_times, det_times, branches = [], [], []
    det = Detector()
    metrics, log = run_episode(Controller(), mass, com, detector=det, collision=collision,
                               pick_offset=offset)
    combined = np.asarray(ctrl_times) + np.asarray(det_times)
    if metrics['diverged'] or combined.shape != (6000, 2):
        raise ValueError('incomplete or diverged episode')
    wall, cpu = combined.T / 1000.
    return {
        'trace_sha256': {k: hexdigest(log[k]) for k in
                         ('q', 'q_d', 'tau', 'qpos', 'e_mm', 'f_table', 'f_tool', 'r', 'detect')},
        'branches_sha256': hexdigest(np.array(branches, dtype=bool)),
        'extra_cycles': int(np.asarray(branches)[:, -1].sum()),
        'cycles': len(wall), 'reset_ms': det.reset_ms,
        'p99_us': float(np.percentile(wall, 99)), 'max_wall_us': float(wall.max()),
        'max_cpu_us': float(cpu.max()), 'wall_ge_1ms': int((wall >= 1000).sum()),
        'cpu_ge_1ms': int((cpu >= 1000).sum()),
        'outliers': [[int(k), wall[k], cpu[k]] for k in np.flatnonzero((wall >= 1000) | (cpu >= 1000))],
    }


def compare(candidate, cases):
    check_integrity(ROOT)
    reference = ROOT / 'experiments/r8_final/C1.py'
    files = {'C1': reference, 'R9': candidate}
    rows = []
    for i, case in enumerate(cases):
        row = {'case': case}
        # Alternate order; include every measured cycle and every outlier.
        for variant in (('C1', 'R9') if i % 2 == 0 else ('R9', 'C1')):
            row[variant] = run_one(case, module(files[variant]).Detector)
        row['identical'] = all(row['C1'][key] == row['R9'][key]
                               for key in ('trace_sha256', 'branches_sha256'))
        rows.append(row)
    servo = [r for r in rows if r['case'][0] == 'servo']
    p99 = {v: max(row[v]['p99_us'] for row in servo) for v in files}
    return {'scope': 'paired full-loop R1 control; not new detection efficacy or a deadline guarantee',
            'source_sha256': {v: hashlib.sha256(p.read_bytes()).hexdigest() for v, p in files.items()},
            'rows': rows, 'identical': all(r['identical'] for r in rows),
            'worst_servo_p99_us': p99,
            'timing_gate': p99['R9'] <= min(.70 * p99['C1'], 800.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'my_detector.py')
    suite = parser.add_mutually_exclusive_group()
    suite.add_argument('--pilot', action='store_true')
    suite.add_argument('--fast1-cases', action='store_true', help='earlier development comparator')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cases = (PILOT if args.pilot else FAST1_CASES if args.fast1_cases else
             json.loads((ROOT / 'results/r9/servo_timing_cases.json').read_text()))
    result = compare(args.candidate, cases)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('identical:', result['identical'], 'worst servo p99 (us):', result['worst_servo_p99_us'])
    return 0 if result['identical'] and result['timing_gate'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
