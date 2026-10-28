"""Control/load/wrench and exact sharing contribution, no collision tuning."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.sweep import (available_memory_bytes, check_integrity, evaluate_control_metrics,
                         timed_classes, timing_stats)
from sim.runner import run_episode
import my_detector


def episode(mode, blend=0., shared=True, mass=8., com=(.02, -.01, .03)):
    base = importlib.import_module('my_'+mode).Controller
    class Variant(base):
        USE_SUPPORT_LOAD = blend > 0
        def reset(self):
            super().reset()
            self.shared_estimator.SUPPORT_BLEND = blend
    class D(my_detector.Detector):
        USE_SHARED = shared
        def reset(self):
            super().reset()
            self.USE_SUPPORT_LOAD = blend > 0
            self.SUPPORT_BLEND = blend
    ctrl_class, det_class = timed_classes(Variant, D)
    ctrl, det = ctrl_class(), det_class()
    result, log = run_episode(ctrl, box_mass=mass, box_com=com, detector=det)
    ct, dt = ctrl._sweep_compute_seconds, det._sweep_update_seconds
    row = dict(mode=mode, blend=blend, shared=shared, mass=mass, com=com,
               **evaluate_control_metrics(result, log))
    row.update(torque_jitter_nm=result['torque_jitter'],
               controller_estimator_samples=ctrl.shared_estimator.sample_count,
               detector_estimator_samples=det.sample_count,
               estimator_total_samples=ctrl.shared_estimator.sample_count + (0 if shared else det.sample_count),
               timing={"controller": timing_stats(ct), "detector": timing_stats(dt),
                       "combined": timing_stats(np.asarray(ct)+np.asarray(dt))})
    last = (log['t'] >= 3.88) & (log['t'] < 3.9)
    row['table_last20ms_mean_n'] = float(np.mean(log['f_table'][last]))
    row['table_last20ms_min_n'] = float(np.min(log['f_table'][last]))
    return row, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--parity-only', action='store_true')
    args = parser.parse_args()
    if available_memory_bytes() < 6*1024**3:
        raise SystemExit('RAM below6GiB')
    check_integrity()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    rows, parity = [], []
    for mode in ('servo', 'torque'):
        blends = (0., .1) if args.parity_only else (0., .1, .25, 1.)
        for blend in blends:
            on, log_on = episode(mode, blend, True)
            rows.append(on)
            np.savez_compressed(args.evidence_dir/f'{mode}_support{blend:g}.npz',
                                **{k: v for k, v in log_on.items() if isinstance(v, np.ndarray)})
            if blend in (0., .1):
                off, log_off = episode(mode, blend, False)
                rows.append(off)
                differences = {k: float(np.max(np.abs(log_on[k].astype(float)-log_off[k].astype(float))))
                               for k in log_on if isinstance(log_on[k], np.ndarray) and k != 't_ctrl'}
                parity.append(dict(mode=mode, blend=blend, max_differences=differences,
                                   exact_all=all(v == 0 for v in differences.values()),
                                   shared_samples=on['estimator_total_samples'],
                                   independent_samples=off['estimator_total_samples']))
    content = dict(official=False, purpose='no-collision support-fusion feasibility and exact sharing contribution',
                   rows=rows, parity=parity)
    (ROOT/'results/D4_SHARED_ABLATION.json').write_text(json.dumps(content, indent=2)+'\n')
    print(json.dumps(content, indent=2))


if __name__ == '__main__':
    main()
