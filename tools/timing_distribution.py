"""Wall-clock compute/update cycle distributions; no real-time guarantee."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools.sweep import available_memory_bytes, check_integrity, timed_classes, timing_stats
from sim.runner import run_episode
import my_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    args = parser.parse_args()
    if available_memory_bytes() < 6*1024**3:
        raise SystemExit('RAM below6GiB')
    check_integrity()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    bins = np.r_[np.linspace(0, 1000, 101), 2000, 5000, 10000, 1000000]
    for mode in ('servo', 'torque'):
        cls = importlib.import_module('my_'+mode).Controller
        for mass in (0., 4., 8.):
            controller_type, detector_type = timed_classes(cls, my_detector.Detector)
            ctrl, det = controller_type(), detector_type()
            run_episode(ctrl, box_mass=mass, detector=det)
            values = dict(controller=np.asarray(ctrl._sweep_compute_seconds),
                          detector=np.asarray(det._sweep_update_seconds))
            values['combined'] = values['controller'] + values['detector']
            np.savez_compressed(args.raw_dir/f'timing_{mode}_{mass:g}.npz', **values)
            components = {}
            for component, seconds in values.items():
                counts, edges = np.histogram(seconds*1e6, bins)
                components[component] = dict(**timing_stats(seconds),
                    cycles_at_least_1ms=int(np.sum(seconds >= .001)),
                    histogram_edges_us=edges.tolist(), histogram_counts=counts.tolist())
            rows.append(dict(controller='my_'+mode, mass_kg=mass, components=components))
    data = dict(official=False, definition='sum of separately perf_counter-timed compute and update calls',
                interpretation='shared desktop wall-clock; all outliers retained, not a hard RT guarantee', rows=rows)
    (ROOT/'results/FINAL_TIMING_DISTRIBUTION.json').write_text(json.dumps(data, indent=2)+'\n')
    print([(r['controller'],r['mass_kg'],r['components']['combined']['p99_us'],
            r['components']['combined']['max_us'],r['components']['combined']['cycles_at_least_1ms']) for r in rows])


if __name__ == '__main__':
    main()
