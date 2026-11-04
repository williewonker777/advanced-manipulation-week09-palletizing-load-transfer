"""Serial descriptive timing of equivalent R2/R3 P-IDC observation replays.

No plant is advanced.  This compares an already-rejected R2 law to the fixed
R3 B-mass setting at a registered mass where alpha is identically one.  It is
not a new candidate configuration or a source of threshold/parameter tuning.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import r2_robust
from tools import r3_study as R
from tools import r3_resources
from tools.r3_stability import make_detector
from tools.r2_stability import make_controller
from tools import v2_study as V
from sim import trajectory as T
import r1_detector


def compare(private_dir):
    document = json.loads((R.D / 'R3_VALIDATION_R1_torque.json').read_text())
    row = document['results'][-1]
    spec = next(value for value in R.specifications() if value['id'] == 'B_mass')
    if row['mass'] < spec['mass_full_kg']:
        raise RuntimeError('equivalent replay requires constant alpha=1')
    raw_path = private_dir / row['private_raw_basename']
    if R.sha(raw_path) != row['private_raw_sha256']:
        raise RuntimeError('validation observation trace hash mismatch')
    with np.load(raw_path, allow_pickle=False) as archive:
        observations = {name: archive[name] for name in (
            'obs_t', 'encoder', 'encoder_velocity', 'motor_torque', 'jts', 'grip')}
    old = r2_robust.make_controller('torque', dict(
        family='C4', law='P-IDC', rate=4.5, damping_scale=1.0))
    new, _ = R.candidate_classes('torque', spec)
    outputs, records = [], []
    for label, controller_class in (('R2_unoptimized', old), ('R3_optimized_alpha1', new)):
        r3_resources.wait_for_capacity()
        timed_controller, timed_detector = V.timed_classes(controller_class, r1_detector.Detector)
        controller = make_controller('torque', row['mass'], tuple(row['com']), timed_controller)
        detector = make_detector(timed_detector, controller, row['mass'], tuple(row['com']))
        applied = []
        for index, time_s in enumerate(observations['obs_t']):
            th, thd, jts = (observations[name][index].copy()
                            for name in ('encoder', 'encoder_velocity', 'jts'))
            grip = bool(observations['grip'][index])
            detector.update(float(time_s), th, thd,
                            observations['motor_torque'][index].copy(), jts, grip)
            applied.append(controller.compute(float(time_s), th, thd, jts, grip,
                                               *T.reference(float(time_s))))
        outputs.append(np.asarray(applied))
        records.append(dict(label=label, timing=V.timing_summary(controller, detector)))
    return dict(schema='R3_REPLAY_TIMING.v1', episodes_run=0,
                method='6000 identical allowed-observation cycles each; R2 then R3, serial, no warmup/outlier removal',
                scope='descriptive implementation-cost comparison only; not closed-loop efficacy or selection',
                limitations='single fixed order and host realization; no randomized benchmark or global real-time guarantee',
                mass_kg=row['mass'], com=row['com'], input_sha256=row['private_raw_sha256'],
                maximum_torque_output_abs_difference_nm=float(np.abs(outputs[0] - outputs[1]).max()),
                global_gc_changes=False, results=records,
                source_sha256={name: R.sha(ROOT / name) for name in (
                    'experiments/r2_robust.py', 'experiments/r2_prne.py',
                    'experiments/r3_generated/B_mass_torque.py', 'tools/r3_replay_timing.py',
                    'r1_torque.py', 'r1_detector.py', 'tools/r2_stability.py',
                    'tools/v2_study.py', 'sim/nominal.py')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('replay benchmark already recorded')
    private_dir = R.require_private(args.private_dir)
    result = compare(private_dir)
    R.write(args.output, result)
    print('output discrepancy:', result['maximum_torque_output_abs_difference_nm'])
    for row in result['results']:
        print(row['label'], row['timing']['combined']['wall'])


if __name__ == '__main__':
    main()
