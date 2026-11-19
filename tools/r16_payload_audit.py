"""Portable ORIGINAL-stock audit for the frozen R16 3.447 s payload sensor.

The pooled normal-only coefficients are shipped transparently as derived JSON.
An ORIGINAL stock kit can reproduce the 20 stock evaluation rows, but cannot
independently regenerate the 12 perturbed source normals used by calibration.
No raw sensor captures or kit files are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

try:
    from tools import r13_grasp_audit as base
except ImportError:
    import r13_grasp_audit as base


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('none', 'explicit_zero', 'xy', 'up', 'down')
SOURCE_PATHS = {
    'torque': 'r1_torque.py', 'detector': 'my_detector.py',
    'base_audit': 'tools/r13_grasp_audit.py',
    'estimator': 'tools/r13_grasp_estimator.py',
    'stream': 'experiments/r14_online/stream.py',
    'robust': 'experiments/r16_robust_prefix/robust.py',
    'portable': 'experiments/r16_robust_prefix/portable.py',
    'plan': 'experiments/r16_robust_prefix/plan.json',
    'calibration': 'experiments/r16_robust_prefix/portable_calibration.json',
    'reference': 'experiments/r16_robust_prefix/portable_stock_reference.json',
    'kit_manifest': 'tools/kit_manifest.sha256',
}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def validate_inputs(plan, calibration, reference):
    if (plan.get('schema') != 'r16-robust-prefix-plan.v1'
            or tuple(plan.get('variants', ())) != VARIANTS
            or len(plan.get('fresh_groups', ())) != 4):
        raise ValueError('frozen R16 four-group stock plan required')
    ids = []
    for case in plan['fresh_groups']:
        ids.append(case.get('id')); mass = float(case['mass']); com = np.asarray(case['com'], float)
        shift = np.asarray(case['shift'], float)
        if (mass <= 0 or not np.isfinite(mass) or com.shape != (3,) or shift.shape != (3,)
                or not np.isfinite(com).all() or not np.isfinite(shift).all()):
            raise ValueError('finite payload and pickup shift required')
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate group id')
    if (calibration.get('schema') != 'r16-robust-prefix-calibration.v1'
            or float(calibration.get('deadline_s', -1)) != 3.447
            or float(calibration.get('normal_guard_mm', 99)) > 2.5):
        raise ValueError('frozen bounded R16 calibration required')
    provenance = calibration.get('portable_provenance', {})
    if (provenance.get('truth_used_in_fit') is not False
            or provenance.get('raw_records_shipped') is not False
            or 'cannot independently regenerate' not in provenance.get('stock_regeneration', '')):
        raise ValueError('transparent pooled-calibration provenance required')
    if (reference.get('schema') != 'r16-portable-stock-reference.v1'
            or len(reference.get('rows', ())) != 20
            or len(reference.get('explicit_zero_audit', ())) != 4):
        raise ValueError('frozen stock parity reference required')


def offset(case, variant):
    if variant == 'none':
        return None
    if variant == 'explicit_zero':
        return (0., 0., 0.)
    x, y, z = map(float, case['shift'])
    return (x, y, {'xy': 0., 'up': z, 'down': -z}[variant])


def physics_digest(log):
    return {key: hashlib.sha256(np.asarray(log[key]).tobytes()).hexdigest()
            for key in ('q', 'tau', 'f_tool', 'f_table', 'r', 'detect')}


def measures(rows):
    norm = lambda key: np.array([np.linalg.norm(row[key]) for row in rows])
    truth, error, action = norm('true_xy_mm'), norm('point_error_mm'), norm('action_error_mm')
    tiny, large = truth < 1., truth >= 3.
    misses = sum(row['accept'] and not row['covered'] for row in rows)
    signed = sum(bool(flag and row['accept'] and np.dot(row['true_xy_mm'], row['model_action_xy_mm']) > 0)
                 for flag, row in zip(large, rows))
    return {'count': len(rows), 'vector_rmse_mm': float(np.sqrt(np.mean(error**2))),
            'max_action_error_mm': float(action.max()), 'accepted': int(sum(row['accept'] for row in rows)),
            'abstained': int(sum(not row['accept'] for row in rows)),
            'accepted_interval_misses': int(misses), 'false_tiny_actions': int(sum(flag and row['accept'] for flag, row in zip(tiny, rows))),
            'big_n': int(large.sum()), 'big_signed': int(signed),
            'qualified': bool(np.sqrt(np.mean(error**2)) <= 1.5 and action.max() <= 2.
                              and misses == 0 and not any(flag and row['accept'] for flag, row in zip(tiny, rows))
                              and signed == int(large.sum()))}


def _row_projection(row, fields):
    return {key: row[key] for key in fields}


def run_stock(plan, calibration, reference):
    """Execute only the authenticated stock runner inside the temporary overlay."""
    from sim.runner import run_episode
    import collision_detector
    from experiments.r16_robust_prefix.portable import feature_from_samples
    from experiments.r16_robust_prefix.robust import decide

    validate_inputs(plan, calibration, reference)
    controller = importlib.import_module('torque_controller').Controller
    if getattr(controller, 'MODE', None) != 'torque':
        raise ValueError('R1 torque controller required')
    rows, capsules = [], []
    for case in plan['fresh_groups']:
        for variant in VARIANTS:
            obj, samples = base.capture(controller)
            metrics, log = run_episode(obj, float(case['mass']), tuple(case['com']),
                                       collision_detector.Detector(), pick_offset=offset(case, variant))
            if metrics['diverged'] or len(samples['t']) not in (3550, 3551):
                raise ValueError('incomplete stock episode')
            prediction = decide(feature_from_samples(samples, case), calibration)
            rows.append({'id': case['id'], 'variant': variant, **prediction,
                         'run': base.compact_metrics(metrics)})
            capsules.append({'geometry': base.score_capsule(log), 'com': np.asarray(case['com']),
                             'digests': physics_digest(log), 'tool_peak_n': float(np.max(log['f_tool']))})

    prediction_blob = json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    prediction_seal = sha(prediction_blob)
    truths = [base.observed_geometry(item['geometry'], item['com']) for item in capsules]
    baselines, explicit = {}, []
    for row, actual, capsule in zip(rows, truths, capsules, strict=True):
        identifier, variant = row['id'], row['variant']; absolute = np.asarray(actual['local_com_shift_mm'])[:2]
        if variant == 'none':
            baselines[identifier] = {'com': absolute, 'digests': capsule['digests'],
                                     'tool_peak_n': capsule['tool_peak_n']}
        truth = absolute-baselines[identifier]['com']; point = np.asarray(row['relative_xy_mm'])-truth
        action = np.asarray(row['model_action_xy_mm'])-truth
        row.update(true_xy_mm=truth.tolist(), point_error_mm=point.tolist(),
                   action_error_mm=action.tolist(), covered=bool(np.linalg.norm(point) <= row['halfwidth_radius_mm']))
        if variant == 'explicit_zero':
            initial = baselines[identifier]
            explicit.append({'id': identifier, 'true_xy_change_mm': truth.tolist(),
                             'none_physics_digests': initial['digests'],
                             'explicit_zero_physics_digests': capsule['digests'],
                             'none_tool_peak_n': initial['tool_peak_n'],
                             'explicit_zero_tool_peak_n': capsule['tool_peak_n']})

    fields = reference['row_fields']; expected = reference['rows']
    projected = [_row_projection(row, fields) for row in rows]
    row_parity = projected == expected
    explicit_parity = explicit == reference['explicit_zero_audit']
    if not row_parity or not explicit_parity:
        raise AssertionError('stock reproduction differs from frozen research fields')
    return {'rows': rows, 'metrics': measures(rows), 'explicit_zero_audit': explicit,
            'prediction_seal_sha256': prediction_seal, 'predictions_sealed_before_truth': True,
            'research_row_parity': row_parity, 'research_explicit_zero_parity': explicit_parity}


def source_inputs():
    paths = {key: ROOT/relative for key, relative in SOURCE_PATHS.items()}; paths['audit'] = Path(__file__)
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()):
        raise ValueError('regular frozen public audit inputs required')
    return base.snapshot_files(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True, help='licensed ORIGINAL stock kit')
    parser.add_argument('--out', type=Path, required=True, help='NEW output directory')
    args = parser.parse_args(argv); kit, out = base.validate_paths(args.kit, args.out, parser)
    blobs, pins = source_inputs(); plan = json.loads(blobs['plan']); calibration = json.loads(blobs['calibration']); reference = json.loads(blobs['reference'])
    validate_inputs(plan, calibration, reference)
    sys.path.insert(0, str(ROOT))
    from tools.validate_submission import prepare_overlay
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='r16-payload-audit-', dir=out.parent) as folder:
        overlay = Path(folder); prepare_overlay(kit, overlay, 'my', 'torque')
        if base.digest(overlay/'tools/kit_manifest.sha256') != pins['kit_manifest']:
            raise ValueError('kit manifest differs from frozen audit input')
        destinations = {'torque': 'torque_controller.py', 'detector': 'collision_detector.py',
            'base_audit': 'tools/r13_grasp_audit.py', 'estimator': 'tools/r13_grasp_estimator.py',
            'stream': 'experiments/r14_online/stream.py', 'robust': 'experiments/r16_robust_prefix/robust.py',
            'portable': 'experiments/r16_robust_prefix/portable.py', 'plan': 'plan.json',
            'calibration': 'calibration.json', 'reference': 'reference.json', 'audit': 'payload_audit.py'}
        for package in ('tools', 'experiments', 'experiments/r14_online', 'experiments/r16_robust_prefix'):
            path = overlay/package; path.mkdir(parents=True, exist_ok=True); (path/'__init__.py').write_text('')
        for relative in destinations.values():
            (overlay/relative).parent.mkdir(parents=True, exist_ok=True)
        for key, relative in destinations.items():
            base.install_snapshot(blobs[key], pins[key], overlay/relative)
        base.install_snapshot(blobs['base_audit'], pins['base_audit'], overlay/'r13_grasp_audit.py')
        code = ('import json,payload_audit,sys;from pathlib import Path;'
                'json.dump(payload_audit.run_stock(json.loads(Path("plan.json").read_text()),'
                'json.loads(Path("calibration.json").read_text()),json.loads(Path("reference.json").read_text())),sys.stdout,allow_nan=False)')
        process = subprocess.run([sys.executable, '-B', '-c', code], cwd=overlay, env=environment,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode:
            raise RuntimeError('stock payload audit failed:\n'+process.stderr)
        result = json.loads(process.stdout)

    scope = {'mode': 'torque', 'episodes': 20, 'calibration_episodes': 0, 'evaluation_episodes': 20,
             'runner': 'authenticated ORIGINAL stock sim.runner.run_episode', 'inference_deadline_s': 3.447,
             'predictions_sealed_before_truth': True, 'raw_sensor_captures_emitted': False,
             'perturbed_calibration_regenerable_from_stock': False, 'collision_episodes': 0}
    evaluation = {'schema': 'r16-payload-evaluation-audit.v1', 'pins': pins, 'scope': scope,
                  'prediction_seal_sha256': result['prediction_seal_sha256'], 'rows': result['rows'],
                  'explicit_zero_audit': result['explicit_zero_audit'],
                  'research_row_parity': result['research_row_parity'],
                  'research_explicit_zero_parity': result['research_explicit_zero_parity']}
    summary = {'schema': 'r16-payload-audit.v1', 'pins': pins, 'scope': scope,
               'calibration_provenance': calibration['portable_provenance'],
               'prediction_seal_sha256': result['prediction_seal_sha256'], 'metrics': result['metrics'],
               'research_row_parity': result['research_row_parity'],
               'research_explicit_zero_parity': result['research_explicit_zero_parity'],
               'decision': 'stock sensing reproduction passed' if result['metrics']['qualified'] else 'stock sensing reproduction failed'}
    out.mkdir()
    for name, document in (('evaluation.json', evaluation), ('summary.json', summary)):
        with (out/name).open('x') as stream:
            json.dump(document, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'episodes': 20, 'qualified': result['metrics']['qualified'],
                      'vector_rmse_mm': result['metrics']['vector_rmse_mm'],
                      'accepted': result['metrics']['accepted'], 'abstained': result['metrics']['abstained'],
                      'research_row_parity': result['research_row_parity']}))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
