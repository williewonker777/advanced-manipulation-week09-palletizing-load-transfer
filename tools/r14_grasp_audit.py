"""Portable stock-kit validation of R14 registered grasp-height sensing.

R13 raw and R14 registered estimates are computed from the same recorded public
signals.  Twelve nominal pickups per mode calibrate both models; four disjoint
geometry groups with zero, lateral, up and down variants are evaluated.  Actual
free-box state is opened only after all predictions for a mode are complete.

The stock runner supports the authored ``pick_offset`` but not the private
research perturbation fixture.  Results therefore validate the registered
method on authenticated stock dynamics and explicitly do not claim parity with
the perturbed research evidence.
"""
import argparse
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
except ImportError:  # Absolute CLI path or isolated overlay package.
    try:
        import grasp_audit_base as base
    except ModuleNotFoundError:
        import r13_grasp_audit as base


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('zero', 'xy', 'up', 'down')


def validate_plan(plan):
    if (plan.get('schema') != 'r14-airborne.v1' or len(plan.get('fit', ())) != 12
            or len(plan.get('test', ())) != 4):
        raise ValueError('unexpected frozen R14 plan shape')
    identifiers = []
    for split in ('fit', 'test'):
        for case in plan[split]:
            identifiers.append((split, case.get('id')))
            mass = float(case['mass'])
            com = np.asarray(case['com'], dtype=float)
            if mass <= 0 or not np.isfinite(mass) or com.shape != (3,) or not np.isfinite(com).all():
                raise ValueError('invalid payload in frozen R14 plan')
            if split == 'test':
                shift = np.asarray(case['shift'], dtype=float)
                if shift.shape != (3,) or not np.isfinite(shift).all():
                    raise ValueError('invalid pickup displacement in frozen R14 plan')
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('duplicate case id in frozen R14 plan')


def strict_action(result):
    return float(result['extra_height_mm']) if result['accept'] else 0.0


def policy_metrics(rows, key):
    selected = [row[key] for row in rows]
    vertical = [(row, row[key]) for row in rows if row['variant'] in ('up', 'down')]
    controls = [(row, row[key]) for row in rows if row['variant'] in ('zero', 'xy')]
    large = [(row, row[key]) for row in rows if abs(row['truth_mm']) >= 2.5]
    tiny = [(row, row[key]) for row in rows if abs(row['truth_mm']) < 1.0]
    point_error = [result['extra_height_mm'] - row['truth_mm'] for row, result in
                   [(row, row[key]) for row in rows]]
    action_error = [result['applied_mm'] - row['truth_mm'] for row, result in
                    [(row, row[key]) for row in rows]]
    rms = lambda values: float(np.mean(np.square(values)) ** .5)
    same_sign = lambda row, result: np.sign(result['extra_height_mm']) == np.sign(row['truth_mm'])
    misses = sum(result['accept'] and not result['covered'] for result in selected)
    large_good = sum(result['accept'] and same_sign(row, result)
                     and abs(result['applied_mm'] - row['truth_mm']) <= 2.0
                     for row, result in large)
    vertical_error = [result['extra_height_mm'] - row['truth_mm'] for row, result in vertical]
    return {
        'cases': len(rows),
        'accepted': int(sum(result['accept'] for result in selected)),
        'accepted_interval_misses': int(misses),
        'point_rmse_mm': rms(point_error),
        'vertical_point_rmse_mm': rms(vertical_error),
        'control_max_abs_point_error_mm': max(abs(result['extra_height_mm'] - row['truth_mm'])
                                               for row, result in controls),
        'action_rmse_mm': rms(action_error),
        'max_abs_action_error_mm': max(map(abs, action_error)),
        'vertical_sign_correct': int(sum(same_sign(row, result) for row, result in vertical)),
        'vertical_accepted_sign_correct': int(sum(result['accept'] and same_sign(row, result)
                                                  for row, result in vertical)),
        'large_cases': len(large),
        'large_signed_accurate_actions': int(large_good),
        'tiny_cases': len(tiny),
        'tiny_nonzero_actions': int(sum(result['accept'] for _, result in tiny)),
        'original_strict_gate': bool(
            sum(result['accept'] and same_sign(row, result) for row, result in vertical) == len(vertical)
            and rms(vertical_error) <= 2.0
            and max(abs(result['extra_height_mm'] - row['truth_mm'])
                    for row, result in controls) <= 2.0
            and misses == 0),
        'functional_gate': bool(
            rms(point_error) <= 2.0 and max(map(abs, action_error)) <= 2.0 and misses == 0
            and bool(large) and large_good >= .75 * len(large)
            and not any(result['accept'] for _, result in tiny)),
    }


def run_mode(mode, plan):
    """Execute 28 episodes and compare both estimators on identical features."""
    from sim import config as config
    from sim.runner import run_episode
    import collision_detector
    import registration
    from tools import r13_grasp_estimator as old

    validate_plan(plan)
    controller = importlib.import_module(mode + '_controller').Controller
    if getattr(controller, 'MODE', None) != mode:
        raise ValueError('controller mode mismatch')

    fit_features = []
    fit_runs = []
    for case in plan['fit']:
        obj, samples = base.capture(controller)
        metrics, _ = run_episode(obj, float(case['mass']), tuple(case['com']),
                                 collision_detector.Detector())
        if metrics['diverged'] or len(samples['t']) not in (3550, 3551):
            raise ValueError('incomplete/diverged calibration episode')
        fit_features.append(base.public_features(old, samples, case))
        fit_runs.append({'id': case['id'], 'mass': case['mass'], 'com': case['com'],
                         'run': base.compact_metrics(metrics)})

    raw_model, raw_guard, raw_loo = old.calibration(fit_features, config.JTS_NOISE)
    registered_model = registration.calibrate(fit_features, config.JTS_NOISE)

    rows = []
    score_capsules = []
    for case in plan['test']:
        for variant in VARIANTS:
            obj, samples = base.capture(controller)
            metrics, log = run_episode(obj, float(case['mass']), tuple(case['com']),
                                       collision_detector.Detector(),
                                       pick_offset=base.offset(case, variant))
            if metrics['diverged'] or len(samples['t']) not in (3550, 3551):
                raise ValueError('incomplete/diverged evaluation episode')
            feature = base.public_features(old, samples, case)
            raw = old.estimate(feature, raw_model, raw_guard)
            raw['applied_mm'] = strict_action(raw)
            raw['soft_interval_action_mm'] = float(
                np.sign(raw['extra_height_mm']) *
                max(0.0, abs(raw['extra_height_mm']) - raw['half_width_mm'])
                if raw['fitted_displacement_within_domain'] else 0.0)
            registered = registration.decide(feature, registered_model)
            rows.append({'id': case['id'], 'variant': variant,
                         'raw': raw, 'registered': registered,
                         'run': base.compact_metrics(metrics)})
            score_capsules.append((base.score_capsule(log), np.asarray(case['com'])))

    truths = [base.observed_geometry(capsule, com) for capsule, com in score_capsules]
    baselines = {}
    for row, actual in zip(rows, truths):
        identifier, variant = row['id'], row['variant']
        measured = actual['bottom_shift_at_placement_mm']
        if variant == 'zero':
            baselines[identifier] = measured
        truth = measured - baselines[identifier]
        row.update(truth_mm=truth, true_bottom_shift_at_placement_mm=measured,
                   true_local_com_shift_mm=actual['local_com_shift_mm'],
                   max_observed_pose_span_mm=max(actual['sample_span_mm']))
        for key in ('raw', 'registered'):
            result = row[key]
            result['point_error_mm'] = result['extra_height_mm'] - truth
            result['action_error_mm'] = result['applied_mm'] - truth
            result['covered'] = bool(result['interval_mm'][0] <= truth <= result['interval_mm'][1])

    raw_calibration = {name: np.asarray(value).tolist() for name, value in raw_model.items()}
    raw_calibration.update(normal_guard_mm=float(raw_guard), loo_signed_mm=raw_loo,
                           schema='r13-raw-calibration.v1')
    metrics = {key: policy_metrics(rows, key) for key in ('raw', 'registered')}
    metrics['registered_minus_raw_point_rmse_mm'] = (
        metrics['registered']['point_rmse_mm'] - metrics['raw']['point_rmse_mm'])
    return {
        'calibration': {'mode': mode, 'runs': fit_runs, 'raw': raw_calibration,
                        'registered': registered_model},
        'evaluation': {'mode': mode, 'rows': rows, 'metrics': metrics},
    }


def assemble_inputs():
    paths = {name: ROOT / ('my_' + name + '.py') for name in ('servo', 'torque', 'detector')}
    paths.update({
        'estimator': ROOT / 'tools/r13_grasp_estimator.py',
        'registration': ROOT / 'experiments/r14_airborne/registration.py',
        'plan': ROOT / 'experiments/r14_airborne/plan.json',
        'base_audit': ROOT / 'tools/r13_grasp_audit.py',
        'audit': Path(__file__),
        'kit_manifest': ROOT / 'tools/kit_manifest.sha256',
    })
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()):
        raise ValueError('regular public candidate and R14 audit inputs required')
    return base.snapshot_files(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True, help='licensed ORIGINAL icpbl_sim kit')
    parser.add_argument('--out', type=Path, required=True, help='NEW output directory')
    args = parser.parse_args(argv)
    kit, out = base.validate_paths(args.kit, args.out, parser)
    blobs, pins = assemble_inputs()
    plan = json.loads(blobs['plan'])
    validate_plan(plan)

    sys.path.insert(0, str(ROOT))
    from tools.validate_submission import prepare_overlay

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)
    results = {}
    for mode in ('servo', 'torque'):
        with tempfile.TemporaryDirectory(prefix='r14-grasp-audit-', dir=out.parent) as folder:
            overlay = Path(folder)
            prepare_overlay(kit, overlay, 'my', mode)
            if base.digest(overlay / 'tools/kit_manifest.sha256') != pins['kit_manifest']:
                raise ValueError('kit manifest differs from frozen audit input')
            destinations = {
                mode: overlay / (mode + '_controller.py'),
                'detector': overlay / 'collision_detector.py',
                'estimator': overlay / 'tools/r13_grasp_estimator.py',
                'registration': overlay / 'registration.py',
                'plan': overlay / 'plan.json',
                'base_audit': overlay / 'grasp_audit_base.py',
                'audit': overlay / 'audit.py',
            }
            (overlay / 'tools/__init__.py').write_text('')
            for key, destination in destinations.items():
                base.install_snapshot(blobs[key], pins[key], destination)
            code = ('import audit,json,sys; from pathlib import Path; '
                    'json.dump(audit.run_mode(' + repr(mode) +
                    ',json.loads(Path("plan.json").read_text())),sys.stdout,allow_nan=False)')
            process = subprocess.run([sys.executable, '-c', code], cwd=overlay, env=environment,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.returncode:
                raise RuntimeError(mode + ' audit failed:\n' + process.stderr)
            results[mode] = json.loads(process.stdout)

    ignored = [{'split': split, 'id': case['id'], 'research_perturb': case.get('perturb', {})}
               for split in ('fit', 'test') for case in plan[split]
               if case.get('perturb', {})]
    scope = {
        'episodes': 56,
        'calibration_episodes': 24,
        'evaluation_episodes': 32,
        'inference_deadline_s': 3.55,
        'runner': 'authenticated stock sim.runner.run_episode',
        'same_signal_comparison': True,
        'control_parity': ('raw and registered estimates are offline consumers of one shared run; '
                           'controller, detector and physics trace are identical by construction'),
        'absolute_height_caveat': ('absolute_bottom_mm is a public-model latch-to-taught estimate, '
                                   'not a measurement of absolute physical box height; qualification '
                                   'uses paired actual bottom-face change only'),
        'research_parity': False,
        'parity_limit': ('Stock runner applies authored pick_offset but does not expose the private '
                         'stiffness/viscous/Coulomb/RNG perturbation fixture. Numeric parity with R14 '
                         'research captures is not claimed.'),
        'ignored_research_perturbations': ignored,
        'causal_split': ('12 zero-offset stock runs fit each mode; all raw and registered held '
                         'predictions are complete before free-box score geometry is evaluated'),
        'retained_data': 'derived calibration/evaluation JSON only; no raw captures',
        'decision': 'validation only; no promotion or threshold change',
    }
    calibration = {'schema': 'r14-grasp-calibration-audit.v1', 'pins': pins, 'scope': scope,
                   'modes': {mode: results[mode]['calibration'] for mode in results}}
    evaluation = {'schema': 'r14-grasp-evaluation-audit.v1', 'pins': pins, 'scope': scope,
                  'modes': {mode: results[mode]['evaluation'] for mode in results}}
    summary = {'schema': 'r14-grasp-audit.v1', 'pins': pins, 'scope': scope,
               'metrics': {mode: results[mode]['evaluation']['metrics'] for mode in results}}
    out.mkdir()
    for name, document in (('calibration.json', calibration), ('evaluation.json', evaluation),
                           ('summary.json', summary)):
        with (out / name).open('x') as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print('Completed 56 stock-runner episodes; wrote R14 grasp audit:', out)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
