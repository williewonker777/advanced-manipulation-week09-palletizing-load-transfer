"""Portable stage-gated R14 registered TORQUE check on an ORIGINAL stock kit.

Executable sources and the supplied stock calibration are immutable snapshots
installed in an authenticated temporary overlay.  The authored neutral runner
compares R10, unregistered R12 hybrid, and registered R14 arms.  Impact episodes
open only after every normal gate passes.  Results contain derived JSON only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

try:
    from tools import r13_grasp_audit as assembly
except ImportError:
    import r13_grasp_audit as assembly


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = {
    'bias', 'center', 'range', 'sigma', 'normal_bottom_intercept_slope_mm',
    'normal_guard_mm', 'loo_signed_mm', 'schema',
}
SOURCE_PATHS = {
    'r1_torque': 'r1_torque.py',
    'my_detector': 'my_detector.py',
    'hybrid': 'experiments/r12_hybrid/controller.py',
    'hybrid_init': 'experiments/r12_hybrid/__init__.py',
    'stream': 'experiments/r14_online/stream.py',
    'stream_init': 'experiments/r14_online/__init__.py',
    'registration': 'experiments/r14_airborne/registration.py',
    'airborne_plan': 'experiments/r14_airborne/plan.json',
    'estimator': 'tools/r13_grasp_estimator.py',
    'placement': 'tools/r11_placement.py',
    'coefficients': 'models/r12/counterfactual_coefficients.json',
    'controller': 'experiments/r14_stock_handoff/controller.py',
    'detector': 'experiments/r14_stock_handoff/detector.py',
    'geometry': 'experiments/r14_stock_handoff/geometry.py',
    'controller_test': 'experiments/r14_stock_handoff/executed_checks.py',
    'plan': 'experiments/r14_stock_handoff/plan.json',
    'qualification': 'experiments/r14_stock_handoff/qualification.py',
    'handoff_init': 'experiments/r14_stock_handoff/__init__.py',
    'paired_events': 'tools/r14_paired_events.py',
    'event_metrics': 'tools/r5_metrics.py',
    'r14_audit': 'tools/r14_grasp_audit.py',
    'r13_audit': 'tools/r13_grasp_audit.py',
    'kit_manifest': 'tools/kit_manifest.sha256',
}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def finite_array(value, shape, name, *, lower=None, upper=None):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('invalid stock calibration field: ' + name)
    if lower is not None and np.any(array < lower):
        raise ValueError('value below bound: ' + name)
    if upper is not None and np.any(array > upper):
        raise ValueError('value above bound: ' + name)
    return array


def validate_plan(plan):
    if (plan.get('schema') != 'r14-stock-handoff-plan.v1'
            or plan.get('status') != 'FROZEN_BEFORE_STOCK_EPISODES'
            or plan.get('created_before_any_stock_handoff_episode') is not True
            or len(plan.get('normal_cases', ())) != 6
            or len(plan.get('impact_cases', ())) != 6):
        raise ValueError('invalid or unfrozen stock handoff plan')
    for name in ('normal_cases', 'impact_cases'):
        identifiers = []
        for case in plan[name]:
            identifiers.append(case.get('id'))
            finite_array(case['mass'], (), name + '.mass', lower=.1, upper=20.)
            finite_array(case['com'], (3,), name + '.com', lower=-.2, upper=.2)
            finite_array(case.get('offset') or [0., 0., 0.], (3,), name + '.offset',
                         lower=-.05, upper=.05)
            if name == 'impact_cases':
                collision = case.get('collision')
                if not isinstance(collision, list) or len(collision) != 2:
                    raise ValueError('invalid impact definition')
                finite_array(collision[0], (), 'collision.time', lower=0., upper=6.)
                finite_array(collision[1], (3,), 'collision.force', lower=-100., upper=100.)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('duplicate stock case id')


def validate_calibration(blob, plan, pins):
    """Return the torque model only after schema, provenance and bounds checks."""
    try:
        document = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid stock calibration JSON') from error
    if plan.get('calibration', {}).get('sha256') != sha(blob):
        raise ValueError('calibration does not match frozen stock handoff plan')
    if document.get('schema') != 'r14-grasp-calibration-audit.v1':
        raise ValueError('R14 stock calibration audit schema required')
    if set(document.get('modes', {})) != {'servo', 'torque'}:
        raise ValueError('complete stock calibration modes required')
    torque = document['modes']['torque']
    if torque.get('mode') != 'torque' or set(torque) != {'mode', 'runs', 'raw', 'registered'}:
        raise ValueError('stock TORQUE calibration required')
    model = torque['registered']
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError('unexpected registered calibration fields')
    if model.get('schema') != 'r14-registration-calibration.v1':
        raise ValueError('registered calibration schema required')
    finite_array(model['bias'], (2, 227, 6), 'bias', lower=-100., upper=100.)
    centre = float(finite_array(model['center'], (), 'center', lower=.1, upper=20.))
    domain = finite_array(model['range'], (2,), 'range', lower=.1, upper=20.)
    if not domain[0] <= centre <= domain[1] or not domain[0] < domain[1]:
        raise ValueError('calibration mass domain does not contain center')
    finite_array(model['sigma'], (6,), 'sigma', lower=np.finfo(float).tiny, upper=100.)
    finite_array(model['normal_bottom_intercept_slope_mm'], (2,), 'normal expectation',
                 lower=-100., upper=100.)
    finite_array(model['normal_guard_mm'], (), 'normal guard', lower=0., upper=2.5)
    finite_array(model['loo_signed_mm'], (12,), 'LOO guard', lower=-2.5, upper=2.5)

    expected = {
        'torque': pins['r1_torque'], 'detector': pins['my_detector'],
        'estimator': pins['estimator'], 'registration': pins['registration'],
        'plan': pins['airborne_plan'], 'audit': pins['r14_audit'],
        'base_audit': pins['r13_audit'], 'kit_manifest': pins['kit_manifest'],
    }
    calibration_pins = document.get('pins')
    if not isinstance(calibration_pins, dict) or any(calibration_pins.get(k) != v
                                                     for k, v in expected.items()):
        raise ValueError('calibration source pins do not match assembled public sources')
    return document, model


def source_inputs(calibration_path):
    paths = {key: ROOT / relative for key, relative in SOURCE_PATHS.items()}
    paths['wrapper'] = Path(__file__)
    paths['calibration'] = calibration_path
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()):
        raise ValueError('regular frozen handoff sources and calibration required')
    return assembly.snapshot_files(paths)


def install_tree(overlay, blobs, pins):
    destinations = {
        'r1_torque': 'r1_torque.py', 'my_detector': 'my_detector.py',
        'hybrid': 'experiments/r12_hybrid/controller.py',
        'hybrid_init': 'experiments/r12_hybrid/__init__.py',
        'stream': 'experiments/r14_online/stream.py',
        'stream_init': 'experiments/r14_online/__init__.py',
        'registration': 'experiments/r14_airborne/registration.py',
        'estimator': 'tools/r13_grasp_estimator.py',
        'placement': 'tools/r11_placement.py',
        'coefficients': 'models/r12/counterfactual_coefficients.json',
        'controller': 'experiments/r14_stock_handoff/controller.py',
        'detector': 'experiments/r14_stock_handoff/detector.py',
        'geometry': 'experiments/r14_stock_handoff/geometry.py',
        'controller_test': 'experiments/r14_stock_handoff/test_controller.py',
        'plan': 'experiments/r14_stock_handoff/plan.json',
        'qualification': 'experiments/r14_stock_handoff/qualification.py',
        'handoff_init': 'experiments/r14_stock_handoff/__init__.py',
        'paired_events': 'tools/r14_paired_events.py',
        'event_metrics': 'tools/r5_metrics.py',
        'calibration': 'calibration.json',
    }
    for relative in destinations.values():
        (overlay / relative).parent.mkdir(parents=True, exist_ok=True)
    for key, relative in destinations.items():
        assembly.install_snapshot(blobs[key], pins[key], overlay / relative)
    # The public source remains neutrally named.  The authenticated same bytes
    # are also copied to the authored runner's hard-coded private-overlay name.
    assembly.install_snapshot(blobs['qualification'], pins['qualification'],
                              overlay / 'experiments/r14_stock_handoff/run.py')


def run_stage(overlay, environment, stage, normal=None):
    output = overlay / (stage + '.json')
    command = [sys.executable, '-B', '-m', 'experiments.r14_stock_handoff.run',
               '--stage', stage, '--calibration', 'calibration.json', '--out', output.name]
    if normal is not None:
        command.extend(('--normal-result', normal.name))
    process = subprocess.run(command, cwd=overlay, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(stage + ' authored runner failed:\n' + process.stderr)
    try:
        document = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('authored runner did not produce valid ' + stage + ' output') from error
    if document.get('stage') != stage or not isinstance(document.get('passed'), bool):
        raise ValueError('invalid authored ' + stage + ' result schema')
    return document


def run_paired_stage(overlay, environment, normal, public_gates):
    """Run only after both same-overlay normal gate documents have passed."""
    public_path = overlay / 'public_gates.json'
    with public_path.open('x') as stream:
        json.dump(public_gates, stream, indent=2, allow_nan=False)
        stream.write('\n')
    output = overlay / 'impact.json'
    command = [sys.executable, '-B', '-m', 'tools.r14_paired_events',
               '--calibration', 'calibration.json', '--normal-result', normal.name,
               '--public-gates', public_path.name, '--out', output.name]
    process = subprocess.run(command, cwd=overlay, env=environment, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError('paired impact runner failed:\n' + process.stderr)
    try:
        document = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('paired runner did not produce valid impact output') from error
    if (document.get('schema') != 'r14-registered-paired-impact.v1'
            or document.get('stage') != 'impact'
            or not isinstance(document.get('passed'), bool)
            or document.get('normal_sha256') != sha(normal.read_bytes())
            or document.get('public_gates_sha256') != sha(public_path.read_bytes())):
        raise ValueError('invalid or stale paired impact result')
    return document


def strip_timing(value):
    """Remove host timing only; decisions and physical metrics remain comparable."""
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items()
                if key not in {'ctrl_p99_us', 'stream_timing_us'}}
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value


def failed_gates(normal):
    failures = []
    if not normal['unloaded']['passed']:
        failures.append({'id': 'unloaded', 'gates': ['byte_parity']})
    for row in normal['rows']:
        bad = [name for name, passed in row['gates'].items() if not passed]
        if bad:
            failures.append({'id': row['id'], 'gates': bad})
    return failures


def match_false_events(candidate, reference, tolerance_s=.020):
    """Every candidate false event needs a distinct nearby reference event."""
    candidate = finite_array(candidate, (len(candidate),), 'candidate false events',
                             lower=0., upper=6.)
    reference = finite_array(reference, (len(reference),), 'reference false events',
                             lower=0., upper=6.)
    unused = list(map(float, reference))
    pairs = []
    for event in map(float, candidate):
        choices = [(abs(event-other), index, other) for index, other in enumerate(unused)
                   if abs(event-other) <= tolerance_s]
        if not choices:
            return {'passed': False, 'pairs': pairs, 'unmatched_candidate': event,
                    'tolerance_ms': 1000*tolerance_s}
        delta, index, other = min(choices)
        pairs.append({'candidate_s': event, 'reference_s': other, 'delta_ms': 1000*delta})
        unused.pop(index)
    return {'passed': True, 'pairs': pairs, 'unmatched_candidate': None,
            'tolerance_ms': 1000*tolerance_s}


def tool_preserved(candidate, reference, tolerance_ms=5.):
    if candidate is None or reference is None:
        return candidate is None and reference is None
    return bool(float(candidate) <= float(reference) + tolerance_ms)


def public_normal_gates(normal, pins, normal_sha256=None):
    """Recheck every arm directly; never trust only the authored pass flag."""
    expected_sources = {
        'r1_torque.py': pins['r1_torque'],
        'my_detector.py': pins['my_detector'],
        'experiments/r12_hybrid/controller.py': pins['hybrid'],
        'experiments/r14_online/stream.py': pins['stream'],
        'experiments/r14_airborne/registration.py': pins['registration'],
        'models/r12/counterfactual_coefficients.json': pins['coefficients'],
        'experiments/r14_stock_handoff/controller.py': pins['controller'],
        'experiments/r14_stock_handoff/detector.py': pins['detector'],
        'experiments/r14_stock_handoff/geometry.py': pins['geometry'],
        'experiments/r14_stock_handoff/test_controller.py': pins['controller_test'],
        'experiments/r14_stock_handoff/plan.json': pins['plan'],
        'experiments/r14_stock_handoff/run.py': pins['qualification'],
    }
    if (normal.get('schema') != 'r14-stock-handoff-normal.v1'
            or normal.get('stage') != 'normal'
            or normal.get('pins', {}).get('sources') != expected_sources
            or normal.get('pins', {}).get('calibration') != pins['calibration']
            or len(normal.get('rows', ())) != 6):
        raise ValueError('authored normal result identity or shape mismatch')
    rows = []
    for row in normal['rows']:
        if set(row.get('arms', {})) != {'r10', 'hybrid', 'registered'}:
            raise ValueError('three normal comparator arms required')
        r10, hybrid, candidate = (row['arms'][key] for key in ('r10', 'hybrid', 'registered'))
        fa_r10 = match_false_events(candidate['false_alarm_times'], r10['false_alarm_times'])
        fa_hybrid = match_false_events(candidate['false_alarm_times'], hybrid['false_alarm_times'])
        gates = {
            'authored_case': bool(row.get('passed')),
            'FA_r10_20ms': fa_r10['passed'],
            'FA_hybrid_20ms': fa_hybrid['passed'],
            'saturation_r10': candidate['saturation_pct'] <= r10['saturation_pct'],
            'saturation_hybrid': candidate['saturation_pct'] <= hybrid['saturation_pct'],
            'tool_r10': tool_preserved(candidate['tool_delay_ms'], r10['tool_delay_ms']),
            'tool_hybrid': tool_preserved(candidate['tool_delay_ms'], hybrid['tool_delay_ms']),
        }
        rows.append({'id': row['id'], 'gates': gates, 'FA_r10': fa_r10,
                     'FA_hybrid': fa_hybrid, 'passed': all(gates.values())})
    return {
        'schema': 'r14-registered-public-normal-gates.v1',
        'rows': rows,
        'passed': bool(normal.get('unloaded', {}).get('passed') and all(r['passed'] for r in rows)),
        'normal_sha256': normal_sha256,
        'calibration_sha256': pins['calibration'],
        'impact_comparator_validated': True,
        'impact_validation_scope': ('tool logic exercised on an already-exposed R12 stock pulse; '
                                    'not R14 candidate impact validation'),
        'impact_reason': ('Source-pinned strict comparator available; it still runs only after '
                          'every authored and supplementary normal gate passes.'),
    }


def make_summary(normal, public_gates, pins, impact=None):
    stable = strip_timing(normal)
    failed_public = [{'id': row['id'],
                      'gates': [name for name, passed in row['gates'].items() if not passed]}
                     for row in public_gates['rows'] if not row['passed']]
    return {
        'schema': 'r14-registered-stock-check.v1', 'mode': 'torque',
        'passed': bool(public_gates['passed'] and impact is not None and impact['passed']),
        'authored_normal_passed': normal['passed'],
        'public_normal_passed': public_gates['passed'],
        'impact_stage': ('not_opened_normal_failed' if impact is None
                         else 'completed_passed' if impact['passed'] else 'completed_failed'),
        'impact_comparator_validated': public_gates['impact_comparator_validated'],
        'impact_validation_scope': public_gates['impact_validation_scope'],
        'failed_normal_gates': failed_gates(normal), 'pins': pins,
        'failed_public_gates': failed_public,
        'calibration_sha256': pins['calibration'],
        'non_timing_sha256': sha(json.dumps(stable, sort_keys=True,
                                            separators=(',', ':'), allow_nan=False).encode()),
        'comparators': ['R10 baseline', 'unregistered R12 hybrid', 'registered R14 hybrid'],
        'non_timing_scope': 'authored normal result only; supplementary public gates are separate',
        'source_mapping': {
            'public': 'experiments/r14_stock_handoff/qualification.py',
            'temporary_overlay': 'experiments/r14_stock_handoff/run.py',
            'sha256': pins['qualification'],
            'byte_identical': True,
        },
        'scope': ('Original stock kit only. Impact opens only after every normal gate. '
                  'No raw traces, private fixture, licensed source, or host paths are retained.'),
        'decision': ('finite stock gates passed; no broader promotion implied'
                     if impact is not None and impact['passed']
                     else 'fail closed; no promotion or threshold change'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True, help='licensed ORIGINAL stock kit')
    parser.add_argument('--calibration', type=Path, required=True,
                        help='stock calibration.json from tools/r14_grasp_audit.py')
    parser.add_argument('--out', type=Path, required=True, help='NEW output directory')
    args = parser.parse_args(argv)
    kit, out = assembly.validate_paths(args.kit, args.out, parser)
    calibration = args.calibration.resolve()
    if args.calibration.is_symlink() or not calibration.is_file():
        parser.error('regular stock calibration file required')
    blobs, pins = source_inputs(calibration)
    plan = json.loads(blobs['plan'])
    validate_plan(plan)
    validate_calibration(blobs['calibration'], plan, pins)

    sys.path.insert(0, str(ROOT))
    from tools.validate_submission import prepare_overlay

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='r14-registered-', dir=out.parent) as folder:
        overlay = Path(folder)
        prepare_overlay(kit, overlay, 'my', 'torque')
        if assembly.digest(overlay / 'tools/kit_manifest.sha256') != pins['kit_manifest']:
            raise ValueError('kit manifest differs from frozen wrapper input')
        install_tree(overlay, blobs, pins)
        normal = run_stage(overlay, environment, 'normal')
        normal_path = overlay / 'normal.json'
        public_gates = public_normal_gates(normal, pins, sha(normal_path.read_bytes()))
        impact = (run_paired_stage(overlay, environment, normal_path, public_gates)
                  if public_gates['passed'] else None)

    document = make_summary(normal, public_gates, pins, impact)
    out.mkdir()
    for name, value in (('normal.json', normal), ('public_gates.json', public_gates),
                        ('summary.json', document), ('impact.json', impact)):
        if value is None:
            continue
        with (out / name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'authored_normal_passed': normal['passed'],
                      'public_normal_passed': public_gates['passed'],
                      'impact_stage': document['impact_stage'],
                      'passed': document['passed']}))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
