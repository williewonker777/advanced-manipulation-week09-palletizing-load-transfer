"""Replay the exposed controls and six fixed fresh contrasts with an external unmodified kit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.r19_combined_gate import prepare_overlay as assembly
from tools import r18_recovery_audit as r18
from tools.r13_grasp_audit import digest, install_snapshot
from tools.r15_early_check import validate_paths

# Freeze the candidate, earlier reference and the independently selected cohort/scorer.
FROZEN = {
    'experiments/r19_impulse_end/state.py': 'e3211c06eff4b3974ed8afd3af66fd68c8da933851abe971e170e19fbbbfeb05',
    'experiments/r19_impulse_end/detector.py': '7568d2e883ff150d2473a6f17bebb760ba425a4887122324d564d80819bbe909',
    'experiments/r20_bridge_challenge/challenge.py': '63da2f879754ab89f7e8f2c19406ddd8def2dcd5fad92030a752ed497e865158',
    'experiments/r20_bridge_challenge/plan.json': '9f21d2911ea753df559410506dc02ea9f62f8b5259b7c71c89b2e6dc83127808',
    'experiments/r20_slow_handoff/state.py': 'ddaecbfb5ba49b05a592251a275b43133af781ecc0d2c8696edbbc3eda0452ab',
    'experiments/r20_slow_handoff/detector.py': 'd03c76fb199797ee012d473034c391a1167705969e908e03b57940dccb3ca7c6',
    'experiments/r20_slow_handoff/plan.json': 'ef498711c85d8c7e82e5ee23882782c023535a58190199f45e86230a9fc67f0b',
    'experiments/r20_force_certificate/certificate.py':
        '5d679601ea48969b41a12f943e1ba35481cc7caa8feb912e8dd626233bcc418a',
}
EXTRA = ('experiments/r19_close_pairs/assay.py', 'experiments/r19_close_pairs/plan.json',
         'experiments/r19_impulse_gate/plan.json', 'experiments/r20_event_bridge/state.py',
         'experiments/r20_event_bridge/detector.py', 'experiments/r20_slow_handoff/qualify.py')


FORCE_GATES = {'complete_finite_strict_grid', 'signed_profiles_exact',
               'force_free_prefixes', 'all_required_prefixes_exact',
               'distinct_non_noop_impacts', 'no_sign_inversion'}
TWINS = ('none', 'first', 'second', 'both')


def check_direct_certificate(row):
    cert = row['direct_force_certificate']
    force_valid = row['checks']['force_and_prefix_valid']
    if force_valid:
        if cert != {'status': 'not_required', 'episodes': 0, 'passed': None}:
            raise ValueError('certificate applied to a force-valid row')
        return 0
    legacy = all(set(row['failed_original_gates'][arm]) & FORCE_GATES ==
                 {'signed_profiles_exact'} for arm in ('r19', 'slow'))
    if not legacy:
        if (cert != {'status': 'not_run_non_roundoff_force_failure',
                     'episodes': 0, 'passed': False}
                or row['supplemental_continuation_retained']):
            raise ValueError('non-roundoff force failure was certified')
        return 0
    if (cert.get('status') != 'direct_stock_rerun' or cert.get('episodes') != 4
            or cert.get('passed') is not all(cert.get('checks', {}).values())
            or set(cert.get('twins', {})) != set(TWINS)
            or cert.get('assessment_sha256') != row['assessment_sha256']['r19']
            or row['assessment_sha256']['r19'] != row['assessment_sha256']['slow']):
        raise ValueError('invalid direct stock certificate')
    authorization = cert.get('authorization_evidence', {})
    if (authorization.get('initial_force_gate_failures') != {
            'r19': ['signed_profiles_exact'], 'slow': ['signed_profiles_exact']}
            or authorization.get('initial_signed_and_scalar_exact') != {
                'r19': True, 'slow': True}
            or authorization.get('initial_stock_scalar_all_twins_exact') != {
                'r19': True, 'slow': True}
            or authorization.get('initial_r19_and_slow_records_equal') is not True
            or authorization.get('initial_r19_to_slow_physical_equal') is not True):
        raise ValueError('incomplete initial authorization evidence')
    required = {'authorized_legacy_batched_norm_only',
                'assessment_sha256_matches_initial_r19',
                'initial_r19_and_slow_assessment_bit_equal',
                'all_four_twins_6000_ticks',
                't_native_second_committed_scalar_exact',
                'stock_scalar_exact_in_actual_assessment',
                'legacy_batched_norm_mismatch_reproduced'}
    if set(cert['checks']) != required or not all(cert['checks'].values()):
        raise ValueError('incomplete direct stock certificate checks')
    mismatch = False
    for twin in TWINS:
        trace = cert['twins'][twin]
        exact = trace.get('exact', {})
        if (trace.get('samples') != 6000 or set(exact) !=
                {'t', 'native', 'extra', 'combined', 'scalar'}
                or trace.get('all_arrays_exact') is not all(exact.values())
                or not trace['all_arrays_exact'] or not trace.get('stock_scalar_bit_exact')
                or set(trace.get('actual_array_sha256', {})) != set(exact)
                or any(len(value) != 64 or any(ch not in '0123456789abcdef' for ch in value)
                       for value in trace['actual_array_sha256'].values())
                or any(not isinstance(tick, int) or not 0 <= tick < 6000
                       for tick in trace.get('batched_norm_mismatch_ticks', []))):
            raise ValueError('incomplete per-twin exact force certificate')
        mismatch = mismatch or bool(trace.get('batched_norm_mismatch_ticks'))
    if not mismatch:
        raise ValueError('legacy batched norm mismatch not reproduced')
    return 4


def check_stages(result):
    stages = result['stages']
    order = ('exposed', 'normal', 'fresh')
    if (result.get('schema') != 'r20-slow-handoff-qualification.v2' or not stages
            or list(stages) != list(order[:len(stages)])
            or result['eligible_for_submission'] is not False
            or result.get('independent_detection_gain_claimed') is not False):
        raise ValueError('invalid stages')
    base_count = certificate_count = 0
    for i, (name, stage) in enumerate(stages.items()):
        rows = stage['rows']
        n = 2 if name == 'normal' else 8
        if (len(rows) != (18 if name == 'exposed' else 6) or
                len({r['id'] for r in rows}) != len(rows) or any(r['episodes'] != n for r in rows)):
            raise ValueError('incomplete cohort or episode count')
        if name == 'normal':
            if any(r['passed'] != all(r['candidates']['slow']['gates'].values()) for r in rows):
                raise ValueError('unsubstantiated normal verdict')
            passed = all(r['passed'] for r in rows)
        else:
            if any(r['retained'] != all(r['checks'].values()) or
                   any(r['strict'][k] != (r['one_per_slot'][k] and not r['failed_original_gates'][k])
                       for k in ('r19', 'slow')) for r in rows):
                raise ValueError('substituted retention or strict score')
            passed = all(r['retained'] and (name != 'fresh' or r['strict']['slow']) for r in rows)
        if stage['passed'] is not passed:
            raise ValueError('substituted stage verdict')
        if name == 'exposed':
            certificate_count = sum(check_direct_certificate(row) for row in rows)
            supplemental = all(row['supplemental_continuation_retained'] for row in rows)
            for row in rows:
                expected = bool(row['retained'] or (
                    row['direct_force_certificate'].get('passed') is True
                    and all(value for key, value in row['checks'].items()
                            if key != 'force_and_prefix_valid')))
                if row['supplemental_continuation_retained'] is not expected:
                    raise ValueError('substituted supplemental retention')
            strict = {arm: sum(row['strict'][arm] for row in rows)
                      for arm in ('r19', 'slow')}
            if (stage.get('certified_continuation_passed') is not supplemental
                    or stage.get('certificate_episodes') != certificate_count
                    or stage.get('strict_counts') != strict):
                raise ValueError('invalid exposed certificate summary')
            can_continue = supplemental
        else:
            can_continue = passed
        if not can_continue and i + 1 != len(stages):
            raise ValueError('invalid progression after failed stage')
        base_count += n * len(rows)
    exposed = stages['exposed']
    failing = [row['id'] for row in exposed['rows'] if not row['retained']]
    initial = result.get('initial_without_direct_certificate')
    if initial != {'public_gate_halted_after': 'exposed', 'passed': exposed['passed'],
                   'failing_ids': failing} or initial['passed'] is not False:
        raise ValueError('initial no-certificate halt not preserved')
    if (base_count != result.get('base_episodes')
            or certificate_count != result.get('certificate_episodes')
            or base_count + certificate_count != result['episodes']
            or (len(stages) < 3 and can_continue)):
        raise ValueError('incomplete gate trace')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit', type=Path, required=True)
    ap.add_argument('--height-calibration', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    args = ap.parse_args(argv)
    try:
        kit, height, out = validate_paths(args.kit, args.height_calibration, args.out)
        with tempfile.TemporaryDirectory(prefix='r20-handoff-', dir=out.parent) as tmp:
            receipt, base = Path(tmp) / 'auth.json', Path(tmp) / 'kit'
            assembly.main(['--kit', str(kit), '--height', str(height), '--early',
                           str(ROOT / 'results/r15/early_stock_calibration.json'), '--overlay', str(base),
                           '--auth', str(receipt)])
            auth = json.loads(receipt.read_bytes())
            pins = auth['source_pins']
            for rel in (*r18.PAIR_FILES, *FROZEN, *EXTRA):
                f = ROOT / rel
                if f.is_symlink() or not f.is_file() or digest(f) != FROZEN.get(rel, digest(f)):
                    raise ValueError('changed or missing frozen input: ' + rel)
                target = base / rel
                target.parent.mkdir(exist_ok=True, parents=True)
                pins[rel] = digest(f)
                install_snapshot(f.read_bytes(), pins[rel], target)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL='disable')
            env.pop('PYTHONPATH', None)
            subprocess.run([sys.executable, '-B', '-m', 'experiments.r20_slow_handoff.qualify',
                            '--out', 'result.json', '--jobs', str(args.jobs)], cwd=base, env=env, check=True)
            result = json.loads((base / 'result.json').read_bytes())
            check_stages(result)
            if any(digest(ROOT / k) != v or digest(base / k) != v for k, v in pins.items()):
                raise ValueError('source changed during simulation')
        doc = {'schema': 'r20-slow-handoff-audit.v2', 'original_demo': False,
               'direct_force_certificate_is_supplemental': True, 'result': result,
               'source_sha256': pins, 'evaluator_sha256': auth['evaluator_sha256'],
               'kit_manifest_sha256': auth['kit_manifest_sha256'],
               'height_sha256': auth['height_source_sha256'], 'early_sha256': auth['early_source_sha256']}
        with out.open('x') as fp:
            json.dump(doc, fp, indent=2, allow_nan=False)
            fp.write('\n')
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        ap.error(str(exc))


if __name__ == '__main__':
    main()
