"""Qualified/repeatable comparison; failures in the old protocol are never converted into successes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
from pathlib import Path

import numpy as np

from experiments.r20_bridge_challenge import challenge
from experiments.r20_force_certificate import certificate

PLAN = 'experiments/r20_bridge_challenge/plan.json'
MODEL = 'results/r17/release_observation_model.json'
ANCHORS = {'pair_return_opposed': [33], 'i55': [41], 'gap80_weak': [24]}
FORCE_GATES = ('complete_finite_strict_grid', 'signed_profiles_exact',
               'force_free_prefixes', 'all_required_prefixes_exact',
               'distinct_non_noop_impacts', 'no_sign_inversion')
TWINS = ('none', 'first', 'second', 'both')


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def legacy_batched_norm_only(row):
    """Classify from recorded evidence before authorizing any extra episode."""
    for arm in ('r19', 'slow'):
        record = row['arms'][arm]
        gates = record['r18_gates']
        if (not record['signed_and_scalar_exact'] or gates['signed_profiles_exact']
                or not all(gates[key] for key in FORCE_GATES
                           if key != 'signed_profiles_exact')
                or not all(record['scalar_forces'][twin]['stock_scalar_bit_exact']
                           for twin in TWINS)
                or not any(record['scalar_forces'][twin]['different_batched_samples']
                           for twin in TWINS)):
            return False
    return bool(row['assessment_sha256']['r19'] == row['assessment_sha256']['slow']
                and row['arms']['r19'] == row['arms']['slow']
                and row['candidates']['slow']['gates']['physical_trajectories_equal'])


def direct_certificate(case, height, early, previous):
    detector = challenge.resolve_arms(('r19', 'slow'))['r19']
    traces, record, raw = challenge.run_arm(detector, case, height, early, MODEL, 20, 320)
    assessment = challenge.paired_replay.signature(raw)
    twins = {}
    for twin in TWINS:
        expected = certificate.exact_trace(case, twin)
        trace = traces[twin]
        actual = {'t': np.asarray(trace['t']),
                  'native': np.asarray(trace['f_ext_xyz_native']),
                  'extra': np.asarray(trace['f_ext_xyz_second']),
                  'combined': np.asarray(trace['f_ext_xyz_committed']),
                  'scalar': np.asarray(trace['f_ext'])}
        exact = {key: bool(np.array_equal(actual[key], expected[key])) for key in actual}
        mismatches = certificate.mismatch_rows(expected)
        twins[twin] = {
            'samples': len(actual['t']), 'exact': exact,
            'all_arrays_exact': all(exact.values()),
            'actual_array_sha256': {key: array_sha(value) for key, value in actual.items()},
            'batched_norm_mismatch_ticks': [row['tick'] for row in mismatches],
            'max_batched_norm_difference_N': max(
                (row['absolute_difference_N'] for row in mismatches), default=0.0),
            'stock_scalar_bit_exact': record['scalar_forces'][twin]['stock_scalar_bit_exact'],
        }
    checks = {
        'authorized_legacy_batched_norm_only': legacy_batched_norm_only(previous),
        'assessment_sha256_matches_initial_r19': assessment == previous['assessment_sha256']['r19'],
        'initial_r19_and_slow_assessment_bit_equal': (
            previous['assessment_sha256']['r19'] == previous['assessment_sha256']['slow']),
        'all_four_twins_6000_ticks': all(row['samples'] == 6000 for row in twins.values()),
        't_native_second_committed_scalar_exact': all(
            row['all_arrays_exact'] for row in twins.values()),
        'stock_scalar_exact_in_actual_assessment': all(
            row['stock_scalar_bit_exact'] for row in twins.values()),
        'legacy_batched_norm_mismatch_reproduced': any(
            row['batched_norm_mismatch_ticks'] for row in twins.values()),
    }
    authorization = {
        'initial_force_gate_failures': {
            arm: [key for key in FORCE_GATES
                  if not previous['arms'][arm]['r18_gates'][key]]
            for arm in ('r19', 'slow')},
        'initial_signed_and_scalar_exact': {
            arm: previous['arms'][arm]['signed_and_scalar_exact']
            for arm in ('r19', 'slow')},
        'initial_stock_scalar_all_twins_exact': {
            arm: all(previous['arms'][arm]['scalar_forces'][twin]['stock_scalar_bit_exact']
                     for twin in TWINS) for arm in ('r19', 'slow')},
        'initial_r19_and_slow_records_equal': previous['arms']['r19'] == previous['arms']['slow'],
        'initial_r19_to_slow_physical_equal': (
            previous['candidates']['slow']['gates']['physical_trajectories_equal']),
    }
    return {'status': 'direct_stock_rerun', 'episodes': 4, 'checks': checks,
            'authorization_evidence': authorization, 'passed': all(checks.values()),
            'assessment_sha256': assessment, 'twins': twins}


def compact(row, force_certificate=None):
    records = row['arms']
    ref, new = records['r19'], records['slow']
    paired = row['candidates']['slow']
    slots = paired['slot_status']
    strict = {key: all(row['original_protocol'][key].values()) for key in ('r19', 'slow')}
    still_pass = all(not old or new['r18_gates'][name] for name, old in ref['r18_gates'].items())
    checks = {
        'force_and_prefix_valid': all(x['force_and_prefix_valid'] for x in row['original_protocol'].values()),
        'all_previously_passing_r18_gates_retained': still_pass,
        'latency_false_tool_tail': paired['gates']['retain_latency_false_tool_tail'],
        'identical_physics': paired['gates']['physical_trajectories_equal'],
        'no_extra_raw_rise': all(len(s['r20_raw_ms']) <= len(s['r19_raw_ms']) for s in slots.values()),
        'no_lost_one_per_slot': not any(s['loss'] for s in slots.values()),
        'no_reclassified_reference_noise': paired['gates']['no_disallowed_reference_raw_reclassification'],
    }
    if row['id'] in ANCHORS:
        target = new['slots']['both_second']
        checks['protected_anchor'] = (target['quiet_before_onset'] and target['raw_delays_ms'] ==
                                      target['credited_delays_ms'] == ANCHORS[row['id']])
    original_retained = all(checks.values())
    if force_certificate is None:
        force_certificate = {'status': 'not_required', 'episodes': 0, 'passed': None}
    supplemental = bool(
        original_retained or (force_certificate.get('passed') is True
                              and all(value for key, value in checks.items()
                                      if key != 'force_and_prefix_valid')))
    return {'id': row['id'], 'episodes': row['episodes'], 'checks': checks,
            'retained': original_retained,
            'direct_force_certificate': force_certificate,
            'supplemental_continuation_retained': supplemental, 'strict': strict,
            'one_per_slot': {k: x['exactly_one_per_slot'] for k, x in row['original_protocol'].items()},
            'failed_original_gates': {k: [g for g, ok in a['r18_gates'].items() if not ok]
                                      for k, a in records.items()},
            'slots': slots, 'assessment_sha256': row['assessment_sha256']}


def evaluate(task):
    stage, case = task
    h, e = (challenge.read(n + '.json') for n in ('height', 'early'))
    arms = challenge.resolve_arms(('r19', 'slow'))
    if stage == 'normal':
        return challenge.evaluate_normal(case, stage, h, e, MODEL, arms)
    row = challenge.evaluate_case(case, stage, h, e, MODEL, arms)
    if stage != 'exposed':
        return compact(row)
    invalid_force = not all(x['force_and_prefix_valid'] for x in row['original_protocol'].values())
    if not invalid_force:
        return compact(row)
    if not legacy_batched_norm_only(row):
        return compact(row, {'status': 'not_run_non_roundoff_force_failure',
                             'episodes': 0, 'passed': False})
    return compact(row, direct_certificate(case, h, e, row))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, choices=(1, 2), default=1)
    args = ap.parse_args(argv)
    p = challenge.read(PLAN)
    challenge.validate_plan(p)
    read = lambda name: challenge.read('experiments/' + name + '/plan.json')
    past = read('r18_paired_impulses')
    known = (past['fresh_paired_cases'] + past['diagnostic_sequences'] +
             read('r19_close_pairs')['new'] + read('r19_impulse_gate')['new'])
    if len(known) != len({c['id'] for c in known}) or len(known) != 18:
        raise ValueError('incomplete exposed set')
    result = {'schema': 'r20-slow-handoff-qualification.v2', 'stages': {},
              'eligible_for_submission': False, 'independent_detection_gain_claimed': False}
    with ProcessPoolExecutor(args.jobs) as pool:
        for name, cases in (('exposed', known), ('normal', p['fresh']), ('fresh', p['fresh'])):
            rows = list(pool.map(evaluate, [(name, c) for c in cases]))
            legacy_passed = (all(r['passed'] for r in rows) if name == 'normal' else
                             all(r['retained'] and (name != 'fresh' or r['strict']['slow'])
                                 for r in rows))
            continuation = (legacy_passed if name != 'exposed' else
                            all(r['supplemental_continuation_retained'] for r in rows))
            stage = {'passed': legacy_passed, 'rows': rows}
            if name == 'exposed':
                stage.update(certified_continuation_passed=continuation,
                             certificate_episodes=sum(
                                 r['direct_force_certificate']['episodes'] for r in rows),
                             strict_counts={arm: sum(r['strict'][arm] for r in rows)
                                            for arm in ('r19', 'slow')})
                result['initial_without_direct_certificate'] = {
                    'public_gate_halted_after': 'exposed', 'passed': legacy_passed,
                    'failing_ids': [r['id'] for r in rows if not r['retained']],
                }
            result['stages'][name] = stage
            print(name, len(rows), 'passed', legacy_passed,
                  'continuation', continuation, flush=True)
            if not continuation:
                break
    result['base_episodes'] = sum(
        row['episodes'] for stage in result['stages'].values() for row in stage['rows'])
    result['certificate_episodes'] = sum(
        stage.get('certificate_episodes', 0) for stage in result['stages'].values())
    result['episodes'] = result['base_episodes'] + result['certificate_episodes']
    with args.out.open('x') as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
        fp.write('\n')


if __name__ == '__main__':
    main()
