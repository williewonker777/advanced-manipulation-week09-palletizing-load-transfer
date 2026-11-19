"""Strict matched-normal comparator for the optional R14 impact stage.

This module does not decide whether impacts may be opened.  The public wrapper
must first prove every normal gate in the same temporary overlay.  Once opened,
each controller/detector arm is rerun twice: an unforced reference and the
declared forced episode.  Only derived records are returned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sim.runner import COLLISION_HOLD, COLLISION_RAMP
from experiments.r14_stock_handoff import qualification
from tools.r5_metrics import event_result


ARMS = ('r10', 'hybrid', 'registered')
PREFIX_KEYS = qualification.KEYS
FALSE_EVENT_TOLERANCE_S = .020
DELAY_TOLERANCE_MS = 5.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match_false_events(candidate, reference, tolerance_s=FALSE_EVENT_TOLERANCE_S):
    """Injectively match every candidate false rise to a reference rise."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (candidate.ndim != 1 or reference.ndim != 1
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError('finite one-dimensional false-event times required')
    unused = list(map(float, reference))
    pairs = []
    for event in map(float, candidate):
        choices = [(abs(event-other), index, other)
                   for index, other in enumerate(unused)
                   if abs(event-other) <= tolerance_s + 1e-12]
        if not choices:
            return {'passed': False, 'pairs': pairs, 'unmatched_candidate': event,
                    'tolerance_ms': 1000.*tolerance_s}
        delta, index, other = min(choices)
        pairs.append({'candidate_s': event, 'reference_s': other,
                      'delta_ms': 1000.*delta})
        unused.pop(index)
    return {'passed': True, 'pairs': pairs, 'unmatched_candidate': None,
            'tolerance_ms': 1000.*tolerance_s}


def expected_force(t, collision):
    """Exact public runner trapezoid, expressed as force magnitude."""
    t = np.asarray(t, dtype=float)
    onset, vector = collision
    vector = np.asarray(vector, dtype=float)
    # Preserve the runner's scalar min/max and norm-after-vector-scale order.
    # Reassociating this as ``weight * norm(vector)`` changes low bits and
    # would make an exact replay check reject the unchanged public runner.
    values = []
    for sample in t:
        age = float(sample)-float(onset)
        weight = min(age/COLLISION_RAMP, 1.,
                     (2.*COLLISION_RAMP+COLLISION_HOLD-age)/COLLISION_RAMP)
        values.append(np.linalg.norm(max(weight, 0.)*vector))
    return np.asarray(values)


def force_realization(t, observed, collision):
    t = np.asarray(t, dtype=float)
    observed = np.asarray(observed, dtype=float)
    expected = expected_force(t, collision)
    if t.shape != (6000,) or observed.shape != t.shape or not np.isfinite(observed).all():
        raise ValueError('complete finite force trace required')
    exact = bool(np.array_equal(observed, expected))
    onset = float(collision[0])
    prefix = t <= onset
    support = expected > 0.
    return {
        'passed': exact,
        'exact_trapezoid': exact,
        'requested_peak_N': float(np.linalg.norm(collision[1])),
        'observed_peak_N': float(observed.max()),
        'zero_through_onset': bool(np.all(observed[prefix] == 0.)),
        'active_first_s': None if not support.any() else float(t[np.flatnonzero(support)[0]]),
        'active_last_s': None if not support.any() else float(t[np.flatnonzero(support)[-1]]),
    }


def equal_prefix(normal, forced, onset):
    """Compare all retained physical and detector signals through zero-force onset."""
    t = np.asarray(normal['t'])
    other_t = np.asarray(forced['t'])
    if t.shape != (6000,) or not np.array_equal(t, other_t):
        raise ValueError('matched fixed-grid episodes required')
    end = int(np.searchsorted(t, float(onset), side='right'))
    fields = {key: bool(np.array_equal(np.asarray(normal[key])[:end],
                                       np.asarray(forced[key])[:end]))
              for key in PREFIX_KEYS}
    return {'passed': all(fields.values()), 'samples_through_onset': end, 'fields': fields}


def preserve_tool(candidate, reference, tolerance_ms=DELAY_TOLERANCE_MS):
    """No lost true tool event, added tool prealarm, or >5 ms delay."""
    present = not reference['tool_present'] or candidate['tool_present']
    detected = not reference['tool_detected'] or candidate['tool_detected']
    prealarm = not candidate['tool_prealarm'] or reference['tool_prealarm']
    delay = (reference['tool_delay_ms'] is None
             or candidate['tool_delay_ms'] is not None
             and candidate['tool_delay_ms'] <= reference['tool_delay_ms']+tolerance_ms+1e-9)
    return bool(present and detected and prealarm and delay)


def assess(events, prefixes, forces, statuses):
    """Apply the frozen public impact contract to three independently scored arms."""
    if set(events) != set(ARMS) or set(prefixes) != set(ARMS):
        raise ValueError('R10, hybrid, and registered impact arms required')
    r10, hybrid, candidate = (events[name] for name in ARMS)
    fa_r10 = match_false_events(candidate['false_alarm_times'], r10['false_alarm_times'])
    fa_hybrid = match_false_events(candidate['false_alarm_times'], hybrid['false_alarm_times'])
    gates = {
        'status': all(statuses[name] == 'ok' for name in ARMS),
        'force_realized': all(forces[name]['passed'] for name in ARMS),
        'matched_prefix': all(prefixes[name]['passed'] for name in ARMS),
        # A retained miss is not a certified impact result.  Both baseline and
        # candidate must have a distinguishable new rise in the force window.
        'r10_detected': r10['outcome'] == 'detected',
        'registered_detected': candidate['outcome'] == 'detected',
        'delay_r10_5ms': (r10['outcome'] == 'detected'
                          and candidate['outcome'] == 'detected'
                          and candidate['delay_ms'] <= r10['delay_ms']
                          + DELAY_TOLERANCE_MS + 1e-9),
        'no_active_prealarm': not candidate['prealarm'],
        'FA_r10_20ms': fa_r10['passed'],
        'FA_hybrid_20ms': fa_hybrid['passed'],
        'tool_r10': preserve_tool(candidate, r10),
        'tool_hybrid': preserve_tool(candidate, hybrid),
    }
    return {'passed': all(gates.values()), 'gates': gates,
            'FA_r10': fa_r10, 'FA_hybrid': fa_hybrid}


def impact_case(case, model):
    collision = case['collision']
    records, prefixes, forces, statuses = {}, {}, {}, {}
    for kind in ARMS:
        normal_record, normal_log = qualification.episode(kind, case, model)
        forced_record, forced_log = qualification.episode(kind, case, model, collision)
        statuses[kind] = ('ok' if normal_record['status'] == forced_record['status'] == 'ok'
                          else 'failed')
        prefixes[kind] = equal_prefix(normal_log, forced_log, collision[0])
        forces[kind] = force_realization(forced_log['t'], forced_log['f_ext'], collision)
        records[kind] = event_result(forced_log['t'], forced_log['detect'],
                                     {'collision': collision}, forced_log['f_tool'],
                                     normal_log['detect'])
    assessment = assess(records, prefixes, forces, statuses)
    return {'id': case['id'], 'collision': collision, 'arms': records,
            'prefix': prefixes, 'force': forces, 'status': statuses,
            'gates': assessment['gates'], 'FA_r10': assessment['FA_r10'],
            'FA_hybrid': assessment['FA_hybrid'], 'passed': assessment['passed']}


def run(plan, model, normal_path, public_path, calibration_sha):
    normal_path, public_path = Path(normal_path), Path(public_path)
    normal = json.loads(normal_path.read_text())
    public = json.loads(public_path.read_text())
    if (normal.get('schema') != 'r14-stock-handoff-normal.v1'
            or normal.get('stage') != 'normal' or normal.get('passed') is not True
            or public.get('schema') != 'r14-registered-public-normal-gates.v1'
            or public.get('passed') is not True
            or public.get('normal_sha256') != sha(normal_path)
            or public.get('calibration_sha256') != calibration_sha):
        raise ValueError('same-invocation authored and public normal gates must pass')
    rows = [impact_case(case, model) for case in plan['impact_cases']]
    return {
        'schema': 'r14-registered-paired-impact.v1', 'stage': 'impact',
        'rows': rows, 'passed': all(row['passed'] for row in rows),
        'normal_sha256': sha(normal_path), 'public_gates_sha256': sha(public_path),
        'calibration_sha256': calibration_sha,
        'scope': ('fresh matched no-impact/forced stock episodes; R10 and hybrid '
                  'comparators; derived records only'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--normal-result', type=Path, required=True)
    parser.add_argument('--public-gates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error('fresh impact output required')
    plan = json.loads((qualification.HERE/'plan.json').read_text())
    model, calibration_sha, _ = qualification.calibration(args.calibration)
    result = run(plan, model, args.normal_result, args.public_gates, calibration_sha)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'stage': 'impact', 'passed': result['passed'],
                      'failed': [row['id'] for row in result['rows'] if not row['passed']]}))


if __name__ == '__main__':
    main()
