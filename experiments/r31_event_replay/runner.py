"""Disclosed D4 cases and actual matched controls; no stored candidate outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r22_census import check
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r30_yaw_headroom.candidate import Detector as Headroom
from experiments.r31_held_identity.candidate import Detector as Identity


ARMS = {'c3': C3, 'headroom': Headroom, 'identity': Identity}
CASES = {'examples': (266, 287, 298, 316),
         'all-duplicates': (266, 269, 270, 276, 279, 280, 286, 287, 289, 290, 297, 298, 308, 316)}


def select(cases, ids):
    references = {check.old.reference_key(c): c for c in cases if c['kind'] == 'reference'}
    by_id = {c['case_id']: c for c in cases}
    selected = set(ids)
    for i in ids:
        if by_id[i]['kind'] != 'collision':
            raise ValueError('expected collision')
        selected.add(references[check.old.reference_key(by_id[i])]['case_id'])
    return [by_id[i] for i in sorted(selected)], references


def record(c):
    return c['case_id'], {key: check.old.episode_record(cls, c) for key, cls in ARMS.items()}


def audit(new, old):
    s, ref = new['score'], old['score']
    guards = check.compare(s, ref)
    guards['latency_plus_5_ms'] = not ref['strict'] or (s['strict'] and s['delay_ms'] <= ref['delay_ms'] + 5)
    guards['duration_plus_5_samples'] = not ref['strict'] or new['active_samples'] <= old['active_samples'] + 5
    guards['tool_time_plus_5_ms'] = not ref['tool']['detected'] or (
        s['tool']['detected'] and new['first_tool_rise'] <= old['first_tool_rise'] + 5)
    return [k for k,v in guards.items() if not v]


def run(scope, jobs):
    cases = check.build_cases('my_torque')
    if hashlib.sha256(check.old.canonical_cases(cases)).hexdigest() != check.MATRIX:
        raise ValueError('changed input matrix')
    if scope == 'full-d4':
        if len(cases) != 385:
            raise ValueError('incomplete full input matrix')
        references = {check.old.reference_key(c): c for c in cases if c['kind'] == 'reference'}
    else:
        cases, references = select(cases, CASES[scope])
    if jobs == 1:
        records = dict(map(record, cases))
    else:
        with ProcessPoolExecutor(jobs) as pool:
            records = dict(pool.map(record, cases))
    rows = []
    for c in cases:
        cid = c['case_id']
        own = records[cid]
        ref = references[check.old.reference_key(c)]['case_id'] if c['kind'] == 'collision' else cid
        if any(r['physical_sha256'] != own['c3']['physical_sha256'] for r in own.values()):
            raise ValueError('different trajectories')
        arms = {}
        for name, r in own.items():
            score = check.score(c, r, records[ref][name])
            # Stock logs a scalar norm one vector at a time. check.score hashes
            # precisely that independently reconstructed per-tick scalar, AND
            # verifies every observed signed input; vectorized norms may round differently.
            if not all(score['valid'].values()):
                raise ValueError('incomplete, wrong signed force, or different before onset')
            arms[name] = {'score': score, 'alarm_intervals': r['detect_intervals'],
                          'batched_scalar_bit_identical': r['stock_scalar_matches_signed'],
                          'active_samples': sum(b-a for a,b in r['detect_intervals']),
                          'first_tool_rise': min(r['tool_rise_indices'], default=None)}
        rows.append(dict(id=cid, input=c, control_id=ref, arms=arms, physics_sha256=own['c3']['physical_sha256']))
    totals = {n: dict(strict=sum(r['arms'][n]['score']['strict'] for r in rows),
                      duplicate=sum(r['arms'][n]['score']['duplicate'] for r in rows),
                      raw_duplicate=sum(r['arms'][n]['score']['raw_duplicate'] for r in rows))
              for n in ARMS}
    floors = {key: [{'id': r['id'], 'failed': failures} for r in rows
                    if (failures := audit(r['arms']['identity'], r['arms'][key]))]
              for key in ('c3', 'headroom')}
    return dict(schema='r31-disclosed-physical-replay.v1', guards_vs=floors, scope=scope, course_original=False,
                default_changed=False, arms=list(ARMS), episodes=len(rows)*len(ARMS),
                controller='my_torque', matrix_sha256=check.MATRIX, totals=totals, rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scope', choices=(*CASES, 'full-d4'))
    p.add_argument('--jobs', type=int, choices=range(1, 7), default=1)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.out.is_symlink():
        p.error('fresh output required')
    with a.out.open('x') as f:
        json.dump(run(a.scope, a.jobs), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
