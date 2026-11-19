"""Input-certified R22 comparison against both R10 and R21; no case-specific detector logic."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

import my_detector
from experiments.r21_full_protection import force_audit, runner_v2 as old
from experiments.r21_yaw_impulse.coarse import Detector as Yaw
from tools.check_integrity import check_integrity
from tools.d4_robustness import build_cases

ROOT = Path(__file__).resolve().parents[2]
MATRIX = 'bc4b3dbb925cfb8609f535e1420f8b3f20785769b7f339708ca914b6992db791'
CACHE = 'c62dd430d7f3b9a4553c0562558f481a4518a893050d1cf8b28e82d3c6c44fcf'
COHORT = {174, 330, 339, 349}
TIME = np.arange(6000, dtype=float) * .001
TIME_PIN = old.array_sha(TIME)


def unmatched_indices(new, old_indices, radius=1):
    """Maximum-cardinality monotone one-to-one pairing; duplicates cannot reuse a reference."""
    ref = sorted(old_indices)
    j = 0
    unmatched = []
    for x in sorted(new):
        while j < len(ref) and ref[j] < x - radius:
            j += 1
        if j < len(ref) and ref[j] <= x + radius:
            j += 1
        else:
            unmatched.append(x)
    return unmatched


@lru_cache(maxsize=196)
def schedule(t, force):
    signed = (np.zeros((len(TIME), 3)) if t is None else
              force_audit.independent_expected(TIME, (t, force)))
    if not np.array_equal(signed, old.expected_signed(TIME, None if t is None else (t, force))):
        raise ValueError('independent signed schedules differ')
    scalar = np.fromiter((np.linalg.norm(v) for v in signed), dtype=float, count=len(TIME))
    active = np.flatnonzero(scalar > 0)
    return (old.array_sha(scalar), int(active[0]) if len(active) else None,
            int(active[-1]) if len(active) else None)


def score(case, arm, normal):
    col = case['collision']
    fhash, first, last = schedule(None if col is None else col[0], None if col is None else tuple(col[1]))
    ready = arm['complete_fixed_grid'] and not arm['metrics']['diverged'] and arm['sample_count'] == 6000
    port = arm['physical_sha256']
    valid = {'complete': bool(ready and port['t'] == TIME_PIN), 'actual_signed': arm['signed_expected_exact'],
             'per_tick_scalar': port['f_ext'] == fhash,
             'active_force_bounds': first == arm['force_first_nonzero_index'] and last == arm['force_last_nonzero_index'],
             'prefix': True, 'attribution_window': True}
    onset = arm['onset_index']
    if col is not None:
        nominal_time = int(np.searchsorted(TIME, col[0]))
        if onset != nominal_time or first not in (onset, onset+1):
            raise ValueError('unexpected onset index')
        end = np.flatnonzero((TIME >= col[0]) & (TIME < col[0]+.32))[-1]
        valid['attribution_window'] = bool(arm['true_window_first_index'] == onset and arm['true_window_last_index'] == end)
        mode = 'before_onset' if first == onset else 'through_zero_force_onset'
        a = arm['prefix_sha256'][str(onset)][mode]
        b = normal['prefix_sha256'][str(onset)][mode]
        valid['prefix'] = set(a) == set(b) == set(old.PREFIX) and a == b
    rises = arm['artificial_rise_indices']
    genuine = [k for k in rises if not old.has_twin(normal, k)]
    strict = bool(col is not None and not arm['masked_at_onset'] and genuine)
    return {'valid':valid, 'strict':strict,
            'delay_ms':int(genuine[0]-onset) if strict else None,
            'duplicate':max(0,len(genuine)-1),'raw_duplicate':max(0,len(rises)-1),
            'false':arm['false_rise_indices'],'delayed_false':arm['delayed_false_indices'],
            'masked':arm['masked_at_onset'],'stuck':bool(arm['detect_intervals'] and arm['detect_intervals'][-1][1]==6000),
            'tool':arm['tool']}


def compare(a, ref):
    return {
        'strict_positive_retained': not ref['strict'] or a['strict'],
        'no_new_false': not unmatched_indices(a['false'],ref['false']),
        'no_new_delayed_false': not unmatched_indices(a['delayed_false'], ref['delayed_false']),
        'no_new_mask': not a['masked'] or ref['masked'],
        'no_new_duplicate': a['duplicate'] <= ref['duplicate'] and a['raw_duplicate'] <= ref['raw_duplicate'],
        'tool_retained': not ref['tool']['detected'] or a['tool']['detected'],
        'no_new_tool_prealarm': not a['tool']['prealarm'] or ref['tool']['prealarm'],
        'no_new_stuck_tail': not a['stuck'] or ref['stuck']}


def candidate(label):
    if label == 'history':
        from experiments.r21_yaw_continuity.candidate import Detector
    elif label == 'guarded':
        from experiments.r22_guarded.candidate import Detector
    elif label == 'light':
        from experiments.r22_light.candidate import Detector
    elif label == 'core':
        from experiments.r22_combined.core import Detector
    elif label == 'integrated':
        from experiments.r22_combined.candidate import Detector
    else:
        raise ValueError('unknown registered policy')
    return Detector


def simulate(args):
    case, label, with_refs = args
    types = {'candidate':candidate(label)}
    if with_refs:types.update({'r10':my_detector.Detector,'r21':Yaw})
    return {k:old.episode_record(det,case) for k,det in types.items()}


def analyze(cases, records):
    refs = {old.reference_key(c):r for c,r in zip(cases,records) if c['kind']=='reference'}
    rows=[]
    for c,r in zip(cases,records,strict=True):
        ref = refs[old.reference_key(c)] if c['kind']=='collision' else r
        s={k:score(c,v,ref[k]) for k,v in r.items()}
        physical = all(v['physical_sha256']==r['r10']['physical_sha256'] for v in r.values())
        vs={a:compare(s['candidate'],s[a]) for a in ('r10','r21')}
        rows.append({'id':c['case_id'],'kind':c['kind'],'mass':c['mass'],'scores':s,'vs':vs,'physical':physical,
                     'input_certified':physical and all(all(a['valid'].values()) for a in s.values()),
                     'protection':all(all(gates.values()) for gates in vs.values())})
    summary = {}
    cols=[r for r in rows if r['kind']=='collision']
    for k in ('r10','r21','candidate'):
        summary[k]={'strict':sum(r['scores'][k]['strict'] for r in cols),
                    'false':sum(len(r['scores'][k]['false']) for r in rows),
                    'duplicate':sum(r['scores'][k]['duplicate'] for r in cols),
                    'tool_detected':sum(r['scores'][k]['tool']['detected'] for r in rows)}
    delta={r['id']:r['scores']['candidate']['delay_ms']-r['scores']['r21']['delay_ms'] for r in cols
           if r['scores']['candidate']['strict'] and r['scores']['r21']['strict']}
    return {'cases':len(rows),'collisions':len(cols),'summary':summary,
            'input_failures':[r['id'] for r in rows if not r['input_certified']],
            'protection_failures':[r['id'] for r in rows if not r['protection']],
            'gains_vs_r21':[r['id'] for r in cols if r['scores']['candidate']['strict'] and not r['scores']['r21']['strict']],
            'losses_vs_r21':[r['id'] for r in cols if not r['scores']['candidate']['strict'] and r['scores']['r21']['strict']],
            'signed_latency_change_vs_r21_ms':delta, 'rows':rows}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('history','core','integrated','guarded','light'),required=True)
    p.add_argument('--cohort',choices=('four','six','full'),default='full')
    p.add_argument('--pinned-r21-cache',type=Path)
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():p.error('new output required')
    verify=check_integrity(ROOT)
    if verify['status']!='ok':raise ValueError('original kit integrity')
    all_cases=build_cases('my_torque'); computed=hashlib.sha256(old.canonical_cases(all_cases)).hexdigest()
    if computed!=MATRIX or len(all_cases)!=385:raise ValueError('reference matrix differs')
    if a.cohort in ('four','six'):
        focus=COHORT if a.cohort=='four' else COHORT|{319,335}
        needed={old.reference_key(c) for c in all_cases if c['case_id'] in focus and c['kind']=='collision'}
        cases=[c for c in all_cases if c['case_id'] in focus or c['kind']=='reference' and old.reference_key(c) in needed]
    else:cases=all_cases
    prior=None
    if a.pinned_r21_cache is not None:
        if old.sha(a.pinned_r21_cache)!=CACHE:raise ValueError('noncanonical reference cache')
        raw=json.loads(a.pinned_r21_cache.read_bytes())
        if raw['episodes']!=770 or [c['case_id'] for c in all_cases]!=[r['case']['case_id'] for r in raw['rows']]:
            raise ValueError('cached cohort differs')
        prior={r['case']['case_id']:r['arms'] for r in raw['rows']}
    with ProcessPoolExecutor(a.workers) as pool:
        records=list(pool.map(simulate,((c,a.arm,prior is None) for c in cases)))
    if prior:
        for c,r in zip(cases,records):
            baseline=prior[c['case_id']];r.update(r10=baseline['r10'],r21=baseline['yaw'])
    result=analyze(cases,records)
    doc={'schema':'r22-d4-input-certified.v1','arm':a.arm,'cohort':a.cohort,'matrix_sha256':MATRIX,
         'protected_kit':verify,'reference_cache_sha256':None if not prior else CACHE,
         'simulated_episodes':len(cases)*(1 if prior else 3),'candidate_promoted':False,**result}
    with a.out.open('x') as f:json.dump(doc,f,indent=2,allow_nan=False);f.write('\n')
    print({k:doc[k] for k in ('arm','simulated_episodes','summary','input_failures','protection_failures')})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
