import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from experiments.r18_single_pass import paired_replay
from experiments.r19_close_pairs import assay
from experiments.r20_force_certificate import certificate
from experiments.r21_surface_pairs.study import KEYS, local_episode
from experiments.r21_yaw_continuity.candidate import Detector as Continuous
from experiments.r21_yaw_impulse.coarse import Detector as Reset


def pair(spec):
    case, mode = spec
    first, second = case['impacts']
    assignment = {'none': (None, None), 'first': (first, None), 'second': (None, second), 'both': (first, second)}
    records, traces, scores = {}, {}, {}
    for arm, typ in (('reset', Reset), ('continuous', Continuous)):
        logs = {k: local_episode(case, mode, typ, ab) for k, ab in assignment.items()}
        traces[arm] = logs
        records[arm], raw = assay.characterize(case, logs, 20, 320)
        scores[arm] = paired_replay.signature(raw)
    r, c = records['reset'], records['continuous']
    exact = all(certificate.exact_equal({k: log[v] for k, v in KEYS.items()}, certificate.exact_trace(case, twin))
                for logs in traces.values() for twin, log in logs.items())
    equal = all(np.array_equal(traces['reset'][tw][k], traces['continuous'][tw][k])
                for tw in assignment for k in paired_replay.PHYSICAL + ('r',))
    less = all(len(c['slots'][k]['raw_delays_ms']) <= len(r['slots'][k]['raw_delays_ms']) for k in assay.SLOTS)
    retain = assay.all_comparisons([assay.compare(c, r)])
    strict = {k: exact and assay.one_per_pulse(v) and all(val for gate, val in v['r18_gates'].items()
                                                           if gate != 'signed_profiles_exact')
              for k, v in records.items()}
    gates = {'exact_force': exact, 'physics_and_residual_unchanged': equal, 'retain_genuine_quiet_false_tool': retain,
             'no_added_duplicate': less, 'preserve_prior_strict': not strict['reset'] or strict['continuous'],
             'strict_new': strict['continuous']}
    return {'id': case['id'], 'mode': mode, 'episodes': 8, 'gates': gates, 'passed': all(gates.values()), 'strict': strict,
            'scores': scores, 'slots': {k: v['slots'] for k, v in records.items()}}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('new output file required')
    here = Path(__file__).resolve().parent
    known = json.loads((here.parent / 'r21_surface_pairs/plan.json').read_bytes())['cases']
    fresh = json.loads((here / 'plan.json').read_bytes())['fresh']
    result = {}
    with ProcessPoolExecutor(2) as pool:
        for stage, cases in [('exposed', known), ('fresh', fresh)]:
            rows = list(pool.map(pair, [(case, mode) for case in cases for mode in ('stock', 'changed')]))
            result[stage] = {'passed': all(r['passed'] for r in rows), 'rows': rows}
            if not result[stage]['passed']:
                break
    with a.out.open('x') as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
        fp.write('\n')
    print({k: v['passed'] for k, v in result.items()})
