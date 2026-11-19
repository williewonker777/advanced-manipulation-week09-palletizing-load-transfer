"""Qualify no-force trials first; replay existing R21 weak stimuli only on pass."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r21_weak_challenge import evaluator as ev
from experiments.r21_yaw_impulse.coarse import Detector as Yaw
from experiments.r22_tangent.frozen import Detector as Candidate

ARMS = {'r10': ev.STATIC['r10'], 'r21': Yaw, 'r22_tangent': Candidate}


def neg(case):
    return ev.negative_row(case, ARMS)


def forced(case):
    row = ev.forced_row(case, ARMS, 20)
    old = row['arms']['r21']; new = row['arms']['r22_tangent']
    t = case['time']
    win = lambda arm: sum(t <= v < t + .32 for v in arm['force']['all_rise_times'])
    row['vs_yaw'] = {
        'preserve': not old['force']['strict_true_positive'] or new['force']['strict_true_positive'],
        'unmatched_normal_false': ev.unmatched(new['none']['false_rise_times'], old['none']['false_rise_times']),
        'unmatched_force_false': ev.unmatched(new['force']['false_rise_times'], old['force']['false_rise_times']),
        'duplicate_change': max(0,win(new)-1) - max(0,win(old)-1),
        'candidate_only': new['force']['strict_true_positive'] and not old['force']['strict_true_positive']}
    return row


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); a=p.parse_args()
    src = Path(__file__).resolve().parent
    plan = json.loads((src/'plan.json').read_bytes())
    cases = [{'id':f'normal{i}','role':'audit','mass':m,'com':c,'pick_offset':None} for i,(m,c) in enumerate(plan['qualification'])]
    with ProcessPoolExecutor(2) as pool:
        normals = list(pool.map(neg,cases))
        ok = all(r['passed'] and not ev.unmatched(r['arms']['r22_tangent']['false_rise_times'],
                                                   r['arms']['r21']['false_rise_times']) for r in normals)
        result = {'normal_gate':ok,'normal':normals,'forced':[]}
        if ok:
            known = json.loads((src.parent/'r21_weak_challenge/plan.json').read_bytes())
            result['forced'] = list(pool.map(forced, known['forced']))
    with a.out.open('x') as fp:json.dump(result,fp,indent=2,allow_nan=False);fp.write('\n')
    print('normal gate',ok,'forced',len(result['forced']),'gains',
          [r['id'] for r in result['forced'] if r['vs_yaw']['candidate_only']])
