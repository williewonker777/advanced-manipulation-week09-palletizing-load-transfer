"""Score the previously disclosed R22 challenge; it is not a new independent set."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments.r22_challenge import evaluator as old
from experiments.r22_light.candidate import Detector as Light
from experiments.r23_detached.candidate import Detector, Free


# The unchanged scoring engine expects these three slot names. Label its real
# detector instances explicitly here and rename every per-arm record on return.
ARMS = {'r10': Light, 'r21_yaw': Free, 'r22': Detector}
NAMES = {'r10': 'light', 'r21_yaw': 'free', 'r22': 'onset'}


def events(record, kind):
    if kind == 'single':
        return {'single': record['event']}
    return record['events'] if kind == 'paired' else {}


def false_lists(record, kind):
    if kind == 'single':
        return {'none': record['none_false_s'], 'force': record['force_false_s']}
    return record['false_s'] if kind == 'paired' else {'none': record['false_times_s']}


def strict(event):
    return event['quiet_before'] and event['exactly_one']


def compare(current, reference, kind):
    new, prior = events(current, kind), events(reference, kind)
    extra = lambda event: max(0, len(event['raw_rise_indices']) - 1)
    result = {
        'no_strict_loss_or_delay_over_5ms': all(not strict(old_event) or (
            strict(new[slot]) and new[slot]['delays_ms'][0] <= old_event['delays_ms'][0] + 5)
            for slot, old_event in prior.items()),
        'no_extra_duplicates': all(extra(new[k]) <= extra(prior[k]) for k in prior),
        'no_added_pre_onset_alarm': all(not new[k]['high_before'] or prior[k]['high_before'] for k in prior),
        'no_added_false': all(old.match_false_events(seq, false_lists(reference, kind)[k])['passed']
                              for k, seq in false_lists(current, kind).items()),
    }
    if 'tool' in reference:
        a, b = current['tool'], reference['tool']
        result['tool_presence_matched'] = a['present'] == b['present']
        result['no_tool_loss'] = not b['detected'] or (a['detected'] and a['delay_ms'] <= b['delay_ms'] + 5)
        result['no_added_tool_prealarm'] = not a['prealarm'] or b['prealarm']
    if kind == 'negative':
        result['no_added_tail'] = not current['stuck_tail'] or reference['stuck_tail']
    else:
        result['no_added_tail'] = all(not value or reference['stuck_tail'][k]
                                      for k, value in current['stuck_tail'].items())
    return result


def one(args):
    kind, case, variant, plan = args
    row = old.one((kind, case, variant, ARMS, plan))
    rec = {NAMES[key]: value for key, value in row['records'].items()}
    result = {'id': case['id'], 'variant': variant['kind'], 'kind': kind,
              'episodes': row['episodes'], 'physical_sha256': row['r10_physics_sha256'],
              'input_certified': row['comparison']['physical_and_force_exact']
              and all(v for k, v in row['comparison'].items() if 'prefix' in k),
              'records': rec, 'guards': {a: compare(rec[a], rec['light'], kind) for a in ('free', 'onset')}}
    return result


def summarize(rows, arm):
    singles = [row for row in rows if row['kind'] == 'single']
    pairs = [row for row in rows if row['kind'] == 'paired']
    return {'single': sum(strict(row['records'][arm]['event']) for row in singles),
            'single_n': len(singles),
            'pair_slots': sum(sum(strict(e) for e in row['records'][arm]['events'].values()) for row in pairs),
            'pair_n': len(pairs),
            'complete_pair': sum(all(strict(e) for e in row['records'][arm]['events'].values()) for row in pairs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', choices=('target','full'), required=True)
    ap.add_argument('--workers', choices=(1,2,3), type=int, default=2)
    ap.add_argument('--out', required=True, type=Path); a = ap.parse_args(argv)
    if a.out.exists() or a.out.is_symlink():
        raise ValueError('new output required')
    plan = json.loads(old.PLAN.read_bytes())
    jobs = [(k,c,v,plan) for k in ('negative','single','paired') for c in plan[k]
            for v in plan['variants'].values() if a.panel == 'full' or c['id'] == 's_after_release_diag']
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(one, jobs))
    episodes = sum(r['episodes'] for r in rows)
    if (len(rows), episodes) != ((66,486) if a.panel=='full' else (3,18)):
        raise ValueError('incomplete panel')
    doc = {'schema':'r23-exposed-old-cohort.v1','panel':a.panel,'unseen':False,'episodes':episodes,
           'old_plan_sha256':old.sha(old.PLAN), 'all_input_certified':all(r['input_certified'] for r in rows),
           'totals':{n:summarize(rows,n) for n in NAMES.values()},
           'regressions':{a:[(r['variant'],r['id'],[k for k, v in r['guards'][a].items() if not v])
                             for r in rows if not all(r['guards'][a].values())] for a in ('free','onset')},
           'rows':rows}
    with a.out.open('x') as f:
        json.dump(doc, f, indent=2, allow_nan=False); f.write('\n')
    print({k:doc[k] for k in ('panel','episodes','all_input_certified','totals','regressions')})


if __name__=='__main__':
    main()
