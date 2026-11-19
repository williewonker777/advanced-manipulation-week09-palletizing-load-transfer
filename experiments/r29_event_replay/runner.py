"""Evaluate the original force/no-force/isolated/both/long twins, not precomputed results."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from experiments.r26_holdout import evaluator as lab
from experiments.r26_yaw_acquisition.candidate import Detector as C3
from experiments.r29_detached_release.candidate import Detector as Endpoint
from experiments.r29_disjoint_events.candidate import Detector as Composed
from experiments.r29_native_event_link.candidate import Detector as Linked
from experiments.r29_loaded_rising_certificate.candidate import Detector as RisingAudit

HERE = Path(__file__).resolve().parent
ARMS = {'released': Endpoint, 'linked': Linked, 'composed': Composed, 'rising-audit': RisingAudit}
SCOPES = ('known-detached', 'known-place', 'known-full', 'prospective-detached')


def canon(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def one(job):
    kind, case, mech, plan, arm, anchor = job
    row = lab.one((kind, case, mech, {'r25_joint': C3, 'r26': ARMS[arm]}, plan))
    if anchor is not None and (row['r25_physics_sha256'] != anchor['physics'] or
                               canon(row['records']['r25_joint']) != anchor['c3_record']):
        raise ValueError('original R26 C3 result/physical trace did not reproduce')
    for record in row['records'].values():
        if (not record['physical_exact_vs_r25']
                or not all(v['passed'] for k, v in record.items() if k.startswith('prefix'))
                or not all(v['force_exact']['passed'] for v in record.get('twins', {'none':record}).values())):
            raise ValueError('physical or causal integrity failure')
    row['records'] = {'c3':row['records']['r25_joint'], arm:row['records']['r26']}
    return row


def execute(scope, arm, workers):
    if arm not in ARMS or scope not in SCOPES or (scope == 'prospective-detached' and arm != 'released'):
        raise ValueError('new tests were registered for the standalone released arm only')
    plan = json.loads(lab.PLAN.read_bytes())
    anchors = json.loads((HERE/'anchors.json').read_bytes())
    if hashlib.sha256(lab.PLAN.read_bytes()).hexdigest() != anchors['plan_sha256']:
        raise ValueError('incorrect historical case plan')
    prospective = scope == 'prospective-detached'
    if prospective:
        spec = json.loads((HERE.parent/'r29_detached_release/prospective.json').read_bytes())
        cases = [('paired', c) for c in spec['paired']]
        mechanics = [plan['variants'][m] for m in spec['mechanics']]
    else:
        cases = [(kind, c) for kind in ('negative','single','paired') for c in plan[kind]
                 if scope == 'known-full' or c['id'] == {'known-detached':'p26_close_detached_reverse',
                                                         'known-place':'p26_close_place_cross'}[scope]]
        mechanics = list(plan['variants'].values())
    tasks = [(k,c,m,plan,arm,None if prospective else anchors['rows'][f"{k}:{c['id']}:{m['kind']}"])
             for k,c in cases for m in mechanics]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(one, tasks))
    return dict(schema='r29-disclosed-torque-physical-replay.v1', scope=scope, arm=arm,
                episodes=sum(r['episodes'] for r in rows), rows=rows,
                runtime=lab.require_runtime(), default_changed=False, course_demo=False,
                disposition='rejected_on_signed_guards' if arm=='rising-audit' else 'research_only')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--scope', choices=SCOPES, required=True)
    p.add_argument('--jobs', choices=(1,2), type=int, default=1); p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    if a.out.exists() or a.out.is_symlink(): p.error('fresh output required')
    doc = execute(a.scope, a.arm, a.jobs)
    with a.out.open('x') as f:
        json.dump(doc, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(doc['scope'],doc['arm'],doc['episodes'])


if __name__ == '__main__':
    main()
